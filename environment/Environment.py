"""A trading environment"""
from typing import Optional
import dataclasses

import gym
from gym import spaces
from acme.utils import loggers
from absl import flags
FLAGS = flags.FLAGS
import copy
from acme.utils import loggers

import numpy as np

from environment.Trading import MainPortfolio, Greek, SwapKeys

#@dataclasses.dataclass
#class StepResult:
    # """Logging step metrics for analysis
    # """
    # episode: int = 0
    # t: int = 0
    # hed_action: float = 0.
    # #hed_share: float = 0.
    # action_swaption_hed: float = 0.
    # position_swaption_liab: float = 0.
    # bound_low : float = 0.
    # bound_high : float = 0.
    # # action_swap_hed: float = 0.
    # # action_swap_liab: float = 0.
    # # swap_price: float = 0.
    # # swap_position: float = 0.
    # # swap_pnl: float = 0.
    # # liab_port_gamma: float = 0.
    # # liab_port_vega: float = 0.
    # # liab_port_pnl: float = 0.
    # # hed_cost: float = 0.
    # # hed_port_gamma: float = 0.
    # # hed_port_vega: float = 0.
    # # hed_port_pnl: float = 0.
    # # gamma_before_hedge: float = 0.
    # # gamma_after_hedge: float = 0.
    # # vega_before_hedge: float = 0.
    # # vega_after_hedge: float = 0.
    # # step_pnl: float = 0.
    # # state_price_hed: float = 0.
    # # state_price_liab: float = 0.
    # # state_delta: float = 0.
    # # state_delta_local_hed: float = 0.
    # # state_delta_local_liab: float = 0.
    # # state_hed_delta: float = 0.
    # # state_liab_delta: float = 0.
    # # state_gamma: float = 0.
    # state_gamma_local: float = 0.
    # state_hed_gamma: float = 0.
    # # state_vega: float = 0.
    # state_vega_local: float = 0.
    # state_hed_vega: float = 0.
    #pass
@dataclasses.dataclass
class StepResult:
    episode: int = 0
    t: int = 0

    # === Agent output and hedge taken ===
    hed_action: float = 0.  # raw action[0] ∈ [0,1]
    bound_low: float = 0.
    bound_high: float = 0.
    gamma_hedge_unit: float = 0.  # gamma of the hedge instrument (per unit)
    action_swaption_hed: float = 0.  # notional traded for gamma hedge (computed)

    # === Gamma before and after hedge ===
    gamma_before_hedge: float = 0.  # total portfolio gamma before hedge
    gamma_after_hedge: float = 0.  # total portfolio gamma after hedge
    liab_port_gamma: float = 0.  # gamma from liability-side swaptions
    hed_port_gamma: float = 0.  # gamma from hedging book swaptions


    # === State features (gamma info seen by the agent) ===
    state_gamma_local: float = 0.  # portfolio gamma (possibly kernel-weighted)
    state_hed_gamma: float = 0.  # gamma of the hedge instrument (per unit)
    position_swaption_liab: float = 0.




class TradingEnv(gym.Env):
    """
    This is the Gamma & Vega Trading Environment.
    """

    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day;
    def __init__(self, utils, logger: Optional[loggers.Logger] = None):

        super(TradingEnv, self).__init__()
        self.logger = logger
        # seed and start
        self.seed(utils.seed)

        # simulated data: array of asset price, option price and delta paths (num_path x num_period)
        # generate data now
        self.portfolio = MainPortfolio(utils)
        self.utils = utils

        # other attributes
        #self.num_path = self.portfolio.a_price.shape[0]
        self.num_path = self.portfolio.hed_port.base_options.shape[0]
        # set num_period: initial time to maturity * daily trading freq + 1 (see get_sim_path() in utils.py)
        self.num_period = self.portfolio.hed_port.base_options.shape[1]

        self.sim_episode = -1

        # track time step within an episode (it's step)
        self.t = None

        # time to maturity array
        #self.ttm_array = np.arange(self.utils.init_ttm, -self.utils.frq, -self.utils.frq)
        # self.ttm_array = utils.ttm_mat TODO: check if this is used anywhere
        # Action space: HIGH value has to be adjusted with respect to the option used for hedging
        # self.action_space = spaces.Box(low=np.array([0]), 
        #                                high=np.array([1.0]), dtype=np.float32)
        self.action_space = spaces.Box( # swaption hedge, 1y swap hedge, 2y swap hedge
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        max_gamma = self.portfolio.liab_port.max_gamma
        max_vega = self.portfolio.liab_port.max_vega
        # minimum price per expiry across all episodes
        min_price_swap_hed = np.min(self.portfolio.underlying.swap_data_hed)
        min_price_swap_liab = np.min(self.portfolio.underlying.swap_data_liab)
        max_price_swap_hed = np.max(self.portfolio.underlying.swap_data_hed)
        max_price_swap_liab = np.max(self.portfolio.underlying.swap_data_liab)

        obs_lowbound = np.array([ min_price_swap_hed,
                                    min_price_swap_liab,
                                    -np.inf, 
                                    -np.inf, 
                                    -np.inf, 
                                    -np.inf,
                                    -np.inf,
                                 -1 * max_gamma , 
                                 -np.inf,
                                 -np.inf])
        obs_highbound = np.array([      
                                    max_price_swap_hed,  
                                    max_price_swap_liab,                               
                                    np.inf, 
                                    np.inf, 
                                    np.inf, 
                                    np.inf,
                                    np.inf,
                                  max_gamma ,
                                    np.inf,
                                  np.inf])
        # concat prices  and gamma

        if FLAGS.vega_obs:
            obs_lowbound = np.concatenate([obs_lowbound, [-1 * max_vega ,
                                                          -np.inf,
                                                          -np.inf]])
            obs_highbound = np.concatenate([obs_highbound, [max_vega,
                                                            np.inf,
                                                            np.inf]])
        obs_highbound = np.concatenate([obs_highbound, [utils.num_period]]) # t
        obs_lowbound = np.concatenate([obs_lowbound, [0]]) # t   
        self.observation_space = spaces.Box(low=obs_lowbound,high=obs_highbound, dtype=np.float32)
        
        # Initializing the state values
        #self.num_state = 5 if FLAGS.vega_obs else 3
        # modified to include time, delta sum, and delta vector
        self.num_state = len(obs_highbound)
        self.state = []


        # was commented out in the original code
        # self.reset()

    def seed(self, seed):
        # set the np random seed
        np.random.seed(seed)

    def reset(self):
        """
        reset function which is used for each episode (spread is not considered at this moment)
        """

        # repeatedly go through available simulated paths (if needed)
        self.sim_episode = (self.sim_episode + 1) % self.num_path
        self.portfolio.reset(self.sim_episode)

        self.t = 0

        self.portfolio.liab_port.add(self.sim_episode, self.t, -1) # just primes the episode

        return self.portfolio.get_state(self.t)

    def step(self, action):
        """
        profit and loss period reward
        """
        result = StepResult(
            episode=self.sim_episode,
            t=self.t,
            hed_action=action,
        )
        # action constraints
        # gamma_action_bound = -self.portfolio.get_gamma(self.t)/self.portfolio.hed_port.options[self.sim_episode, self.t].gamma_path[self.t]/self.utils.contract_size
        # action_low = [0, gamma_action_bound]
        # action_high = [0, gamma_action_bound]
        
        # if FLAGS.vega_obs:
        #     # vega bounds
        #     vega_action_bound = -self.portfolio.get_vega(self.t)/self.portfolio.hed_port.options[self.sim_episode, self.t].vega_path[self.t]/self.utils.contract_size
        #     action_low.append(vega_action_bound)
        #     action_high.append(vega_action_bound)

        # low_val = np.min(action_low)
        # high_val = np.max(action_high)

        # hed_share = low_val + action[0] * (high_val - low_val)
        t = self.t
        ep = self.sim_episode
        hed_option = self.portfolio.hed_port.base_options
        liab_option = self.portfolio.liab_port.base_options  
        #contract_size = self.utils.contract_size

        # === Gamma hedge ===
        gamma_hedge_unit = hed_option[ep, t, t, Greek.GAMMA]  # gamma for swaption to be traded
        gamma_liab_unit = liab_option[ep, t, t, Greek.GAMMA]  # not used as liab is already added before this
        portfolio_gamma = self.portfolio.get_gamma_local(t)

        gamma_action_bound = -portfolio_gamma / (gamma_hedge_unit)
        result.gamma_hedge_unit = gamma_hedge_unit

        action_low = [0,gamma_action_bound]
        action_high = [0,gamma_action_bound]

        # === Vega hedge ===
        if FLAGS.vega_obs:
            vega_hedge_unit = hed_option[ep, t, t, Greek.VEGA]  # vega for swaption to be traded
            portfolio_vega = self.portfolio.get_vega_local(t) 

            vega_action_bound = -portfolio_vega / (vega_hedge_unit)
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)

        # === Delta hedge ===
        delta_hedge_unit = hed_option[ep, t, t, Greek.DELTA]  
        delta_liab_unit = liab_option[ep, t, t, Greek.DELTA]
        


        #portfolio_delta = self.portfolio.get_delta(t) # array of deltas for all options in the portfolio

        



        # === Hedge scaling from normalized action ===
        low_val = np.min(action_low)
        high_val = np.max(action_high)
        result.bound_low = low_val
        result.bound_high = high_val
        action_swaption_hedge = low_val + action[0] * (high_val - low_val)
        if t == 2:
            print(f"gamma_bound: {gamma_action_bound:.3f}, hed_action: {action[0]:.3f}, hedge: {action_swaption_hedge:.3f}")
        



        # action[1] bound
        # delta that is added by the hedging swaption
        delta_swaption_offset_hed =  delta_hedge_unit * action_swaption_hedge # delta added by the swaption traded in this period
        # effect of the hedging swaption 
        delta_hed_local = self.portfolio.get_delta_local_hed(t)
        delta_hed_total = delta_hed_local + delta_swaption_offset_hed
        
        
        action_swap_hedge =  -action[1] * delta_hed_total/self.portfolio.underlying.active_path_hed[t, t, SwapKeys.DELTA] # divide to scale by the delta sensitivity of the underlying

        # action[2] bound
        delta_swaption_offset_liab = delta_liab_unit # delta added by the swaption traded in this period
        delta_liab_local = self.portfolio.get_delta_local_liab(t)
        delta_liab_total = delta_liab_local + delta_swaption_offset_liab # local delta + delta from liab
        action_swap_liab =  - action[2] * delta_liab_total//self.portfolio.underlying.active_path_liab[t, t, SwapKeys.DELTA] # divide to scale by the delta sensitivity of the underlying

        
        
        # current prices at t
        result.gamma_before_hedge = copy.deepcopy(self.portfolio.get_gamma(self.t))

        result.vega_before_hedge = self.portfolio.get_vega(self.t)
        result.hed_port_gamma_before = self.portfolio.hed_port.get_gamma(self.t)
        result.liab_port_gamma_before = self.portfolio.liab_port.get_gamma(self.t)

        if t == 2:
            print(f"position before hedge: {self.portfolio.hed_port.positions[self.sim_episode,t,t]}")

        result.step_pnl = reward = self.portfolio.step(action_swaption_hed=action_swaption_hedge,
                                                        action_swap_hed=action_swap_hedge,
                                                        action_swap_liab=action_swap_liab, 
                                                        t=self.t, 
                                                        result=result)
        result.gamma_after_hedge = self.portfolio.get_gamma(self.t)
        assert abs(result.gamma_after_hedge - (
        result.gamma_before_hedge + result.hed_port_gamma - result.hed_port_gamma_before
        )) < 1e-6
        if t == 2:
            print(f"position after hedge : {self.portfolio.hed_port.positions[self.sim_episode,t,t]}\n")
        result.vega_after_hedge = self.portfolio.get_vega(self.t)
        
        self.t = self.t + 1

        state = self.portfolio.get_state(self.t)
        if self.t == self.num_period - 1:
            done = True
            #state[1:] = 0 this is handled in the tensors. 
        else:
            done = False
        
        result.state_price_hed = state[0]
        result.state_price_liab = state[1]
        result.state_delta = state[2]
        result.state_delta_local_hed = state[3]
        result.state_delta_local_liab = state[4]
        result.state_hed_delta = state[5]
        result.state_liab_delta = state[6]
        result.state_gamma = state[7]
        result.state_gamma_local = state[8]
        result.state_hed_gamma = state[9]

        if FLAGS.vega_obs:
            result.state_vega = state[10]
            result.state_vega_local = state[11]
            result.state_hed_vega = state[12]
        # TODO: look into this, it might be slowing down the code
        # for other info later
        info = {"path_row": self.sim_episode}
        if self.logger:
            self.logger.write(dataclasses.asdict(result))
        return state, reward, done, info
