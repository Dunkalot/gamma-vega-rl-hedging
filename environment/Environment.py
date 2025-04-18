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

def _safe_div(num, den, default=0.0):
    # 1) elementwise division (possibly INF/NAN) in C
    # 2) nan/±inf→default in C
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = num / den
    return np.nan_to_num(raw, nan=default, posinf=default, neginf=default)

@dataclasses.dataclass
class StepResult:
    episode: int = 0
    t: int = 0
    hed_action: float = 0.0
    gamma_before_hedge: float = 0.0
    gamma_after_hedge: float = 0.0
    vega_before_hedge: float = 0.0
    vega_after_hedge: float = 0.0




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
        self.num_path = self.portfolio.hed_port._base_options.shape[0]
        # set num_period: initial time to maturity * daily trading freq + 1 (see get_sim_path() in utils.py)
        self.num_period = self.portfolio.hed_port._base_options.shape[1]

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
        min_price_swap_hed = np.min(self.portfolio.underlying.swap_data_hed[:,:,:, SwapKeys.RATE])
        min_price_swap_liab = np.min(self.portfolio.underlying.swap_data_liab[:,:,:, SwapKeys.RATE])
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

        

        return self.portfolio.get_state(self.t)

    def step(self, action):
        """
        profit and loss period reward
        """
        #print(f"step {self.t} episode {self.sim_episode}, action: {action}")
        result = StepResult(
            episode=self.sim_episode,
            t=self.t,
            hed_action=action,
        )
        
        over_hedge_scale = 1.5
        t = self.t

        hed_port = self.portfolio.hed_port


        
        gamma_hedge_unit = hed_port.get_gamma(t, position_scale=False, single_value=True)
        portfolio_gamma = self.portfolio.get_gamma_local_hed(t) # gamma sensitivity around the hedging swaption

        vega_hedge_unit = hed_port.get_vega(t, position_scale=False, single_value=True) # vega for swaption to be traded
        portfolio_vega = self.portfolio.get_vega_local_hed(t) 





        gamma_hedge_ratio = _safe_div(-portfolio_gamma, gamma_hedge_unit)
        vega_action_bound  = _safe_div(-portfolio_vega , vega_hedge_unit) if FLAGS.vega_obs else 0.0
        action_space = [0, gamma_hedge_ratio, vega_action_bound]

        action_space = np.max(np.abs(action_space))

        #action_swaption_hedge = low_val + action[0] * (high_val - low_val)
        action_swaption_hedge =  over_hedge_scale*action[0] * gamma_hedge_ratio # counter hedge risk direction

        delta_hedge_unit = hed_port.get_delta(t, position_scale=False, single_value=True) # delta for swaption to be traded


        # action[1] bound
        # delta that is added by the hedging swaption
        delta_swaption_offset_hed =  delta_hedge_unit * action_swaption_hedge # delta added by the swaption traded in this period

        delta_hed_local = self.portfolio.get_delta_local_hed(t)
        delta_hed_total = delta_hed_local + delta_swaption_offset_hed 
        
        
 
        delta_liab_local = self.portfolio.get_delta_local_liab(t)
        
        
        delta_liab_hed_unit_sensitivity = self.portfolio.get_hed_liab_relative_sensitivity(delta_hedge_unit) # sensitivity to the added delta from the hedging swaption
        delta_liab_total = delta_liab_local + delta_liab_hed_unit_sensitivity  # local delta + delta from liab
       


        

        action_swap_hedge = -over_hedge_scale * action[1] * _safe_div(
            delta_hed_total,
            self.portfolio.underlying.active_path_hed[self.t, self.t, SwapKeys.DELTA])

        action_swap_liab = -over_hedge_scale * action[2] * _safe_div(
            delta_liab_total,
            self.portfolio.underlying.active_path_liab[self.t, self.t, SwapKeys.DELTA])
        
        
        result.bound_low = low_val
        result.bound_high = high_val
        result.step_pnl = reward = self.portfolio.step(
            action_swaption_hed=action_swaption_hedge,
            action_swap_hed=action_swap_hedge,
            action_swap_liab=action_swap_liab,
            t=self.t,
            result=result,
        )
  
        
        self.t = self.t + 1

        state = self.portfolio.get_state(self.t)
        if self.t == self.num_period - 1:
            done = True
            #state[1:] = 0 this is handled in the tensors. 
        else:
            done = False
        

        # TODO: look into this, it might be slowing down the code
        # for other info later
        info = {"path_row": self.sim_episode}
        if self.logger:
            self.logger.write(dataclasses.asdict(result))
        return state, reward, done, info
