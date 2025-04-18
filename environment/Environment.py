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
import logging
import numpy as np

from environment.Trading import MainPortfolio, Greek, SwapKeys



@dataclasses.dataclass
class StepResult:
    episode: int = 0
    t: int = 0
    action0_swaption_hed_dir: float = 0.0
    action1_swaption_hed_mag: float = 0.0
    action2_swap_hed: float = 0.0
    action3_swap_liab: float = 0.0
    
    cost_swaption_hed: float = 0.0
    cost_swap: float = 0.0
    
    step_pnl: float = 0.0
    step_pnl_hed_swaption: float = 0.0
    step_pnl_liab_swaption: float = 0.0
    step_pnl_hed_swap: float = 0.0
    step_pnl_liab_swap: float = 0.0
    
    
    delta_local_hed_before_hedge: float = 0.0
    delta_local_hed_after_hedge: float = 0.0
    delta_local_liab_before_hedge: float = 0.0
    delta_local_liab_after_hedge: float = 0.0
    delta_before_hedge: float = 0.0
    delta_after_hedge: float = 0.0


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
        self.utils = utils
        # prepare portfolio and underlying iterables
        self.portfolio = MainPortfolio(utils)
        self.print_nanwarning = True
        self.portfolio.reset(0)
        # number of episodes available
        hedge_mm, *_ = utils.generate_swaption_market_data()
        self.num_path   = len(hedge_mm)
        # number of steps per episode inferred from memmap shape
        self.num_period = hedge_mm.shape[1]
        self.sim_episode = -1
        self.t = None
        # action: [swaption, swap1, swap2, swap3]
        self.action_space = spaces.Box(low=np.zeros(4), high=np.ones(4), dtype=np.float32)
        # obs space bounds
        max_g = self.utils.contract_size * 20 * 100 * 50
        max_v = self.utils.contract_size * 20 * 100  * 50
        # rate bounds from memmap

        low = [0, 0] + [-np.inf]*5 + [-max_g, -np.inf, -np.inf]
        high= [1, 1] + [np.inf]*5 + [max_g, np.inf, np.inf]
        if FLAGS.vega_obs:
            low += [-max_v, -np.inf, -np.inf]
            high+= [ max_v,  np.inf,  np.inf]
        low += [0]; high += [self.num_period]
        self.observation_space = spaces.Box(low=np.array(low, dtype=np.float32),
                                            high=np.array(high,dtype=np.float32))

    def seed(self, seed):
        # set the np random seed
        np.random.seed(seed)

    def reset(self):
        self.sim_episode = (self.sim_episode + 1) % self.num_path
        self.portfolio.reset(self.sim_episode)
        self.t = 0
        self.print_nanwarning = True
        return self.portfolio.get_state(self.t)

    def step(self, action):
        """
        profit and loss period reward
        """
        #print(f"step {self.t} episode {self.sim_episode}, action: {action}")
        result = StepResult(
            episode=self.sim_episode,
            t=self.t,
        )
        result.action0_swaption_hed_dir = action[0]
        result.action1_swaption_hed_mag = action[1]
        result.action2_swap_hed = action[2]
        result.action3_swap_liab = action[3]
        if self.print_nanwarning and np.isnan(action).any():
            self.print_nanwarning = False
            print(f"action is NaN! This warning is turned off until next episode")
        over_hedge_scale = 1.5
        t = self.t

        hed_port = self.portfolio.hed_port


        
        gamma_hedge_unit = hed_port.get_gamma(t, position_scale=False, single_value=True)
        portfolio_gamma = self.portfolio.get_gamma_local_hed(t) # gamma sensitivity around the hedging swaption

        vega_hedge_unit = hed_port.get_vega(t, position_scale=False, single_value=True) # vega for swaption to be traded
        portfolio_vega = self.portfolio.get_vega_local_hed(t) 

        gamma_hedge_ratio = np.divide(portfolio_gamma, gamma_hedge_unit * self.utils.contract_size)
        vega_hedge_ratio  = np.divide(portfolio_vega , vega_hedge_unit * self.utils.contract_size) if FLAGS.vega_obs else np.float32(0.0)


        hedge_direction =  -(action[0] * gamma_hedge_ratio + (1 - action[0]) * vega_hedge_ratio)

        hedge_magnitude = over_hedge_scale * action[1]
        action_swaption_hedge = hedge_magnitude * hedge_direction

        delta_hedge_unit = hed_port.get_delta(t, position_scale=False, single_value=True) # delta for swaption to be traded


        # action[1] bound
        # delta that is added by the hedging swaption
        delta_swaption_offset_hed =  delta_hedge_unit * action_swaption_hedge # delta added by the swaption traded in this period

        delta_hed_local = self.portfolio.get_delta_local_hed(t)
        delta_hed_total = delta_hed_local + delta_swaption_offset_hed 
        
        
 
        delta_liab_local = self.portfolio.get_delta_local_liab(t)
        
        
        delta_liab_hed_unit_sensitivity = self.portfolio.get_hed_liab_relative_sensitivity(t,delta_hedge_unit) # sensitivity to the added delta from the hedging swaption
        delta_liab_total = delta_liab_local + delta_liab_hed_unit_sensitivity  # local delta + delta from liab
       

        action_swap_hedge = -over_hedge_scale * action[2] *  delta_hed_total/ (self.portfolio.underlying.active_path_hed[self.t, self.t, SwapKeys.DELTA] * self.utils.contract_size)

        action_swap_liab = -over_hedge_scale * action[3] * delta_liab_total/ (self.portfolio.underlying.active_path_liab[self.t, self.t, SwapKeys.DELTA] * self.utils.contract_size)
        
        
        # ============================================================================
        #           Log and step
        #===============================================================================
        
        result.delta_local_hed_before_hedge = delta_hed_local
        result.delta_local_liab_before_hedge = self.portfolio.get_delta_local_liab(t)
        result.delta_before_hedge = self.portfolio.get_delta(t)

        result.gamma_before_hedge = portfolio_gamma
        result.vega_before_hedge = portfolio_vega

        result.step_pnl = reward = self.portfolio.step(
            action_swaption_hed=np.float32(action_swaption_hedge),
            action_swap_hed=np.float32(action_swap_hedge),
            action_swap_liab=np.float32(action_swap_liab),
            t=self.t,
            result=result,
        )
        result.delta_local_hed_after_hedge = self.portfolio.get_delta_local_hed(t)
        result.delta_local_liab_after_hedge = self.portfolio.get_delta_local_liab(t)
        result.delta_after_hedge = self.portfolio.get_delta(t)

        result.gamma_after_hedge = self.portfolio.get_gamma_local_hed(t)
        result.vega_after_hedge = self.portfolio.get_vega_local_hed(t)
  
        
        
        self.t = self.t + 1

        state = self.portfolio.get_state(self.t)
        if self.t == self.num_period - 1:
            done = True
            state[2:-1] = 0 # all all greeks to 0
            #state[1:] = 0 this is handled in the tensors. 
        else:
            done = False
        

        info = {"path_row": self.sim_episode}
        if self.logger:
            self.logger.write(dataclasses.asdict(result))
        return state, reward, done, info
