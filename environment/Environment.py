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

import tensorflow as tf

@dataclasses.dataclass
class StepResult:
    episode: int = 0
    t: int = 0
    action_swaption: float = 0.0
    action_swap_hed: float = 0.0
    action_swap_liab: float = 0.0
    action_swap_hed_mag: float = 0.0
    action_swap_liab_mag: float = 0.0

    cost_swaption_hed: float = 0.0
    cost_swap_hed: float = 0.0
    cost_swap_liab: float = 0.0

    step_pnl: float = 0.0
    step_pnl_hed_swaption: float = 0.0
    step_pnl_liab_swaption: float = 0.0
    step_pnl_hed_swap: float = 0.0
    step_pnl_liab_swap: float = 0.0

    delta_local_hed_before: float = 0.0
    delta_local_hed_after: float = 0.0
    delta_local_liab_before: float = 0.0
    delta_local_liab_after: float = 0.0
    delta_before: float = 0.0
    delta_after: float = 0.0

    gamma_before: float = 0.0
    gamma_after: float = 0.0
    vega_before: float = 0.0
    vega_after: float = 0.0

    action_gamma: float = 0.0
    action_vega: float = 0.0
    gamma_ratio: float = 0.0
    vega_ratio: float = 0.0


class TrainLog:
    @staticmethod
    def _log_before(self,result):
        pass
    @staticmethod
    def _log_after(self,result):
        pass
class EvalLog:
    @staticmethod
    def _log_before(self, result):
        
            result.gamma_before, result.vega_before, result.delta_local_hed_before, result.delta_local_liab_before = self.portfolio.get_kernel_greek_risk()
            result.delta_before = self.portfolio.get_delta(self.t)
    @staticmethod
    def _log_after(self, result):

        self.portfolio.update_risk_vectors(self.t) # risk change following action
        result.gamma_after, result.vega_after, result.delta_local_hed_after, result.delta_local_liab_after = self.portfolio.get_kernel_greek_risk()
        result.delta_after = self.portfolio.get_delta(self.t) # portfolio delta
        self.logger.write(dataclasses.asdict(result))

class TradingEnv(gym.Env):
    """
    This is the Gamma & Vega Trading Environment.
    """

    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day;
    def __init__(self, utils, log_bef=None, log_af=None, logger: Optional[loggers.Logger] = None):

        super(TradingEnv, self).__init__()
        self.log_bef = log_bef
        self.log_af = log_af 
        self.logger = logger
        self.logger_present = True if self.logger else False 
        print("TRAINING WITH LOGGER:", self.logger_present)
        print("")
        self.utils = utils
        # prepare portfolio and underlying iterables
        self.portfolio = MainPortfolio(utils)
        self.print_nanwarning = True
        #self.portfolio.reset(-1)
        # number of episodes available
        hedge_mm, *_ = utils.generate_swaption_market_data()
        self.num_path   = len(hedge_mm)
        # number of steps per episode inferred from memmap shape
        self.num_period = hedge_mm.shape[1]
        self.sim_episode = -1
        self.t = None
        self.action_space = spaces.Box(
            low=-np.inf,    
            high=+np.inf,
            shape=(1),
            dtype=np.float32
        )
        # obs space bounds

        self.observation_space = spaces.Box(
            low=-np.inf,    
            high=+np.inf,
            shape=(323,),
            dtype=np.float32
        )


    def seed(self, seed):
        # set the np random seed
        np.random.seed(seed)

    def reset(self):
        self.sim_episode = (self.sim_episode + 1) % self.num_path
        self.portfolio.reset(self.sim_episode)
        self.t = 0
        self.print_nanwarning = True
        #print("episode reset!", self.sim_episode)
        return self.portfolio.get_state(self.t)
    

    def step(self, action):
        """
        profit and loss period reward
        """
        t = self.t
        result = StepResult( episode=self.sim_episode, t=t)

        result.action_gamma, result.action_vega, result.gamma_ratio ,result.vega_ratio,result.action_swaption, result.action_swap_hed, result.action_swap_liab = action
        assert not np.isnan(action).any(), action
        #if self.print_nanwarning and np.isnan(action).any():
        #    self.print_nanwarning = False
        #    print(f"action is NaN! This warning is turned off until next episode")
        
        #if self.logger: # dont waste resources

        self.log_bef(self,result) 
        result.step_pnl = reward = self.portfolio.step(
            action_swaption_hed=result.action_swaption,
            action_swap_hed=result.action_swap_hed,
            action_swap_liab=result.action_swap_liab,
            t=self.t,
            result=result,
        )

        self.log_af(self,result)
            
        self.t = self.t + 1

        state = self.portfolio.get_state(self.t)
        if self.t == self.num_period - 1:
            done = True
            state[7:] = 0 # all greeks to 0
            state[:4] = 0
        else:
            done = False
        

        info = {"path_row": self.sim_episode}
            
        return state, reward, done, info
