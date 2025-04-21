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
    swaption_action: float = 0.0
    delta_action_hed: float = 0.0
    
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
        #self.portfolio.reset(-1)
        # number of episodes available
        hedge_mm, *_ = utils.generate_swaption_market_data()
        self.num_path   = len(hedge_mm)
        # number of steps per episode inferred from memmap shape
        self.num_period = hedge_mm.shape[1]
        self.sim_episode = -1
        self.t = None
        # action: [swaption, swap1, swap2, swap3]
        self.action_space = spaces.Box(low=np.zeros(7), high=np.ones(7), dtype=np.float32)
        # obs space bounds
        # rate bounds from memmap


        self.observation_space = spaces.Box(
            low=-np.inf,    
            high=+np.inf,
            shape=(322,),
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
        print("episode reset!", self.sim_episode)
        return self.portfolio.get_state(self.t)

    def step(self, action):
        """
        profit and loss period reward
        """
        t = self.t
        result = StepResult( episode=self.sim_episode, t=t)
        result.action_mag, result.action_dir, result.gamma, result.vega, result.swaption_action, result.delta_action_hed, result.delta_action_liab = action

        if self.print_nanwarning and np.isnan(action).any():
            self.print_nanwarning = False
            print(f"action is NaN! This warning is turned off until next episode")

        result.step_pnl = reward = self.portfolio.step(
            action_swaption_hed=result.swaption_action,
            action_swap_hed=result.delta_action_hed,
            action_swap_liab=result.delta_action_liab,
            t=self.t,
            result=result,
        )
        
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
