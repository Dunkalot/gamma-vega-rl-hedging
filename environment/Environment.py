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
import cProfile
import pstats
import io # For capturing output
from environment.Trading import MainPortfolio, Greek, SwapKeys

import tensorflow as tf

@dataclasses.dataclass
class StepResult:
    episode: int = 0
    t: int = 0
    action_swaption: float = 0.0

    cost_swaption_hed: float = 0.0
    cost_swap_hed: float = 0.0
    step_pnl: float = 0.0
    step_pnl_hed_swaption: float = 0.0
    step_pnl_liab_swaption: float = 0.0
    step_pnl_hed_swap: float = 0.0

    delta_local_hed_before: float = 0.0
    delta_local_hed_after: float = 0.0
    delta_before: float = 0.0
    delta_after: float = 0.0

    gamma_local_hed_before: float = 0.0
    gamma_local_hed_after: float = 0.0
    gamma_before: float = 0.0
    gamma_after: float = 0.0
    gamma_hed: float = 0.0
    gamma_liab: float = 0.0
    gamma_hed_before: float = 0.0
    gamma_liab_before: float = 0.0
    gamma_hed_after: float = 0.0
    gamma_liab_after: float = 0.0
    vega_local_hed_before: float = 0.0
    vega_local_hed_after: float = 0.0
    vega_before: float = 0.0
    vega_after: float = 0.0

    action_gamma: float = 0.0
    
    # State fields
    rate_norm: float = 0.0
    rate_liab_norm: float = 0.0
    hed_cost_norm: float = 0.0
    gamma_unit_norm: float = 0.0
    vega_unit_norm: float = 0.0
    gamma_port_norm: float = 0.0
    vega_port_norm: float = 0.0
    iv_norm: float = 0.0
    iv_liab_norm: float = 0.0
    ttm: float = 0.0
    position_swaption_hed: float = 0.0
    position_swapion_liab: float = 0.0
    position_swap_hed: float = 0.0
    action_mag: float = 0.0
    action_dir: float = 0.0

class TrainLog:
    @staticmethod
    def _log_before(self,result,t):
        pass
    @staticmethod
    def _log_after(self,result,t):
        pass
class EvalLog:
    @staticmethod
    def _log_before(self, result:StepResult,t):
        
            result.gamma_before ,result.gamma_local_hed_before, result.delta_local_hed_before, result.vega_before, result.vega_local_hed_before = self.portfolio.get_kernel_greek_risk(t)
            result.delta_before = self.portfolio.get_delta(self.t)
            result.gamma_hed_before = self.portfolio.hed_port.get_gamma(t)
            result.gamma_liab_before = self.portfolio.liab_port.get_gamma(t)
    @staticmethod
    def _log_after(self, result:StepResult,t):

        result.gamma_after,result.gamma_local_hed_after, result.delta_local_hed_after,  result.vega_after, result.vega_local_hed_after = self.portfolio.get_kernel_greek_risk(t)
        result.delta_after = self.portfolio.get_delta(self.t) # portfolio delta
        result.gamma_hed_after = self.portfolio.hed_port.get_gamma(t)
        result.gamma_liab_after = self.portfolio.liab_port.get_gamma(t)
        self.logger.write(dataclasses.asdict(result))

class TradingEnv(gym.Env):
    """
    This is the Gamma & Vega Trading Environment.
    """

    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day;
    def __init__(self, utils, log_bef=None, log_af=None, logger: Optional[loggers.Logger] = None):
        self.actions = np.zeros(utils.swap_shape[0]*20)#*np.nan
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
        print(self.num_period)
        self.sim_episode = -1 + (utils.test_episode_offset if utils.test else 0)
        print(f"episode offset {self.sim_episode}")
        self.t = None
        self.action_space = spaces.Box(
            low=0.0,    
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        # obs space bounds

        self.observation_space = spaces.Box(
            low=-np.inf,    
            high=+np.inf,
            shape=(10,),
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
        if self.sim_episode%20 == 0:

            a = self.actions
            p1, p5, p50, p95, p99 = np.nanpercentile(a, [1, 5, 50, 95, 99])
            mean, std, min_val, max_val = np.nanmean(a), np.nanstd(a), np.nanmin(a), np.nanmax(a)
            eps = 1e-6
            frac_clipped_min = np.sum(a <= -1+eps) / a.size
            frac_clipped_max = np.sum(a >=  1-eps) / a.size
            
            print(f"Action stats - Mean: {mean:.4f}, Std: {std:.4f}")
            print(f"Min: {min_val:.4f}, Max: {max_val:.4f}")
            print(f"Percentiles - p1: {p1:.4f}, p5: {p5:.4f}, p50: {p50:.4f}, p95: {p95:.4f}, p99: {p99:.4f}")
            print(f"Clipped values - Min: {frac_clipped_min:.4f} ({frac_clipped_min*100:.1f}%), Max: {frac_clipped_max:.4f} ({frac_clipped_max*100:.1f}%)")
            
            self.actions = self.actions * np.nan

        return self.portfolio.get_state(self.t)
    

    def step(self, action):
        """
        profit and loss period reward
        """
        t = self.t
        action_mag, action_dir = action[0], action[1]
        result = StepResult( episode=self.sim_episode, t=t)
        result.action_mag = action_mag
        result.action_dir = action_dir
        self.actions[t] = action[0]
        # gamma to notional 
        gamma_bound =  -self.portfolio.get_gamma_local_hed(t)/(
            self.portfolio.hed_port._base_options[t,t,Greek.GAMMA] * self.utils.contract_size)
        vega_bound = -self.portfolio.get_vega_local_hed(t)/(
            self.portfolio.hed_port._base_options[t,t,Greek.VEGA] * self.utils.contract_size)
        
        hedge_dir = gamma_bound * action_dir + vega_bound * (1-action_dir)

        bounds = [0, gamma_bound, vega_bound]
        high = np.max(bounds)
        low = np.min(bounds)

        result.action_swaption = action_mag*hedge_dir#low + action * (high - low)

        assert not np.isnan(action).any(), action
        #if self.logger: # dont waste resources
        self.log_bef(self,result,t) 
        result.step_pnl = reward = self.portfolio.step(
            action_swaption_hed=result.action_swaption,
            t=self.t,
            result=result,
        )

            
        self.t = self.t + 1

        state = self.portfolio.get_state(self.t)
        result.rate_norm, result.rate_liab_norm, result.hed_cost_norm, result.gamma_unit_norm, result.gamma_port_norm, result.vega_unit_norm, result.vega_port_norm, result.iv_norm, result.iv_liab_norm, result.ttm= state
        if self.t == self.num_period-1:
            done = True
            #state[[2,3,4]] = 0 # swaptions doesnt expire at episode end, so greeks say. 
        else:
            done = False
        
        self.log_af(self,result,t)
        info = {"path_row": self.sim_episode}
        return state, reward, done, info
