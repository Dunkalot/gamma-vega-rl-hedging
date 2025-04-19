""" Utility Functions & Imports"""
import random
import gc
import numpy as np
import psutil
import os
import glob
from absl import flags
FLAGS = flags.FLAGS
import numpy as np
from environment.lmmsabr import LMMSABR, make_nss_yield_df, compute_6m_forward_dataframe
random.seed(1)


class Utils:
    # def __init__(self, init_ttm, np_seed, num_sim, mu=0.0, init_vol=0.2, 
    #              s=10, k=10, r=0, q=0, t=52, frq=1, spread=0,
    #              hed_ttm=60, beta=1, rho=-0.7, volvol=0.6, ds=0.001, 
    #              poisson_rate=1, moneyness_mean=1.0, moneyness_std=0.0, ttms=None, 
    #              num_conts_to_add = -1, contract_size = 100,
    #              action_low=0, action_high=3, kappa = 0.0, svj_rho = -0.1, mu_s=0.2, sigma_sq_s=0.1, lambda_d=0.2, gbm = False, sabr=False):
    def __init__(self, n_episodes =1000, tau=0.5,
        resolution=26,
        tenor=4,
        sim_time = 1,
        t_max=None,
        beta=0.5,
        B=0.5, swap_hedge_expiry=1, swap_client_expiry=2, poisson_rate=1,spread=0, seed=42, swap_spread =0.0001):
        
        self.seed = seed

        self.out_dir = "data/swaption_memmap"
        
        
        
        print(f"utils initiated with {spread=}, {poisson_rate=}, {n_episodes=}")
        
        print(f"\nMemory usage before lmm: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
        self.swap_spread = swap_spread

        
        self.contract_size = 1
        self.spread = spread
        self.poisson_rate = poisson_rate
        self.n_episodes = n_episodes
        self.lmm:LMMSABR = LMMSABR(tau=tau,
            resolution=resolution,
            tenor=tenor,
            sim_time=sim_time,
            t_max=t_max,
            beta=beta,
            B=B,
            swap_hedge_expiry=swap_hedge_expiry,
            swap_client_expiry=swap_client_expiry
        )
        self.swap_shape = self.lmm.swap_sim_shape
        self.hed_greeks = 6
        self.swap_dims = 5
        self.dt = self.lmm.dt

        self.num_period = self.lmm.swap_sim_shape[0] # number of steps
        print(f"Memory usage after: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")


    def generate_swaption_market_data(self):
        """
        Load swaption market episodes from disk directory `out_dir`.

        Assumes files named:
        - swaption_hed.dat   -> shape (n_episodes, T, T, hed_greeks)
        - swaption_liab.dat  -> same as above
        - swap_hedge.dat     -> shape (n_episodes, T, T, swap_dims)
        - swap_liab.dat      -> same as above
        - net_direction.dat  -> shape (n_episodes, T, T)

        Args:
            out_dir (str): path containing the .dat files or timestamped subdir.
            n_episodes (int): number of episodes.
            swap_shape (tuple): (T, T) shape of time/time.
            hed_greeks (int): number of greeks in swaption data (e.g. 6).
            swap_dims (int): number of dims in swap data (e.g. 5).
        Returns:
            tuple[np.memmap]: (hedge_swaption_mm, liab_swaption_mm,
                                hedge_swap_mm,   liab_swap_mm,
                                net_direction_mm)
        """
        out_dir = self.out_dir
        n_episodes = self.n_episodes
        swap_shape = self.swap_shape
        hed_greeks = self.hed_greeks
        swap_dims = self.swap_dims  
        # If out_dir has subdirectories, pick latest timestamp
        candidates = sorted(glob.glob(os.path.join(out_dir, '*')))
        data_dir = candidates[-1] if os.path.isdir(candidates[-1]) else out_dir

        T1, T2 = swap_shape[0], swap_shape[0] # hedge and liability are split into two square matrices
        # Load memmaps with known shapes and dtype float32
        hedge_swaption_mm = np.memmap(
            os.path.join(data_dir, 'swaption_hed.dat'),
            dtype=np.float32, mode='r',
            shape=(n_episodes, T1, T2, hed_greeks)
        )
        liab_swaption_mm = np.memmap(
            os.path.join(data_dir, 'swaption_liab.dat'),
            dtype=np.float32, mode='r',
            shape=(n_episodes, T1, T2, hed_greeks)
        )
        hedge_swap_mm = np.memmap(
            os.path.join(data_dir, 'swap_hedge.dat'),
            dtype=np.float32, mode='r',
            shape=(n_episodes, T1, T2, swap_dims)
        )
        liab_swap_mm = np.memmap(
            os.path.join(data_dir, 'swap_liab.dat'),
            dtype=np.float32, mode='r',
            shape=(n_episodes, T1, T2, swap_dims)
        )
        net_direction_mm = np.memmap(
            os.path.join(data_dir, 'net_direction.dat'),
            dtype=np.float32, mode='r',
            shape=(n_episodes, T1, T2)
        )
        return (
            hedge_swaption_mm,
            liab_swaption_mm,
            hedge_swap_mm,
            liab_swap_mm,
            net_direction_mm
        )
        
