""" Utility Functions & Imports"""
import random
import gc
import numpy as np
import psutil


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
        B=0.5, swap_hedge_expiry=1, swap_client_expiry=2, poisson_rate=1,spread=0, seed=42):
        self.seed = seed
        print(f"utils initiated with {spread=}, {poisson_rate=}, {n_episodes=}")
        
        print(f"\nMemory usage before lmm: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

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
        self.dt = self.lmm.dt

        self.num_period = self.lmm.swap_sim_shape[0] # number of steps
        print(f"Memory usage after: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")


    def generate_swaption_market_data(self):
        df_fwd = compute_6m_forward_dataframe(make_nss_yield_df())
        print("sampling starting conditions...")
        self.lmm.sample_starting_conditions(df_fwd, curve_samples=np.minimum(self.n_episodes,len(df_fwd)))
        print("priming the initial state...")
        self.lmm.prime()
        print(f"Memory usage after priming: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")
        hedge_swaption, liab_swaption, hedge_swap, liab_swap = self.lmm.generate_episodes(self.n_episodes)

        
        # generate poisson arrival options for the liab_swaption
        poisson_draws = np.random.poisson(lam=self.poisson_rate, size=(liab_swaption.shape[0], liab_swaption.shape[2]))
        # Expand dimensions to match liab_swaption shape
        poisson_draws = np.expand_dims(poisson_draws, axis=1)  # Add timestep dimension
        # Binomial draws: number of +1s per entry
        num_pos = np.random.binomial(poisson_draws, 0.5)
        # Net direction: (2 * num_pos - total options)
        net_direction = 2 * num_pos - poisson_draws
        net_direction = np.tile(net_direction, (1,liab_swaption.shape[2], 1))  # Expand to match liab_swaption shape
        net_direction[:, np.triu_indices(liab_swaption.shape[2], k=1)[0], np.triu_indices(liab_swaption.shape[2], k=1)[1]] = 0
        print(f"Memory usage after generating data: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")
        print("Data generation successful, deleting lmm object to save memory...")
        del self.lmm
        gc.collect()
        print(f"Memory usage after deleting lmm: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")
        return hedge_swaption, liab_swaption, hedge_swap, liab_swap, net_direction
    
