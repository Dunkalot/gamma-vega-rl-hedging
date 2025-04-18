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



import os
import gc
import numpy as np
import h5py
import math
from datetime import datetime
from tqdm import tqdm
from typing import Union, Sequence, Tuple, Any

class EpisodeArray:
    """
    Lazy-loading, NumPy-like array interface for HDF5 datasets.
    Fetches data in batches to avoid loading all episodes into RAM.
    Supports multi-dimensional indexing and behaves like a full NumPy array.
    """
    def __init__(self, dataset: h5py.Dataset, batch_size: int = 1000):
        self._ds = dataset
        self.batch_size = batch_size
        self.cache_start = 0
        self.cache_end = 0
        self.cache = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._ds.shape

    @property
    def dtype(self) -> Any:
        return self._ds.dtype

    def __len__(self) -> int:
        return self._ds.shape[0]

    def __getitem__(self, idx: Union[int, slice, Sequence[int], Tuple]) -> np.ndarray:
        # Multi-axis tuple indexing
        if isinstance(idx, tuple):
            first, *rest = idx
            if not isinstance(first, int):
                return self._ds[idx]
            if first < 0:
                first += len(self)
            if not (self.cache_start <= first < self.cache_end):
                start = (first // self.batch_size) * self.batch_size
                end = min(start + self.batch_size, len(self))
                self.cache = self._ds[start:end]
                self.cache_start, self.cache_end = start, end
            local = first - self.cache_start
            return self.cache[local][tuple(rest)]
        # slice or fancy indexing
        if isinstance(idx, slice) or isinstance(idx, (list, np.ndarray)):
            return self._ds[idx]
        # single index on first axis
        i = idx
        if i < 0:
            i += len(self)
        if not (self.cache_start <= i < self.cache_end):
            start = (i // self.batch_size) * self.batch_size
            end = min(start + self.batch_size, len(self))
            self.cache = self._ds[start:end]
            self.cache_start, self.cache_end = start, end
        local = i - self.cache_start
        return self.cache[local]

    def slice(self, start: int, end: int) -> np.ndarray:
        """Explicitly load full array for episodes [start:end]."""
        return self._ds[start:end]


class DataManager:
    """
    Generates and lazily serves swaption market data via HDF5-backed EpisodeArrays.
    """
    def __init__(self,
                 lmm,
                 n_episodes: int,
                 cache_path: str = "swaption_data.h5",
                 block: int = 1000):
        self.lmm = lmm
        self.n_episodes = n_episodes
        self.cache_path = cache_path
        self.block = block
        self._file = None

    def generate_and_save(self) -> str:
        """
        Run LMM in blocks, save to timestamped HDF5, return filepath.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(self.cache_path)
        path = f"{base}_{ts}{ext}"

        with h5py.File(path, 'w') as hf:
            hf.attrs['created'] = datetime.now().isoformat()

            # sample a single episode to get individual shapes
            sample_data = self.lmm.generate_episodes(1)
            names = ['swaption_hed', 'swaption_liab', 'swap_hedge', 'swap_liab', 'net_direction']
            shapes = {name: sample_data[i].shape[1:] for i, name in enumerate(names)}
            dtype = sample_data[0].dtype

            # create extendable datasets per name with matching shape
            dsets = {}
            for name in names:
                dsets[name] = hf.create_dataset(
                    name,
                    shape=(0,)+shapes[name],
                    maxshape=(None,)+shapes[name],
                    chunks=(self.block,)+shapes[name],
                    compression='gzip',
                    compression_opts=4,
                    dtype=dtype
                )

            total = self.n_episodes
            ep = 0
            pbar = tqdm(total=math.ceil(total/self.block), desc='Generating', unit='blk')
            while ep < total:
                b = min(self.block, total - ep)
                hd, ld, hs, ls, nd = self.lmm.generate_episodes(b)
                arrays = [hd, ld, hs, ls, nd]
                for arr, name in zip(arrays, names):
                    ds = dsets[name]
                    old = ds.shape[0]
                    ds.resize(old + b, axis=0)
                    ds[old:old + b] = arr
                ep += b
                pbar.update(1)
            pbar.close()
            del self.lmm
            gc.collect()
        return path

    def open(self, path: str = None) -> None:
        """
        Open HDF5 and expose EpisodeArray objects matching original arrays.
        """
        p = path or self.cache_path
        self._file = h5py.File(p, 'r')
        self.swaption_hed = EpisodeArray(self._file['swaption_hed'], self.block)
        self.swaption_liab = EpisodeArray(self._file['swaption_liab'], self.block)
        self.swap_hedge = EpisodeArray(self._file['swap_hedge'], self.block)
        self.swap_liab = EpisodeArray(self._file['swap_liab'], self.block)
        self.net_direction = EpisodeArray(self._file['net_direction'], self.block)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()






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
        B=0.5, swap_hedge_expiry=1, swap_client_expiry=2, poisson_rate=1,spread=0, seed=42, swap_spread=0.0001, contract_size=10000
        ):

        self.seed = seed
        print(f"utils initiated with {spread=}, {poisson_rate=}, {n_episodes=}")
        
        print(f"\nMemory usage before lmm: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

        self.swap_spread = swap_spread
        self.spread = spread
        print("SWAPTION SPREAD IS: ", self.swap_spread, "SWAP SPREAD IS: ", self.spread)

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
        df_fwd = compute_6m_forward_dataframe(make_nss_yield_df())
        self.lmm.sample_starting_conditions(df_fwd, curve_samples=min(self.n_episodes,len(df_fwd)))
        self.lmm.prime()
        self.num_period = self.lmm.swap_sim_shape[0] # number of steps
        print(f"Memory usage after: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")


    
    def generate_swaption_market_data(self,
                                    cache_path: str = "./data/swaption_data.h5",
                                    block: int = 1000)-> Tuple[EpisodeArray, EpisodeArray, EpisodeArray, EpisodeArray, EpisodeArray]:
        """
        Wrapper that generates and returns lazy EpisodeArray objects in place of full NumPy arrays.

        Returns:
            swaption_hed, swaption_liab, swap_hedge, swap_liab, net_direction
        """
        lmm = self.lmm
        n_episodes = self.n_episodes
        dm = DataManager(lmm, n_episodes, cache_path, block)
        path = dm.generate_and_save()
        dm.open(path)
        return (
            dm.swaption_hed,
            dm.swaption_liab,
            dm.swap_hedge,
            dm.swap_liab,
            dm.net_direction,
        )


    # def generate_swaption_market_data(self):
    #     df_fwd = compute_6m_forward_dataframe(make_nss_yield_df())
    #     print("sampling starting conditions...")
    #     self.lmm.sample_starting_conditions(df_fwd, curve_samples=np.minimum(self.n_episodes,len(df_fwd)))
    #     print("priming the initial state...")
    #     self.lmm.prime()
    #     print(f"Memory usage after priming: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")
    #     hedge_swaption, liab_swaption, hedge_swap, liab_swap = self.lmm.generate_episodes(self.n_episodes)

        
    #     # generate poisson arrival options for the liab_swaption
    #     poisson_draws = np.random.poisson(lam=self.poisson_rate, size=(liab_swaption.shape[0], liab_swaption.shape[2]))
    #     # Expand dimensions to match liab_swaption shape
    #     poisson_draws = np.expand_dims(poisson_draws, axis=1)  # Add timestep dimension
    #     # Binomial draws: number of +1s per entry
    #     num_pos = np.random.binomial(poisson_draws, 0.5)
    #     # Net direction: (2 * num_pos - total options)
    #     net_direction = 2 * num_pos - poisson_draws
    #     net_direction = np.tile(net_direction, (1,liab_swaption.shape[2], 1))  # Expand to match liab_swaption shape
    #     net_direction[:, np.triu_indices(liab_swaption.shape[2], k=1)[0], np.triu_indices(liab_swaption.shape[2], k=1)[1]] = 0
    #     print(f"Memory usage after generating data: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")
    #     print("Data generation successful, deleting lmm object to save memory...")
    #     del self.lmm
    #     gc.collect()
    #     print(f"Memory usage after deleting lmm: {psutil.Process().memory_info().rss / 1e6:.2f} MB\n")
    #     return hedge_swaption, liab_swaption, hedge_swap, liab_swap, net_direction
    
