from abc import ABC, abstractmethod
from absl import flags
FLAGS = flags.FLAGS
import numpy as np
from enum import IntEnum
import tensorflow as tf


class AssetInterface(ABC):
    """Asset Interface
    Asset contains its price, and risk profiles.
    Any kind of asset can inherit this interface, i.e. single option, portfolio constructed by multi-assets and etc. 
    """

    @abstractmethod
    def step(self, t):
        """Step on time t and generate asset's P&L as reward"""
        pass

    @abstractmethod
    def get_value(self, t):
        """asset value at time t"""
        pass

    @abstractmethod
    def get_delta(self, t):
        """asset delta at time t"""
        pass
    
    @abstractmethod
    def get_gamma(self, t):
        """asset gamma at time t"""
        pass
    
    @abstractmethod
    def get_vega(self, t):
        """asset vega at time t"""
        pass


class SwapKeys(IntEnum):
    PRICE = 0
    PNL = 1
    ACTIVE = 2
    DELTA = 3,
    RATE = 4
class Swaps(AssetInterface):
    """Swap instrument book (hedge + liability legs), episode-aware."""
    def __init__(self, price_path_hed_iterable, price_path_liab_iterable, utils):
        super().__init__()
        self.utils = utils
        self._price_path_hed_mm  = price_path_hed_iterable
        self._price_path_liab_mm = price_path_liab_iterable
        self.active_path_hed     = None
        self.active_path_liab    = None
        self.position_hed        = None
        self.position_liab       = None
        self._episode            = None

    def set_path(self, sim_episode: int):
        """Load one episode of swap paths into memory."""
        self._episode = sim_episode
        self.active_path_hed  = self._price_path_hed_mm[sim_episode]
        self.active_path_liab = self._price_path_liab_mm[sim_episode]
        steps, swaps, _ = self.active_path_hed.shape
        self.position_hed  = np.zeros((steps, swaps), dtype=np.float32)
        self.position_liab = np.zeros((steps, swaps), dtype=np.float32)

    def step(self, t: int):
        raise NotImplementedError("step() not implemented for Swaps class. Use step_hed() and step_liab() instead.")


    def step_hed(self, t: int):
        pnl_hed  = np.sum(self.active_path_hed [t, :, SwapKeys.PNL]  * self.position_hed )
        return pnl_hed
    def step_liab(self, t: int):
        pnl_liab = np.sum(self.active_path_liab[t, :, SwapKeys.PNL] * self.position_liab )
        return pnl_liab

    def get_value(self, t: int) -> np.ndarray:
        val_hed  = self.active_path_hed [t, t, SwapKeys.PRICE]  * self.position_hed
        val_liab = self.active_path_liab[t, t, SwapKeys.PRICE] * self.position_liab 
        return np.array([val_hed, val_liab])

    def get_delta_vec(self, t: int):
        vec_hed  = self.position_hed[t,:]  * self.active_path_hed [t, :, SwapKeys.DELTA]
        vec_liab = self.position_liab[t,:] * self.active_path_liab[t, :, SwapKeys.DELTA] 
        swap_delta_vec = np.concatenate([vec_hed, vec_liab])
        return swap_delta_vec
    def get_delta(self, t: int):    
        raise NotImplementedError("get_delta() not implemented for Swaps class. Use get_delta_vec() instead.")

    def get_gamma(self, t: int):
        return 0

    def get_vega(self, t: int):
        return 0

    def get_rate_hed(self, t: int):
        return self.active_path_hed [t, t, SwapKeys.RATE] * 100 # for better sensitivity 

    def get_rate_liab(self, t: int):
        return self.active_path_liab[t, t, SwapKeys.RATE] * 100

    def add(self, t: int, action_swap_hed: float, action_swap_liab: float):
        self.position_hed [t:, t] = action_swap_hed * self.utils.contract_size
        self.position_liab[t:, t] = action_swap_liab * self.utils.contract_size
        cost_hed   = -abs(self.utils.swap_spread   * self.position_hed[t,t] )
        cost_liab  = -abs(self.utils.swap_spread * self.position_liab[t,t] )
        return cost_hed, cost_liab

    def get_position_hed(self, t: int):
        return self.position_hed[t, t].copy()

    def get_position_liab(self, t: int):
        return self.position_liab[t, t].copy()


class Greek(IntEnum):
    PRICE = 0
    DELTA = 1
    GAMMA = 2
    VEGA = 3
    PNL = 4
    ACTIVE = 5


class SwaptionPortfolio:
    def __init__(self, utils, swaption_data_iterable):
        """
        Args:
            utils: your utils object
            swaption_data_iterable: something you can index by episode (e.g. np.memmap or list of arrays)
        """
        self.utils = utils
        # Keep the full dataset on‐disk or as an iterable
        self._base_memmap = swaption_data_iterable

        # Will be a (steps, swaps, greeks) array once set_episode() is called
        self._base_options: np.ndarray = None  
        self._episode = None  

        self.steps = utils.lmm.swap_sim_shape[0]
        # Per‐episode buffers (steps × swaps)
        self._positions: np.ndarray = np.zeros((self.steps, self.steps), dtype=np.float32)
        
    def reset(self):
        """Reset portfolio state within the current episode."""
        self._positions[:] = np.float32(0.0)

    def set_episode(self, episode: int):
        """
        Load episode `episode` into RAM and re-init position buffers.
        Must be called before any other method in a new episode.
        """
        self._episode = episode

        self._base_options = self._base_memmap[episode]  # shape: (steps, swaps, greeks)


    def add(self, t: int, num_contracts: float):
        """
        Place `num_contracts` at timestep t.  
        Must have called set_episode() already.
        """
        # mark for all future timesteps
        self._positions[t:, t] = num_contracts * self.utils.contract_size

        price = self._base_options[t, t, Greek.PRICE]
        cost  = -abs(self.utils.spread * price * self._positions[t, t] )
        return cost

    def get_metric(self, t: int, greek: int,
                   position_scale=True,
                   summed=True,
                   single_value=False):
        """Exactly as before, but no episode index anywhere."""
        if single_value:
            val   = self._base_options[t, t, greek]
            scale = self._positions[t, t]  if position_scale else np.float32(1.0)
            return val * scale 
        else:
            vals  = self._base_options[t, :, greek]
            scale = self._positions[t, :]  if position_scale else np.float32(1.0)
            out   = vals * scale
            return out.sum() if summed else out

    # convenience wrappers
    def get_delta(self, t, position_scale=True, single_value=False):
        return self.get_metric(t, Greek.DELTA, position_scale, True, single_value)
    
    def get_delta_vec(self, t, position_scale=True):
        return self.get_metric(t, Greek.DELTA, position_scale, False, False)
    
    def get_gamma(self, t, position_scale=True, single_value=False):
        """Compute portfolio gamma at timestep t (summed if single_value is False)."""
        return self.get_metric(t, Greek.GAMMA, position_scale, summed=True, single_value=single_value)

    def get_gamma_vec(self, t, position_scale=True):
        """Retrieve the full gamma vector at timestep t."""
        return self.get_metric(t, Greek.GAMMA, position_scale, summed=False, single_value=False)

    def get_vega(self, t, position_scale=True, single_value=False):
        """Compute portfolio vega at timestep t (summed if single_value is False)."""
        return self.get_metric(t, Greek.VEGA, position_scale, summed=True, single_value=single_value)

    def get_vega_vec(self, t, position_scale=True):
        """Retrieve the full vega vector at timestep t."""
        return self.get_metric(t, Greek.VEGA, position_scale, summed=False, single_value=False)

    def step(self, t: int):
        vals = self._base_options[t, :, Greek.PNL]
        pos  = self._positions[t, :]
        return (vals * pos ).sum() 

    def get_value(self, t: int):
        vals = self._base_options[t, :, Greek.PRICE]
        pos  = self._positions[t, :]
        return (vals * pos ).sum()

    def get_current_position(self, t: int):
        return self._positions[t, t].copy()




class SwaptionLiabilityPortfolio(SwaptionPortfolio):
    """Liability portfolio with Poisson‐arrival positions, loaded per episode."""
    def __init__(self, utils, swaption_memmap, positions_memmap):
        super().__init__(utils, swaption_memmap)
        # positions_memmap[i] is a (steps, swaps) array for episode i
        self._positions_memmap = positions_memmap  
        # placeholders, filled in set_episode()
        self.max_gamma = None  
        self.max_vega   = None  

    def set_episode_liab(self, episode: int):
        # 1) load the episode's Greeks into RAM
        self.set_episode(episode)

        # 2) load this episode's static liability positions
        #    (could be a memmap slice or an in‑memory array)
        pos = self._positions_memmap[episode]
        self._positions = np.array(pos, dtype=np.float32)           # (steps, swaps)

    def reset(self):
        # do not zero out positions—they're fixed liabilities.
        # Just clear the episode marker so nothing else runs by accident.
        self._episode = None

    def add(self, t: int, num_contracts: float):
        """Liabilities only attach at t==0, no cost."""
        assert self._episode is not None, "call set_episode() first"
        assert t == 0, "Liability portfolio can only be added at the start"
        return np.float32(0.0)  # no cost




class MainPortfolio(AssetInterface):
    """Main Portfolio
    This is the total portfolio contains three components:
    1. Liability portfolio: Poisson arrival underwritten options
    2. Hedging option portfolio: ATM options that are purchased by agent for hedging purpose
    3. Underlying stock: Automatic delta neutral hedging position
    """
    def __init__(self, utils):
        """Constructor

        Args:
            utils (utils.Utils): contains environment configurations and util functions
        """
        super().__init__()
        self.utils = utils
        self.dt = self.utils.dt
        #self.a_price, self.vol = utils.init_env()
        hedge_swaption, liab_swaption, hedge_swap, liab_swap, liab_swaption_position = utils.generate_swaption_market_data()
        print("initializing classes")
        self.liab_port = SwaptionLiabilityPortfolio(utils, liab_swaption, liab_swaption_position)
        self.hed_port: SwaptionPortfolio = SwaptionPortfolio(utils, hedge_swaption) 
        print("done initializing classes")
        # since we are concatenating hedge and liability matrix, we need to define an offset that for t gets the t column in hedge and t + offset in liability
        self.liab_offset_idx = self.utils.swap_shape[0]

        self.underlying = Swaps(hedge_swap, liab_swap, utils=utils)  # WE USING SWAPS INSTEAD
        self.kernel_beta = np.float32(1.0) # standard deviation of the kernel
        self.use_rbf_kernel = False
        self.gamma_vector = np.zeros(105, dtype=np.float32)
        self.vega_vector = np.zeros(105, dtype=np.float32)
        self.delta_vector = np.zeros(105, dtype=np.float32)

        print("Main portfolio initialized with kernel beta of ", self.kernel_beta, " using" , ("rbf kernel" if self.use_rbf_kernel else "triangle kernel") if self.kernel_beta > 0 else "no kernel")
        print("REMEMBER TO ADD BACK THE SENSITIVITY FOR LIAB AND HEDGE WHEN USING THE RBF KERNEL")
        assert self.use_rbf_kernel == False, "RBF kernel logic not yet implemented, there needs to be a system for accounting for cross-sensitivity of delta hedges with swaps with the get_hed_liab_relative_sensitivity func"
        assert self.kernel_beta > 0, "are you sure you want to run without kernel?"
        self.kernel = (self.rbf_kernel if self.use_rbf_kernel else self.triangle_kernel) if self.kernel_beta >0 else lambda t, vec_len: np.ones(vec_len, dtype=np.float32)
    # -------------- kernel helpers ---------------
    def rbf_kernel(self, center_idx, vector_len):
        if self.kernel_beta <= 0:
            return np.ones(vector_len, dtype=np.float32)
        indices = np.arange(vector_len, dtype=np.float32)
        distances = (indices - center_idx) * self.utils.dt
        return np.exp(-(distances ** 2) / (2 * self.kernel_beta ** 2), dtype=np.float32)

    def triangle_kernel(self, center_idx, vector_len):
        indices = np.arange(vector_len, dtype=np.float32)
        half_len = np.float32(vector_len / 2)
        distances = np.abs(indices - center_idx)
        return np.float32(np.clip(1 - distances / half_len, 0, 1))

    def compute_local_sensitivity(self, vec, t):
        weights = self.kernel(t, len(vec))
        return np.dot(weights, vec)

    def get_kernel_sensitivity(self, idx, idx_perpective) -> float:
        """function that compus the sensitivity to the value at idx the perspective of the idx_perspective using the current kernel on a single value"""
        if self.kernel_beta <= 0:
            return 1
        kernel = (self.rbf_kernel if self.use_rbf_kernel else self.triangle_kernel)(idx_perpective, int(2 * self.utils.swap_shape[0]))[idx]
        return kernel



    def get_value(self, t):
        """portfolio value at time t"""
        return self.hed_port.get_value(t) + self.liab_port.get_value(t) + self.underlying.get_value(t)


    def get_delta_vec(self, t):
        """portfolio delta vector at time t"""
        swap_delta =  self.underlying.get_delta_vec(t)
        #print("=============\nswap delta vec")
        #print(swap_delta[:5])
        swaption_delta = np.concatenate([self.hed_port.get_delta_vec(t), self.liab_port.get_delta_vec(t)])
        ##print("swaption delta")
        #print(swaption_delta[:5])
        delta_concat =  swaption_delta+swap_delta
        #print(delta_concat[:5])
        return delta_concat
    


    def get_delta(self, t):
        """portfolio delta at time t"""
        delta = np.sum(self.get_delta_vec(t))
        return delta
    
    def get_delta_local_hed(self, t):
        """compute the delta around the hedging swaption expiry"""
        return self.compute_local_sensitivity(self.get_delta_vec(t), t)
    
    def get_delta_local_liab(self, t):
        """Compute the delta around the liability swaption expiry"""
        return self.compute_local_sensitivity(self.get_delta_vec(t), t+self.liab_offset_idx)
    
    def get_hed_liab_relative_sensitivity(self, t,scalar):
        """Compute the delta sensitivity of the liability swaption to the hedging swaption delta"""
        # delta_hedge_unit is the delta of the hedging swaption
        # delta_liab_local is the local delta of the liability swaption
        # delta_liab_hed_unit_sensitivity is the sensitivity of the liability swaption to the hedging swaption delta
        delta_liab_local = self.get_kernel_sensitivity(t,t+ self.liab_offset_idx)
        return delta_liab_local * scalar


    def get_delta_local_spot (self, t):
        """compute the delta at the spot time. NOTE: this assumes that the liability swaption is 1 year away from the hedging swaption, and that the hedging swaption
        has 1 year to expiry."""
        return self.compute_local_sensitivity(self.get_delta_vec(t), t-self.liab_offset_idx)


    def get_gamma(self, t):
        """portfolio gamma at time t"""
        return self.hed_port.get_gamma(t) + self.liab_port.get_gamma(t)
    
    def get_gamma_vec(self, t):
        """portfolio gamma vector at time t"""
        swaption_gamma = np.concatenate([self.hed_port.get_gamma_vec(t), self.liab_port.get_gamma_vec(t)])
        return swaption_gamma
    
    def get_gamma_local_hed(self, t):
        """portfolio gamma around the hedging swaption expiry"""
        return self.compute_local_sensitivity(self.get_gamma_vec(t), t)


    def get_vega(self, t):
        """portfolio vega at time t"""
        return self.hed_port.get_vega(t) + self.liab_port.get_vega(t)
    
    def get_vega_vec(self, t):
        """portfolio vega vector at time t"""
        swaption_vega = np.concatenate([self.hed_port.get_vega_vec(t), self.liab_port.get_vega_vec(t)])
        return swaption_vega
    
    def get_vega_local_hed(self, t):
        """portfolio vega at time t"""
        return self.compute_local_sensitivity(self.get_vega_vec(t), t)
    
    def update_kernel_delta_vector(self,t):
        self.delta_vector[52-t:] = self.get_delta_vec(t)[:53+t]
        return self.delta_vector

    def update_kernel_gamma_vector(self,t):
        self.gamma_vector[52-t:] = self.get_gamma_vec(t)[:53+t]
        return self.gamma_vector
    
    def update_kernel_vega_vector(self,t):
        self.vega_vector[52-t:] = self.get_vega_vec(t)[:53+t]
        return self.vega_vector

    def update_risk_vectors(self,t):
        self.update_kernel_gamma_vector(t)
        self.update_kernel_vega_vector(t)
        self.update_kernel_delta_vector(t)
        

    def reset(self, sim_episode: int):
        """Reset all components at the beginning of episode `sim_episode`."""
        self.sim_episode = sim_episode
        self.underlying.set_path(sim_episode)

        self.hed_port.reset()
        self.hed_port.set_episode(sim_episode)

        self.liab_port.reset()
        self.liab_port.set_episode_liab(sim_episode)
        self.gamma_vector[:] = np.float32(0.)
        self.delta_vector[:] = np.float32(0.)
        self.vega_vector[:]  = np.float32(0.)
        # prime the delta vector
        
    
    def get_kernel_greek_risk(self):
        vol_kernel = self.utils.vol_kernel
        volvol_kernel = self.utils.volvol_kernel
        
        gamma_local, delta_local_hed = vol_kernel(np.concatenate([self.gamma_vector[None,:],self.delta_vector[None,:]]))
        vega_local = volvol_kernel(self.vega_vector)
        delta_local_liab = vol_kernel(self.delta_vector, anchor_index=104)
        return gamma_local, vega_local, delta_local_hed, delta_local_liab
    
    def get_state(self, t: int) -> np.ndarray:
        rate_hed   = self.underlying.get_rate_hed(t)
        rate_liab  = self.underlying.get_rate_liab(t)  

        # scale sensitivities by the number of contracts that will be bought
        gamma_unit_hed  = self.hed_port._base_options[t,t,Greek.GAMMA] * self.utils.contract_size
        vega_unit_hed = self.hed_port._base_options[t,t,Greek.VEGA] * self.utils.contract_size
        delta_unit_hed  = self.underlying.active_path_hed[t, t, SwapKeys.DELTA] * self.utils.contract_size
        delta_unit_liab = self.underlying.active_path_liab[t, t, SwapKeys.DELTA] * self.utils.contract_size
        delta_unit_hed_swaption = self.hed_port._base_options[t,t,Greek.DELTA] * self.utils.contract_size
        state = np.array([gamma_unit_hed, vega_unit_hed, delta_unit_hed_swaption, delta_unit_hed,delta_unit_liab,
                        t*self.dt,  rate_hed, rate_liab], dtype=np.float32)

        self.update_risk_vectors(t)

        state = np.concatenate([state, self.gamma_vector, self.vega_vector, self.delta_vector])
        return state.astype(np.float32)


    
    
    def step(self, action_swaption_hed, action_swap_hed, action_swap_liab, t: int, result):
        """Apply actions, compute PnL and reward at time t."""
        

        result.cost_swaption_hed = self.hed_port.add(t, action_swaption_hed)
        result.cost_swap_hed, result.cost_swap_liab = self.underlying.add(t, action_swap_hed, action_swap_liab)
       
        
        
        # PnL contributions
        result.step_pnl_hed_swaption  = self.hed_port.step(t)
        result.step_pnl_liab_swaption = self.liab_port.step(t)
        result.step_pnl_hed_swap      = self.underlying.step_hed(t)
        result.step_pnl_liab_swap     = self.underlying.step_liab(t)
        

        reward = (
            result.cost_swaption_hed +
            result.cost_swap_hed +
            result.cost_swap_liab +
            result.step_pnl_hed_swaption  +
            result.step_pnl_liab_swaption +
            result.step_pnl_hed_swap +
            result.step_pnl_liab_swap
        )
        
   
        return np.float32(reward)
