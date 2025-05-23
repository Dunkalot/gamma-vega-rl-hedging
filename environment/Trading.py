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
    DELTA = 2,
    RATE = 3,
    
class Swaps(AssetInterface):
    """Swap instrument book (hedge only), episode-aware."""
    def __init__(self, price_path_hed_iterable, utils):
        super().__init__()
        self.utils = utils
        self._price_path_hed_mm = price_path_hed_iterable
        self.active_path_hed = None
        self.position_hed = None
        self._episode = None

    def set_path(self, sim_episode: int):
        """Load one episode of swap paths into memory."""
        self._episode = sim_episode
        self.active_path_hed = self._price_path_hed_mm[sim_episode]
        steps, swaps, _ = self.active_path_hed.shape
        self.position_hed = 0.

    def step(self, t: int):
        """Step on time t and return PnL."""
        return self.step_hed(t)

    def step_hed(self, t: int):
        pnl_hed = np.sum(self.active_path_hed[t, 0, SwapKeys.PNL] * self.position_hed)
        return pnl_hed

    def get_value(self, t: int) -> np.ndarray:
        val_hed = self.active_path_hed[t, 0, SwapKeys.PRICE] * self.position_hed
        return np.array([val_hed])

        
    def get_delta(self, t: int):
        """Return total delta across all positions."""
        return self.position_hed * self.active_path_hed[t, 0, SwapKeys.DELTA]

    def get_gamma(self, t: int):
        return 0

    def get_vega(self, t: int):
        return 0

    def get_rate_hed(self, t: int):
        return self.active_path_hed[t, 0, SwapKeys.RATE] 

    def add(self, t: int, action_swap_hed: float):
        """Add a new swap position at time t."""
        self.position_hed += action_swap_hed * self.utils.contract_size
        cost_hed = -abs(self.utils.swap_spread * action_swap_hed)
        return cost_hed

    def get_position_hed(self, t: int):
        return self.position_hed


class Greek(IntEnum):
    PRICE = 0
    DELTA = 1
    GAMMA = 2
    VEGA = 3
    PNL = 4
    IV = 5


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
        DVVOL = self._base_options[t, t, Greek.VEGA] * self._base_options[t, t, Greek.IV]
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
        return self.get_metric(t, Greek.DELTA, position_scale, summed=True, single_value=single_value)
    
    
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
        hedge_swaption, liab_swaption, hedge_swap, liab_swap, liab_swaption_position, self.kernel_hed_all, self.kernel_vol_all, self.ttm_mat = utils.generate_swaption_market_data()
        
        print("initializing classes")
        self.liab_port = SwaptionLiabilityPortfolio(utils, liab_swaption, liab_swaption_position * utils.contract_size)
        self.hed_port: SwaptionPortfolio = SwaptionPortfolio(utils, hedge_swaption) 
        print("done initializing classes")
        # since we are concatenating hedge and liability matrix, we need to define an offset that for t gets the t column in hedge and t + offset in liability
        self.liab_offset_idx = self.utils.swap_shape[0]

        # Initialize with hedge swap only
        self.underlying = Swaps(hedge_swap, utils=utils)  
        self.underlying_liab =  Swaps(liab_swap, utils=utils) # just needed for the rate
        self.kernel_beta = np.float32(1.0)  # standard deviation of the kernel
        self.use_rbf_kernel = False
        
        self.gamma_vector = np.zeros(105, dtype=np.float32)
        self.vega_vector = np.zeros(105, dtype=np.float32)
        self.delta_vector = np.zeros(105, dtype=np.float32)

    def kernel_coef_hed(self, t):
        """regression weights wrt hedging asset"""
        return self.kernel_hed[t]
    
    def kernel_coef_vol(self, t):
        """regression weights wrt liability asset"""
        return self.kernel_vol[t]

    def get_ttm_vec(self, t):
        return self.ttm_mat[t]

    def get_value(self, t):
        """portfolio value at time t"""
        return self.hed_port.get_value(t) + self.liab_port.get_value(t) + self.underlying.get_value(t)[0]

    def get_delta_vec(self, t):
        """portfolio delta vector at time t"""
        swap_delta = self.underlying.get_delta(t)
        return np.array([self.hed_port.get_delta(t)+swap_delta, self.liab_port.get_delta(t)])

    def get_delta(self, t):
        """portfolio delta at time t"""
        delta = np.sum(self.get_delta_vec(t))
        return delta

    def get_gamma(self, t):
        """portfolio gamma at time t"""
        return self.hed_port.get_gamma(t) + self.liab_port.get_gamma(t)
    
    def get_gamma_vec(self, t):
        """portfolio gamma vector at time t"""
        swaption_gamma = np.array([self.hed_port.get_gamma(t), self.liab_port.get_gamma(t)])
        return swaption_gamma

    def get_gamma_local_hed(self, t):
        """portfolio gamma around the hedging swaption expiry"""
        return np.inner(self.kernel_coef_hed(t)**2, self.get_gamma_vec(t))

    def get_delta_local_hed(self, t):
        """portfolio delta around the hedging swaption expiry"""
        return np.inner(self.kernel_coef_hed(t), self.get_delta_vec(t))


    def get_vega(self,t):
        return self.hed_port.get_vega(t) + self.liab_port.get_vega(t)

    def get_vega_local_hed(self,t):
        """portfolio vega vector at time t"""
        swaption_vega = np.array([self.hed_port.get_vega(t), self.liab_port.get_vega(t)])
        return np.inner(self.kernel_coef_vol(t),swaption_vega)




    def reset(self, sim_episode: int):
        """Reset all components at the beginning of episode `sim_episode`."""
        self.sim_episode = sim_episode
        self.underlying.set_path(sim_episode)
        self.underlying_liab.set_path(sim_episode)
        self.hed_port.reset()
        self.hed_port.set_episode(sim_episode)

        self.liab_port.reset()
        self.liab_port.set_episode_liab(sim_episode)
        # prime the delta vector
        self.kernel_hed = self.kernel_hed_all[sim_episode]
        self.kernel_vol = self.kernel_vol_all[sim_episode]
        T = len(self.kernel_hed)
        
        # Transform kernel_hed
        self.kernel_hed = np.concatenate((np.ones_like(self.kernel_hed), self.kernel_hed), axis=1)
        self.kernel_vol = np.concatenate((np.ones_like(self.kernel_vol), self.kernel_vol), axis=1)
        np.set_printoptions(precision=10, suppress=True)
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    def get_state(self, t: int) -> np.ndarray:
        rate_hed = self.underlying.get_rate_hed(t)*300
        rate_liab = self.underlying_liab.get_rate_hed(t)*300
        gamma_unit_hed = self.hed_port._base_options[t, t, Greek.GAMMA]/14
        vega_unit_hed = self.hed_port._base_options[t, t, Greek.VEGA] / 0.006
        gamma_port = self.get_gamma_local_hed(t) /85000
        vega_port = self.get_vega_local_hed(t)/40
        ttm = self.get_ttm_vec(t)[0]

        hed_cost = (self.hed_port._base_options[t, t, Greek.PRICE] )  * self.utils.spread*100*100*10
        iv_norm = (self.hed_port._base_options[t, t, Greek.IV])*25
        iv_liab_norm = (self.liab_port._base_options[t, t, Greek.IV])*25
        ttm_norm =  ttm*14
        state_raw = np.array([rate_hed,rate_liab, hed_cost, gamma_unit_hed, gamma_port, vega_unit_hed, vega_port, iv_norm, iv_liab_norm,ttm_norm])
        return state_raw.astype(np.float32)

    def get_kernel_greek_risk(self, t):
        return self.get_gamma(t), self.get_gamma_local_hed(t), self.get_delta_local_hed(t), self.get_vega(t), self.get_vega_local_hed(t)

    def solve_delta_action(self, t):
        """
        Solve for the hedge swap position that neutralizes delta exposure.
        With only one swap, this becomes a simpler calculation.
        """
        delta_unit_hed = self.underlying.active_path_hed[t, 0, SwapKeys.DELTA] * self.utils.contract_size
        
        # Compute current delta exposure
        delta_local_hed = self.get_delta_local_hed(t)
        
        # Simple formula: action = -delta_exposure / unit_delta
        action_swap_hed = -delta_local_hed / delta_unit_hed
        
        return action_swap_hed

    def step(self, action_swaption_hed, t: int, result):
        """Apply actions, compute PnL and reward at time t."""
        
        result.cost_swaption_hed = self.hed_port.add(t, action_swaption_hed)
        
        # swaption is added, we now know the full delta vector that needs to be neutralized
        action_swap_hed = self.solve_delta_action(t)
        result.cost_swap_hed = self.underlying.add(t, action_swap_hed)
        result.cost_swap_liab = 0.0  # No liability swap
        
        result.position_swaption_hed = self.hed_port.get_current_position(t)
        result.position_swaption_liab = self.liab_port.get_current_position(t)
        result.position_swap_hed = self.underlying.get_position_hed(t)

        # PnL contributions
        result.step_pnl_hed_swaption = self.hed_port.step(t)
        result.step_pnl_liab_swaption = self.liab_port.step(t)
        result.step_pnl_hed_swap = self.underlying.step_hed(t)
        result.step_pnl_liab_swap = 0.0  # No liability swap
        
        reward = (
            result.cost_swaption_hed +
            result.cost_swap_hed +
            result.step_pnl_hed_swaption +
            result.step_pnl_liab_swaption +
            result.step_pnl_hed_swap
        )
        result.cost_ratio = np.abs(result.cost_swaption_hed) / np.abs(reward) if reward != 0 else 0
   
        return np.float32(reward)
