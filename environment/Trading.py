from abc import ABC, abstractmethod
from absl import flags
FLAGS = flags.FLAGS
import numpy as np
from enum import IntEnum


def _safe_div(num, den, default=0.0):
    # 1) elementwise division (possibly INF/NAN) in C
    # 2) nan/±inf→default in C
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = num / den
    return np.nan_to_num(raw, nan=default, posinf=default, neginf=default)

def _safe_mul(a, b):
    # 1) elementwise product in C; 2) nan→0 in C
    return np.nan_to_num(a * b, nan=0.0)

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
    """Swap instrument book (hedge + liability legs)."""

    def __init__(self, price_path_hed, price_path_liab, utils) -> None:
        super().__init__()
        self.utils = utils
        self.swap_data_hed: np.ndarray = price_path_hed
        self.swap_data_liab: np.ndarray = price_path_liab

        # positions are kept as [steps, swaps]
        self.position_hed = np.zeros((self.swap_data_hed.shape[1], self.swap_data_hed.shape[2]))
        self.position_liab = np.zeros((self.swap_data_liab.shape[1], self.swap_data_liab.shape[2]))


    def set_path(self, sim_episode):
        """Select the simulated path for *sim_episode*."""
        # Align hedging and liability trajectories on time index `t`.
        self.active_path_hed  = self.swap_data_hed [sim_episode]
        self.active_path_liab = self.swap_data_liab[sim_episode]
        self.position_hed = np.zeros((self.swap_data_hed.shape[1], self.swap_data_hed.shape[2]))
        self.position_liab = np.zeros((self.swap_data_liab.shape[1], self.swap_data_liab.shape[2]))
        self.episode = sim_episode

    def step(self, t):
        """Return PnL from t → t+1 (hedge + liability)."""
        pnl_hed  = np.nansum(_safe_mul(self.active_path_hed [t, :, SwapKeys.PNL], self.position_hed))
        pnl_liab = np.nansum(_safe_mul(self.active_path_liab[t, :, SwapKeys.PNL], self.position_liab))
        return pnl_hed + pnl_liab


    def get_value(self, t) -> np.ndarray:
        """Return [hedge_value, liability_value] at time *t*."""
        val_hed  = self.active_path_hed [t, t, SwapKeys.PRICE] * self.position_hed * self.utils.contract_size
        val_liab = self.active_path_liab[t, t, SwapKeys.PRICE] * self.position_liab * self.utils.contract_size
        return np.array([val_hed, val_liab])
    
    def get_delta_vec(self, t):
        """Delta vector at time *t* (concatenated hed + liab)."""
        delta_vec = np.concatenate([
            self.position_hed  * self.active_path_hed [:, :, SwapKeys.DELTA],
            self.position_liab * self.active_path_liab[:, :, SwapKeys.DELTA]
        ], axis=1)
        return delta_vec[t]

    # trivial Greeks
    def get_gamma(self, t):
        return 0
    def get_delta(self, t):  # not used
        raise NotImplementedError("use get_delta_vec instead")
    def get_vega(self, t):
        return 0
    def get_rate_hed(self, t):
        return self.active_path_hed [t, t, SwapKeys.RATE]
    def get_rate_liab(self, t):
        return self.active_path_liab[t, t, SwapKeys.RATE]
    

    def add(self, t, action_swap_hed, action_swap_liab):
        """Stamp *immutable* positions at inception (column t)."""
        self.position_hed [:, t] = action_swap_hed
        self.position_liab[:, t] = action_swap_liab
        price_hed  = self.active_path_hed [t, t, SwapKeys.PRICE]
        price_liab = self.active_path_liab[t, t, SwapKeys.PRICE]
        cost_hed   = -abs(self.utils.swap_spread * price_hed  * action_swap_hed)
        cost_liab  = -abs(self.utils.swap_spread * price_liab * action_swap_liab)

        return cost_hed + cost_liab
    
    def get_position_hed(self, t):
        """Get the current position of the portfolio at timestep t."""
        return self.position_hed[t, t].copy()
    def get_position_liab(self, t):
        """Get the current position of the portfolio at timestep t."""
        return self.position_liab[t, t].copy()


class Greek(IntEnum):
    PRICE = 0
    DELTA = 1
    GAMMA = 2
    VEGA = 3
    PNL = 4
    ACTIVE = 5


class SwaptionPortfolio:
    def __init__(self, utils, swaption_data):
        self.utils = utils
        self._base_options = np.nan_to_num(swaption_data)  # Static option Greek profiles
        self._episode = 0  # Protected attribute
        
        # Protected positions and active matrices
        self._positions = np.zeros_like(self._base_options[..., 0])  # shape: (episodes, steps, swaps)
        self._active = np.zeros_like(self._positions)  # same shape as _positions

    def reset(self):
        """Reset portfolio state between episodes."""
        self._positions[:] = 0
        self._active[:] = 0

    def set_episode(self, episode):
        """Set the current episode index."""
        self._episode = episode

    def add(self, sim_episode, t, num_contracts):
        """
        Add a new position.
        
        For the given simulation episode and timestep `t`, increase the position by 
        num_contracts and mark the active flag for all future timesteps.
        """
        self._episode = sim_episode
        #print("Adding position at episode:", sim_episode, "timestep:", t, "num_contracts:", num_contracts)
        self._positions[sim_episode, t:, t] = num_contracts
        self._active[sim_episode, t:, t] = 1  # Mark these positions as active

        price = self._base_options[sim_episode, t, t, Greek.PRICE]
        transaction_cost = -abs(self.utils.spread * price * num_contracts * self.utils.contract_size)
        return transaction_cost

    def get_metric(self, t, greek, position_scale=True, summed=True, single_value=False):
        """
        Retrieve a metric value for a given Greek at timestep `t` with flexible options.
        
        Parameters:
            t (int): The timestep to compute the metric.
            greek: The constant/index representing the Greek to retrieve (e.g., Greek.DELTA).
            position_scale (bool): If True, scales the Greek values by the positions; 
                                   otherwise, multiplies by 1 (i.e., ignores positions).
            summed (bool): If True, returns the sum over the vector of values; if False, 
                           returns the raw vector. Not used when single_value is True.
            single_value (bool): If True, uses the diagonal index (i.e., [episode, t, t, greek])
                                 rather than the full vector ([episode, t, :, greek]).
        
        Returns:
            float or np.ndarray: The computed metric. If single_value is True or summed is True,
                                 a scalar value is returned; otherwise, a vector is returned.
        """
        if single_value:
            # Retrieve single element along the diagonal
            value = self._base_options[self._episode, t, t, greek]
            active = self._active[self._episode, t, t] if position_scale else 1
            scale = self._positions[self._episode, t, t] if position_scale else 1
            return value * scale * active
        else:
            # Retrieve full vector along the swap axis
            values = self._base_options[self._episode, t, :, greek]
            active = self._active[self._episode, t, :]
            scale = self._positions[self._episode, t, :] * self.utils.contract_size if position_scale else np.ones_like(active)
            result = values * scale * active
            return np.nansum(result) if summed else result

    # Convenience wrappers for specific metrics

    def get_delta(self, t, position_scale=True, single_value=False):
        """Compute portfolio delta at timestep t (summed if single_value is False)."""
        return self.get_metric(t, Greek.DELTA, position_scale, summed=True, single_value=single_value)

    def get_delta_vec(self, t, position_scale=True):
        """Retrieve the full delta vector at timestep t."""
        return self.get_metric(t, Greek.DELTA, position_scale, summed=False, single_value=False)

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

    def step(self, t):
        values = self._base_options[self._episode, t, :, Greek.PNL]
        active = self._active      [self._episode, t, :]
        pos    = self._positions   [self._episode, t, :]
        return np.nansum(_safe_mul(values, _safe_mul(pos, active)))* self.utils.contract_size

    def get_value(self, t):
        values = self._base_options[self._episode, t, :, Greek.PRICE]
        active = self._active      [self._episode, t, :]
        pos    = self._positions   [self._episode, t, :]
        return np.nansum(_safe_mul(values, _safe_mul(pos, active))) * self.utils.contract_size

    def get_current_position(self,t):
        """Get the current position of the portfolio at timestep t."""
        return self._positions[self._episode, t, t].copy()




class SwaptionLiabilityPortfolio(SwaptionPortfolio):
    """Poisson arrival liability portfolio using vectorized swaption tensor."""

    def __init__(self, utils, swaption_data, positions):
        super().__init__(utils, swaption_data)
        # Compute max gamma/vega across all swaptions for this dataset (optional, for scaling bounds)
        self._positions = positions
        self.max_gamma = np.nanmax(np.abs(self._base_options[..., Greek.GAMMA] * self._positions * self.utils.contract_size))
        self.max_vega = np.nanmax(np.abs(self._base_options[..., Greek.VEGA] * self._positions * self.utils.contract_size))


    def reset(self):
        self._episode = None

    def add(self, sim_episode, t, num_contracts):
        """Only effective at the beginning of the episode (t == 0)."""
        assert t == 0, "Liability portfolio can only be added at the beginning of the episode."
        self._episode = sim_episode
        self._active[...] = 1 # positions control the scaling here
        return 0.0  # No transaction cost for liability position




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
        #self.a_price, self.vol = utils.init_env()
        hedge_swaption, liab_swaption, hedge_swap, liab_swap, liab_swaption_position = utils.generate_swaption_market_data()

        self.liab_swaption_position = liab_swaption_position
        self.liab_port = SwaptionLiabilityPortfolio(utils, liab_swaption, positions=liab_swaption_position)
        self.hed_port: SwaptionPortfolio = SwaptionPortfolio(utils, hedge_swaption) 

        # since we are concatenating hedge and liability matrix, we need to define an offset that for t gets the t column in hedge and t + offset in liability
        self.liab_offset_idx = self.hed_port._base_options.shape[2] 

        #self.underlying = Stock(self.a_price)
        self.underlying = Swaps(hedge_swap, liab_swap, utils=utils)  # WE USING SWAPS INSTEAD
        self.sim_episode = -1
        self.kernel_beta = 1 # standard deviation of the kernel
        self.use_rbf_kernel = False

        print("Main portfolio initialized with kernel beta of ", self.kernel_beta, " using" , ("rbf kernel" if self.use_rbf_kernel else "triangle kernel") if self.kernel_beta > 0 else "no kernel")

    # -------------- kernel helpers ---------------
    def rbf_kernel(self, center_idx, vector_len):
        if self.kernel_beta <= 0:
            return np.ones(vector_len)
        indices = np.arange(vector_len)
        distances = (indices - center_idx) * self.utils.dt
        return np.exp(-(distances ** 2) / (2 * self.kernel_beta ** 2))

    def triangle_kernel(self, center_idx, vector_len):
        indices = np.arange(vector_len)
        half_len = vector_len / 2
        distances = np.abs(indices - center_idx)
        return np.clip(1 - distances / half_len, 0, 1)

    def compute_local_sensitivity(self, vec, t):
        if self.kernel_beta <= 0:
            return np.nansum(vec)
        kernel = (self.rbf_kernel if self.use_rbf_kernel else self.triangle_kernel)(t, len(vec))
        return np.dot(kernel, vec)


    def get_value(self, t):
        """portfolio value at time t"""
        return self.hed_port.get_value(t) + self.liab_port.get_value(t) + self.underlying.get_value(t)

    # def get_delta(self, t):
    #     """portfolio delta at time t"""
    #     return self.hed_port.get_delta(t) + self.liab_port.get_delta(t) + self.underlying.get_delta(t)
    def get_delta_vec(self, t):
        """portfolio delta vector at time t"""
        swap_delta =  self.underlying.get_delta_vec(t)
        swaption_delta = np.concatenate([self.hed_port.get_delta_vec(t), self.liab_port.get_delta_vec(t)])
        delta_concat =  swaption_delta+swap_delta
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
    

    def reset(self, sim_episode):
        """Reset portfolio at the begining of a new episode

        1. Clear hedging option portfolio
        2. Clear liability portfolio
        3. Set underlying stock to new episode and clear position
        """
        self.underlying.set_path(sim_episode)

        self.sim_episode = sim_episode
        self.hed_port.set_episode(sim_episode)
        
        self.liab_port.add(self.sim_episode, t = 0, num_contracts=-1) # just primes the episode

    def get_state(self, t):

        rate_hed = np.nan_to_num(self.underlying.get_rate_hed(t), nan=0.0)
        rate_liab = np.nan_to_num(self.underlying.get_rate_liab(t), nan=0.0)

        # for information to determine swap position size
        delta_portfolio = np.nan_to_num(self.get_delta(t), nan=0.0)
        delta_local_spot = np.nan_to_num(self.get_delta_local_spot(t), nan=0.0)
        delta_local_hed = np.nan_to_num(self.get_delta_local_hed(t), nan=0.0)
        delta_local_liab = np.nan_to_num(self.get_delta_local_liab(t), nan=0.0)
        
        
        # total portfolio gamma
        gamma_portfolio = np.nan_to_num(self.get_gamma(t), nan=0.0)
        gamma_local = np.nan_to_num(self.get_gamma_local_hed(t), nan=0.0)
        hed_gamma_unit = np.nan_to_num(self.hed_port.get_gamma(t, position_scale=False, single_value=True), nan=0.0)



        # swaption delta, unscaled
        hed_delta = np.nan_to_num(self.hed_port.get_delta(t, position_scale=False, single_value=True), nan=0.0)
        #liab_delta = self.liab_port.get_delta(t, position_scale=False, single_value=True)
        
        state = [rate_hed, rate_liab, delta_portfolio,delta_local_spot, delta_local_hed, delta_local_liab, hed_delta,#liab_delta,
                gamma_portfolio, gamma_local, hed_gamma_unit]

        if FLAGS.vega_obs:
            vega = np.nan_to_num(self.get_vega(t), nan=0.0)
            vega_local = np.nan_to_num(self.get_vega_local_hed(t), nan=0.0)
            hed_vega = np.nan_to_num(self.hed_port.base_options[self.sim_episode, t, t, Greek.VEGA], nan=0.0)
            state.extend([vega,vega_local, hed_vega])

        
        
        #print("state", state)
        state.append(t)
        # rate_hed, rate_liab, delta, delta_local_spot, delta_local_hed, delta_local_liab, hed_delta,liab_delta, 
        # gamma, gamma_local, hed_gamma, vega, vega_local, hed_vega, t
        return np.array(state, dtype=np.float32)


        


    def step(self, action_swaption_hed, action_swap_hed, action_swap_liab, t, result):
        result.hed_cost = reward = np.nan_to_num(self.hed_port.add(self.sim_episode, t, action_swaption_hed))
        result.swap_position = self.underlying.add(t, action_swap_hed, action_swap_liab)

        result.liab_port_pnl = self.liab_port.step(t)
        result.hed_port_pnl  = self.hed_port.step(t)
        result.swap_pnl      = self.underlying.step(t)
        reward += (np.nan_to_num(result.liab_port_pnl) +
                   np.nan_to_num(result.hed_port_pnl)  +
                   np.nan_to_num(result.swap_pnl))
        return reward