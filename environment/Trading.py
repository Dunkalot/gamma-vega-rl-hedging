from abc import ABC, abstractmethod
from absl import flags
FLAGS = flags.FLAGS
import numpy as np
from enum import IntEnum

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
    DELTA = 3
class Swaps(AssetInterface):
    """Swap
    Swap delta is 1, other greeks are 0.
    Price path is in shape (num_episode, num_step, num_swap).
    Active path stores the current episode swap trajectories, 
    and it is reset in MainPortfolio at the beginning of a new episode

    """
    def __init__(self, price_path_hed, price_path_liab) -> None:
        """Constructor

        Args:
            price_path (np.ndarray): simulated swap prices in shape (num_episode, num_step, num_swap)
        """
        super().__init__()
        self.swap_data_hed: np.ndarray = price_path_hed
        self.swap_data_liab: np.ndarray = price_path_liab
        self.active_path = []
        self.position_hed = np.zeros((self.swap_data_hed.shape[1], self.swap_data_hed.shape[2]))  # shape: [steps, swaps]
        self.position_liab = np.zeros((self.swap_data_liab.shape[1], self.swap_data_liab.shape[2]))


    def set_path(self, sim_episode):
        # joint so the t points to the same time step in both hedging and liability paths
        self.active_path_hed = self.swap_data_hed[sim_episode, :, :,:]
        self.active_path_liab = self.swap_data_liab[sim_episode, :, :, :]

        self.episode = sim_episode
        self.position = 0

    def step(self, t):
        """Step on time t and generate swap P&Ls

        Args:
            t (int): step time t

        Returns:
            float: swap P&Ls from time t to t+1
        """
        # (self.active_path[t + 1] - self.active_path[t]) * self.position
        # this is a vectorized version of the above line
        pnl_hed = self.active_path_hed[t, :, SwapKeys.PNL]*self.position_hed
        pnl_liab = self.active_path_liab[t, :, SwapKeys.PNL]*self.position_liab
        # sum over all swaps
        pnl_hed = np.nansum(pnl_hed)
        pnl_liab = np.nansum(pnl_liab)
        # this is the net P&L of the swap
        pnl = pnl_hed + pnl_liab
        return pnl

    def get_value(self, t) -> float:
        """swap value at time t. Only shows the actively traded swap at time t"""
        #print(self.position_hed)
        val_hed =  self.active_path_hed[t, t, SwapKeys.PRICE]
        val_liab = self.active_path_liab[t, t, SwapKeys.PRICE]
        # sum over all swaps
        val_hed = val_hed
        val_liab = val_hed
        #print(val_hed.shape)
        # this is the net value of the swap
        val = np.array([val_hed, val_liab])
        return val
    
    def get_delta_vec(self, t):
        """swap delta at time t"""
        # concatenate at find the delta at time t, this is mainly important for when when some are not active, as they should be 0 then.
        self.delta_vec = np.concatenate([self.position_hed*self.active_path_hed[:,:,SwapKeys.DELTA] , self.position_liab*self.active_path_liab[:,:,SwapKeys.DELTA]], axis=1)
        #print(self.delta_vec[t].shape)
        return self.delta_vec[t]# TODO: IMPLEMENT ANNUITY (DV01) WEIGHTING LATER

    def get_gamma(self, t):
        """swap gamma at time t"""
        return 0

    def get_delta(self, t):
        raise NotImplementedError("get_delta is not implemented for swaps, use get_delta_vec instead")
    def get_vega(self, t):
        """swap vega at time t"""
        return 0
    def set_position(self,t, action_swap_hed, action_swap_liab):
        """set position of the swap at the inception of swap"""
        pos_hed = action_swap_hed # TODO actually set the action to be something meaningful
        pos_liab = action_swap_liab
        # check if the position is a float
        mask_hed = self.active_path_hed[:,:, SwapKeys.ACTIVE]
        mask_liab = self.active_path_liab[:,:, SwapKeys.ACTIVE]
        # set the position of the swap at the inception of swap, we set the whole column as the position is unchangeable and layter set inactive spots to 0
        self.position_hed[:,t] = pos_hed
        self.position_liab[:,t] = pos_liab
        # make sure the position is set to 0 if the swap is not active
        self.position_hed *= mask_hed
        self.position_liab *= mask_liab
        return self.position_hed[t,t], self.position_liab[t,t] # return the position of the swap at time t, this is used to set the action in the environment


class Greek(IntEnum):
    PRICE = 0
    DELTA = 1
    GAMMA = 2
    VEGA = 3
    PNL = 4
    ACTIVE = 5

import numpy as np

import numpy as np

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
        transaction_cost = -abs(self.utils.spread * price * num_contracts)
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
            scale = self._positions[self._episode, t, :] if position_scale else np.ones_like(active)
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
        """
        Compute the portfolio's PnL at timestep t.
        
        PnL is computed using positions, active flags, and the PnL values from base_options.
        """
        values = self._base_options[self._episode, t, :, Greek.PNL]
        active = self._active[self._episode, t, :]
        pos = self._positions[self._episode, t, :]
        return np.nansum(values * pos * active)

    def get_value(self, t):
        """
        Compute the portfolio value at timestep t based on price.
        """
        values = self._base_options[self._episode, t, :, Greek.PRICE]
        active = self._active[self._episode, t, :]
        pos = self._positions[self._episode, t, :]
        return np.nansum(values * pos * active)

    def get_current_position(self,t):
        """Get the current position of the portfolio at timestep t."""
        return self._positions[self._episode, t, t].copy()




class SwaptionLiabilityPortfolio(SwaptionPortfolio):
    """Poisson arrival liability portfolio using vectorized swaption tensor."""

    def __init__(self, utils, swaption_data, positions):
        super().__init__(utils, swaption_data)
        # Compute max gamma/vega across all swaptions for this dataset (optional, for scaling bounds)
        self.max_gamma = np.nanmax(np.abs(self._base_options[..., Greek.GAMMA]))
        self.max_vega = np.nanmax(np.abs(self._base_options[..., Greek.VEGA]))
        self._positions = positions


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
        self.underlying = Swaps(hedge_swap, liab_swap)  # WE USING SWAPS INSTEAD
        self.sim_episode = -1
        self.kernel_beta = -1 # standard deviation of the kernel
        print("Main portfolio initialized with kernel beta of ", self.kernel_beta)

    def rbf_kernel(self,center_idx, vector_len):
        if self.kernel_beta <= 0:
            return np.ones(vector_len)
        indices = np.arange(vector_len)
        distances = (indices - center_idx) * self.utils.dt
        weights = np.exp(-(distances ** 2) / (2 * self.kernel_beta ** 2))
        return weights

    def triangle_kernel(center_idx, vector_len):
        """
        Triangle kernel that peaks at center_idx and linearly tapers to 0 at both ends of the vector.
        The full support of the kernel is the entire vector length (symmetric around the center).
        """
        indices = np.arange(vector_len)
        half_len = vector_len / 2
        distances = np.abs(indices - center_idx)
        weights = np.clip(1 - distances / half_len, 0, 1)
        return weights



    def compute_local_sensitivity(self, vec, t):
        """Compute the kernel-weighted local delta around a hedgeable point."""
        
        if self.kernel_beta <= 0:
            return np.nansum(vec)
        else:
            kernel = self.rbf_kernel(center_idx=t, vector_len=len(vec))
            return np.dot(kernel, vec)

    def get_hed_liab_relative_sensitivity(self, value):
        #kernel is symmetric and translation-invariant, so we can use the first index
        weight = self.rbf_kernel(center_idx=0, vector_len=self.liab_offset_idx)[-1]
        return weight * value


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

        price_hed, price_liab = self.underlying.get_value(t) 

        # total portfolio gamma
        gamma = self.get_gamma(t) 
        gamma_local = self.get_gamma_local_hed(t) 
        hed_gamma_unit = self.hed_port.get_gamma(t, position_scale=False, single_value=True)


        # for information to determine swap position size
        delta = self.get_delta(t) 
        delta_local_hed = self.get_delta_local_hed(t) 
        delta_local_liab = self.get_delta_local_liab(t) 

        # swaption delta, unscaled
        hed_delta = self.hed_port.get_delta(t, position_scale=False, single_value=True)
        #liab_delta = self.liab_port.get_delta(t, position_scale=False, single_value=True)
        
        state = [price_hed, price_liab, delta, delta_local_hed, delta_local_liab, hed_delta,#liab_delta,
                gamma, gamma_local, hed_gamma_unit]

        if FLAGS.vega_obs:
            vega = self.get_vega(t) 
            vega_local = self.get_vega_local_hed(t) 
            hed_vega = self.hed_port.base_options[self.sim_episode, t, t, Greek.VEGA] 
            state.extend([vega,vega_local, hed_vega])

        #print("state", state)
        state.append(t)
        # price_hed, price_liab, delta, delta_local_hed, delta_local_liab, hed_delta,liab_delta, 
        # gamma, gamma_local, hed_gamma, vega, vega_local, hed_vega, t
        #print("STATE SHAPE:",np.array(state, dtype=np.float32)
        return np.array(state, dtype=np.float32)


        


    def step(self, action_swaption_hed, action_swap_hed,action_swap_liab, t,  result):
        """Step on time t and generate reward

        Args:
            action (float): hedging action
            t (int): time step
            result (StepResult): logging step metrics

        Returns:
            float: P&L as step reward
        """
        #result.stock_price = self.a_price[self.sim_episode, t]
        result.swap_price = self.underlying.get_value(t)
        result.hed_cost = reward = self.hed_port.add(self.sim_episode, t, action_swaption_hed)
        result.action_swaption_hed = action_swaption_hed
        result.position_swaption_liab = self.liab_port.get_current_position(t)
        result.action_swap_hed = action_swap_hed
        result.action_swap_liab = action_swap_liab
        result.swap_position = self.underlying.set_position(t, action_swap_hed, action_swap_liab) # TODO: the RL agent should set the position of the underlying swap
        result.liab_port_gamma = self.liab_port.get_gamma(t)
        #result.liab_port_vega = self.liab_port.get_vega(t)
        result.hed_port_gamma = self.hed_port.get_gamma(t)
        #result.hed_port_vega = self.hed_port.get_vega(t)
        result.liab_port_pnl = self.liab_port.step(t)
        result.hed_port_pnl = self.hed_port.step(t)
        result.swap_pnl = self.underlying.step(t)
        reward += (result.liab_port_pnl + result.hed_port_pnl + result.swap_pnl)
        #print("reward", reward)
        return reward
