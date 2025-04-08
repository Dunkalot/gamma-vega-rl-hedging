#load libraries
import math
import numpy as np
import warnings
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import zoom

from scipy import interpolate
from functools import partial
from scipy.stats import norm
import ipympl
from mpl_toolkits.mplot3d import Axes3D
import time
def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def doust_corr(beta, n):
    '''
    create nxn doust correlation with beta decay exponential
    n = # of semi-annual expiries
    '''
    tau = np.arange(0, n+1)/2 # start from spot
    a = np.exp(- beta / np.arange(1, len(tau[:-1])+1) )
    doust = np.zeros((n, n))
    dim = doust.shape
    for i in range(doust.shape[0]):
        for j in range(doust.shape[1]):
            if i == j:
                doust[i, j] = 1
            elif i > j:
                doust[i, j] = np.prod(a[j:i])
    #reflect
    doust[np.triu_indices(dim[0], 1)] = doust.T[np.triu_indices(dim[0], 1)]
    return(doust)


def interpolate_correlation_matrix(matrix: np.ndarray, resolution: int) -> np.ndarray:
    """
    Interpolates a correlation matrix using bilinear interpolation.

    Args:
        matrix (np.ndarray): The input correlation matrix (must be square).
        resolution (int): The resolution factor. For a 4x4 and resolution=2, output will be 7x7.

    Returns:
        np.ndarray: Interpolated correlation matrix.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")

    # Compute zoom factor: new_size = original_size + (original_size - 1) * (resolution - 1)
    zoom_factor = resolution

    # Use order=1 for bilinear interpolation
    interpolated = zoom(matrix, zoom=zoom_factor, order=1)

    # Adjust shape to match expected output: new size = original + (n-1)*(res-1)
    target_size = matrix.shape[0] + (matrix.shape[0] - 1) * (resolution - 1)
    interpolated = interpolated[:target_size, :target_size]

    return interpolated



def get_instant_vol_func(tau , params):
    '''
    Return the instantaneous volatility ,
    computed in terms of the parametric
    form proposed by Rebonato , at a given time t.
    @var t: time at which we want to compute the
    instantaneous volatility (in years)
    @var expiry: caplet expiry (in years)
    @var a: parameter a of Rebonato ’s instant. vol. function
    @var b: parameter b of Rebonato ’s instant. vol. function
    @var c: parameter c of Rebonato ’s instant. vol. function
    @var d: parameter d of Rebonato ’s instant. vol. function
    
    #g(T - t) & h(T - t)
    '''
    tau = np.maximum(tau, 0)
    a,b,c,d = params
    instantaneous_vol = (a + b * tau) * np.exp(-c * tau) + d
    return instantaneous_vol


def build_phi_matrix(T, phi_diag, lambda3, lambda4, add_spot=True):
    n = len(T)
    phi = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Ti, Tj = T[i], T[j]
            phi_ii, phi_jj = phi_diag[i], phi_diag[j]
            A = np.sign(phi_ii) * np.sqrt(abs(phi_ii * phi_jj))
            decay = np.exp(-lambda3 * max(Ti - Tj, 0) - lambda4 * max(Tj - Ti, 0))
            phi[i, j] = A * decay

    return phi


def black_price(F, K, sigma, T, r=0.0, option_type="call"):
    """
    Black's formula for European options on forwards.

    Args:
        F (float): Forward rate
        K (float): Strike
        sigma (float): Implied volatility
        T (float): Time to maturity
        r (float): Discount rate (e.g. risk-free rate)
        option_type (str): 'call' or 'put'

    Returns:
        float: Present value of the option
    """
    print("T AND SIGMA",T, sigma)
    if np.isclose(T,0) or np.isclose(sigma,0):
        print("WE AT MATURITY")
        intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
        return np.exp(-r * T) * intrinsic

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

def pairwise_outer(arr):
    """
    Given an array of shape (..., d), return an array of shape (..., d, d),
    where each (...)-indexed vector is expanded to an outer product with itself.

    Parameters:
    - arr: np.ndarray, shape (..., d)

    Returns:
    - out: np.ndarray, shape (..., d, d)
    """
    return arr[..., :, None] * arr[..., None, :]



def create_test_lmm():
    # DATA LOADING AND CLEANING
    path_params = os.path.join(os.getcwd(), "parameters")
    forwards = load_object(path_params+"/spot_forwards.pkl")
    s0_exp = load_object(path_params+"/vol_initial_correction.pkl")
    epsilon_exp = load_object(path_params+"/volvol_initial_correction.pkl")
    doust_fwd_fwd = load_object(path_params+"/fwdfwd_corr.pkl")
    doust_vol_vol = load_object(path_params+"/volvol_corr.pkl")
    corr_fwd_vol = load_object(path_params+"/fwdvol_corr.pkl")
    params_g = load_object(path_params+"/vol_params_g.pkl")
    params_h= load_object(path_params+"/volvol_params_h.pkl")
    spots= load_object(path_params+"/spot_rates.pkl")
    params_g = np.array([-0.00557585, -0.00864318,  0.89466108,  0.00755986])
    params_h = np.array([1.42258187e-08, 3.01935702e01, 4.57201647e00, 4.05843346e-12,])
    epsilon_exp = np.concatenate([epsilon_exp[[0]],epsilon_exp])
    s0_exp = np.concatenate([s0_exp[[0]], s0_exp])



    rho_mat_6m = doust_fwd_fwd[:19, :19]
    theta_mat_6m = doust_vol_vol[:19, :19] #TODO: check if this is correct, or it should remove the first row and column instead
    phi_mat_diag = corr_fwd_vol
    fwd_tenors = np.arange(1,10.5,0.5)
    # self defined
    params_g = np.array([0.005, 0.04, 1, 0.001])
    params_h = np.array([0.001, 3, 3, 0.01])
    s0_exp = np.ones_like(s0_exp)


    # Create new matrices 


    beta_6m = 0.20696204
    beta_0m = 0.25697769
    beta_theta_0m = 0.1556888
    beta_theta_6m = 0.12135651
    n=20
    





    # create phi 



    # interpolate phi diag
    phi_diag = np.diag(corr_fwd_vol)

    phi_diag = np.concatenate([phi_diag[[0]],phi_diag])
    T_phi = np.arange(0,10, 0.5)

    # SHOULD BE IN LMM CLASS

    rho_mat_0m = doust_corr(beta_0m, n)
    theta_mat_0m = doust_corr(beta_theta_0m, n)
    phi_mat_0m = build_phi_matrix(T_phi, phi_diag, 0.0087931, 0.051319)


    # create rates 
    path = os.path.join(os.getcwd(), "raw_dataset")
    df_cap = pd.read_excel(path+"/caplet_raw.xlsx", sheet_name = 2, header = 0)
    df_raw_spot = pd.read_csv(path+"/spot.csv")
    df_raw_spot["Tenor"] = np.array([1/12, 2/12, 3/12, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 20, 30, 50])
    
    lmm = LMMSABR(
        rho_mat=rho_mat_0m,
        theta_mat=theta_mat_0m,
        phi_mat=phi_mat_0m,
        g_params=params_g,
        h_params=params_h,
        epsilon_exp=epsilon_exp,
        k0_exp=s0_exp,
        df_cap=df_cap,
        df_raw_spot=df_raw_spot,
        resolution=2, t_max=3, swap_hedge_expiry=1, swap_client_expiry=1
    )
    def single_run(seed):
        lmm.simulate(seed=seed)
        lmm.get_swap_matrix()
        res = lmm.get_sabr_params()
        return res

# LMMSABR RELATED FUNCS





def create_df_init(df_fwd, df_raw_spot, resolution, tau=0.5):

    # Get the LIBOR 6-month spot rate
    libor_6m_spot = df_raw_spot.loc[df_raw_spot["Tenor"] == 0.5, "Spot"].iloc[0] # floating point comparison only safe due to 0.5 bein representable with 2**-1

    # Create a dataframe with the initial values for the spot rate
    spot_row = pd.DataFrame({
        'Fixing': [0.0],
        'Reset Rate': [libor_6m_spot],  # Convert back to percentage
        'Maturity': [0.5]
    })

    # Initialize df_init with df_cap data
    df_full = df_fwd[['Fixing', 'Reset Rate', 'Maturity']].copy()

    # Concatenate with the spot rate row and reset index
    df_full = pd.concat([spot_row, df_full], ignore_index=True)
    df_full['Reset Rate'] = df_full['Reset Rate'] / 100  # Convert to percentage


    # =============================================================================
    #                               TIME INDEXING
    # =============================================================================
    ts_fwd_expiry = df_full['Fixing'].values

    dt = tau / resolution
    ids_fwd_interp = (ts_fwd_expiry / dt).astype(int) # divide by dt to get indices in the new time unit
    
    
    n_fwd = len(ts_fwd_expiry)-1 # exclude period covering the last forward rate tenor
    ts_fwd_interp= np.linspace(0, n_fwd*tau, int(n_fwd * resolution +1))
    #print(f"{ts_fwd_interp=}")
    assert np.all(np.isin(ts_fwd_expiry, ts_fwd_interp)), "Not all forward expirys are in the time grid"
    # =============================================================================




    # =============================================================================
    #           Create the zcb interpolated curve
    # =============================================================================


    fwd_canon = df_full['Reset Rate'].values
    discount_factors = 1 / (1 + fwd_canon[:-1] * 0.5) # leave out last as we dont use zcb prices after the last forward rate
    zcb_from_fwd = np.concatenate(([1], np.cumprod(discount_factors)))

    zcb_cs = interpolate.CubicSpline(ts_fwd_expiry, zcb_from_fwd)
    zcb_interp = zcb_cs(ts_fwd_interp)

    # =============================================================================


    # =============================================================================
    #          Construct dataframe with tenors, zcb and forward rates
    # =============================================================================
    df = pd.DataFrame({'Tenor': ts_fwd_interp, 'zcb': zcb_interp, 'Forward': np.nan, 's0': np.nan})
    df.loc[ids_fwd_interp, 'Forward'] = fwd_canon
    df.loc[ids_fwd_interp, 'k0'] = s0_exp
    # add column with backfilled forward indices, such that the value in this column is 0 from 0 to 5, 1 from 6 to 11, 2 from 12 to 17, etc.    
    df['i_s'] = (np.arange(len(df)) // resolution)*resolution
    df['i_sp1'] = (np.arange(len(df)) // resolution+1)*resolution
    df['mod_accrual'] = tau - (df['Tenor'] % tau)
    df_temp = df.merge(
    df[['zcb', 'Forward']],
    left_on='i_s',     # Column with pointers to index
    right_index=True,      # Use index from right DataFrame
    how='left',            # Keep all rows from original
    suffixes=('', '_i_s')  # Add suffix to avoid column name conflicts
    )
    df_temp = df_temp.merge(
    df[['zcb']],
    left_on='i_sp1',     # Column with pointers to index
    right_index=True,      # Use index from right DataFrame
    how='left',            # Keep all rows from original
    suffixes=('', '_i_sp1')  # Add suffix to avoid column name conflicts
    )
    df['gamma']= (df_temp['zcb'] / df_temp['zcb_i_sp1'] -1) / df_temp['mod_accrual'] /df_temp['Forward_i_s']


    # =============================================================================
    return df


def interp_func_fac(df_init, resolution=2, tau=0.5, beta=0.5, rho_mat=None, g_func=None, interp_vol = False, zcb_interp = False):
    df = df_init
    #fwd = df['Forward'].values # only used for test, not to be uncommented

    i_s = df['i_s'].values[:-resolution]
    i_sp1 = df['i_sp1'].values[:-resolution]
    i_e = i_sp1

    s = np.arange(len(df)-resolution)
    #print("len s",len(s))
    e = s + resolution

    theta = (df['mod_accrual'].values)
    gamma = (df['gamma'].values)
    gamma_theta = gamma * theta

    #print(f"{len(i_s)}, {len(i_e)}")
        



    def get_interp_rates(fwd):
        p_s_e = (1 + fwd[i_e] * gamma_theta[e]) / (1 + fwd[i_s] * gamma_theta[s]) * 1/(1+fwd[i_e]*tau)
        f_s_e = (1 / p_s_e - 1) / tau 
        return f_s_e
    
    
    if zcb_interp:


        def build_forward_zcb_matrix_from_f_sim(f_sim_input, max_tenor=10):
            """
            Compute a forward zero-coupon bond (ZCB) matrix P[t,e](t) in a fully vectorized manner,
            but only for maturities up to max_tenor (in years) beyond the current time.
            
            Parameters
            ----------
            f_sim : np.ndarray, shape (n, n)
                Simulated forward rate matrix. f_sim[t, i] is the forward rate observed at time t
                for the interval starting at the tenor corresponding to index i.
            gamma_theta : np.ndarray, shape (n,)
                Effective stub accrual factors (γ[i] * θ[i]) for each tenor.
            tau : float
                Canonical accrual period (e.g., 0.5 years).
            resolution : int
                Number of time steps per tau (e.g., 2 means grid spacing of tau/resolution, i.e., 0.25 years if tau=0.5).
            max_tenor : float
                Maximum tenor (in years) ahead of the current time for which to calculate ZCB prices.
            
            Returns
            -------
            P : np.ndarray, shape (n, n)
                The forward zero-coupon bond (ZCB) matrix such that P[t, e] = P(t, e)
                for e >= t and for e <= t + max_steps. Entries for e < t or e > t + max_steps are set to NaN.
            """
            f_sim = f_sim_input.copy()
            f_sim[np.isnan(f_sim)] = 0
            n = f_sim.shape[0]
            gamma_theta_trunc = gamma_theta[:n]  # Ensure gamma_theta is the same length as f_sim
            # --- Stub Start: for each row t, discount from t to the next stub endpoint.
            stub_start = 1.0 / (1.0 + np.diag(f_sim) * gamma_theta_trunc)  # shape (n,)
            
            # --- Stub End: for each (t,e), discount adjustment for the end stub.
            stub_end = 1.0 + f_sim * gamma_theta_trunc[np.newaxis, :]  # shape (n, n)
            
            # --- Canonical Product
            # Define canonical grid indices: these are every 'resolution' step.
            can = np.arange(0, n, resolution)  # e.g., [0, resolution, 2*resolution, ...]
            n_can = len(can)
            
            # Build a matrix M: for each row t, M[t, j] = 1/(1 + f_sim[t, can[j]] * tau)
            M = 1.0 / (1.0 + f_sim[:, can] * tau)  # shape (n, n_can)
            # Cumulative product along the canonical axis
            CP = np.cumprod(M, axis=1)  # shape (n, n_can)
            
            # For each row t, define J_s = floor(t/resolution) + 1 (starting canonical index)
            t_idx = np.arange(n)
            J_s = (t_idx // resolution) + 1  # shape (n,)
            # For each column e, define J_e = floor(e/resolution)
            J_e = np.arange(n) // resolution  # shape (n,)
            
            # Broadcast J_s and J_e to form matrices
            J_s_mat = J_s[:, None]  # shape (n, 1)
            J_e_mat = J_e[None, :]  # shape (1, n)
            
            # Get canonical cumulative product:
            A = CP[np.arange(n)[:, None], J_e_mat]  # shape (n, n)
            B = np.where(J_s_mat > 0,
                        CP[np.arange(n)[:, None], (J_s_mat - 1)],
                        1.0)
            # Canonical product for each (t,e)
            canon_prod = np.where(J_s_mat <= J_e_mat, A / B, 1.0)
            
            # --- Combine All Terms: full ZCB from t to e
            stub_start_matrix = stub_start[:, None]  # shape (n, 1)
            P = stub_start_matrix * canon_prod * stub_end  # shape (n, n)
            
            # Ensure lower-triangular entries (e < t) are NaN and diagonal is 1.
            P = np.triu(P, k=0)
            np.fill_diagonal(P, 1.0)
            
            # --- Apply max_tenor: Only keep P[t,e] for e <= t + max_steps.
            max_steps = int(max_tenor * resolution / tau)
            row_indices = np.arange(n)[:, None]
            col_indices = np.arange(n)[None, :]
            mask = col_indices > (row_indices + max_steps)
            P[mask] = np.nan
            # set lower triangle to nan
            P[np.tril_indices(n, k=-1)] = np.nan
            P[:,-resolution:] = np.nan
            return P





    if interp_vol:
        rho_mat_interpolated = interpolate_correlation_matrix(rho_mat, resolution)
        fwd = df['Forward'].values
        f1 = fwd[i_s]**beta      
        f2 = fwd[i_e]**beta
        w1 = gamma_theta[s] / tau
        w2 = (tau - gamma_theta[e]) / tau
        f_interp = get_interp_rates(fwd)**beta

        term1 = (w1**2) * (f1**2) / f_interp[s]**2
        term2 = (w2**2) * (f2**2) / f_interp[s]**2
        rho = rho_mat_interpolated[i_s, i_e]  
        cross = 2 * w1 * w2 * f1 * f2 * rho / f_interp[s]**2

        tenors = df['Tenor'].values
        ttm_mat = tenors[None, :] - tenors[:, None]
        g_mat = g_func(ttm_mat)

        def get_interp_vol_matrix(s_mat):
            """
            Vectorized volatility interpolation for an entire volatility matrix.

            Parameters
            ----------
            s_mat : np.ndarray of shape (n_steps, n_forwards)
                Matrix of instantaneous volatilities.

            Returns
            -------
            s_interp : np.ndarray of shape (n_steps, len(s))
                Interpolated volatilities at subgrid points.
            """
            s1 = s_mat[:, i_s]  # shape (n_steps, len(s))
            s2 = s_mat[:, i_e]  # shape (n_steps, len(s))

            # Compute numerator vectorized over time
            sigma_sum = (
                term1[None, :] * s1**2 +
                term2[None, :] * s2**2 +
                cross[None, :] * s1 * s2
            )  # shape (n_steps, len(s))

            s_interp = np.sqrt(sigma_sum)

            return s_interp / g_mat[:, s]  # shape (n_steps, len(s))

        if zcb_interp:
            return get_interp_rates, get_interp_vol_matrix, build_forward_zcb_matrix_from_f_sim
        else:
            return get_interp_rates, get_interp_vol_matrix
    
    return get_interp_rates



def get_swap_matrix(f_sim, T_idxs, resolution, tau, tenor, df, expiry=1, beta=0.5, B=0.5):
    """
    Compute time-evolving swap rates from a simulated forward path for a set of swap expiries.

    Parameters:
    - f_sim: np.ndarray, shape (n_steps, n_forwards)
        Simulated forward curve matrix (lower-triangular in time).
    - T_idxs: list or np.ndarray of int
        Forward indices (expiry) at which each swap starts.
    - resolution: int
        Number of simulation steps per accrual period (tau).
    - tau: float
        Accrual period of the swap (e.g., 0.5 for semiannual).
    - tenor: float
        Total length of the swap in years (e.g., 1.0 for a 1y swap).
    - df: pd.DataFrame
        DataFrame containing initial zero-coupon bond prices in column 'zcb'.
    - expiry: float, optional
        If provided, the maximum length of a simulated swap path will be limited to this value (in year units).
    Returns:
    - swap_paths: np.ndarray, shape (n_valid_steps, n_swaps)
        Matrix of swap rates over time for each swap expiry.
    - valid_steps: np.ndarray
        Array of time steps for which all required forward rates exist.
    - used_T_idxs: np.ndarray
        Final T_idxs that were valid and included in the result.
    """
    f_sim = np.asarray(f_sim)
    T_idxs = np.asarray(T_idxs)
    zcbs = df['zcb'].values

    n_steps, n_forwards = f_sim.shape
    n_payments = int(tenor / tau)
    swap_len = n_payments  # number of forward rates needed

    # Compute max usable T_idx based on number of forward rates
    max_T_idx = n_forwards - swap_len * resolution
    T_idxs = T_idxs#[T_idxs <= max_T_idx]
    if len(T_idxs) == 0:
        raise ValueError("No valid T_idxs remain after filtering based on swap length and resolution.")

    # Compute all valid time steps
    t_end = n_steps - swap_len * resolution
    valid_steps = np.arange(0, t_end)

    # Compute forward rate indices for each swap
    forward_offsets = np.arange(n_payments) * resolution
    col_indices = T_idxs[:, None] + forward_offsets[None, :]
    # Check bounds
    if np.any(col_indices >= n_forwards):
        raise IndexError("Computed forward indices exceed available f_sim columns.")

    # Compute ZCB indices needed for annuity weights
    zcb_offsets = np.arange(1,n_payments+1) * resolution
    zcb_indices = T_idxs[:, None] + zcb_offsets[None, :]
    if np.any(zcb_indices >= len(zcbs)):
        raise IndexError("Computed ZCB indices exceed available zcb entries.")
    

    # Compute frozen swap weights
    zcb_sets = np.stack([zcbs[idxs] for idxs in zcb_indices])
    swap_weights = zcb_sets * tau
    annuity = np.sum(swap_weights, axis=1, keepdims=True)
    swap_weights = swap_weights / annuity  # Normalize to 1

    # Gather simulated forward rate and volatility slices
    f_curves = f_sim[valid_steps]
    fwd_subsets = np.stack([f_curves[:, idxs] for idxs in col_indices], axis=1)

    #print(fwd_subsets.shape)
    # Compute weighted sum (dot product): (n_valid_steps, n_swaps)
    swap_paths = np.einsum('tsp,sp->ts', fwd_subsets, swap_weights)
    swap_paths[np.triu_indices_from(swap_paths, k=int(expiry*resolution/tau+1))] = np.nan  # Set upper triangle to NaN

    # Compute W: (n_valid_steps, n_swaps, n_payments)
    swaps_expanded = swap_paths[:, :, None]  # (n_valid_steps, n_swaps, 1)
    W = swap_weights[None, :, :] * (fwd_subsets**beta) / (swaps_expanded**B)  # default: beta=0.5, B=0.5
    #print(swap_weights)
    #W = np.nan_to_num(W)
    return swap_paths, W, annuity


def make_swap_indexer(n_steps, T_idxs, resolution, tau, tenor, return_indices=False):
    """
    this is a function factory to create an indexer that given a matrix of similar structure to f_sim
    will return an indexer function that takes a matrix of simulated values connected to a set of values relevant to the swap

    Parameters:
    - n_steps: int
        Number of simulation steps.
    - T_idxs: np.ndarray
        Array of forward indices where swaps start (e.g. expiries).
    - resolution: int
        Number of simulation steps per accrual period (tau).
    - tau: float
        Accrual period of the swap (e.g., 0.5 for semiannual).
    - tenor: float
        Total length of the swap in years (e.g., 1.0 for a 1y swap).
    - return_indices: bool, optional
        If True, return the valid steps and column indices used for indexing.
    Returns:
    - indexer: callable
        Function to index into the matrix of simulated forward rates.
    - (valid_steps, col_indices): tuple of np.ndarray
        If return_indices is True, returns the valid steps and column indices used for indexing.
    """
    n_payments = int(tenor / tau)
    swap_len = n_payments
    t_end = n_steps - swap_len * resolution
    valid_steps = np.arange(0, t_end)
    offsets = np.arange(n_payments) * resolution
    col_indices = T_idxs[:, None] + offsets[None, :]
    col_indices = np.broadcast_to(col_indices, (len(valid_steps), *col_indices.shape))

    def indexer(mat):
        mat_short = mat[valid_steps]
        mat_short = mat_short[:, None, :]
        return np.take_along_axis(mat_short, col_indices, axis=2) # shape (n_valid_steps, n_swaps, n_payments)
    if return_indices:
        return indexer, (valid_steps, col_indices)
    return indexer 


def build_swap_correlation_tensor(rho, T_idxs, resolution, tau, tenor, n_valid_steps=None):
    """
    Build a (n_valid_steps, n_swaps, n_payments, n_payments) tensor of forward correlations for each swap.

    Parameters:
    - rho: np.ndarray, shape (n_forwards, n_forwards)
        Correlation matrix between forward rates.
    - T_idxs: np.ndarray, shape (n_swaps,)
        Start indices for each swap.
    - resolution: int
        Number of forward steps per tau.
    - tau: float
        Accrual period (in years).
    - tenor: float
        Swap tenor (in years).
    - n_valid_steps: int
        Number of time steps to tile over.

    Returns:
    - rho_tensor: np.ndarray, shape (n_valid_steps, n_swaps, n_payments, n_payments)
    """
    if not n_valid_steps:
        n_valid_steps = len(T_idxs)
    n_swaps = len(T_idxs)
    n_payments = int(tenor / tau)
    rho_subs = np.empty((n_swaps, n_payments, n_payments))

    for i, T_idx in enumerate(T_idxs):
        indices = T_idx + np.arange(n_payments) * resolution
        rho_subs[i] = rho[np.ix_(indices, indices)]

    # Tile over time steps
    rho_tensor = np.tile(rho_subs[None, :, :, :], (n_valid_steps, 1, 1, 1))
    assert rho_tensor.shape[0] == rho_tensor.shape[1], f"Shape mismatch: {rho_tensor.shape[0]} != {rho_tensor.shape[1]}"
    #print(f"rho_tensor shape: {rho_tensor.shape}")
    return rho_tensor



# Batched stuff

def build_swap_correlation_tensor_batched(rho, T_idxs, resolution, tau, tenor, n_valid_steps=None):
    """
    Batched version of build_swap_correlation_tensor.

    Parameters:
    - rho: np.ndarray, shape (S, F, F)
        Correlation matrices for each simulation.
    - T_idxs: np.ndarray, shape (n_swaps,)
        Forward start indices of each swap.
    - resolution: int
        Steps per tau.
    - tau: float
        Accrual period.
    - tenor: float
        Swap length in years.
    - n_valid_steps: int, optional
        Number of valid simulation steps.

    Returns:
    - rho_tensor: np.ndarray, shape (S, n_valid_steps, n_swaps, n_payments, n_payments)
    """
    S, F, _ = rho.shape
    n_swaps = len(T_idxs)
    n_payments = int(tenor / tau)

    if n_valid_steps is None:
        n_valid_steps = len(T_idxs)

    rho_subs = np.empty((S, n_swaps, n_payments, n_payments))

    for j, T_idx in enumerate(T_idxs):
        indices = T_idx + np.arange(n_payments) * resolution  # (n_payments,)
        # Batched slice: (S, n_payments, n_payments)
        rho_subs[:, j] = np.take(rho, indices[:, None], axis=1)[..., indices]

    # Now tile over valid steps: (S, n_valid_steps, n_swaps, n_payments, n_payments)
    rho_tensor = np.tile(rho_subs[:, None, :, :, :], (1, n_valid_steps, 1, 1, 1))

    return rho_tensor


def make_swap_indexer_batched(n_steps, T_idxs, resolution, tau, tenor, return_indices=False):
    """
    Batched version of swap indexer — works on input shaped (n_sims, n_steps, n_forwards).

    Returns an indexer that extracts the floating leg reset values for all simulations.
    """
    n_payments = int(tenor / tau)
    swap_len = n_payments
    t_end = n_steps - swap_len * resolution
    valid_steps = np.arange(0, t_end)

    # (n_swaps, n_payments)
    offsets = np.arange(n_payments) * resolution
    col_indices = T_idxs[:, None] + offsets[None, :]
    
    # (n_valid_steps, n_swaps, n_payments)
    col_indices = np.broadcast_to(col_indices, (len(valid_steps), *col_indices.shape))

    def indexer(mat):
        """
        mat: np.ndarray of shape (n_sims, n_steps, n_forwards)
        returns: (n_sims, n_valid_steps, n_swaps, n_payments)
        """
        n_sims = mat.shape[0]

        # (n_valid_steps, n_steps) → pick rows from each sim
        mat_short = mat[:, valid_steps, :]  # shape: (n_sims, n_valid_steps, n_forwards)

        # Expand for swap index structure
        # We want: (n_sims, n_valid_steps, n_swaps, n_payments)
        # We can do this using np.take_along_axis

        # col_indices: (n_valid_steps, n_swaps, n_payments)
        # Broadcast to (1, n_valid_steps, n_swaps, n_payments)
        col_indices_batched = np.broadcast_to(col_indices, (n_sims, *col_indices.shape))

        # mat_short: (n_sims, n_valid_steps, n_forwards) → expand axis
        mat_short_expanded = mat_short[:, :, None, :]  # (n_sims, n_valid_steps, 1, n_forwards)

        return np.take_along_axis(mat_short_expanded, col_indices_batched, axis=3)

    if return_indices:
        return indexer, (valid_steps, col_indices)
    return indexer


def get_swap_matrix_batched(f_sim, T_idxs, resolution, tau, tenor, df, expiry=1, beta=0.5, B=0.5):
    """
    Batched version of get_swap_matrix.

    Parameters:
    - f_sim: np.ndarray, shape (n_sims, n_steps, n_forwards)
    - T_idxs: np.ndarray of shape (n_swaps,)
    - df: pd.DataFrame with column 'zcb'
    Returns:
    - swap_paths: (n_sims, n_valid_steps, n_swaps)
    - W: (n_sims, n_valid_steps, n_swaps, n_payments)
    """
    f_sim = np.asarray(f_sim)
    T_idxs = np.asarray(T_idxs)
    zcbs = df['zcb'].values

    S, n_steps, n_forwards = f_sim.shape
    n_payments = int(tenor / tau)
    swap_len = n_payments
    max_T_idx = n_forwards - swap_len * resolution

    if len(T_idxs) == 0:
        raise ValueError("No valid T_idxs remain after filtering.")

    t_end = n_steps - swap_len * resolution
    valid_steps = np.arange(0, t_end)

    # (n_swaps, n_payments)
    forward_offsets = np.arange(n_payments) * resolution
    col_indices = T_idxs[:, None] + forward_offsets[None, :]

    if np.any(col_indices >= n_forwards):
        raise IndexError("Forward indices exceed f_sim dimension.")

    # ZCB weights
    zcb_offsets = np.arange(1, n_payments + 1) * resolution
    zcb_indices = T_idxs[:, None] + zcb_offsets[None, :]

    if np.any(zcb_indices >= len(zcbs)):
        raise IndexError("ZCB indices exceed available entries.")

    # Frozen swap weights: (n_swaps, n_payments)
    zcb_sets = np.stack([zcbs[idxs] for idxs in zcb_indices])
    swap_weights = zcb_sets * tau
    swap_weights = swap_weights / swap_weights.sum(axis=1, keepdims=True)

    # Grab forward subsets
    f_curves = f_sim[:, valid_steps, :]  # (S, T', F)
    fwd_subsets = np.stack(
        [np.take(f_curves, idxs, axis=2) for idxs in col_indices], axis=2
    )  # shape: (S, T', n_swaps, n_payments)

    # Compute weighted swap rates: einsum over payments axis
    swap_paths = np.einsum("stnp,sp->stn", fwd_subsets, swap_weights)

    # Mask future expiry values
    expiry_cutoff = int(expiry * resolution / tau) + 1
    for i in range(swap_paths.shape[1]):
        if i > expiry_cutoff:
            swap_paths[:, i, :] = np.nan

    # Compute W
    swaps_expanded = swap_paths[..., None]  # (S, T', n_swaps, 1)
    weights_expanded = swap_weights[None, None, :, :]  # (1, 1, n_swaps, n_payments)
    W = weights_expanded * (fwd_subsets ** beta) / (swaps_expanded ** B)  # shape: (S, T', n_swaps, n_payments)

    return swap_paths, W

# CLASSES
class LMMSABR:
    def __init__(
        self,
        rho_mat,
        theta_mat,
        phi_mat,
        g_params,
        h_params,
        epsilon_exp,
        k0_exp,
        df_cap,
        df_raw_spot,
        tau=0.5,
        tenor=1,
        resolution=2,
        t_max=9.5,
        beta=0.5,
        B=0.5, swap_hedge_expiry=1, swap_client_expiry=2
        
        
    ):
        self.rho_mat = rho_mat
        self.theta_mat = theta_mat
        self.phi_mat = phi_mat
        self.g = partial(get_instant_vol_func, params=g_params)
        self.h = partial(get_instant_vol_func, params=h_params)
        self.epsilon_exp = epsilon_exp
        self.k0_exp = k0_exp

        self.tau = tau
        self.tenor = tenor
        self.resolution = resolution
        self.t_max = t_max
        self.dt = tau / resolution
        self.beta = beta
        self.B = B
        self.swap_hedge_expiry = swap_hedge_expiry
        self.swap_client_expiry = swap_client_expiry
        self.max_swap_expiry = np.maximum(self.swap_hedge_expiry, self.swap_client_expiry)

        # Store raw curve inputs
        self.df_cap = df_cap
        self.df_raw_spot = df_raw_spot

        # Placeholders
        self.df_init = None
        self.f_sim = None
        self.k_mat = None
        self.swap_sim = None
        start_curve_prep = time.time()
        self.prepare_curves()
        start_G_tensor = time.time()
        self.G_tensor = self.precompute_G_tensor()
        start_ggh_tensor = time.time()
        self.ggh_tensor = self.build_V_tensor_from_scalar( tenor=self.tenor, resolution=self.resolution, tau=self.tau)
        print(f"Curve preparation: {start_G_tensor - start_curve_prep:.2f}s")
        print(f"G tensor preparation: {start_ggh_tensor - start_G_tensor:.2f}s")
        print(f"V tensor preparation: {time.time() - start_ggh_tensor:.2f}s")
    def h_ij_vectorized_from_grid(self,t, u_arr, T_i, T_j):
        """
        Compute h_ij for each u in u_arr using the self.t_arr as the integration grid.
        Uses searchsorted for efficient pre-filtering of integration intervals.
        """
        t_arr = self.t_arr
        t_idx = np.searchsorted(t_arr, t, side='left')  # index just after t
        hij_vals = []

        for u in u_arr:
            if u <= t:
                hij_vals.append(0.0)
                continue

            u_idx = np.searchsorted(t_arr, u, side='right')
            s_grid = t_arr[t_idx:u_idx]
            if len(s_grid) == 0:
                hij_vals.append(0.0)
                continue

            h_prod = self.h(T_i - s_grid) * self.h(T_j - s_grid)
            integral = np.trapz(h_prod, s_grid)
            hij_vals.append(np.sqrt(integral / (u - t)))

        return np.array(hij_vals)


    def integral_term_V(self, t_idx, T_idx, i, j):
        """
        Compute the integral:
        ∫ₜᵀ g_i(u)*g_j(u)*[h_ij(t,u)]²*(u-t) du
        using lmm.t_arr as the integration grid.
        """
        t = self.t_arr[t_idx]
        T = self.t_arr[T_idx]
        if T <= t:
            return 0.0

        u_arr = self.t_arr[t_idx:T_idx+1]
        dt_arr = u_arr - t
        T_i = self.t_arr[i]
        T_j = self.t_arr[j]

        h_vals = self.h_ij_vectorized_from_grid(t, u_arr, T_i, T_j)
        h_sq = h_vals**2

        g_i_arr = self.g(T_i - u_arr)
        g_j_arr = self.g(T_j - u_arr)

        integrand = g_i_arr * g_j_arr * h_sq * dt_arr
        return np.trapz(integrand, u_arr)
        
    def build_V_tensor_from_scalar(self, tenor, resolution, tau):
        """
        Build the V_tensor using scalar integral_term_V, memoizing based on
        time-translation invariance.

        Returns:
        - V_tensor: np.ndarray, shape (n_t, n_t, n, n)
        """
        import numpy as np
        tenor = self.tenor
        resolution = self.resolution
        tau = self.tau
        max_expiry = self.max_swap_expiry
        n = int(tenor / tau)
        max_expiry_steps = int(max_expiry * resolution / tau)
        num_t = len(self.t_arr) - n * resolution
        V_tensor = np.full((num_t, num_t, n, n), np.nan)

        cache = {}  # key: (delta_T, delta_i, delta_j) -> float

        for t_idx in range(num_t):
            expiry_limit = min(t_idx + max_expiry_steps+1, num_t)

            for T_idx in range(t_idx, expiry_limit):
                start_idx = T_idx
                end_idx = T_idx + n * resolution
                indices = list(range(start_idx, end_idx, resolution))

                if len(indices) != n:
                    print(f"Skipping incomplete swap at indices: {indices}")
                    continue  # Skip incomplete swaps at boundary

                delta_T = T_idx - t_idx

                for i_local, i_global in enumerate(indices):
                    for j_local, j_global in enumerate(indices):
                        delta_i = i_global - T_idx
                        delta_j = j_global - T_idx

                        key = (delta_T, delta_i, delta_j)

                        if key not in cache:
                            # Compute and store
                            cache[key] = self.integral_term_V(
                                t_idx, T_idx, i_global, j_global
                            )
                        V_tensor[t_idx, T_idx, i_local, j_local] = cache[key]

        return V_tensor

    def precompute_G_tensor(self):
        """
        Precompute a G_tensor using memoization and the self.t_arr as integration grid.
        Uses:
            G[t_idx, T_idx, i_local, j_local] = ∫ₜᵀ g(T_i - u) * g(T_j - u) du
        with T_i, T_j based on T_idx and forward rate spacing.
        """
        import numpy as np

        t_arr = self.t_arr
        resolution = self.resolution
        tau = self.tau
        tenor = self.tenor
        max_expiry = self.max_swap_expiry

        num_t = len(t_arr) - int(tenor * resolution / tau)
        n = int(tenor / tau)
        G_tensor = np.zeros((num_t, num_t, n, n))*np.nan
        # set the diagonal of the n,m,k,l tensor to 0
        G_tensor[np.diag_indices(num_t)] = 0

        cache = {}  # (delta_T_idx, delta_i, delta_j) → float
        max_expiry_idx = int(max_expiry * resolution / tau)  # max expiry in steps
        for t_idx in range(num_t):
            for T_idx in range(t_idx, num_t):
                delta_T_idx = T_idx - t_idx
                # check for max expiry
                if delta_T_idx > max_expiry_idx:
                    continue
                # Use the actual model time grid for integration
                s_idx_start = t_idx   # strictly > t
                s_idx_end = T_idx + 1    # include T
                u_arr = t_arr[s_idx_start:s_idx_end]
                if len(u_arr) < 1:
                    G_tensor[t_idx, T_idx] = np.zeros((n, n))  # no integration needed
                    #print("Skipping integration for empty u_arr at indices:", s_idx_start, s_idx_end)
                    continue  # skip if no points to integrate over

                start_idx = T_idx
                end_idx = T_idx + n * resolution
                indices = list(range(start_idx, end_idx, resolution))
                if len(indices) != n:
                    continue  # incomplete swap

                for i_local, i_global in enumerate(indices):
                    for j_local, j_global in enumerate(indices):
                        delta_i = i_global - T_idx
                        delta_j = j_global - T_idx
                        key = (delta_T_idx, delta_i, delta_j)

                        if key not in cache:
                            T_i = t_arr[i_global]
                            T_j = t_arr[j_global]
                            g_i = self.g(T_i - u_arr)
                            g_j = self.g(T_j - u_arr)
                            cache[key] = np.trapz(g_i * g_j, u_arr)

                        G_tensor[t_idx, T_idx, i_local, j_local] = cache[key]

        return G_tensor



    def prepare_curves(self):
        self.df_init = create_df_init(
            self.df_cap, self.df_raw_spot, resolution=self.resolution, tau=self.tau
        ).query(f"Tenor <= {self.t_max + 1e-6}")
        self.tenors = self.df_init["Tenor"].values
        self.t_arr = self.tenors
        self.ids_fwd_canon = self.df_init["Forward"].dropna().index.values
        self.num_forwards = len(self.ids_fwd_canon)
        self.n_steps = len(self.df_init)

    def precompute_vol_surfaces(self):
        ttm_mat = self.tenors[None, :] - self.tenors[:,None]
        self.ttm_mat = ttm_mat

        self.h_mat = self.h(ttm_mat[1:, self.ids_fwd_canon])
        self.g_mat = self.g(ttm_mat[:, self.ids_fwd_canon])

    def precompute_interpolation(self):
        self.interp_func, self.interp_vol_func, self.zcb_interp_func = interp_func_fac(
            self.df_init,
            resolution=self.resolution,
            tau=self.tau,
            rho_mat=self.rho_mat,
            g_func=self.g,
            interp_vol=True,
            zcb_interp=True,

        )
        self.rho_mat_0m_interpolated = interpolate_correlation_matrix(self.rho_mat, self.resolution)
        self.theta_mat_0m_interpolated = interpolate_correlation_matrix(self.theta_mat, self.resolution)
        self.phi_mat_0m_interpolated = interpolate_correlation_matrix(self.phi_mat, self.resolution)

    def simulate_forwards(self, seed=None):
        np.random.seed(seed)
        dt = self.dt
        dt_sqrt = np.sqrt(dt)

        dZ_f = np.random.multivariate_normal(
            np.zeros(self.num_forwards),
            self.rho_mat[:self.num_forwards, :self.num_forwards],
            self.n_steps-1,
        ) * dt_sqrt
        dW_s = np.random.multivariate_normal(
            np.zeros(self.num_forwards),
            self.theta_mat[:self.num_forwards, :self.num_forwards],
            self.n_steps-1,
        ) * dt_sqrt

        f_0 = self.df_init["Forward"].values
        f_sim = np.full((self.n_steps, len(f_0)), np.nan)
        f_sim[0] = f_0   # temporary adjustment
        self.f_sim = f_sim
        self.dZ_f = dZ_f
        self.dW_s = dW_s

        self._simulate_vol_surface()
        self._simulate_forward_dynamics()

    def _simulate_vol_surface(self):
        
        
        g_mat = self.g_mat
        h_mat = self.h_mat

        k_mat = np.concatenate([self.k0_exp[:self.num_forwards].reshape(1, -1), (self.k0_exp[:self.num_forwards] * np.cumprod(1 + self.epsilon_exp[:self.num_forwards].reshape(1, -1) * self.dW_s * h_mat, axis=0))])

        self.k_mat = k_mat
        self.s_mat = g_mat * k_mat
        self.k_mat_full_res = np.zeros((self.n_steps, self.n_steps))*np.nan
        s_mat_full_res = np.zeros((self.n_steps, self.n_steps))*np.nan
        
        s_mat_full_res[:, self.ids_fwd_canon] = k_mat * self.g_mat

        
        self.k_mat_full_res[:,:-self.resolution] = self.interp_vol_func(s_mat_full_res)

        self.s_mat_full_res = self.k_mat_full_res * self.g(self.ttm_mat)
        
    def _simulate_forward_dynamics(self):
        interp_func = self.interp_func
        k_mat = self.k_mat

        f_sim = self.f_sim
        dZ_f = self.dZ_f

        
        ids_rev = self.ids_fwd_canon[::-1]
        ids_short_rev = ids_rev // self.resolution
        non_canon_idx = np.setdiff1d(np.arange(len(f_sim[0]))[:-self.resolution], self.ids_fwd_canon)
        f_sim[0, non_canon_idx] = self.interp_func(f_sim[0])[non_canon_idx]
        drift_correction = np.zeros(len(ids_rev))
        drift_shared = np.zeros(len(ids_rev))

        for t in range(1, self.n_steps):
            drift_correction.fill(0)
            drift_shared.fill(0)
            # next loop runs from longest to shortest tenor
            for canon_short_idx, canon_idx in zip(ids_short_rev, ids_rev):
                if self.ttm_mat[t, canon_idx] +self.tau+1e-8>= 0:     # TODO <------------ THIS IS IMPORTANT
                    s_t = self.s_mat[t-1, canon_short_idx]
                    dZ_f_t = dZ_f[t-1,canon_short_idx]
                    f_t = f_sim[t-1,canon_idx]
                    f_beta_t = f_t**self.beta
                    
                    drift_f = (-self.g_mat[t, canon_short_idx] * k_mat[t, canon_short_idx] * f_beta_t * drift_shared[canon_short_idx])
                    df_t =  drift_f + f_beta_t*s_t*dZ_f_t
                    
                    f_t_new =  f_t + df_t 
                    f_sim[t,canon_idx] = f_t_new if f_t_new > 0 else 0 # zero absorbing boundary

                    if canon_short_idx > 0:
                        drift_correction[canon_short_idx-1] = self.rho_mat[canon_short_idx-1, canon_short_idx] * self.tau * self.g_mat[t,canon_short_idx] * k_mat[t, canon_short_idx] * f_beta_t / (1 + self.tau * f_t)
                        drift_shared[canon_short_idx-1] = np.sum(drift_correction[canon_short_idx-1:])

            f_sim[t, non_canon_idx] = interp_func(f_sim[t])[non_canon_idx]

    def simulate(self, seed=None):
        #start_time = time.time()
        self.prepare_curves()
        self.precompute_vol_surfaces()
        self.precompute_interpolation()
        self.simulate_forwards(seed=seed)
        return self.f_sim
    
    
    def get_swap_matrix(self):
        
        T_idxs = np.arange(len(self.f_sim)-int(self.tenor/self.tau*self.resolution))
        swap_sim, W, annuity = get_swap_matrix( # TODO: make it so we get weights for both hedge and client swaps
            self.f_sim, T_idxs=T_idxs, resolution=self.resolution, tau=self.tau, tenor=self.tenor, df=self.df_init, expiry = self.swap_hedge_expiry, beta=self.beta, B=self.B
        )
        swap_sim[np.tril_indices_from(swap_sim, k=-1)] = np.nan
        self.swap_sim = swap_sim
        self.W = W
        self.annuity = annuity
        assert self.k_mat_full_res.shape == self.f_sim.shape, f"Shape mismatch: {self.k_mat_full_res.shape} != {self.f_sim.shape}"
        return swap_sim, W
    

    def get_sabr_params(self):
        # ==========================================================
        #          Create tensors for the SABR parameters
        # ==========================================================
        
        swap_idxs = np.arange(self.swap_sim.shape[0])
        swap_indexer = make_swap_indexer(n_steps = self.f_sim.shape[0], T_idxs=swap_idxs,resolution=self.resolution, tau=0.5, tenor=1,)
        k_tensor = swap_indexer(self.k_mat_full_res)

        rho_mat_0m_interpolated = self.rho_mat_0m_interpolated
        rho_tensor = build_swap_correlation_tensor(rho_mat_0m_interpolated, T_idxs=swap_idxs, resolution=self.resolution, tau=0.5, tenor=1)
        k_tensor_prod = pairwise_outer(k_tensor)
        W_tensor_prod = pairwise_outer(self.W)
        # ==========================================================
        # for the m,n,k,l tensor, sum such that we have a m,n tensor
        
        # ===========================================================
        #               compute sigma tensor
        # ==========================================================
        prod = rho_tensor*W_tensor_prod*k_tensor_prod*self.G_tensor

        numerator = np.sum(prod, axis=(2, 3))
        denominator = self.ttm_mat[np.ix_(swap_idxs, swap_idxs)]
        sigma_sq = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),  # fill result with 0 where denominator == 0
            where=denominator != 0
        )
        sigma = np.sqrt(sigma_sq)


        # ==========================================================
        #              compute V and Phi tensor       
        # ==========================================================

        



        
        V_terms = rho_tensor*W_tensor_prod*k_tensor_prod*self.ggh_tensor
        V_sum = np.sum(V_terms, axis=(2, 3))
        V_numerator = np.sqrt(2*V_sum)
        V_denominator = sigma*self.ttm_mat[np.ix_(swap_idxs, swap_idxs)]
        V = np.divide(
            V_numerator, 
            V_denominator,
            out=np.zeros_like(numerator)*np.nan,  # fill result with 0 where denominator == 0
            where=denominator != 0
        )


        # TODO: ALL TENSORS SHOULD BE SLICED TO HAVE [max_expiry:,...], this removes unnecessary computation
        omega_tensor = np.divide(V_terms, V_sum[..., None, None], out=np.zeros_like(V_terms), where=V_sum[..., None, None] != 0)
        phi_tensor = build_swap_correlation_tensor(self.phi_mat_0m_interpolated, T_idxs=swap_idxs, resolution=self.resolution, tau=0.5, tenor=1)
        phi = np.sum(phi_tensor * omega_tensor, axis=(2, 3))
        
        # SWAP INDEX EXPIRY OFFSET
        first_max_expiry_swap_idx = int(self.swap_hedge_expiry/self.tau*self.resolution)
        self.swap_hedge_expiry_idx = first_max_expiry_swap_idx
        
        atm_strikes_hedge = self.swap_sim.diagonal(offset=first_max_expiry_swap_idx)
        atm_strikes_hedge = np.tile(atm_strikes_hedge, (len(self.swap_sim), 1))
        
        
        
        annuity = np.tile(self.annuity.T, (len(self.swap_sim), 1))[:, first_max_expiry_swap_idx:]
        # remove the left_most elements of the sabr param matrices
        sigma = sigma[:, first_max_expiry_swap_idx:]
        V = V[:, first_max_expiry_swap_idx:]
        phi = phi[:, first_max_expiry_swap_idx:]
        swap_sim = self.swap_sim[:, first_max_expiry_swap_idx:]
        col_idxs = list(range(first_max_expiry_swap_idx, max(swap_idxs)+1))
        ttm_mat = self.ttm_mat[np.ix_(swap_idxs, col_idxs)]



        iv = self.sabr_implied_vol(F=swap_sim, K=atm_strikes_hedge, T=ttm_mat, alpha=sigma, beta=self.beta, rho=phi, nu=V)
        
        self.price, self.delta, self.gamma, self.vega = self.black_swaption_price(F=swap_sim, K=atm_strikes_hedge, T=ttm_mat, sigma=iv, annuity=annuity )
        
        # PnL stuff
        zcb = self.zcb_interp_func(self.f_sim)
        annuity = np.sum(swap_indexer(zcb), axis=2) * self.tau

        annuity[np.triu_indices_from(annuity, k=first_max_expiry_swap_idx+1)] = np.nan
        annuity = annuity[:, first_max_expiry_swap_idx:]

        self.swap_value = annuity * (self.swap_sim[:, first_max_expiry_swap_idx:]- atm_strikes_hedge )
        
        inactive = np.isnan(self.price)

        # make returned results into a tensor of matrices so self.price, self.delta, self.gamma, self.vega, inactive is a tensor of shape (5, n_steps, n_swaps)
        results = np.stack([self.price, self.delta, self.gamma, self.vega, inactive], axis=0), self.swap_value
        

        return results


    def sabr_implied_vol(self,
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    rho: np.ndarray,
    nu: np.ndarray,
    ):
        """
        Hagan SABR implied vol (not just ATM) with numpy broadcasting.

        Parameters
        ----------
        F : np.ndarray
            Forward swap rate (e.g., shape (steps, expiries))
        K : np.ndarray
            Strike rate (same shape as F for ATM, or broadcastable)
        T : np.ndarray
            Time to maturity in years
        alpha : np.ndarray
            Instantaneous vol (sigma0 in SABR)
        beta : float
            Elasticity parameter
        rho : np.ndarray
            SABR correlation
        nu : np.ndarray
            SABR vol-of-vol

        Returns
        -------
        np.ndarray
            SABR implied vol, same shape as inputs
        """
        F = np.maximum(F, 1e-8)
        K = np.maximum(K, 1e-8)
        T = np.maximum(T, 1e-8)
        log_FK = np.log(F / K)
        z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * log_FK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        # A and B terms
        FK_beta = (F * K) ** ((1 - beta) / 2)
        A = alpha / (FK_beta * (1 + (1 - beta) ** 2 * log_FK ** 2 / 24 + (1 - beta) ** 4 * log_FK ** 4 / 1920))
        B = (
            1
            + ((1 - beta) ** 2 / 24) * (alpha ** 2 / FK_beta ** 2)
            + (rho * beta * nu * alpha) / (4 * FK_beta)
            + ((2 - 3 * rho ** 2) * nu ** 2 / 24)
        ) * T

        # ATM simplified case
        atm_mask = np.isclose(F, K)
        sigma = np.full_like(F, np.nan)
        sigma[atm_mask] = (
            alpha[atm_mask]
            / (F[atm_mask] ** (1 - beta))
            * (1 + ((2 - 3 * rho[atm_mask] ** 2) / 24) * nu[atm_mask] ** 2 * T[atm_mask])
        )

        # General case
        non_atm = ~atm_mask
        sigma[non_atm] = A[non_atm] * z[non_atm] / x_z[non_atm] * B[non_atm]

        return sigma

    def black_swaption_price(self, F, K, T, sigma, annuity=1.0):
        """
        Black's formula for payer swaption pricing.

        Parameters
        ----------
        F : np.ndarray
            Forward swap rate (shape (...,))
        K : np.ndarray
            Strike swap rate (same shape as F)
        T : np.ndarray
            Time to maturity (in years)
        sigma : np.ndarray
            Implied volatility (same shape as F)
        annuity : np.ndarray or float
            Present value of fixed leg payments (default=1.0)

        Returns
        -------
        price : np.ndarray
            Black swaption price
        delta : np.ndarray
            dPrice/dF
        gamma : np.ndarray
            d²Price/dF²
        vega : np.ndarray
            dPrice/dVol
        """
        F = np.maximum(F, 1e-8)
        K = np.maximum(K, 1e-8)
        # get expiry mask where T is close to 0
        expiry_mask = np.isclose(T, 0)
        T = np.maximum(T, 1e-8)
        
        sigma = np.maximum(sigma, 1e-8)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        n_prime = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)

        price = annuity * (F * norm.cdf(d1) - K * norm.cdf(d2))
        # at ttm == 0, price is just the annuity * max(F-K)
        price[expiry_mask] = annuity[expiry_mask] * np.maximum(F[expiry_mask] - K[expiry_mask], 0)
        delta = annuity * norm.cdf(d1)
        gamma = annuity * n_prime / (F * sigma * sqrt_T)
        vega = annuity * F * n_prime * sqrt_T / 100  # divide by 100 for % vol bump

        return price, delta, gamma, vega



    def plot(self, mat):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Create mesh grid the shape of self.swap_sim
        X = np.arange(mat.shape[0])*self.dt
        Y = np.arange(mat.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = mat.T
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        # Add labels and colorbar
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Expiry')
        ax.set_zlabel('Rate')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        # angle so we look down from above
        ax.view_init(elev=45, azim=210)


#single_run(42)
from joblib import Parallel, delayed

n_jobs = -1  # uses all available CPU cores
n_sims = 100

results = Parallel(n_jobs=n_jobs)(
    delayed(single_run)(seed) for seed in range(n_sims))
# import utils.py
import environment.utils as utils
# reload import 
from importlib import reload
reload(utils)

utils.Utils(lmm.swap_hedge_expiry_idx, np_seed=42, num_sim = lmm.swap_sim.shape[1]).convert_tensor_to_option_objects(results)
utils.Utils(lmm.swap_hedge_expiry_idx, np_seed=42, num_sim = lmm.swap_sim.shape[1]).agg_poisson_dist(np.ones_like(results[0][0]), np.ones_like(results[0][0]))
