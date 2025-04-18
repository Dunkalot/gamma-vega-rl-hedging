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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import interpolate
from functools import partial
from scipy.stats import norm
import ipympl
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
import time
import cv2
from scipy.optimize import minimize
from tqdm import tqdm
from numba import njit, prange
import psutil
import os
from pathlib import Path
import pickle
@njit
def _simulate_core(f_sim, k_mat, g_mat, h_mat, phi_mat, rho_mat, dZ_f, dW_s, n_steps, dt, tau, rev_short, rev_ids, ttm_mat, beta):
    for t in range(n_steps-1):
        drift_sum = 0.0
        for short_i, canon_i in zip(rev_short, rev_ids):
            if ttm_mat[t, canon_i] + tau + 1e-8 < 0:
                continue
            
            
            g_it = g_mat[t, canon_i]
            h_it = h_mat[t, canon_i]
            k_it = k_mat[t, canon_i]

            f_t = f_sim[t, canon_i]
            k_t = k_mat[t, canon_i]
            f_beta = f_t ** beta
            
            gkfb   = g_it * k_it * f_beta
            phk   = phi_mat[short_i, short_i]*h_it * k_it

            
            # calculate drift
            drift_f  = -gkfb * drift_sum
            drift_k = -phk * drift_sum
            #print(drift_f, drift_k)
            

            # calculate change
            df     = drift_f * dt + f_beta * g_it * k_it * dZ_f[t, short_i] # already multiplied by sqrt(dt)
            dk     = drift_k * dt + h_it * k_it * dW_s[t, short_i] # already multiplied by sqrt(dt)
            
            # update values
            f_new  = f_t + df
            k_new  = k_t + dk


            f_sim[t+1, canon_i] = f_new if f_new > 0 else 1e-4
            k_mat[t+1, canon_i] = k_new if k_new > 0 else 1e-4
            
            if short_i > 0:
                corr = rho_mat[short_i-1, short_i] * tau * gkfb / (1 + tau * f_t)
                drift_sum += corr
    return f_sim, k_mat






def make_nss_yield_df(start_date='1986-12-01', end_date='2025-03-01', max_maturity=120):
    df = pd.read_csv('../feds200628.csv', skiprows=9)
    df = df[['Date', 'BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']]
    df['Date'] = pd.to_datetime(df['Date'])#.dt.date
    
    n_years = np.arange(0,max_maturity) / 12
    yields = np.empty((len(df), len(n_years)))
    for i, n in enumerate(n_years):
        tau1, tau2 = df['TAU1'], df['TAU2']
        term1 = (1 - np.exp(-n / tau1)) / (n / tau1)
        term2 = np.exp(-n / tau1)
        term3 = (1 - np.exp(-n / tau2)) / (n / tau2)
        term4 = np.exp(-n / tau2)
        yields[:, i] = df['BETA0'] + df['BETA1'] * term1 + df['BETA2'] * (term1 - term2) + df['BETA3'] * (term3 - term4)

    yield_cols = [f'{int(m*12)}m' for m in n_years]
    yields_df = pd.DataFrame(yields, columns=yield_cols, index=df['Date'])
    yields_df = yields_df[start_date:end_date]#.dropna()#.resample('BME').last().to_period('M')
    return yields_df / 100


def compute_6m_forward_curve(yields_df, tau=0.5):
    maturities = np.array([int(col.strip('m')) for col in yields_df.columns]) / 12
    fwd_df = pd.DataFrame(index=yields_df.index, columns=yields_df.columns[:-6])
    tau_in_months = int(tau * 12)
    for i, T in enumerate(maturities[:-6]):
        y_T = yields_df.iloc[:, i]
        y_Tp = yields_df.iloc[:, i + 6]
        P_T = np.exp(-T * y_T)
        P_Tp = np.exp(-(T + 0.5) * y_Tp)
        fwd_df.iloc[:, i] = (P_T / P_Tp - 1) / 0.5

    return fwd_df.astype(float)


def estimate_X_full_PCA(df_forwards, n_components=5):
    # Optionally skip short end
    #X_maturities_idx = np.arange(3, df_forwards.shape[1])  # skip first 3
    pca_data = df_forwards#.iloc[:, X_maturities_idx]
    
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)

    pca = PCA(n_components=n_components, whiten=False)
    X_full = pca.fit_transform(pca_data_scaled)

    df_pcs = pd.DataFrame(X_full, index=df_forwards.index, columns=[f"PC{i+1}" for i in range(n_components)])
    return df_pcs, pca, scaler


def tag_regime_and_stress(date):
    if isinstance(date, pd.Period):
        date = date.to_timestamp()

    if date < pd.Timestamp('1987-08-11'):
        return 'Volcker', 3
    elif date < pd.Timestamp('1995-06-01'):
        return 'Greenspan I', 1
    elif date < pd.Timestamp('2001-01-01'):
        return 'Greenspan II', 0
    elif date < pd.Timestamp('2004-06-01'):
        return 'Post-dotcom easing', 1
    elif date < pd.Timestamp('2007-07-01'):
        return 'Pre-GFC tightening', 1
    elif date < pd.Timestamp('2010-01-01'):
        return 'GFC', 4
    elif date < pd.Timestamp('2016-01-01'):
        return 'ZIRP/QE1-3', 0
    elif date < pd.Timestamp('2020-01-01'):
        return 'Normalization', 1
    elif date < pd.Timestamp('2020-06-01'):
        return 'COVID crash', 4
    elif date < pd.Timestamp('2022-03-01'):
        return 'Pandemic QE', 0
    else:
        return 'QT/Inflation', 2.5


def compute_6m_forward_dataframe(yields_df, tau=0.5):
    """
    Compute a DataFrame of 6M simple-compounded forward rates with tau-spaced columns.
    Assumes yield curve starts at time 0 and is monthly spaced.

    Parameters
    ----------
    yields_df : pd.DataFrame
        DataFrame of continuous-compounded yields (decimal format), starting at 0m with 1M spacing.
        
    tau : float
        Forward tenor in years (default 0.5 for 6M).
        
    Returns
    -------
    df_fwd : pd.DataFrame
        DataFrame of 6M forward rates with columns tau-spaced (e.g., '0m', '6m', '12m', ...)
    """
    tau_in_months = int(tau * 12)
    total_months = len(yields_df.columns)
    maturities = np.arange(total_months) / 12  # 0m to (n-1)m in years

    # Choose columns spaced every tau months starting from 0
    fwd_col_indices = np.arange(0, total_months - tau_in_months, tau_in_months)
    fwd_col_names = [f"{i}m" for i in fwd_col_indices]

    df_fwd = pd.DataFrame(index=yields_df.index, columns=fwd_col_names)

    # Extract necessary slices
    T = maturities[fwd_col_indices].reshape(1, -1)
    y_T = yields_df.iloc[:, fwd_col_indices].values
    y_Tp = yields_df.iloc[:, fwd_col_indices + tau_in_months].values

    P_T = np.exp(-T * y_T)
    P_Tp = np.exp(-(T + tau) * y_Tp)

    fwd = (P_T / P_Tp - 1) / tau
    df_fwd.loc[:, :] = fwd
    if "0m" in df_fwd.columns:
        df_fwd["0m"] = yields_df.iloc[:, tau_in_months]  # spot 6M yield (continuous)
        df_fwd["0m"] = (np.exp(tau * df_fwd["0m"]) - 1) / tau  # convert to simple-compounded
    df_fwd = df_fwd.dropna().astype(float)
    df_fwd[['regime', 'stress']] = df_fwd.index.to_series().apply(lambda d: pd.Series(tag_regime_and_stress(d)))
    return df_fwd


def sample_forward_curves(forwards_df, regime=None, stress=None, n=100):
    """
    Sample forward curves from historical data with optional filtering on regime or stress.

    Parameters:
    - forwards_df: DataFrame with columns [fwd1, fwd2, ..., regime, stress]
    - regime: optional string or list of regimes to filter on
    - stress: optional numeric (or list of values) to filter on
    - n: number of samples to draw

    Returns:
    - np.ndarray of shape (n, #forwards)
    """
    df = forwards_df.copy()

    if regime is not None:
        if isinstance(regime, str):
            regime = [regime]
        df = df[df['regime'].isin(regime)]

    if stress is not None:
        if isinstance(stress, (int, float)):
            stress = [stress]
        df = df[df['stress'].isin(stress)]

    if len(df) == 0:
        raise ValueError("No data matches the specified regime/stress filter.")

    idx = df.sample(n=n, replace=True).index
    return df.loc[idx, df.columns[:-2]].values  # exclude regime and stress


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
    return doust


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


def interpolate_correlation_matrix_cv(matrix: np.ndarray, resolution: int) -> np.ndarray:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    # Compute the target size based on the resolution factor
    target_size = matrix.shape[0] + (matrix.shape[0] - 1) * (resolution - 1)
    # cv2.resize expects the shape as (width, height); ensure your matrix is in float32 if needed.
    return cv2.resize(matrix.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_LINEAR)


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


def sample_phi_diag_mvn(T, mean=-0.5, sigma=0.1, beta=0.1, clip_bounds=(-0.9, -0.3)):
    """
    Sample diagonal of phi matrix (rate-vol correlation) using a multivariate normal, with vectorized covariance computation.
    """
    T = np.asarray(T)
    n = len(T)

    # Efficiently compute the covariance using the outer difference
    diff = np.subtract.outer(T, T)
    cov = sigma**2 * np.exp(-beta * np.abs(diff))

    # Draw from MVN with specified mean and covariance
    phi_diag = np.random.multivariate_normal(mean=np.full(n, mean), cov=cov)

    # Clip to valid correlation bounds
    return np.clip(phi_diag, clip_bounds[0], clip_bounds[1])


def build_phi_matrix(T, phi_diag, lambda3, lambda4):
    """
    Construct the phi matrix from the given diagonal and decay parameters in a vectorized manner.
    """
    T = np.asarray(T)
    n = len(T)

    # Compute the pairwise differences
    T_diff = np.subtract.outer(T, T)

    # Compute the decay factors using vectorized operations:
    decay = np.exp(-lambda3 * np.maximum(T_diff, 0) - lambda4 * np.maximum(-T_diff, 0))
    
    # Instead of looping, build the A component as an outer product.
    # Note: sqrt(|phi_diag[i] * phi_diag[j]|) equals sqrt(|phi_diag[i]|) * sqrt(|phi_diag[j]|)
    A = np.outer(np.sign(phi_diag) * np.sqrt(np.abs(phi_diag)), np.sqrt(np.abs(phi_diag)))
    
    # Multiply elementwise to get the final phi matrix
    return A * decay


def sample_phi_matrix(
    T, 
    mean=-0.5, 
    sigma=0.1, 
    beta_corr=0.1, 
    lambda3_range=(0.005, 0.05), 
    lambda4_range=(0.01, 0.07), 
    clip_bounds=(-0.9, -0.3)
):
    """
    Sample a full phi matrix, with rate-vol correlation structure for caplets.

    Parameters:
    - T: Tenor grid (e.g., np.arange(0, 10, 0.5))
    - mean, sigma, beta_corr: parameters for phi_diag sampling
    - lambda3_range, lambda4_range: asymmetric decay parameters
    - clip_bounds: restrict sampled diagonal entries

    Returns:
    - phi matrix
    - metadata dict with sampled values
    """
    # Sample diagonal from MVN
    phi_diag = sample_phi_diag_mvn(T, mean, sigma, beta_corr, clip_bounds)

    # Sample asymmetric exponential decay parameters
    lambda3 = np.random.uniform(*lambda3_range)
    lambda4 = np.random.uniform(*lambda4_range)

    # Build full matrix
    phi = build_phi_matrix(T, phi_diag, lambda3, lambda4)

    return phi, {
        "phi_diag": phi_diag,
        "lambda3": lambda3,
        "lambda4": lambda4
    }


def create_df_init_from_forward(fwd, resolution=2, tau=0.5):
    """
    Initialize LMM-compatible DataFrame from a vector of simple-compounded forward rates.
    
    Parameters
    ----------
    fwd : array-like
        1D array of forward rates (simple-compounded) spaced by tau. First rate is the spot tau forward.
    resolution : int, optional
        Number of interpolation steps per tau interval. Default is 2.
    tau : float, optional
        Length of forward interval in years. Default is 0.5 (6 months).
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with Tenor, zcb, Forward, gamma, and interpolation structure (i_s, i_sp1, mod_accrual).
    """
    fwd = np.asarray(fwd)
    #print(fwd)
    n_fwd = len(fwd)
    
    # Step 1: Time grid
    ts_fwd_fixing = np.arange(n_fwd) * tau
    dt = tau / resolution
    ts_fwd_interp= np.linspace(0, n_fwd*tau, int(n_fwd * resolution +1))
    ids_fwd_interp = (ts_fwd_fixing / dt).astype(int)
    
    # Step 2: Compute ZCB curve from forwards
    discount_factors = 1 / (1 + fwd * tau)
    zcb_from_fwd = np.concatenate(([1.0], np.cumprod(discount_factors)))
    zcb_cs = interpolate.CubicSpline(ts_fwd_fixing, zcb_from_fwd[:-1])
    zcb_interp = zcb_cs(ts_fwd_interp)
    
    # Step 3: Build output DataFrame
    df = pd.DataFrame({
        'Tenor': ts_fwd_interp,
        'zcb': zcb_interp,
        'Forward': np.nan,
    })

    df.loc[ids_fwd_interp, 'Forward'] = fwd

    # Step 4: Drift calculation terms
    df['i_s'] = (np.arange(len(df)) // resolution) * resolution
    df['i_sp1'] = df['i_s'] + resolution
    df['mod_accrual'] = tau - (df['Tenor'] % tau)

    df_temp = df.merge(
        df[['zcb', 'Forward']],
        left_on='i_s', right_index=True,
        how='left', suffixes=('', '_i_s')
    ).merge(
        df[['zcb']],
        left_on='i_sp1', right_index=True,
        how='left', suffixes=('', '_i_sp1')
    )

    df['gamma'] = (
        (df_temp['zcb'] / df_temp['zcb_i_sp1'] - 1)
        / df_temp['mod_accrual']
        / df_temp['Forward_i_s']
    )

    return df


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
    df = pd.DataFrame({'Tenor': ts_fwd_interp, 'zcb': zcb_interp, 'Forward': np.nan})
    df.loc[ids_fwd_interp, 'Forward'] = fwd_canon
    #df.loc[ids_fwd_interp, 'k0'] = s0_exp
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


def interp_func_fac(df_init, resolution=2, tau=0.5, beta=0.5, rho_mat_interpolated=None, g_func=None, interp_vol = False, zcb_interp = False):
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
        



    def get_interp_rates(f_sim):
        f_s = f_sim[:, i_s]
        f_e = f_sim[:, i_e]
        p_s_e = (1 + f_e * gamma_theta[e]) / (1 + f_s * gamma_theta[s]) * 1/(1+f_e*tau)
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
        fwd = df['Forward'].values
        f1 = fwd[i_s]**beta      
        f2 = fwd[i_e]**beta
        w1 = gamma_theta[s] / tau
        w2 = (tau - gamma_theta[e]) / tau
        f_interp = get_interp_rates(fwd[None,:])**beta
        f_interp = f_interp.squeeze(axis=0)
        #print(f"w1 shape {w1.shape}, w2 shape {w2.shape}, f1 shape {f1.shape}, f2 shape {f2.shape}, f_interp shape {f_interp.shape}")
        term1 = (w1**2) * (f1**2) / f_interp[s]**2
        term2 = (w2**2) * (f2**2) / f_interp[s]**2
        rho = rho_mat_interpolated[i_s, i_e]  
        cross = 2 * w1 * w2 * f1 * f2 * rho / f_interp[s]**2

        tenors = df['Tenor'].values
        ttm_mat = tenors[None, :] - tenors[:, None]
        g_mat = g_func(ttm_mat)

        def get_interp_vol_matrix(s_mat,f_mat):
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


def get_swap_matrix(f_sim, shape, resolution, tau, tenor, df, expiry_max=1, expiry_min=1, beta=0.5, B=0.5):
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
    expiry_min_idx = int(expiry_min * resolution / tau)
    n_swaps = np.arange(shape[1]+expiry_min_idx)
    zcbs = df['zcb'].values

    n_steps, n_forwards = f_sim.shape
    n_payments = int(tenor / tau)
    swap_len = n_payments  # number of forward rates needed

    # Compute max usable T_idx based on number of forward rates
    #max_T_idx = n_forwards -   swap_len * resolution
    #n_swaps = n_swaps # we remove the first columns afterwards so that the first column has expiry_min time to expiry
    if len(n_swaps) == 0:
        raise ValueError("No valid T_idxs remain after filtering based on swap length and resolution.")

    # Compute all valid time steps
    t_end = shape[0]
    valid_steps = np.arange(0, t_end, dtype=np.int32)

    # Compute forward rate indices for each swap
    forward_offsets = np.arange(n_payments) * resolution
    col_indices = n_swaps[:, None] + forward_offsets[None, :]
    # Check bounds
    if np.any(col_indices >= n_forwards):
        raise IndexError("Computed forward indices exceed available f_sim columns.")

    # Compute ZCB indices needed for annuity weights
    zcb_offsets = np.arange(1,n_payments+1, dtype=np.float32) * resolution
    zcb_indices = n_swaps[:, None] + zcb_offsets[None, :]
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
    swap_paths[np.triu_indices_from(swap_paths, k=int(expiry_max*resolution/tau+1))] = np.nan  # Set upper triangle to NaN

    # Compute W: (n_valid_steps, n_swaps, n_payments)
    swaps_expanded = swap_paths[:, :, None]  # (n_valid_steps, n_swaps, 1)
    W = swap_weights[None, :, :] * (fwd_subsets**beta) / (swaps_expanded**B)  # default: beta=0.5, B=0.5
    #print(swap_weights)
    #W = np.nan_to_num(W)
    

    return swap_paths[:,expiry_min_idx:], W[:,expiry_min_idx:] # we start cut the first columns so the first column has has expiry_min time to expiry


def make_swap_indexer(n_steps, swap_idxs, resolution, tau, tenor, expiry, return_indices=False):
    """
    this is a function factory to create an indexer that given a matrix of similar structure to f_sim
    will return an indexer function that takes a matrix of simulated values connected to a set of values relevant to the swap

    Parameters:
    - n_steps: int
        Number of simulation steps.
    - shape: tuple
        Shape of the swap matrix that the final output will have for the first two dimensions.
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
    valid_steps = swap_idxs[0]
    col_indices = swap_idxs[1]
    offsets = np.arange(n_payments, dtype=np.int32) * resolution
    col_indices = col_indices[:, None] + offsets[None, :]
    col_indices = np.broadcast_to(col_indices, (len(valid_steps), *col_indices.shape))
    def indexer(mat):
        mat_short = mat[valid_steps]
        mat_short = mat_short[:, None, :]
        return np.take_along_axis(mat_short, col_indices, axis=2) # shape (n_valid_steps, n_swaps, n_payments)
    if return_indices:
        return indexer, (valid_steps, col_indices)
    return indexer 


def min_max_rebonato_vol(params, max_allowed=np.inf):
    """
    Compute the minimum and maximum value of the Rebonato instantaneous volatility function
    over τ ≥ 0, and check whether it is strictly within bounds.

    Parameters
    ----------
    params : tuple or list of length 4
        The parameters (a, b, c, d) of the Rebonato volatility function.
    max_allowed : float
        Optional cap on maximum allowed volatility.

    Returns
    -------
    min_val : float
        The minimum value of the volatility function over τ ≥ 0.
    max_val : float
        The maximum value of the volatility function over τ ≥ 0.
    tau_star : float
        The point τ* where the minimum occurs (if within domain).
    is_valid : bool
        True if min_val > 0 and max_val <= max_allowed.
    """
    a, b, c, d = params

    if b == 0:
        # Monotonic decay or flat
        tau_vals = np.array([0, 100])
        vol_vals = (a + b * tau_vals) * np.exp(-c * tau_vals) + d
        min_val = np.min(vol_vals)
        max_val = np.max(vol_vals)
        return min_val, max_val, 0.0, (min_val > 0 and max_val <= max_allowed)

    # Analytical critical point
    tau_star = (b - c * a) / (c * b)

    def sigma(tau):
        return (a + b * tau) * np.exp(-c * tau) + d

    # Evaluate function at key points
    eval_points = [0, 100]
    if tau_star >= 0:
        eval_points.append(tau_star)

    vols = [sigma(t) for t in eval_points]
    min_val = min(vols)
    max_val = max(vols)
    is_valid = (min_val > 0) and (max_val <= max_allowed)

    return min_val, max_val, tau_star, is_valid


def sample_phi_matrix_batch(
    T,
    n_samples=100,
    mean=-0.5,
    sigma=0.1,
    beta_corr=0.1,
    lambda3_range=(0.005, 0.05),
    lambda4_range=(0.01, 0.07),
    clip_bounds=(-0.9, -0.3)
):
    """
    Generate a batch of phi matrices with realistic rate-vol correlation structure.

    Returns:
    - List of phi matrices
    - List of metadata dictionaries
    """
    phi_matrices = []
    metadata_list = []

    for _ in range(n_samples):
        phi, meta = sample_phi_matrix(
            T,
            mean=mean,
            sigma=sigma,
            beta_corr=beta_corr,
            lambda3_range=lambda3_range,
            lambda4_range=lambda4_range,
            clip_bounds=clip_bounds
        )
        phi_matrices.append(phi)
        metadata_list.append(meta)

    return phi_matrices, metadata_list


def sample_exponential_corr_matrix_batch(
    T,
    n_samples=100,
    eta_range=(0.1, 0.6),
    lambda_range=(0.05, 0.5)
):
    """
    Batch sampler for exponential decay correlation matrices.

    Parameters:
    - T: array of tenor times
    - n_samples: number of matrices to generate
    - eta_range: asymptotic correlation lower bound (min, max)
    - lambda_range: exponential decay rate (min, max)

    Returns:
    - List of correlation matrices
    - List of metadata dicts with eta, lambda
    """
    T = np.asarray(T)
    n = len(T)

    matrices = []
    meta_list = []

    for _ in range(n_samples):
        eta = np.random.uniform(*eta_range)
        lam = np.random.uniform(*lambda_range)

        # Build the matrix
        T_i, T_j = np.meshgrid(T, T)
        abs_diff = np.abs(T_i - T_j)
        corr = eta + (1 - eta) * np.exp(-lam * abs_diff)

        matrices.append(corr)
        meta_list.append({'eta': eta, 'lambda': lam})

    return matrices, meta_list


def create_df_inits_from_samples(tau=0.5, resolution=2, n_samples=100):
    df_fwd = compute_6m_forward_dataframe(make_nss_yield_df())
    random_samples = sample_forward_curves(df_fwd, n=n_samples)
    df_init_list = [create_df_init_from_forward(sample, resolution=resolution, tau=tau) for sample in random_samples]
    return df_init_list


def sample_positive_rebonato_params(n_samples=1,max_allowed = 0.1, seed=None, volvol=False):
    """
    Generate Rebonato parameter sets (a, b, c, d) such that the corresponding vol surface is strictly positive.

    Parameters
    ----------
    n_samples : int
        Number of valid samples to generate.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    params_list : list of tuples
        List of (a, b, c, d) tuples with strictly positive volatility surface.
    vol_funcs : list of callables
        List of functions vol(tau) = sigma(tau) with those parameters.
    """
    rng = np.random.default_rng(seed)
    valid_params = []
    vol_funcs = []
    
    while len(valid_params) < n_samples:
        state = 0#rng.choice([0, 1], p=[0.5, 0.5])
        if volvol:
            # Define bounds for low and high states
            low_bounds = {'a': 0.5, 'b': 0.0002, 'c': 1.9, 'd': 0.27}
            high_bounds = {'a': 1.1, 'b': 0.00021, 'c': 2.3, 'd': 0.3}

            # Randomly choose state (0 for low, 1 for high) with probability p=0.5
            if state == 0:  # Low bound state
                a = rng.uniform(low_bounds['a'], low_bounds['a'] + 0.1)
                b = rng.uniform(low_bounds['b'], low_bounds['b'] + 0.00001)
                c = rng.uniform(low_bounds['c'], low_bounds['c'] + 0.1)
                d = rng.uniform(low_bounds['d'], low_bounds['d'] + 0.01)
            else:  # High bound state
                a = rng.uniform(high_bounds['a'] - 0.1, high_bounds['a'])
                b = rng.uniform(high_bounds['b'] - 0.00001, high_bounds['b'])
                c = rng.uniform(high_bounds['c'] - 0.1, high_bounds['c'])
                d = rng.uniform(high_bounds['d'] - 0.01, high_bounds['d'])
        else:
            # Define bounds for low and high states
            low_bounds = {'a': 0.0, 'b': 0.02, 'c': 0.5, 'd': 0.02}
            high_bounds = {'a': 0.04, 'b': 0.15, 'c': 1.2, 'd': 0.026}

            # Randomly choose state (0 for low, 1 for high) with probability p=0.5
            if state == 0:  # Low bound state
                a = rng.uniform(low_bounds['a'], low_bounds['a'] + 0.01)
                b = rng.uniform(low_bounds['b'], low_bounds['b'] + 0.01)
                c = rng.uniform(low_bounds['c'], low_bounds['c'] + 0.1)
                d = rng.uniform(low_bounds['d'], low_bounds['d'] + 0.002)
            else:  # High bound state
                a = rng.uniform(high_bounds['a'] - 0.01, high_bounds['a'])
                b = rng.uniform(high_bounds['b'] - 0.01, high_bounds['b'])
                c = rng.uniform(high_bounds['c'] - 0.1, high_bounds['c'])
                d = rng.uniform(high_bounds['d'] - 0.002, high_bounds['d'])
        param_set = (a, b, c, d)

        # Check positivity
        min_val,max_val, _, is_valid = min_max_rebonato_vol(param_set,max_allowed=max_allowed)
        if is_valid:
            valid_params.append(param_set)
            vol_funcs.append(partial(get_instant_vol_func, params=param_set))

    return valid_params, vol_funcs





class LMMSABR:
    def __init__(
        self,
        tau=0.5,
        resolution=2,
        tenor=1,
        sim_time = 1,
        t_max=None,
        beta=0.5,
        B=0.5, swap_hedge_expiry=1, swap_client_expiry=2
        
        
    ):
        self.samples = None
        self.tau = np.float32(tau)
        self.resolution = np.int32(resolution)
        self.dt = np.float32(tau / resolution)
        self.beta = np.float32(beta)
        self.B = np.float32(B)
        self.sim_time = np.float32(sim_time)
        self.primed = False
        # swap params
        self.prime_swap_data(swap_hedge_expiry, swap_client_expiry, tenor)

        if t_max is None:
            print(f"t_max set to {self.max_swap_expiry + self.tenor + self.sim_time}")
            self.t_max = self.max_swap_expiry + tenor + self.sim_time
        else:
            self.t_max = t_max
        self.t_max = np.float32(self.t_max)
        self.t_arr = np.linspace(0, self.t_max, int(self.t_max/self.dt +1), dtype=np.float32)
        ttm_mat = self.t_arr[None, :] - self.t_arr[:,None]
        self.ttm_mat = ttm_mat
        self.swap_indexer, self.swap_indices = make_swap_indexer(n_steps = len(self.t_arr), swap_idxs = self.swap_idxs,resolution=self.resolution, tau=tau, tenor=self.tenor,expiry=self.max_swap_expiry, return_indices=True)


    def prime_swap_data(self, swap_hedge_expiry, swap_client_expiry, tenor):
        self.tenor = tenor
        assert swap_hedge_expiry != swap_client_expiry, "swap_hedge_expiry and swap_client_expiry should be different"
        self.swap_hedge_expiry = swap_hedge_expiry
        self.swap_liab_expiry = swap_client_expiry
        self.max_swap_expiry = np.maximum(self.swap_hedge_expiry, self.swap_liab_expiry)
        self.min_swap_expiry = np.minimum(self.swap_hedge_expiry, self.swap_liab_expiry)
        self.swap_hedge_expiry_relative = self.swap_hedge_expiry - self.min_swap_expiry
        self.swap_liab_expiry_relative = self.swap_liab_expiry - self.min_swap_expiry
        self.swap_sim_shape = self.t_to_idx(self.sim_time), self.t_to_idx(self.max_swap_expiry - self.min_swap_expiry+ self.sim_time)
        self.swap_idxs = np.arange(self.swap_sim_shape[0]),np.arange(self.t_to_idx(self.min_swap_expiry), self.swap_sim_shape[1]+self.t_to_idx(self.min_swap_expiry))
        
    
    def t_to_idx(self, t):
        """
        Convert time time units to index units.
        """
        return int(t / self.dt)
    

    def sample_starting_conditions(
        self,
        df_fwd=None,
        n_samples=1,
        n_curves=None,
        random_curves = False,
        rho_kwargs=None,
        theta_kwargs=None,
        phi_kwargs=None,
        g_kwargs=None,
        h_kwargs=None,
        fwd_kwargs=None,
        seed=None,
    ):
        """
        Sample and store LMM SABR starting conditions.

        Parameters:
        - n_samples: number of samples
        - rho_kwargs, theta_kwargs: kwargs for exponential correlation sampler
        - phi_kwargs: kwargs for phi matrix sampler
        - g_kwargs, h_kwargs: kwargs for Rebonato parameter samplers
        - fwd_kwargs: kwargs passed to `sample_forward_curves` (e.g. regime, stress)
        - seed: RNG seed for reproducibility
        """
        if df_fwd is None:
            df_fwd = compute_6m_forward_dataframe(make_nss_yield_df())
        T = self.t_arr
        rho_kwargs = rho_kwargs or {}
        theta_kwargs = theta_kwargs or {}
        phi_kwargs = phi_kwargs or {}
        g_kwargs = g_kwargs or {'max_allowed': 0.2}
        h_kwargs = h_kwargs or {'max_allowed': 1.0}
        fwd_kwargs = fwd_kwargs or {}
        
        self.df_fwd = df_fwd
        rho_batch, rho_meta = sample_exponential_corr_matrix_batch(T, n_samples, **rho_kwargs)
        theta_batch, theta_meta = sample_exponential_corr_matrix_batch(T, n_samples, **theta_kwargs)
        phi_batch, phi_meta = sample_phi_matrix_batch(T, n_samples, **phi_kwargs)
        g_params, _ = sample_positive_rebonato_params(n_samples=n_samples, seed=seed, **g_kwargs)
        h_params, _ = sample_positive_rebonato_params(n_samples=n_samples, volvol=True, seed=seed, **h_kwargs)
        if random_curves:
            fwd_samples = sample_forward_curves(df_fwd, n=n_samples if n_curves == None else n_curves, **fwd_kwargs)
        else:
            print("random_curves set to False, using the first n_curves of the dataframe")
            df_fwd_idx = df_fwd.index[:n_curves]
            fwd_samples = df_fwd.loc[df_fwd_idx, df_fwd.columns[:-2]].values
        
        self.df_init_list = [create_df_init_from_forward(fwd, resolution=self.resolution, tau=self.tau) for fwd in fwd_samples]
        
        self.samples = [
            {
                "rho_mat": rho_batch[i],
                "theta_mat": theta_batch[i],
                "phi_mat": phi_batch[i],
                "g_params": g_params[i],
                "h_params": h_params[i],
                "df_init": self.df_init_list[i],
                "meta": {
                    "rho_meta": rho_meta[i],
                    "theta_meta": theta_meta[i],
                    "phi_meta": phi_meta[i],
                },
            }
            for i in range(n_samples)
        ]
    
    
    def summary(self):
        """
        Print a summary of the sampled starting conditions:
        - Regime counts
        - Stress distribution
        - Optional metadata diagnostics
        """
        if self.samples is None or len(self.samples) == 0:
            print("No samples loaded.")
            return

        # Extract regimes and stress levels from df_init metadata
        regimes = []
        stress_levels = []

        for sample in self.samples:
            df = sample["df_init"]
            if "regime" in df.columns and "stress" in df.columns:
                # Extract from df_init if tagging is preserved
                regimes.append(df["regime"].iloc[0])
                stress_levels.append(df["stress"].iloc[0])
            else:
                # Fallback if not tagged (e.g., test cases)
                regimes.append("Unknown")
                stress_levels.append(np.nan)

        import pandas as pd
        regime_counts = pd.Series(regimes).value_counts()
        stress_counts = pd.Series(stress_levels).value_counts().sort_index()

        print("Sampled Regime Breakdown:")
        print(regime_counts.to_string())
        print("\nSampled Stress Levels:")
        print(stress_counts.to_string())

    
    def prime(self, sample_idx=0):
        if self.samples is None:
            raise ValueError("You must call sample_starting_conditions() first.")
        if not (0 <= sample_idx < len(self.samples)):
            raise IndexError(f"sample_idx={sample_idx} is out of bounds.")

        sample = self.samples[sample_idx]
        self._primed_sample = sample  # Optional: stash for later reference

        # Assign to model state — depends on how your class expects these
        self.rho_mat = sample["rho_mat"]
        self.theta_mat = sample["theta_mat"]
        self.phi_mat = sample["phi_mat"]
        self.g_params = sample["g_params"]
        self.h_params = sample["h_params"]
        self.df_init = sample["df_init"]
        self.g = partial(get_instant_vol_func, params=self.g_params)
        self.h = partial(get_instant_vol_func, params=self.h_params)
        self.interpolate_corr_matrices()
        self.prepare_curves()
        self.rho_tensor = self.build_swap_correlation_tensor(self.rho_mat_interp)
        self.theta_tensor = self.build_swap_correlation_tensor(self.theta_mat_0m_interpolated)
        self.phi_tensor = self.build_swap_correlation_tensor(self.phi_mat_0m_interpolated)
        self.G_tensor = self.precompute_G_tensor()
        self.ggh_tensor = self.build_V_tensor_from_scalar(tenor=self.tenor, resolution=self.resolution, tau=self.tau)[np.ix_(self.swap_idxs[0], self.swap_idxs[1])]
        self.primed = True
 
    def get_sample_meta(self, sample_idx=0):
        if self.samples is None:
            raise ValueError("No samples loaded.")
        return self.samples[sample_idx]["meta"]


    def integral_term_V(self, t_idx, T_idx, i, j):
        """
        Compute the integral:
            ∫ₜᵀ g_i(u)*g_j(u)*[h_ij(t,u)]²*(u-t) du,
        where
            h_{ij}(t,u) = sqrt( (∫ₜᵤ h(T_i-s)*h(T_j-s) ds) / (u-t) ).
        
        This vectorized version uses cumulative trapezoidal integration on self.t_arr.
        
        Parameters
        ----------
        t_idx : int
            Index in self.t_arr corresponding to time t.
        T_idx : int
            Index in self.t_arr corresponding to time T.
        i : int
            Index for T_i in self.t_arr.
        j : int
            Index for T_j in self.t_arr.
        
        Returns
        -------
        result : float
            The computed integral.
        """
        # Get the time grid segment from t to T:
        t_arr = self.t_arr  # full time grid, assumed 1D numpy array
        t = t_arr[t_idx]
        T = t_arr[T_idx]
        
        if T <= t:
            return 0.0

        # Define the integration domain (u = integration variable)
        u_arr = t_arr[t_idx:T_idx+1]
        t = t_arr[t_idx]
        delta = u_arr - t  # vector of (u - t)

        # Compute h_i(u) and h_j(u) over the integration range
        h_i_all = self.h(t_arr[i] - u_arr)
        h_j_all = self.h(t_arr[j] - u_arr)

        # Elementwise product
        h_prod = h_i_all * h_j_all

        # Cumulative trapezoidal integration over u_arr
        integrals_full = cumulative_trapezoid(h_prod, u_arr, initial=0)

        # Compute ∫ₜᵘ [...] as I(u) - I(t) = integrals_full - integrals_full[0]
        I_segment = integrals_full  # already matches u_arr
        
        # Compute the integrated value over [t, u] as I(u) - I(t)
        # Avoid division by 0 by setting very small delta values to np.nan (or zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(I_segment - integrals_full[t_idx],
                            delta,
                            out=np.zeros_like(delta),
                            where=delta>1e-10)
        
        # Compute h_ij(t, u) for each u in u_arr:
        h_ij_vals = np.sqrt(ratio)
        
        # Compute g_i(u) and g_j(u) on the same slice:
        g_i_vals = self.g(self.t_arr[i] - u_arr)
        g_j_vals = self.g(self.t_arr[j] - u_arr)
        
        # Now, the integrand becomes: 
        # g_i(u)*g_j(u)*[h_ij(t,u)]^2*(u-t)
        integrand = g_i_vals * g_j_vals * (h_ij_vals**2) * delta
        
        # Finally, integrate using the trapezoidal rule over u_arr:
        result = np.trapz(integrand, u_arr)
        return result

    
    def build_swap_correlation_tensor(self, corr_mat):
        """
        Build a (n_valid_steps, n_swaps, n_payments, n_payments) tensor of forward correlations for each swap.

        Parameters:
        - corr_mat: np.ndarray, shape (n_forwards, n_forwards)
            Correlation matrix between forward rates.

        Returns:
        - corr_tensor: np.ndarray, shape (n_valid_steps, n_swaps, n_payments, n_payments)
        """
        n_valid_steps = len(self.swap_idxs[0])
        
        n_swaps = len(self.swap_idxs[1])
        n_payments = int(self.tenor / self.tau)
        corr_subs = np.empty((n_swaps, n_payments, n_payments))
        expiry_idxs = self.swap_idxs[1]
        for i, T_idx in enumerate(expiry_idxs):
            indices = T_idx + np.arange(n_payments, dtype=np.int32) * self.resolution
            corr_subs[i] = corr_mat[np.ix_(indices, indices)]

        # Tile over time steps
        corr_tensor = np.tile(corr_subs[None, :, :, :], (n_valid_steps, 1, 1, 1))
        return corr_tensor        
    

    def build_V_tensor_from_scalar(self, tenor, resolution, tau):
        """
        Build the V_tensor using scalar integral_term_V, memoizing based on
        time-translation invariance.

        Returns:
        - V_tensor: np.ndarray, shape (n_t, n_t, n, n)
        """
        tenor = self.tenor
        resolution = self.resolution
        tau = self.tau
        max_expiry = self.max_swap_expiry
        n = int(tenor / tau)
        max_expiry_steps = int(max_expiry * resolution / tau)
        num_t = len(self.t_arr) - n * resolution
        n_swap_t = len(self.swap_idxs[0])
        V_tensor = np.full((n_swap_t, num_t, n, n), np.nan)
        cache = {}  # key: (delta_T, delta_i, delta_j) -> float

        for t_idx in range(n_swap_t):
            expiry_limit = min(t_idx + max_expiry_steps+1, num_t)

            for T_idx in range(t_idx, expiry_limit):

                start_idx = T_idx
                end_idx = T_idx + n * resolution
                indices = list(range(start_idx, end_idx, resolution))


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

        t_arr = self.t_arr
        resolution = self.resolution
        tau = self.tau
        tenor = self.tenor
        max_expiry = self.max_swap_expiry

        num_t = len(t_arr) - int(tenor * resolution / tau)#-self.t_to_idx(self.min_swap_expiry)
        n_swap_t = len(self.swap_idxs[0])
        n = int(tenor / tau)
        G_tensor = np.zeros((n_swap_t, num_t, n, n))*np.nan
        # set the diagonal of the n,m,k,l tensor to 0
        G_tensor[np.diag_indices(n_swap_t)] = 0

        cache = {}  # (delta_T_idx, delta_i, delta_j) → float
        max_expiry_idx = int(max_expiry * resolution / tau)  # max expiry in steps
        for t_idx in range(n_swap_t):
            for T_idx in range(np.maximum(t_idx -self.t_to_idx(self.min_swap_expiry), 0), num_t):
                delta_T_idx = T_idx - t_idx
                # check for max expiry
                if delta_T_idx > max_expiry_idx:
                    continue
                # Use the actual model time grid for integration
                s_idx_start = t_idx   # strictly > t
                s_idx_end = T_idx + 1    # include T
                u_arr = t_arr[s_idx_start:s_idx_end]
                if len(u_arr) < 1:
                    G_tensor[t_idx, T_idx] = np.nan#np.zeros((n, n))  # no integration needed
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
        idxs = np.ix_(self.swap_idxs[0],self.swap_idxs[1])
        return G_tensor[idxs]


    def prepare_curves(self, shuffle_df=False, seed=None):
        np.random.seed(seed)
        if shuffle_df:
            self.df_init = self.df_init_list[np.random.randint(len(self.df_init_list))]
        else:
            self.df_init = self.df_init
        self.df_init: pd.DataFrame = self.df_init.query(f"Tenor <= {self.t_max + 1e-6}")
        self.tenors = self.df_init["Tenor"].values
        self.t_arr = self.tenors
        self.ids_fwd_canon = self.df_init["Forward"].dropna().index.values
        self.num_forwards = len(self.ids_fwd_canon)
        self.n_steps = len(self.df_init)


        self.interp_func, self.interp_vol_func, self.zcb_interp_func = interp_func_fac(
            self.df_init,
            resolution=self.resolution,
            tau=self.tau,
            rho_mat_interpolated=self.rho_mat_interp,
            g_func=self.g,
            interp_vol=True,
            zcb_interp=True,beta=1
        )


    def precompute_instant_vol(self):
        

        self.h_mat = self.h(self.ttm_mat)
        self.g_mat = self.g(self.ttm_mat)


    def interpolate_corr_matrices(self):
        self.rho_mat_interp = interpolate_correlation_matrix_cv(self.rho_mat, self.resolution)
        self.theta_mat_0m_interpolated = interpolate_correlation_matrix_cv(self.theta_mat, self.resolution)
        self.phi_mat_0m_interpolated = interpolate_correlation_matrix_cv(self.phi_mat, self.resolution)

    
    def simulate_forwards(self, seed=None, minimum_starting_rate=0.01):
        np.random.seed(seed)
        dt = self.dt
        dt_sqrt = np.sqrt(dt)

        self.dZ_f = np.random.multivariate_normal(
            np.zeros(self.num_forwards),
            self.rho_mat[:self.num_forwards, :self.num_forwards],
            self.n_steps-1) * dt_sqrt
        
        self.dW_s = np.random.multivariate_normal(
            np.zeros(self.num_forwards),
            self.theta_mat[:self.num_forwards, :self.num_forwards],
            self.n_steps-1) * dt_sqrt

        f_0 = self.df_init["Forward"].values
        # shift f_0 up by the difference between the minimum starting rate and the minimum starting rate
        f_0_min = np.nanmin(f_0)
        # calculated the needed shift
        shift = max(minimum_starting_rate - f_0_min, 0)
        # apply the shift
        f_0 = f_0 + shift
        
        self.f_sim = np.full((self.n_steps, len(f_0)), np.nan)
        self.k_mat = np.full_like(self.f_sim, np.nan)
        self.f_sim[0] = f_0   # temporary adjustment
        self.k_mat[0] = np.ones_like(f_0, dtype=np.float32)

        self._simulate_forward_dynamics()
        self._interpolate_vol()  # ex-post interpolation of vol



        
    def _simulate_forward_dynamics(self):

        ids         = self.ids_fwd_canon
        ids_short   = ids // self.resolution
        rev_idx     = ids[::-1]
        rev_short   = ids_short[::-1]
        
        self.f_sim, self.k_mat = _simulate_core(
            f_sim=self.f_sim, 
            k_mat=self.k_mat, 
            g_mat=self.g_mat, 
            h_mat=self.h_mat, 
            phi_mat=self.phi_mat, 
            rho_mat=self.rho_mat, 
            dZ_f=self.dZ_f, 
            dW_s=self.dW_s, 
            n_steps=self.n_steps, 
            dt=self.dt, 
            tau=self.tau, 
            rev_short=rev_short, 
            rev_ids=rev_idx, 
            ttm_mat=self.ttm_mat, 
            beta=self.beta
        )
    
    def _interpolate_vol(self):

        s_mat = self.k_mat * self.g_mat
        self.s_mat = s_mat
        self.k_mat_interp = self.interp_vol_func(s_mat, self.f_sim)

        self.s_mat_interp = self.k_mat_interp * self.g(self.ttm_mat)[:,:self.k_mat_interp.shape[1]]
        all_idx     = np.arange(self.f_sim.shape[1])
        non_canon   = np.setdiff1d(all_idx[:-self.resolution], self.ids_fwd_canon)
        self.f_sim[:, non_canon] = self.interp_func(self.f_sim)[:,non_canon]

        

    def simulate(self, shuffle_df=True,seed=None,minimum_starting_rate=0.01):
        #start_time = time.time()
        self.prepare_curves(shuffle_df=shuffle_df, seed=seed)
        self.precompute_instant_vol()
        self.simulate_forwards(seed=seed, minimum_starting_rate=minimum_starting_rate)

        return self.f_sim
    
    

    def get_swap_matrix(self):
        """
        Returns
        -------
        swap_sim : ndarray (n_valid_steps, n_swaps)
        W_dynamic: ndarray (n_valid_steps, n_swaps, n_payments)
        """

        # zcb and indexing, creating n_steps, n_swaps, n_fixings, n_fixings tensor
        P             = self.zcb_interp_func(self.f_sim)
        P_sets        = self.swap_indexer(P)         # P^f(t, T_{leg})
        fwd_subsets   = self.swap_indexer(self.f_sim)  # L^leg_t

        #  annuity and weights
        w_t           = self.tau * P_sets
        annuity_t     = w_t.sum(axis=2, keepdims=True)
        w_t          /= annuity_t              # (t_valid, n_swaps, n_payments)


        #  swap rate
        swap_paths    = (w_t * fwd_subsets).sum(axis=2)
        swap_paths[np.triu_indices_from(swap_paths, k= self.t_to_idx(self.max_swap_expiry-self.min_swap_expiry)+1)] = np.nan
        annuity_t[np.triu_indices_from(swap_paths, k= self.t_to_idx(self.max_swap_expiry-self.min_swap_expiry)+1)] = np.nan


        # weights for Sigma_0 for the SABR parameter creation
        swap_lvl      = swap_paths[:, :, None]     # (t, swap, 1)
        W_dynamic     = w_t * (fwd_subsets**self.beta) / (swap_lvl**self.B)

        #  store & return the valid slice -----------------------------------
        self.swap_sim = swap_paths
        self.W        = W_dynamic
        self.annuity = annuity_t.squeeze(-1)



        return self.swap_sim, self.W, self.annuity


    def get_sabr_params(self):
        # ==========================================================
        #          Create tensors for the SABR parameters
        # ==========================================================
        
        swap_idxs_0 = np.arange(self.swap_sim.shape[0])
        swap_idxs_1 = np.arange(self.swap_sim.shape[1])+self.t_to_idx(self.min_swap_expiry)
        k_tensor = self.swap_indexer(self.k_mat_interp)


        rho_tensor = self.rho_tensor
        theta_tensor = self.theta_tensor
        phi_tensor = self.phi_tensor
        k_tensor_prod = pairwise_outer(k_tensor)
        W_tensor_prod = pairwise_outer(self.W)
        # ==========================================================
        # for the m,n,k,l tensor, sum such that we have a m,n tensor
        
        # ===========================================================
        #               compute sigma tensor
        # ==========================================================
        #print(f"theta_tensor shape: {theta_tensor.shape}, k_tensor shape: {k_tensor.shape}, rho_tensor shape: {rho_tensor.shape}, W_tensor_prod shape: {W_tensor_prod.shape}, G_tensor shape: {self.G_tensor.shape}")
        prod = rho_tensor*W_tensor_prod*k_tensor_prod*self.G_tensor

        numerator = np.sum(prod, axis=(2, 3))

        denominator = self.ttm_mat[np.ix_(swap_idxs_0, swap_idxs_1)]
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

        



        
        V_terms = rho_tensor*theta_tensor*W_tensor_prod*k_tensor_prod*self.ggh_tensor
        V_sum = np.sum(V_terms, axis=(2, 3))
        V_numerator = np.sqrt(2*V_sum)
        V_denominator = sigma*self.ttm_mat[np.ix_(swap_idxs_0, swap_idxs_1)]
        V = np.divide(
            V_numerator, 
            V_denominator,
            out=np.zeros_like(numerator)*np.nan,  # fill result with 0 where denominator == 0
            where=denominator != 0
        )
        # print shape of all component tensors
        #print(f"V_terms shape: {V_terms.shape}, V_sum shape: {V_sum.shape}, V_numerator shape: {V_numerator.shape}, V_denominator shape: {V_denominator.shape}")

        # TODO: ALL TENSORS SHOULD BE SLICED TO HAVE [max_expiry:,...], this removes unnecessary computation
        omega_tensor = np.divide(V_terms, V_sum[..., None, None], out=np.zeros_like(V_terms), where=V_sum[..., None, None] != 0)
        
        phi = np.sum(phi_tensor * omega_tensor, axis=(2, 3))
        
        # SWAP INDEX EXPIRY OFFSET
        swap_hedge_expiry_relative_idx = self.t_to_idx(self.swap_hedge_expiry_relative)
        swap_liab_expiry_relative_idx = self.t_to_idx(self.swap_liab_expiry_relative)

        

        ttm_mat = self.ttm_mat[np.ix_(swap_idxs_0, swap_idxs_1)]
        def risk_metrics(offset):
            atm_strikes = self.swap_sim.diagonal(offset=offset)
            atm_strikes = np.tile(atm_strikes, (self.swap_sim.shape[0], 1))
            # remove upper triangular part of the matrix
            atm_strikes[np.triu_indices_from(atm_strikes, k=1)] = np.nan
            # sabr
            start = offset
            end = offset + self.swap_sim.shape[0] # each day a new swap, so there must be as many strikes as time steps
            sigma_ofs = sigma[:, start:end]
            V_ofs = V[:, start:end]
            phi_ofs = phi[:, start:end]
            # swap
            ttm_mat_ofs = ttm_mat[:, start:end]
            annuity_ofs = self.annuity[:, start:end]
            # SWAPTION pricing
            swap_sim_ofs = self.swap_sim[:, start:end]
            #print("sigmas")
            #print(np.max(np.nan_to_num(sigma_ofs)), np.min(np.nan_to_num(sigma_ofs)))
            iv = self.sabr_implied_vol(F=swap_sim_ofs, K=atm_strikes, T=ttm_mat_ofs, alpha=sigma_ofs, beta=self.beta, rho=phi_ofs, nu=V_ofs)
            price, delta, gamma, vega = self.black_swaption_price(F=swap_sim_ofs, K=atm_strikes, T=ttm_mat_ofs, sigma=iv, annuity=annuity_ofs)
            active = ~np.isnan(price)
                    # Initialize with zeros (or nan if you prefer to explicitly mask unused rows)
            swaption_pnl = np.zeros_like(price)
            
            # Compute safe differences
            swaption_pnl[:-1] = np.subtract(
                price[1:, :],       # tomorrow's price
                price[:-1, :],      # today's price
                out=swaption_pnl[:-1, :],
                where=~np.isnan(price[1:, :]) & ~np.isnan(price[:-1, :])
            )

            
            # SWAP pricing
            swap_value = annuity_ofs * (swap_sim_ofs- atm_strikes )

            swap_pnl = np.zeros_like(swap_value)
            swap_pnl[:-1] = np.subtract(
                swap_value[1:, :],       # tomorrow's price
                swap_value[:-1, :],      # today's price
                out=swap_pnl[:-1, :],
                where=~np.isnan(swap_value[1:, :]) & ~np.isnan(swap_value[:-1, :])
            )

            return np.stack([price, delta, gamma, vega, swaption_pnl, active], axis=-1), np.stack([swap_value, swap_pnl, active, annuity_ofs, swap_sim_ofs], axis=-1)
        
        
        hedge_metrics = risk_metrics(swap_hedge_expiry_relative_idx)
        liab_metrics = risk_metrics(swap_liab_expiry_relative_idx)

        return hedge_metrics, liab_metrics


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
        X = np.arange(mat.shape[0], dtype=np.float32)*self.dt
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
    
    


    def generate_episodes(
        self,
        n_episodes: int = 1000,
        poisson_rate: float = 1.0,
        block: int = 1000,
        out_dir: str = "data/swaption_memmap"
    ) -> str:
        """
        Stream-generate `n_episodes` into five .dat files under `out_dir`, with a tqdm
        progress bar and a final pickle of `self.lmm` for later reuse.
        """
        # prepare output directory
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        assert self.primed, "You must call prime() before generating episodes."
        # 1) sample one episode to get dtype & per-episode shapes
        self.simulate(seed=0)
        self.get_swap_matrix()
        (h0_sh, h0_sw), (l0_sh, l0_sw) = self.get_sabr_params()
        T     = self.swap_sim_shape[0]
        print(T)
        dtype = h0_sh.dtype
        shape6 = (T, T, h0_sh.shape[-1])   # = (T, T, 6)
        shape5 = (T, T, h0_sw.shape[-1])   # = (T, T, 5)

        # 2) create (or overwrite) five memmap files at full size
        total6  = (n_episodes, *shape6)
        total5  = (n_episodes, *shape5)
        total_nd = (n_episodes, T, T)

        # Create a timestamped subdirectory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        timestamped_out_dir = f"{out_dir}/{timestamp}"
        Path(timestamped_out_dir).mkdir(parents=True, exist_ok=True)

        mm_h   = np.memmap(f"{timestamped_out_dir}/swaption_hed.dat",   mode="w+", dtype=dtype, shape=total6)
        mm_l   = np.memmap(f"{timestamped_out_dir}/swaption_liab.dat", mode="w+", dtype=dtype, shape=total6)
        mm_hs  = np.memmap(f"{timestamped_out_dir}/swap_hedge.dat",     mode="w+", dtype=dtype, shape=total5)
        mm_ls  = np.memmap(f"{timestamped_out_dir}/swap_liab.dat",      mode="w+", dtype=dtype, shape=total5)
        mm_nd  = np.memmap(f"{timestamped_out_dir}/net_direction.dat",  mode="w+", dtype=dtype, shape=total_nd)


        # 3) loop over episodes in blocks, with tqdm
        for start in tqdm(range(0, n_episodes, block),
                          desc="Generating episode blocks",
                          unit="block"):
            b = min(block, n_episodes - start)

            # generate net_direction for this block
            pd = np.random.poisson(lam=poisson_rate, size=(b, T))[:, None, :]
            num_pos  = np.random.binomial(pd, 0.5)
            nd_block = 2 * num_pos - pd
            iu = np.triu_indices(T, k=1)
            nd_block = np.tile(nd_block, (1,T, 1)) 
            nd_block[:, iu[0], iu[1]] = 0
            nd_block = np.nan_to_num(nd_block).astype(dtype)

            # prepare in‑RAM buffers
            hd_b = np.empty((b, *shape6), dtype=dtype)
            ld_b = np.empty((b, *shape6), dtype=dtype)
            hs_b = np.empty((b, *shape5), dtype=dtype)
            ls_b = np.empty((b, *shape5), dtype=dtype)

            # fill buffers
            for i in range(b):
                ep_idx = start + i
                self.simulate(seed=ep_idx)
                self.get_swap_matrix()
                (h_sh, h_sw), (l_sh, l_sw) = self.get_sabr_params()
                hd_b[i] = np.nan_to_num(h_sh)
                ld_b[i] = np.nan_to_num(l_sh)
                hs_b[i] = np.nan_to_num(h_sw)
                ls_b[i] = np.nan_to_num(l_sw)

            # write and flush this block
            mm_h  [start:start+b] = hd_b
            mm_l  [start:start+b] = ld_b
            mm_hs [start:start+b] = hs_b
            mm_ls [start:start+b] = ls_b
            mm_nd [start:start+b] = nd_block

            mm_h .flush()
            mm_l .flush()
            mm_hs.flush()
            mm_ls.flush()
            mm_nd.flush()

        # 4) pickle the LMMModel itself for later reuse
        pickle_path = Path(timestamped_out_dir) / "lmm_model.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(self.samples, f)

        print(f"All episodes streamed to disk in '{timestamped_out_dir}'.")
        print(f"LMMModel object saved to '{pickle_path}'.")
        return timestamped_out_dir
