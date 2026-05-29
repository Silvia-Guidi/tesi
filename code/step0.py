import numpy as np
import pandas as pd
from scipy.stats import invwishart
from priors import (
    minnesota_prior,
    inverse_wishart_prior,
    bernoulli_prior,
    stochastic_volatility_prior,
    ar1_residual_variances
)

#=============================================
# This file builds every data structure that the Gibb Sampler steps will read
#=============================================




def initialize_model (
    y_raw: np.ndarray,
    X_exo_raw: np.ndarray | None,
    selected_lags : list,
    hparams: dict, 
    rng: np.random.Generator
) -> dict:
    """
    Build and return a dictionary that holds every model object
    
    Returns:
    state : dict - all model quantities, mutable across Gibbs interactions
    """
    
    # 1. DIMENSIONS
    T_full = y_raw.shape[0]
    ny = y_raw.shape[1]     # number of endo vars
    n_lags = len(selected_lags)
    max_lag = max(selected_lags)
    T = T_full - max_lag    # effective sample size
    nz = X_exo_raw.shape[1] if X_exo_raw is not None else 0   # number of exo vars
    n_lags_exo = n_lags
    
    
    # 2. DATA MATRICES
    Y = y_raw[max_lag:, :]  #shape (T, ny)
    
    X_endo = np.hstack([
        y_raw[max_lag - lag : T_full - lag, :]      #y_{t-1}, shape (T, ny)
        for lag in selected_lags
    ])      # shape (T, ny*n_lags)
    
    # X_exo has shape (T, nz * q), column-blocks ordered by lag (lag 1, lag 2, ...).
    # X_exo_raw must already be aligned to y_raw (same date index) when passed in.
    if nz > 0:
        X_exo = np.hstack([
            X_exo_raw[max_lag - lag : T_full - lag, :]    # X_{t-lag}, shape (T, nz)
            for lag in selected_lags
        ])                                                # shape (T, nz * q)
    else:
        X_exo = np.zeros((T, 0))
    
    #3. PRIORS
    sigma2_ar1 = ar1_residual_variances(y_raw, max_lag)
    hparams['sigma2_ar1'] = sigma2_ar1
        
    
    # --- Minnesota prior for Φ: endogenous coefficients ---
    B_phi, Omega_phi = minnesota_prior(
        n_endo= ny,
        n_exo=0,       
        n_lags_endo= n_lags, n_lags_exo= 0,
        selected_lags = selected_lags,
        hparams= hparams,
    )
    
    # --- Minnesota prior for Gamma: exogenous coefficients ---
    if nz > 0:
        _, Omega_joint = minnesota_prior(
            n_endo=ny, n_exo=nz,
            n_lags_endo=n_lags, n_lags_exo=n_lags_exo,
            selected_lags=selected_lags,
            hparams=hparams,
        )
        # The exogenous block is the bottom-right corner of Omega_joint:
        # rows/cols from (ny*n_lags) to (ny*n_lags + nz*n_lags_exo)
        start = ny * n_lags
        Omega_gamma = Omega_joint[start:, start:]
    else:
        Omega_gamma = np.zeros((0, 0))
    
    # --- Inverse-Wishart prior for Σ_u ---
    S_prior, alpha_prior = inverse_wishart_prior(ny, hparams)
    
    # --- Bernoulli prior for graph edges ((G0, GΦ,lag)) ---
    bern_prior = bernoulli_prior(hparams)
    pi_bernoulli = bern_prior['pi']
    
    # --- Stochastic volatility priors ---
    sigma_prior_sv, h_init = stochastic_volatility_prior(hparams)
    
    
    # 4. GRAPH INIT
    def expand_G0(G0_matrix: np.ndarray, n_vars: int = 1) -> np.ndarray:
        n_countries = G0_matrix.shape[0]
        ny = n_countries * n_vars
        G0_expanded = np.zeros((ny, ny), dtype=int)
        
        for c1 in range(n_countries):
            for c2 in range(n_countries):
                
                rows = slice(c1 * n_vars, (c1 + 1) * n_vars)
                cols = slice(c2 * n_vars, (c2 + 1) * n_vars)
                
                if c1 == c2:
                    block = np.ones((n_vars, n_vars), dtype=int)
                    np.fill_diagonal(block, 0)
                    G0_expanded[rows, cols] = block
                elif G0_matrix[c1, c2] == 1: 
                    G0_expanded[rows, cols] = np.ones((n_vars, n_vars), dtype=int)

        return G0_expanded       
        
    G0_matrix = pd.read_csv("data/network_data/G0_matrix.csv", index_col=0
                            ).values.astype(int)
    n_countries = G0_matrix.shape[0]

    # Derive n_vars from the actual data: ny / n_countries must be integer.
    if ny % n_countries != 0:
        raise ValueError(
            f"ny={ny} is not a multiple of n_countries={n_countries}; "
            f"check DATA_MODE filtering vs G0_matrix.csv."
        )
    n_vars = ny // n_countries

    G0_expanded = expand_G0(G0_matrix, n_vars=n_vars)
    
    G0 = np.zeros((ny, ny), dtype=int)
    G_Phi   = [np.zeros((ny, ny), dtype=np.int8) for _ in selected_lags]
    G_Gamma = [np.zeros((ny, nz), dtype=np.int8) for _ in range(n_lags_exo)]
    
    
    # 5. COEFF INIT
    Phi = [np.zeros((ny, ny)) for _ in selected_lags]
    Gamma = [np.zeros((ny, nz)) for _ in range(n_lags_exo)]
    
    
    # 6. RESIDUAL COVARIANCE INIT
    seed_init = int(rng.integers(0, 2**31 - 1))
    Sigma_u = invwishart.rvs(df=alpha_prior, scale=S_prior, random_state=seed_init)
    Sigma_u = 0.5 * (Sigma_u + Sigma_u.T)
    
    
    # 7. STOCHASTIC VOLATILITY INIT
    h = np.full(T, h_init)
    
    # sigma_h2 : variance of log-vol innovations
    #            initialized to prior mean of IG: E[IG(shape,scale)] = scale/(shape-1)
    if sigma_prior_sv['shape'] > 1:
        sigma_h2 = sigma_prior_sv['scale'] / (sigma_prior_sv['shape'] - 1)
    else:
        sigma_h2 = sigma_prior_sv['scale'] / (sigma_prior_sv['shape'] + 1)
    assert sigma_h2 > 0, f"sigma_h2 must be positive, got {sigma_h2}"

    
    # 8. STATE DICT
    
    state = {
        # --- Dimentions ---
        'T':                T,
        'ny':               ny,
        'nz':               nz,
        'n_lags':           n_lags,
        'n_lags_exo':       n_lags_exo,
        'selected_lags':    selected_lags,
        'max_lag':          max_lag,
    
        
        # --- Data ---
        'Y':                Y,
        'X_endo':           X_endo,
        'X_exo':            X_exo,
        
        # --- Graph structures ---
        'G0':             G0, 
        'G0_matrix':      G0_matrix,
        'G0_expanded':    G0_expanded,      
        'G_Phi':          G_Phi,
        'G_Gamma':        G_Gamma,    
 
        # -- Structural coefficients ---
        'Phi':            Phi, 
        'Gamma':          Gamma,     
 
        # -- Residual covariance ---
        'Sigma_u':        Sigma_u,  
 
        # -- Stochastic volatility ---
        'h':              h,    
        'sigma_h2':       0.1,
        'V_h0':           10,
        'sv_propsd':      0.2,
        'sv_iter':        0,
        'BURNIN_for_SV':  hparams.get('sv_burnin_adapt', 1000), 
 
        # -- Priors ---
        'B_phi':          B_phi,
        'Omega_phi':      Omega_phi,
        'Omega_gamma':    Omega_gamma,
        'S_prior':        S_prior,
        'alpha_prior':    alpha_prior,
        'pi_bernoulli':   pi_bernoulli,
        'sigma_prior_sv': sigma_prior_sv,
    }
 
    return state