import numpy as np
from numpy.linalg import lstsq
from scipy.stats import invwishart
from priors import (
    minnesota_prior,
    inverse_wishart_prior,
    bernoulli_prior,
    stochastic_volatility_prior,
    df_prior,
    global_shrinkage_prior
)

#=============================================
#This file builds every data structure that the Gibb Sampler steps will read
#=============================================

def initialize_model (
    y_raw: np.ndarray,
    x_raw: np.ndarray,
    p: int,
    q: int,
    hparams: dict
) -> dict:
    """
    Build and return a dictionary that holds every model object
    
    Returns:
    state : dict - all model quantities, mutable across Gibbs interactions
    """
    
    # 1. DIMENSIONS
    T_full = y_raw.shape[0]
    ny = y_raw.shape[1]     # number of endo vars
    nx = x_raw.spae[1]      # number of exo vars
    max_lag = max(p, q)
    T = T_full - max_lag    # effective sample size
    
    
    # 2. DATA MATRICES
    Y = y_raw[max_lag:, :]  #shape (T, ny)
    
    X_endo = np.hstack([
        y_raw[max_lag - i : T_full - i, :]      #y_{t-1}, shape (T, ny)
        for i in range (1, p + 1)
    ])      # shape (T, ny*p)
    
    X_exo= np.hstack([
        x_raw[max_lag - j : T_full - j, :]      #X_{t-j}, shape (T, nx)
        for j in range (1, q + 1)
    ])     # shape (T, nx*q)
    
    
    #3. PRIORS
    
    def ar1_residual_variances(y_raw, max_lag):
        """
        Estimate resudial variance of a univariate AR(1) for each endo var
        """
        T_full, ny = y_raw.shape
        sigma2 = np.zeros(ny)
        
        for i in range(ny):
            y = y_raw[max_lag:, i]      # dep vars
            y_lag = y_raw[max_lag -1 : -1, i].reshape(-1,1)  # one lag
            a, _, _, _= lstsq(y_lag, y, rcond = None)
        
    
    # --- Minnesota prior for Φ: endogenous coefficients ---
    B_phi, Omega_phi = minnesota_prior(
        n_endo= ny,
        n_exo=0,       # exo handled separately below
        n_lags_endo= p, n_lags_exo= 0
        hparams= hparams,
    )
    
    # --- Minnesota prior for Γ: exogenous coefficients ---
    B_gamma, Omega_gamma = minnesota_prior(
        n_endo= 0,
        n_exo=nx,
        n_lags_endo=0, n_lags_exo= q
        hparams=hparams,
    )
    
    # --- Inverse-Wishart prior for Σ_u ---
    S_prior, alpha_prior = inverse_wishart_prior(ny, hparams)