import numpy as np
import pandas as pd
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
    selected_lags : list,
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
    n_lags = len(selected_lags)
    max_lag = max(selected_lags)
    T = T_full - max_lag    # effective sample size
    
    
    # 2. DATA MATRICES
    Y = y_raw[max_lag:, :]  #shape (T, ny)
    
    Xendo = np.hstack([
        y_raw[max_lag - lag : T_full - lag, :]      #y_{t-1}, shape (T, ny)
        for lag in selected_lags
    ])      # shape (T, ny*n_lags)
    
    
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
            residuals = y - y_lag @ a
            sigma2[i]= np.var(residuals)
            
            return sigma2
    
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
    
    # --- Inverse-Wishart prior for Σ_u ---
    S_prior, alpha_prior = inverse_wishart_prior(ny, hparams)
    
    # --- Bernoulli prior for graph edges ((G0, GΦ,lag)) ---
    bern_prior = bernoulli_prior(hparams)
    pi_bernoulli = bern_prior['pi']
    
    # --- Stochastic volatility priors ---
    phi_prior_sv, mu_prior_sv, sigma_prior_sv, h_init = stochastic_volatility_prior(hparams)
    
    #--- Degrees-of-freedom prior for λ_t ---
    nu_prior, nu_init = df_prior(hparams)
    
    # --- Global shrinkage prior for λ_t given ν ---
    lambda_prior, lambda_init = global_shrinkage_prior(nu_init)
    
    
    # 4. GRAPH INIT
    def expand_G0(G0_matrix: np.ndarray, n_vars: int = 2) -> np.ndarray:
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
        
    G0_matrix = pd.read_csv ("data/netwrok_data/G0_matrix.csv", index_col=0
                             ).values.astype(int)
    
    G0_expanded = expand_G0(G0_matrix, n_vars=2)
    
    G0 = G0_expanded.copy()
    G_Phi = [np.zeros((ny,ny), dtype = int) for _ in selected_lags]
    
    
    # 5. COEFF INIT
    Phi = [np.zeros((ny, ny)) for _ in selected_lags]
    
    
    # 6. RESIDUAL COVARIANCE INIT
    Sigma_u = invwishart.rvs(df=alpha_prior, scale=S_prior)
    
    
    # 7. STOCHASTIC VOLATILITY INIT
    h = np.full(T, h_init)
    
    # phi_h : AR(1) persistence of log-vol, initialized to prior mean of Beta(20,1.5)
    #         E[Beta(a,b)] = a/(a+b)
    phi_h = phi_prior_sv['a']/ (phi_prior_sv['a'] + phi_prior_sv['b'])
    
    mu_h = mu_prior_sv['mean']
    
    # sigma_h2 : variance of log-vol innovations
    #            initialized to prior mean of IG: E[IG(shape,scale)] = scale/(shape-1)
    sigma_h2 = sigma_prior_sv['scale'] / (sigma_prior_sv['shape'] - 1)
    
    
    # 8. STUDENT-t MIXING VAR INIT
    lambda_t = np.ones(T)
    
    nu = float(nu_init)
    
    
    # 9. STATE DICT
    
    state = {
        # --- Dimentions ---
        'T':                T,
        'ny':               ny,
        'n_lags':           n_lags,
        'selected_lags':    selected_lags,
        'max_lag':          max_lag,
        
        # --- Data ---
        'Y':                Y,
        'Xendo':            Xendo,
        
        # --- Graph structures ---
        'G0':             G0, 
        'G0_matrix':      G0_matrix,
        'G0_expanded':    G0_expanded,      
        'G_Phi':          G_Phi,    
 
        # -- Structural coefficients ---
        'Phi':            Phi,      
 
        # -- Residual covariance ---
        'Sigma_u':        Sigma_u,  
 
        # -- Stochastic volatility ---
        'h':              h,        
        'phi_h':          phi_h,
        'mu_h':           mu_h,
        'sigma_h2':       sigma_h2,
 
        # -- Student-t mixing ---
        'lambda_t':       lambda_t, 
        'nu':             nu,
 
        # -- Priors ---
        'B_phi':          B_phi,
        'Omega_phi':      Omega_phi,
        'S_prior':        S_prior,
        'alpha_prior':    alpha_prior,
        'pi_bernoulli':   pi_bernoulli,
        'phi_prior_sv':   phi_prior_sv,
        'mu_prior_sv':    mu_prior_sv,
        'sigma_prior_sv': sigma_prior_sv,
        'nu_prior':       nu_prior,
        'lambda_prior':   lambda_prior,
        'sigma2_ar1':     sigma2_ar1,
    }
 
    return state