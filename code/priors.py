import numpy as np
from numpy.linalg import lstsq


def minnesota_prior(n_endo, n_exo, n_lags_endo, n_lags_exo, hparams, selected_lags=None):
    pi_shrink = hparams['pi_shrink']      
    lag_decay = hparams['lag_decay']       
    exog_weight = hparams['exog_weight']   
    
    # Validation
    assert pi_shrink > 0,   "pi_shrink must be positive"
    assert lag_decay > 0,   "lag_decay must be positive"
    assert exog_weight > 0, "exog_weight must be positive"
    
    total_coefs = (n_endo * n_lags_endo) + (n_exo * n_lags_exo)
    omega_diag = np.zeros(total_coefs)
    
    if selected_lags is None:
        selected_lags = list(range(1, n_lags_endo + 1))
    
    # shrinkage endo - lag decay
    for k, lag in enumerate(selected_lags):
        idx = k * n_endo
        omega_diag[idx : idx + n_endo] = pi_shrink / (lag ** lag_decay)
        
    # shrinkage exo - lag decay scaled by exo_weight
    for lag in range (1, n_lags_exo + 1):
        idx = n_endo * n_lags_endo + (lag - 1) * n_exo
        omega_diag[idx : idx + n_exo] = (pi_shrink * exog_weight) / (lag ** lag_decay)
        
    B = np.zeros((total_coefs, n_endo))
    Omega = np.diag (omega_diag)
    return B, Omega


def inverse_wishart_prior(n_endo, hparams):
    alpha = n_endo+hparams['alpha_offset']
    S = np.diag(hparams['sigma2_ar1'])
    
    assert alpha > n_endo + 1, \
        "alpha must be > n_endog + 1 for the IW distribution to have a finite mean"
        
    return S, alpha


def bernoulli_prior(hparams):
    return {'pi': hparams['pi_bernoulli']}


def stochastic_volatility_prior(hparams):
    sv_params = hparams['stochastic_volatility']
    
    # --- validation ---
    assert sv_params['phi_a'] > 0 and sv_params['phi_b'] > 0, \
    "phi_a and phi_b must be positive"
    # --- 
    
    # phi_h ~ Beta(a, b)
    phi_prior = {
        'a': sv_params['phi_a'],
        'b': sv_params['phi_b']
    }
    # mu_h~ N(mu_0, V_mu)
    mu_prior = {
        'mean': sv_params['mu_0'],
        'var': sv_params['mu_var']
    }
    # sigma_h^2 ~ IG(v/2, S/2)
    sigma_prior = {
        'shape': sv_params['sigma_v']/2,
        'scale' : sv_params['sigma_s']/2
    }
    h_init = 0.0
    return phi_prior, mu_prior, sigma_prior, h_init

def ar1_residual_variances(y_raw, max_lag):
        """
        Estimate resudial variance of a univariate AR(1) for each endo var
        """
        T_full, ny = y_raw.shape
        sigma2 = np.zeros(ny)
        
        for i in range(ny):
            y = y_raw[max_lag:, i]      # dep vars
            y_lag = y_raw[max_lag -1 : -1, i].reshape(-1,1)  # one lag
            X = np.column_stack([np.ones_like(y_lag), y_lag])  # intercept + lag
            coef, _, _, _ = lstsq(X, y, rcond=None)
            residuals = y - X @ coef
            sigma2[i]= np.var(residuals)
            
        return sigma2


def df_prior(hparams):
    df_params = hparams['degrees_of_freedom']
    
    # --- validation ---
    assert df_params['min_nu'] >= 2,  "min_nu must be >= 2 (finite variance)"
    assert df_params['max_nu'] > df_params['min_nu'], "max_nu must be > min_nu"
    assert df_params['initial_nu'] >= df_params['min_nu'], "initial_nu out of range"
    # ---
    
    nu_prior = {
        'low': df_params['min_nu'], 
        'high': df_params['max_nu']
    }
    nu_init = df_params['initial_nu']
    
    return nu_prior, nu_init


def global_shrinkage_prior(nu):
    lambda_prior = {
        'alpha': nu/2,
        'beta': nu/2
    }
    lambda_init = 1
    return lambda_prior, lambda_init