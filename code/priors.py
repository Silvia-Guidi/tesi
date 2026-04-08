import numpy as np


def minnesota_prior(n_endo, n_exo, n_lags_endo, n_lags_exo, hparams):
    pi_shrink = hparams['pi_shrink']      
    lag_decay = hparams['lag_decay']       
    exog_weight = hparams['exog_weight']   
    
    total_coefs = (n_endo * n_lags_endo) + (n_exo * n_lags_exo)
    omega_diag = np.zeros(total_coefs)
    
    # shrinkage endo - lag decay
    for lag in range (1, n_lags_endo + 1):
        idx = (lag - 1) * n_endo
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
    return S, alpha


def bernoulli_prior(hparams):
    return {'pi': hparams['pi_bernoulli']}


def stochastic_volatility_prior(hparams):
    sv_params = hparams['stochastic_volatility']
    
    # --- validation ---
    assert sv_params['phi_a'] == 20 and sv_params['phi_b'] == 1.5, \
    "phi_h prior should be Beta(20, 1.5)"
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