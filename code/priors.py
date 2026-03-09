import numpy as np

def minnesota_prior(n_endog, n_exog, n_lags, hparams):
    pi_shrink = hparams['pi_shrink']      
    lag_decay = hparams['lag_decay']       
    exog_weight = hparams['exog_weight']   
    
    total_coefs = (n_endog * n_lags) + n_exog
    # Prior Mean (B)
    B = np.zeros((total_coefs, n_endog))
    # Prior Covariance (Omega)
    omega_diag = np.zeros(total_coefs)
    # Shrinkage Phi
    for p in range(1, n_lags + 1):
        idx = (p - 1) * n_endog
        omega_diag[idx : idx + n_endog] = pi_shrink / (p**lag_decay)
    # Shrinkage Gamma 
    omega_diag[n_endog * n_lags:] = pi_shrink * exog_weight 
    
    Omega = np.diag(omega_diag)
    return B, Omega

def sigma_prior(n_endog, hparams):
    alpha= n_endog+hparams['alpha_offset']
    S=np.eye(n_endog)*hparams['s_scale']
    return S, alpha

def informed_bernoulli_prior(size, hparams):
    pi = hparams['pi_bernoulli']
    inclusion_vector = np.random.binomial(1, pi, size=size)
    return inclusion_vector

def stochastic_volatility_prior(hparams):
    sv_params = hparams['stochastic_volatility']
    
    # phi_h ~ Beta(a, b)
    phi_prior = {
        'a': sv_params['phi_a'],
        'b': sv_params['phi_b']
    }
    #mu_h~ N(mu_0, V_mu)
    mu_prior = {
        'mean': sv_params['mu_0'],
        'var': sv_params['mu_var']
    }
    #sigma_h^2 ~ IG(v/2, S/2)
    sigma_prior = {
        'shape': sv_params['sigma_v']/2,
        'scale' : sv_params['sigma_s']/2
    }
    h_init = 0.0
    return phi_prior, mu_prior, sigma_prior, h_init

def df_prior(hparams):
    df_params = hparams['degrees_od_freedom']
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
    lambda_init= 1
    return lambda_prior, lambda_init