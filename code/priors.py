import numpy as np


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
    """
    Random-walk SV prior (Gianfreda-Ravazzolo-Rossini 2023 style):
        h_t = h_{t-1} + eta_t,   eta_t ~ N(0, sigma_h^2)
        sigma_h^2 ~ IG(shape, scale)
        h_0 ~ N(0, V_h0)

    No AR(1) drift (mu_h) and no persistence (phi_h): the level is
    anchored only by the diffuse prior on h_0 and by the data.

    Returns
    -------
    sigma_prior : dict with keys 'shape' and 'scale' for sigma_h^2 ~ IG.
    h_init      : float, initial value for h_t (0 = neutral level).
    """
    sv_params = hparams['stochastic_volatility']

    # --- validation ---
    assert sv_params['shape'] > 0, "IG shape must be positive"
    assert sv_params['scale'] > 0, "IG scale must be positive"

    sigma_prior = {
        'shape': sv_params['shape'],
        'scale': sv_params['scale'],
    }
    h_init = 0.0
    return sigma_prior, h_init

def ar1_residual_variances(y_raw, max_lag):
    """
    Estimate resudial variance of a univariate AR(1) for each endo var
    """
    T_full, ny = y_raw.shape
    # Dependent and lagged matrices (each column = one variable)
    Y_dep = y_raw[max_lag:, :]                      # (T_eff, ny)
    Y_lag = y_raw[max_lag - 1 : -1, :]              # (T_eff, ny)

    # Per-variable univariate AR(1) with intercept, solved analytically:
    # y_i = a_i + b_i * y_{i,-1} + eps_i.
    # Closed-form OLS: no loop, no lstsq.
    mean_dep = Y_dep.mean(axis=0)
    mean_lag = Y_lag.mean(axis=0)
    cov = ((Y_dep - mean_dep) * (Y_lag - mean_lag)).mean(axis=0)
    var_lag = ((Y_lag - mean_lag) ** 2).mean(axis=0)

    b = cov / var_lag
    a = mean_dep - b * mean_lag

    resid = Y_dep - (a + b * Y_lag)
    return resid.var(axis=0)
