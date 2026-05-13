"""
STEP 6 (RANDOM WALK SV) -- Stochastic Volatility for the BGVAR.

Model
-----
For each time t = 1, ..., T:
    u_t  | h_t  ~  N_{ny}(0,  exp(h_t) * Sigma_u)        # reduced-form residuals
    h_t          =  h_{t-1} + eta_t,    eta_t ~ N(0, sigma_h^2)         # RANDOM WALK
    h_0          ~  N(0, V_h0)                                          # diffuse prior on level

Following Gianfreda, Ravazzolo & Rossini (2023, Oxford Bulletin), we drop
both the AR(1) drift (mu_h) and persistence (phi_h) and let h_t evolve as
a random walk.  This kills the identification ridge that flattens h_t in
the AR(1) specification: the level is anchored only by the prior on h_0
and by the data, with nothing else competing for it.

Gibbs sweep:
    1. Build the pseudo-observation y*_t = log( u_t' Sigma_u^{-1} u_t / ny ).
    2. Sample mixture indicators s_t in {0..6} (KSC 7-component mixture).
    3. Sample h_{0:T} jointly via Forward-Filter-Backward-Sampler (FFBS).
    4. Sample sigma_h^2 from its conjugate Inverse-Gamma posterior.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve


# ==========================================================================
# Kim-Shephard-Chib (1998) 7-component normal-mixture approximation of
# log(chi^2_1). These constants are FIXED -- do not retune.
# ==========================================================================
_KSC_PROB = np.array([0.00730, 0.10556, 0.00002, 0.04395,
                      0.34001, 0.24566, 0.25750])
_KSC_MEAN = np.array([-10.12999, -3.97281, -8.56686,  2.77786,
                       0.61942,   1.79518, -1.08819])
_KSC_VAR  = np.array([5.79596,  2.61369,  5.17950,  0.16735,
                      0.64009,  0.34023,  1.26261])
_KSC_LOGPI    = np.log(_KSC_PROB)
_KSC_LOG_HALF = 0.5 * np.log(2.0 * np.pi)


# ==========================================================================
# Step 6.1 -- pseudo-observation y*_t = log( u_t' Sigma_u^{-1} u_t / ny ).
# Under the model q_t = u_t' Sigma_u^{-1} u_t  ~  exp(h_t) * chi^2_{ny},
# so y*_t = h_t + zeta_t with zeta_t = log( chi^2_{ny}/ny ).
# The KSC 7-mixture approximates the law of zeta_t for ny=1; for ny>1 we
# treat the Mahalanobis sum as a single noisy observation and rely on the
# mixture as an approximation (standard practice in multivariate SV).
# ==========================================================================
def _build_y_star(U: np.ndarray, Sigma_u_inv: np.ndarray,
                  offset: float = 1e-6) -> np.ndarray:
    """y*_t = log( u_t' Sigma_u^{-1} u_t / ny + offset )."""
    T, ny = U.shape
    US    = U @ Sigma_u_inv                        # (T, ny)
    q_t   = np.einsum('ti,ti->t', US, U)           # (T,) Mahalanobis sums
    return np.log(q_t / ny + offset)


# ==========================================================================
# Step 6.2 -- sample mixture indicators s_t in {0, ..., 6}.
#   P(s_t = k | y*_t, h_t)  ∝  pi_k * N(y*_t - h_t ; m_k, v_k)
# Vectorised with the Gumbel-max trick for a single argmax over k.
# ==========================================================================
def _sample_indicators(y_star: np.ndarray, h: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    e = y_star - h                                              # (T,)
    log_norm = -_KSC_LOG_HALF - 0.5 * np.log(_KSC_VAR)          # (7,)
    diff     = e[:, None] - _KSC_MEAN                           # (T, 7)
    log_lik  = log_norm - 0.5 * diff * diff / _KSC_VAR          # (T, 7)
    log_w    = _KSC_LOGPI + log_lik                             # (T, 7)
    g        = -np.log(-np.log(rng.uniform(size=log_w.shape)))  # Gumbel(0,1)
    return np.argmax(log_w + g, axis=1)


# ==========================================================================
# Step 6.3 -- FFBS for h_{1:T} under the RANDOM WALK transition.
#
# Conditional on s, the state-space is linear-Gaussian:
#     y*_t  =  h_t + m_{s_t}  +  e_t,    e_t  ~ N(0, v_{s_t})    (observation)
#     h_t   =  h_{t-1}        +  eta_t,  eta_t ~ N(0, sigma_h^2)  (state, RW)
#
# Forward Kalman recursions:
#     a_pred  = a_filt[t-1]                       (RW: no phi or mu)
#     P_pred  = P_filt[t-1] + sigma_h^2
#     F       = P_pred + v_{s_t}
#     K       = P_pred / F
#     a_filt  = a_pred + K * (y*_t - m_{s_t} - a_pred)
#     P_filt  = (1 - K) * P_pred
#
# Backward simulation smoother (De Jong-Shephard):
#     J        = P_filt[t] / (P_filt[t] + sigma_h^2)
#     mean_bs  = a_filt[t] + J * (h[t+1] - a_filt[t])
#     var_bs   = (1 - J) * P_filt[t]
#     h[t]     = mean_bs + sqrt(var_bs) * standard_normal
#
# Initialisation: h_0 has a diffuse prior  h_0 ~ N(0, V_h0).
# We initialise the filter at t = 0 with a_pred = 0, P_pred = V_h0.
# ==========================================================================
def _ffbs_h_rw(y_star: np.ndarray, s: np.ndarray,
               sigma_h2: float, V_h0: float,
               rng: np.random.Generator) -> np.ndarray:
    T  = y_star.shape[0]
    ms = _KSC_MEAN[s]
    vs = _KSC_VAR[s]

    a_filt = np.empty(T)
    P_filt = np.empty(T)

    # Diffuse prior at t = 0 (no t = -1 needed: a_pred = 0, P_pred = V_h0)
    a_pred = 0.0
    P_pred = V_h0

    # ---- Forward filter ----
    for t in range(T):
        if t > 0:
            a_pred = a_filt[t - 1]                # RW: predicted mean = previous filtered mean
            P_pred = P_filt[t - 1] + sigma_h2     # RW: predicted variance grows by sigma_h^2
        F = P_pred + vs[t]
        K = P_pred / F
        innov = (y_star[t] - ms[t]) - a_pred
        a_filt[t] = a_pred + K * innov
        P_filt[t] = (1.0 - K) * P_pred

    # ---- Backward simulation smoother ----
    h = np.empty(T)
    h[-1] = a_filt[-1] + np.sqrt(P_filt[-1]) * rng.standard_normal()
    for t in range(T - 2, -1, -1):
        # Under RW, the smoother gain simplifies (no phi factor):
        #   P_pred(t+1 | t) = P_filt[t] + sigma_h^2
        #   J               = P_filt[t] / P_pred(t+1 | t)
        denom = P_filt[t] + sigma_h2
        J     = P_filt[t] / denom
        mean_bs = a_filt[t] + J * (h[t + 1] - a_filt[t])
        var_bs  = max((1.0 - J) * P_filt[t], 0.0)
        h[t]    = mean_bs + np.sqrt(var_bs) * rng.standard_normal()
    return h


# ==========================================================================
# Step 6.4 -- conjugate Inverse-Gamma update for sigma_h^2.
#
# Random walk: eta_t = h_t - h_{t-1}  ~  N(0, sigma_h^2)  for t = 1..T-1.
# (We do NOT include t = 0 in the sum because h_0 is governed by its own
#  diffuse prior, not by sigma_h^2.)
#
# With prior sigma_h^2 ~ IG(shape0, scale0):
#     posterior shape = shape0 + (T-1)/2
#     posterior scale = scale0 + 0.5 * sum_t eta_t^2
# ==========================================================================
def _sample_sigma_h2(h: np.ndarray,
                     sigma_prior: dict,
                     rng: np.random.Generator) -> float:
    eta = np.diff(h)                                       # (T-1,)  RW innovations
    post_shape = sigma_prior['shape'] + 0.5 * eta.shape[0]
    post_scale = sigma_prior['scale'] + 0.5 * float(eta @ eta)
    return post_scale / rng.standard_gamma(post_shape)


# ==========================================================================
# MAIN SAMPLER -- one Gibbs sweep
# ==========================================================================
def step6_sample_SV(state: dict, rng: np.random.Generator) -> dict:
    """
    One Gibbs sweep over the random-walk stochastic-volatility block:
        1. y*_t            (pseudo-observation built from current residuals)
        2. s_t             (mixture indicator, KSC 7-component)
        3. h_{1:T}         (FFBS under RW transition)
        4. sigma_h^2       (conjugate IG update)

    Writes back into `state`:  h, sigma_h2 (and lambda_t = 1 for compat).
    """
    from step3 import compute_residuals             # lazy import

    ny       = state['ny']
    Sigma_u  = state['Sigma_u']
    h        = state['h']
    sigma_h2 = float(state['sigma_h2'])

    sigma_prior = state['sigma_prior_sv']
    # Diffuse prior variance on h_0 (read from state if present, else 10).
    V_h0 = float(state.get('V_h0', 10.0))

    # Pre-compute Sigma_u^{-1} once for y* this iteration.
    cho, low = cho_factor(Sigma_u, lower=True, check_finite=False)
    Sigma_u_inv = cho_solve((cho, low), np.eye(ny), check_finite=False)
    Sigma_u_inv = 0.5 * (Sigma_u_inv + Sigma_u_inv.T)

    # Fresh residuals from the current Phi / G_Phi.
    U = compute_residuals(state)

    # ---- 1. pseudo-observation ----
    y_star = _build_y_star(U, Sigma_u_inv)

    # ---- 2. mixture indicators ----
    s = _sample_indicators(y_star, h, rng)

    # ---- 3. h_{1:T} via FFBS under RW transition ----
    h = _ffbs_h_rw(y_star, s, sigma_h2, V_h0, rng)

    # ---- 4. sigma_h^2 (conjugate IG) ----
    sigma_h2 = _sample_sigma_h2(h, sigma_prior, rng)

    # ---- Write back into state ----
    state['h']        = h
    state['sigma_h2'] = sigma_h2
    # Lambda mixing is disabled in this RW-SV version. Keep lambda_t = 1
    # so that step3 / step4 (which scale by sqrt(exp(h) * lam)) still work.
    state['lambda_t'] = np.ones_like(h)

    return {
        'h_mean':   float(h.mean()),
        'h_std':    float(h.std()),
        'sigma_h2': sigma_h2,
    }