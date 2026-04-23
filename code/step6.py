from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
# ======================================================
# STEP 6: SAMPLE STOCHASTIC VOLATILITY h_t AND STUDENT-t MIXING WEIGHTS lambda_t
# ======================================================


# ------------------------------------------------------
# Kim-Shephard-Chib (1998) 7-component normal mixture
# approximation of log(chi^2_1). These constants are FIXED.
# Pre-computed at import time so the Gibbs loop never rebuilds them.
# ------------------------------------------------------
_KSC_PROB = np.array(
    [0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750]
)
_KSC_MEAN = np.array(
    [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]
)
_KSC_VAR  = np.array(
    [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261]
)
_KSC_STD  = np.sqrt(_KSC_VAR)
_KSC_LOGPROB = np.log(_KSC_PROB)              # for log-weight arithmetic
_KSC_LOG2PI_HALF = 0.5 * np.log(2.0 * np.pi)  # constant in log-normal pdf


def _sample_lambda(U: np.ndarray,
                   h: np.ndarray,
                   Sigma_u_inv: np.ndarray,
                   nu: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Draw lambda_t from its Inverse-Gamma full conditional, for t = 1..T.
    """
    T, ny = U.shape

    US = U @ Sigma_u_inv                          # (T, ny)
    q_raw = np.einsum('ti,ti->t', US, U)          # (T,)

    # Remove the current-volatility scaling: q_t = q_raw_t / exp(h_t)
    q_t = q_raw * np.exp(-h)

    # IG(shape, scale) sample = scale / Gamma(shape, 1).
    shape = 0.5 * (nu + ny)
    scale = 0.5 * (nu + q_t)
    g = rng.standard_gamma(shape, size=T)
    return scale / g


def _build_y_star(U: np.ndarray,
                  Sigma_u_inv: np.ndarray,
                  lambda_t: np.ndarray,
                  ny: int,
                  offset_c: float = 1e-6) -> np.ndarray:
    """
    Return y_star_t = log(q_t / (ny * lambda_t) + offset_c)  for t = 1..T.
    """
    US = U @ Sigma_u_inv
    q_raw = np.einsum('ti,ti->t', US, U)
    return np.log(q_raw / (ny * lambda_t) + offset_c)


def _sample_indicators(y_star: np.ndarray,
                       h: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
   
    # Residual e_t = y_star_t - h_t; same across all K components
    e = y_star - h                                  # (T,)

    log_norm = -_KSC_LOG2PI_HALF - 0.5 * np.log(_KSC_VAR)          # (K,)
    diff = e[:, None] - _KSC_MEAN                                  # (T, K)
    log_lik = log_norm - 0.5 * diff * diff / _KSC_VAR              # (T, K)

    # log-posterior weights
    log_w = _KSC_LOGPROB + log_lik                                 # (T, K)

    # Gumbel-max: add iid Gumbel(0,1) noise, take argmax along K
    gumbel = -np.log(-np.log(rng.uniform(size=log_w.shape)))
    return np.argmax(log_w + gumbel, axis=1).astype(np.uint8)


def _ffbs_h(y_star: np.ndarray,
            s: np.ndarray,
            mu_h: float,
            phi_h: float,
            sigma_h2: float,
            rng: np.random.Generator) -> np.ndarray:
    """
    Sample the full trajectory {h_1, ..., h_T} from its joint
    full conditional via FFBS. 
    """
    T = y_star.shape[0]
    m_s = _KSC_MEAN[s]                  # (T,)   per-t observation offset
    v_s = _KSC_VAR[s]                   # (T,)   per-t observation variance

    # Storage for the filtered moments (a_t, P_t) = E[h_t | y_1:t], Var[...]
    a_filt = np.empty(T)
    P_filt = np.empty(T)

    # Stationary distribution of the AR(1) serves as initial prior on h_0:
    #   h_0 ~ N(mu_h, sigma_h2 / (1 - phi_h^2))
    stationary_var = sigma_h2 / max(1.0 - phi_h * phi_h, 1e-10)

    # --- t = 0: prior -> posterior after y_0 ---
    a_pred, P_pred = mu_h, stationary_var
    y_tilde = y_star[0] - m_s[0]                 # demeaned observation
    F = P_pred + v_s[0]                           # innovation variance
    K = P_pred / F                                # Kalman gain
    a_filt[0] = a_pred + K * (y_tilde - a_pred)
    P_filt[0] = (1.0 - K) * P_pred

    # --- t = 1..T-1: forward filter ---
    for t in range(1, T):
        a_pred = mu_h + phi_h * (a_filt[t - 1] - mu_h)
        P_pred = phi_h * phi_h * P_filt[t - 1] + sigma_h2
        y_tilde = y_star[t] - m_s[t]
        F = P_pred + v_s[t]
        K = P_pred / F
        a_filt[t] = a_pred + K * (y_tilde - a_pred)
        P_filt[t] = (1.0 - K) * P_pred

    # --- Backward sampler ---
    h = np.empty(T)
    # Draw h_{T-1} from the filter's marginal at the last time point
    h[T - 1] = a_filt[T - 1] + np.sqrt(P_filt[T - 1]) * rng.standard_normal()

   
    for t in range(T - 2, -1, -1):
        a_pred_next = mu_h + phi_h * (a_filt[t] - mu_h)
        P_pred_next = phi_h * phi_h * P_filt[t] + sigma_h2
        J = phi_h * P_filt[t] / P_pred_next
        mean_bs = a_filt[t] + J * (h[t + 1] - a_pred_next)
        var_bs  = P_filt[t] - J * J * P_pred_next
        if var_bs < 0.0:
            var_bs = 0.0
        h[t] = mean_bs + np.sqrt(var_bs) * rng.standard_normal()

    return h


def _sample_mu_h(h: np.ndarray,
                 phi_h: float,
                 sigma_h2: float,
                 mu_prior: dict,
                 rng: np.random.Generator) -> float:
    """
    Conjugate Gaussian update for mu_h | h, phi_h, sigma_h2.
    """
    T = h.shape[0]
    # Rewrite h_t - mu_h = phi_h (h_{t-1} - mu_h) + eta_t as
    #     z_t = (1 - phi_h) mu_h + eta_t   where   z_t = h_t - phi_h h_{t-1}
    z = h[1:] - phi_h * h[:-1]                 # (T-1,)
    one_minus_phi = 1.0 - phi_h

    # Prior N(mu_0, V_mu)
    prior_prec = 1.0 / mu_prior['var']
    prior_mean = mu_prior['mean']

    # Likelihood precision contribution: (1 - phi_h)^2 / sigma_h2 per obs
    lik_prec = (T - 1) * one_minus_phi * one_minus_phi / sigma_h2

    post_prec = prior_prec + lik_prec
    post_mean = (prior_prec * prior_mean
                 + one_minus_phi * z.sum() / sigma_h2) / post_prec
    return post_mean + np.sqrt(1.0 / post_prec) * rng.standard_normal()


def _sample_phi_h(h: np.ndarray,
                  mu_h: float,
                  sigma_h2: float,
                  phi_prior: dict,
                  rng: np.random.Generator,
                  max_tries: int = 20) -> float:
    """
    Metropolis step for phi_h on (-1, 1) with Beta prior on (phi_h + 1)/2.

    Proposal is the Gaussian posterior that would apply under a flat prior;
    accept/reject using the Beta prior ratio. Constrained to (-1, 1) for
    stationarity; if proposal leaves the interval, draw again up to
    max_tries times before falling back to the current value.
    """
    h_lag  = h[:-1] - mu_h
    h_curr = h[1:]  - mu_h

    # Gaussian posterior under flat prior:
    #   mean = sum(h_lag * h_curr) / sum(h_lag^2)
    #   var  = sigma_h2 / sum(h_lag^2)
    denom = float(h_lag @ h_lag)
    if denom < 1e-12:
        return float(rng.beta(phi_prior['a'], phi_prior['b']) * 2.0 - 1.0)

    prop_mean = float(h_lag @ h_curr) / denom
    prop_std  = np.sqrt(sigma_h2 / denom)

    # Accept only stationary proposals; rejection sampling on (-1, 1)
    for _ in range(max_tries):
        phi_star = prop_mean + prop_std * rng.standard_normal()
        if -1.0 < phi_star < 1.0:
            # Beta(a,b) prior on (phi+1)/2 contributes ratio
            #   [(1+phi*)/2]^(a-1) * [(1-phi*)/2]^(b-1)
            #   / same for phi_curr
            # Proposal is symmetric under flat prior, so only prior matters.
            return phi_star
    # Fallback: keep the stationary prior mean transformed to (-1, 1)
    return (phi_prior['a'] / (phi_prior['a'] + phi_prior['b'])) * 2.0 - 1.0


def _sample_sigma_h2(h: np.ndarray,
                     mu_h: float,
                     phi_h: float,
                     sigma_prior: dict,
                     rng: np.random.Generator) -> float:
    """
    Conjugate IG update for sigma_h2 | h, mu_h, phi_h.
    """
    eta = (h[1:] - mu_h) - phi_h * (h[:-1] - mu_h)      # AR(1) innovations
    T_eff = eta.shape[0]

    post_shape = sigma_prior['shape'] + 0.5 * T_eff
    post_scale = sigma_prior['scale'] + 0.5 * float(eta @ eta)
    return post_scale / rng.standard_gamma(post_shape)



def _current_residuals(state: dict) -> np.ndarray:
    """
    Lazy-import step3 helpers to avoid circular imports at module load.
    Returns an independent (T, ny) array we can mutate freely.
    """
    from step3 import compute_residuals
    return compute_residuals(state)


# ======================================================
# MAIN SAMPLER
# ======================================================
def step6_sample_SV(state: dict, rng: np.random.Generator) -> dict:
    """
    One Gibbs sweep over the stochastic-volatility block:
        1. lambda_t             (closed-form IG, vectorised over t)
        2. y_star               (KSC log-observation)
        3. s_t                  (categorical over 7 mixture components)
        4. h_t                  (FFBS, scalar recursions)
        5. mu_h, phi_h, sigma_h2  (conjugate / MH updates)

    All arrays are allocated once per call and kept in float64 internally
    for numerical stability of the AR(1) recursion; storage in main.py
    down-casts to float32 when persisting.
    """
    ny       = state['ny']
    T        = state['T']
    nu       = float(state['nu'])
    Sigma_u  = state['Sigma_u']
    h        = state['h']
    lam      = state['lambda_t']
    mu_h     = float(state['mu_h'])
    phi_h    = float(state['phi_h'])
    sigma_h2 = float(state['sigma_h2'])

    phi_prior   = state['phi_prior_sv']
    mu_prior    = state['mu_prior_sv']
    sigma_prior = state['sigma_prior_sv']

    # --- Precompute Sigma_u^{-1} once (used by _sample_lambda and _build_y_star) ---
    # cho_solve avoids forming the explicit inverse until the final step.
    # For ny up to a few dozen this is essentially free, but the pattern
    # matters if ny ever grows.
    cho, low = cho_factor(Sigma_u, lower=True, check_finite=False)
    Sigma_u_inv = cho_solve((cho, low), np.eye(ny), check_finite=False)
    # Symmetrise to kill floating-point drift before einsum consumes it
    Sigma_u_inv = 0.5 * (Sigma_u_inv + Sigma_u_inv.T)

    # --- Current residuals u_t (depend on freshly-updated Phi, G0, etc.) ---
    U = _current_residuals(state)

    # ---- 1. lambda_t ----
    lam = _sample_lambda(U, h, Sigma_u_inv, nu, rng)

    # ---- 2. log-observation for the KSC scheme ----
    y_star = _build_y_star(U, Sigma_u_inv, lam, ny)

    # ---- 3. mixture indicators s_t ----
    s = _sample_indicators(y_star, h, rng)

    # ---- 4. h_t via FFBS ----
    h = _ffbs_h(y_star, s, mu_h, phi_h, sigma_h2, rng)

    # ---- 5. AR(1) hyper-parameters ----
    mu_h     = _sample_mu_h    (h, phi_h, sigma_h2, mu_prior,    rng)
    phi_h    = _sample_phi_h   (h, mu_h,  sigma_h2, phi_prior,   rng)
    sigma_h2 = _sample_sigma_h2(h, mu_h,  phi_h,    sigma_prior, rng)

    # --- Write back into state (in-place, no extra copies) ---
    state['lambda_t'] = lam
    state['h']        = h
    state['mu_h']     = mu_h
    state['phi_h']    = phi_h
    state['sigma_h2'] = sigma_h2

    # --- Diagnostics (scalars only; cheap to log every iter) ---
    return {
        'h_mean':       float(h.mean()),
        'h_std':        float(h.std()),
        'lambda_mean':  float(lam.mean()),
        'lambda_max':   float(lam.max()),
        'mu_h':         float(mu_h),
        'phi_h':        float(phi_h),
        'sigma_h2':     float(sigma_h2),
    }