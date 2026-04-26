from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
# ======================================================
# STEP 6: SAMPLE STOCHASTIC VOLATILITY h_t AND STUDENT-t MIXING WEIGHTS lambda_t
#
# Model (per observation t):
#   u_t  ~  N(0, exp(h_t) * lambda_t * Sigma_u)        # u_t = Y_t - X_t Phi'
#   h_t  =  mu_h + phi_h * (h_{t-1} - mu_h) + eta_t,    eta_t ~ N(0, sigma_h2)
#   lambda_t  ~  IG(nu/2, nu/2)                         # Student-t mixing
#
# IDENTIFICATION NOTE
# -------------------
# The pair (h_t, lambda_t) is only weakly identified: the joint likelihood
# is invariant under
#       h_t  -> h_t - c
#       lam_t -> lam_t * exp(c)
# because exp(h_t) * lam_t is preserved. Without an explicit identification
# constraint the Gibbs sampler drifts slowly along this ridge, which makes
# h_t flatten over iterations and phi_h collapse toward 0.
#
# Fix: after sampling lam_t, re-centre it so that mean(log lam) = 0 and
# absorb the shift into h_t. This is the standard "geometric-mean
# centring" trick. The likelihood is exactly preserved point-by-point;
# only the slow random walk along the ridge is killed.
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


# ------------------------------------------------------
# 1. Lambda_t  ~  IG((nu + ny) / 2, (nu + q_t) / 2)
# ------------------------------------------------------
def _sample_lambda(U: np.ndarray,
                   h: np.ndarray,
                   Sigma_u_inv: np.ndarray,
                   nu: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised Inverse-Gamma draw for lambda_t, t = 1..T.

    Full conditional:
        lambda_t | rest ~ IG(shape=(nu+ny)/2, scale=(nu + q_t)/2)
        with q_t = u_t' Sigma_u^{-1} u_t / exp(h_t).
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
    lam = scale / g

    # Two-sided clip: upper cap 50 prevents outlier spikes; lower floor
    # 1e-3 prevents numerical underflow that would crash y_star = log(q/lam).
    # Without the lower floor, the median-centring step downstream can
    # send small lam_t below the float64 threshold when shift is large.
    return np.clip(lam, 1e-3, 50.0)


# ------------------------------------------------------
# 2. Linearised observation y_star_t for the KSC scheme
# ------------------------------------------------------
def _build_y_star(U: np.ndarray,
                  Sigma_u_inv: np.ndarray,
                  lambda_t: np.ndarray,
                  ny: int,
                  offset_c: float = 1e-6) -> np.ndarray:
    """
    Return y_star_t = log(q_t / (ny * lambda_t) + offset_c)  for t = 1..T.

    The +offset_c is the standard "offset" trick (Kim, Shephard & Chib
    1998) that prevents log(0) when residuals are numerically tiny.
    """
    US = U @ Sigma_u_inv
    q_raw = np.einsum('ti,ti->t', US, U)
    return np.log(q_raw / (ny * lambda_t) + offset_c)


# ------------------------------------------------------
# 3. Mixture indicators s_t (one of K=7 components per t)
# ------------------------------------------------------
def _sample_indicators(y_star: np.ndarray,
                       h: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Categorical draw of s_t in {0..6} from
        P(s_t = k) ∝ pi_k * N(y_star_t - h_t | m_k, V_k).

    Implemented via the Gumbel-max trick (single argmax, no per-row
    normalisation), which is the fastest correct way to sample from
    a categorical when only the unnormalised log-weights are known.
    """
    # Residual e_t = y_star_t - h_t; same across all K components
    e = y_star - h                                                  # (T,)

    log_norm = -_KSC_LOG2PI_HALF - 0.5 * np.log(_KSC_VAR)           # (K,)
    diff = e[:, None] - _KSC_MEAN                                   # (T, K)
    log_lik = log_norm - 0.5 * diff * diff / _KSC_VAR               # (T, K)

    # log-posterior weights (unnormalised)
    log_w = _KSC_LOGPROB + log_lik                                  # (T, K)

    # Gumbel-max: add iid Gumbel(0,1) noise, take argmax along K
    gumbel = -np.log(-np.log(rng.uniform(size=log_w.shape)))
    return np.argmax(log_w + gumbel, axis=1).astype(np.uint8)


# ------------------------------------------------------
# 4. h_t  via Forward Filter Backward Sampler (FFBS)
# ------------------------------------------------------
def _ffbs_h(y_star: np.ndarray,
            s: np.ndarray,
            mu_h: float,
            phi_h: float,
            sigma_h2: float,
            rng: np.random.Generator) -> np.ndarray:
    """
    Sample the joint trajectory {h_1, ..., h_T} from its full conditional
    via FFBS on the linearised state-space:
        y_star_t  =  h_t + m_{s_t} + e_t,    e_t ~ N(0, V_{s_t})
        h_t       =  mu_h + phi_h (h_{t-1} - mu_h) + eta_t,
                                              eta_t ~ N(0, sigma_h2)
    """
    T = y_star.shape[0]
    m_s = _KSC_MEAN[s]                  # (T,)   per-t observation offset
    v_s = _KSC_VAR[s]                   # (T,)   per-t observation variance

    # Storage for the filtered moments (a_t, P_t) = E[h_t | y_1:t], Var[...]
    a_filt = np.empty(T)
    P_filt = np.empty(T)

    # Stationary distribution of the AR(1) serves as the initial prior on h_0:
    #   h_0 ~ N(mu_h, sigma_h2 / (1 - phi_h^2))
    stationary_var = sigma_h2 / max(1.0 - phi_h * phi_h, 1e-10)

    # --- t = 0: prior -> posterior after y_0 ---
    a_pred, P_pred = mu_h, stationary_var
    y_tilde = y_star[0] - m_s[0]                  # demeaned observation
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


# ------------------------------------------------------
# 5a. mu_h | h, phi_h, sigma_h2  (conjugate Normal)
# ------------------------------------------------------
def _sample_mu_h(h: np.ndarray,
                 phi_h: float,
                 sigma_h2: float,
                 mu_prior: dict,
                 rng: np.random.Generator) -> float:
    """
    Conjugate Gaussian update for mu_h | h, phi_h, sigma_h2.
    Rewrite the AR(1) as
        z_t = (1 - phi_h) * mu_h + eta_t,    z_t = h_t - phi_h h_{t-1}
    so that mu_h has a closed-form Gaussian posterior.
    """
    T = h.shape[0]
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


# ------------------------------------------------------
# 5b. phi_h | h, mu_h, sigma_h2  (Metropolis with Beta prior)
# ------------------------------------------------------
def _sample_phi_h(h: np.ndarray,
                  mu_h: float,
                  sigma_h2: float,
                  phi_prior: dict,
                  rng: np.random.Generator,
                  max_tries: int = 20) -> float:
    """
    Metropolis-Hastings step for phi_h on (-1, 1) with prior
        (phi_h + 1) / 2  ~  Beta(a, b).

    Proposal is the Gaussian posterior under a flat prior; the Beta prior
    enters only in the accept/reject ratio. The proposal is symmetric
    given the data, so the MH ratio reduces to the prior ratio at the
    candidate vs the current state.
    """
    h_lag  = h[:-1] - mu_h
    h_curr = h[1:]  - mu_h

    denom = float(h_lag @ h_lag)
    if denom < 1e-12:
        # Degenerate: no AR(1) signal in h, fall back to a prior draw
        return float(rng.beta(phi_prior['a'], phi_prior['b']) * 2.0 - 1.0)

    prop_mean = float(h_lag @ h_curr) / denom
    prop_std  = np.sqrt(sigma_h2 / denom)

    a, b = phi_prior['a'], phi_prior['b']

    # Current value implied by the existing draw of h is approximated by
    # the OLS estimate (used only inside the prior ratio).
    phi_curr = float(np.clip(prop_mean, -0.999, 0.999))

    # Propose, then accept/reject under the Beta prior on (phi+1)/2
    for _ in range(max_tries):
        phi_star = prop_mean + prop_std * rng.standard_normal()
        if not (-1.0 < phi_star < 1.0):
            continue

        # log prior ratio on (phi+1)/2 ~ Beta(a, b)
        log_ratio = (
            (a - 1.0) * (np.log1p(phi_star) - np.log1p(phi_curr))
          + (b - 1.0) * (np.log1p(-phi_star) - np.log1p(-phi_curr))
        )
        if np.log(rng.uniform()) < log_ratio:
            return phi_star

    # No accepted proposal in max_tries: keep the OLS-style mean
    return phi_curr


# ------------------------------------------------------
# 5c. sigma_h2 | h, mu_h, phi_h  (conjugate Inverse-Gamma)
# ------------------------------------------------------
def _sample_sigma_h2(h: np.ndarray,
                     mu_h: float,
                     phi_h: float,
                     sigma_prior: dict,
                     rng: np.random.Generator) -> float:
    """
    Conjugate IG update:
        sigma_h2 | h, mu_h, phi_h  ~  IG(shape0 + (T-1)/2, scale0 + 0.5 * SS)
    where SS = sum_t (h_t - mu_h - phi_h (h_{t-1} - mu_h))^2.
    """
    eta = (h[1:] - mu_h) - phi_h * (h[:-1] - mu_h)      # AR(1) innovations
    T_eff = eta.shape[0]

    post_shape = sigma_prior['shape'] + 0.5 * T_eff
    post_scale = sigma_prior['scale'] + 0.5 * float(eta @ eta)
    return post_scale / rng.standard_gamma(post_shape)


# ------------------------------------------------------
# Helper: get fresh residuals from the current state
# ------------------------------------------------------
def _current_residuals(state: dict) -> np.ndarray:
    """
    Lazy import of step3.compute_residuals to avoid circular imports
    at module load time. Returns an independent (T, ny) array.
    """
    from step3 import compute_residuals
    return compute_residuals(state)


# ======================================================
# MAIN SAMPLER
# ======================================================
def step6_sample_SV(state: dict, rng: np.random.Generator) -> dict:
    """
    One Gibbs sweep over the stochastic-volatility block:
        1. lambda_t              (closed-form IG, vectorised over t)
        2. y_star                (KSC log-observation)
        3. s_t                   (categorical over 7 mixture components)
        4. h_t                   (FFBS, scalar recursions)
        5. IDENTIFICATION fix    (re-centre log lam to mean 0, absorb in h)
        6. mu_h, phi_h, sigma_h2 (conjugate / MH updates on the centred h)

    Returns a small dict of scalar diagnostics for the progress logger.
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

    # ---- 5. IDENTIFICATION FIX (v2: robust to outliers) ----------------
    # Re-centre log lam using the MEDIAN (not the mean) so a few large
    # lambda_t values can't dominate the shift and crush everyone else.
    # Floor lam from below after centring to avoid numerical underflow
    # in the next iteration's y_star = log(q_raw / (ny * lam)).
    log_lam = np.log(lam)
    shift   = float(np.median(log_lam))      # robust to outliers
    lam     = lam * np.exp(-shift)           # mean(log lam) ~ 0 (in the bulk)
    lam     = np.maximum(lam, 1e-3)          # prevent underflow in next iter
    h       = h + shift                      # absorb the shift into h_t
    # ---------------------------------------------------------------------

    # ---- 6. AR(1) hyper-parameters (on the CENTRED h_t) -----------------
    # Order matters: mu_h first, then phi_h (uses fresh mu_h), then
    # sigma_h2 (uses fresh mu_h and phi_h).
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