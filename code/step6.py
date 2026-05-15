"""
STEP 6 (RANDOM WALK SV, SINGLE-SITE METROPOLIS-HASTINGS)
========================================================
Stochastic Volatility for the BGVAR -- minimalist version.

Model
-----
For each time t = 1, ..., T:
    u_t  | h_t  ~  N_{ny}(0,  exp(h_t) * Sigma_u)        # reduced-form residuals
    h_t          =  h_{t-1} + eta_t,  eta_t ~ N(0, sigma_h^2)         # random walk
    h_0          ~  N(0, V_h0)                                        # diffuse prior on level

Why this version
----------------
The previous step6 used the Kim-Shephard-Chib (KSC) 7-component mixture
augmentation followed by Forward-Filter-Backward-Sampling (FFBS) on the
resulting linear-Gaussian state space.  That is the textbook fast block
sampler, but it has many moving parts: a KSC pseudo-observation tuned for
ny=1, a Kalman forward pass, a backward simulation smoother, and tight
sensitivity to initial conditions and diffuse priors.  Any small bug in
any of those collapses the entire path toward the prior mean.

Here we replace both KSC and FFBS with the simplest correct sampler in
the SV literature: single-site Metropolis-Hastings on each h_t, following
Jacquier-Polson-Rossi (1994).  Each h_t is updated independently using
ONLY its Markov blanket (h_{t-1}, h_{t+1}, q_t).  No Kalman recursions,
no auxiliary mixture, no backward sweep.  If anything goes wrong, the
problem is local to that single update.

Gibbs sweep (one call to step6_sample_SV):
    1. Compute q_t = u_t' Sigma_u^{-1} u_t                (Mahalanobis sum)
    2. Single-site MH update of h_{0:T-1}                 (the volatility path)
    3. Adaptive proposal scaling during burn-in           (Roberts-Rosenthal)
    4. Conjugate Inverse-Gamma update of sigma_h^2
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve


# Target acceptance rate for single-site MH on a univariate target.
# Roberts, Gelman & Gilks (1997) show this is asymptotically optimal.
_MH_TARGET_ACCEPT = 0.44

# How aggressively we rescale tau when it is off-target during burn-in.
# Multiplicative step: tau_new = tau_old * exp(+/- _ADAPT_STEP).
_ADAPT_STEP = 0.05


# ==========================================================================
# Step 6.1 -- log full conditional of a single h_t  (up to additive const).
#
# For 1 <= t <= T-2 (interior point) the conditional log-density is:
#
#     log pi(h_t | rest) = -(ny/2) * h_t                       # likelihood: det
#                          - q_t / (2 * exp(h_t))              # likelihood: quad
#                          - (h_t - h_{t-1})^2 / (2*sigma_h2)  # prior past
#                          - (h_{t+1} - h_t)^2 / (2*sigma_h2)  # prior future
#
# Derivation:
#   p(u_t | h_t) propto |exp(h_t)*Sigma_u|^{-1/2} * exp(-0.5*q_t/exp(h_t))
#                 = exp(-0.5*ny*h_t) * exp(-0.5*q_t*exp(-h_t)) * (Sigma_u term)
#   The Sigma_u term and any constant do not depend on h_t, hence dropped.
#   The RW prior adds two quadratic terms in h_t (one from h_{t-1} -> h_t,
#   one from h_t -> h_{t+1}).
#
# Boundary cases:
#   t = 0    : the "past" term becomes -h_t^2 / (2*V_h0)
#              (h_0 ~ N(0, V_h0) diffuse prior; no h_{-1} exists)
#   t = T-1  : the "future" term is absent (no h_T exists)
#
# We pass h_prev / h_next as floats with np.nan signalling boundaries.
# A single function therefore handles all three cases via simple branches.
# ==========================================================================
def _log_target_ht(h_t: float,
                   h_prev: float, h_next: float,
                   q_t: float, ny: int,
                   sigma_h2: float, V_h0: float) -> float:
    # ---- Likelihood contribution ---------------------------------------
    # q_t is fixed during the MH update (it depends on residuals and
    # Sigma_u, not on h_t), so we treat it as a constant here.
    ll = -0.5 * ny * h_t - 0.5 * q_t * np.exp(-h_t)

    # ---- Prior from the past (or diffuse N(0, V_h0) prior at t = 0) ----
    if np.isnan(h_prev):
        lp_past = -0.5 * h_t * h_t / V_h0
    else:
        d = h_t - h_prev
        lp_past = -0.5 * d * d / sigma_h2

    # ---- Prior from the future (absent at t = T-1) ---------------------
    if np.isnan(h_next):
        lp_fut = 0.0
    else:
        d = h_next - h_t
        lp_fut = -0.5 * d * d / sigma_h2

    return ll + lp_past + lp_fut


# ==========================================================================
# Step 6.2 -- one full sweep of single-site MH over h_{0:T-1}.
#
# Proposal: symmetric random walk on h_t
#     h_t* = h_t + tau * N(0, 1)
# Acceptance ratio (symmetric proposal => no proposal correction):
#     log r = log pi(h_t*) - log pi(h_t)
# Accept iff log U < log r, with U ~ Uniform(0, 1).
#
# Loop order: t = 0, 1, ..., T-1.  Single-site Gibbs is valid in any order;
# sequential is cache-friendly and means the update of h_t already sees
# the newly-accepted h_{t-1} from this sweep, which improves mixing.
#
# Returns the updated h array AND the empirical acceptance rate, used by
# the caller for the burn-in adaptive scaling of tau.
# ==========================================================================
def _sample_h_path(h: np.ndarray,
                   q: np.ndarray,
                   sigma_h2: float,
                   V_h0: float,
                   ny: int,
                   proposal_sd: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, float]:
    T = h.shape[0]
    h_new = h.copy()                  # rejected proposals must leave h_t unchanged
    n_accept = 0

    # Pre-draw all random numbers in one shot (numpy vectorised RNG is much
    # faster than calling rng inside the Python loop).
    z_all = rng.standard_normal(T) * proposal_sd      # MH proposal noise
    log_u = np.log(rng.uniform(size=T))               # log of acceptance uniforms

    for t in range(T):
        # Identify Markov-blanket neighbours; NaN signals a boundary.
        h_prev = h_new[t - 1] if t > 0       else np.nan
        h_next = h_new[t + 1] if t < T - 1   else np.nan

        h_old  = h_new[t]
        h_prop = h_old + z_all[t]

        log_pi_old = _log_target_ht(h_old,  h_prev, h_next,
                                    q[t], ny, sigma_h2, V_h0)
        log_pi_new = _log_target_ht(h_prop, h_prev, h_next,
                                    q[t], ny, sigma_h2, V_h0)

        if log_u[t] < (log_pi_new - log_pi_old):
            h_new[t]  = h_prop
            n_accept += 1
        # else: keep h_old (already stored in h_new[t])

    return h_new, n_accept / T


# ==========================================================================
# Step 6.3 -- conjugate Inverse-Gamma update for sigma_h^2.
#
# Under the RW prior, eta_t = h_t - h_{t-1} ~ N(0, sigma_h^2) for t = 1..T-1.
# We do NOT include t = 0 in the sum because h_0 is governed by its own
# N(0, V_h0) diffuse prior, not by sigma_h^2.
#
# Prior:     sigma_h^2 ~ IG(shape0, scale0)
# Posterior: shape_post = shape0 + (T - 1) / 2
#            scale_post = scale0 + 0.5 * sum_t eta_t^2
#
# Sampling trick: if X ~ Gamma(shape_post, 1) then 1/X ~ IG(shape_post, 1),
# so sigma_h^2 = scale_post / X gives the desired IG draw.
# ==========================================================================
def _sample_sigma_h2(h: np.ndarray,
                     sigma_prior: dict,
                     rng: np.random.Generator) -> float:
    eta = np.diff(h)                                       # (T-1,) RW innovations
    post_shape = sigma_prior['shape'] + 0.5 * eta.shape[0]
    post_scale = sigma_prior['scale'] + 0.5 * float(eta @ eta)
    return post_scale / rng.standard_gamma(post_shape)


# ==========================================================================
# MAIN SAMPLER -- one Gibbs sweep
# ==========================================================================
def step6_sample_SV(state: dict, rng: np.random.Generator) -> dict:
    """
    One Gibbs sweep over the random-walk stochastic-volatility block.

    Pipeline:
        1. Recompute residuals u_t from current Phi / Gamma.
        2. Form q_t = u_t' Sigma_u^{-1} u_t.
        3. Single-site MH sweep updating h_{0:T-1}.
        4. (Burn-in only) Adapt the MH proposal SD to target ~44% accept.
        5. Conjugate IG update of sigma_h^2.

    Reads from / writes to `state`:
        h         : (T,) log-volatility path
        sigma_h2  : float, RW innovation variance
        sv_propsd : float, MH proposal SD (adaptive during burn-in)
        sv_iter   : int, Gibbs iteration counter (used to freeze adaptation)
        lambda_t  : (T,) array of ones, kept ONLY for backward compatibility
                    with step3 / step4 which still multiply by sqrt(lam).
    """
    from step3 import compute_residuals             # lazy import (avoid circular dep)

    # ---- Unpack state ---------------------------------------------------
    ny       = state['ny']
    Sigma_u  = state['Sigma_u']
    h        = state['h']
    sigma_h2 = float(state['sigma_h2'])

    sigma_prior = state['sigma_prior_sv']
    V_h0        = float(state.get('V_h0', 10.0))

    # Adaptive MH bookkeeping.  Both fields are created on first call.
    proposal_sd = float(state.get('sv_propsd', 0.20))   # sensible default
    sv_iter     = int(state.get('sv_iter', 0))
    burnin      = int(state.get('BURNIN_for_SV', 1000)) # adapt only up to this iter

    # ---- 1. Mahalanobis sums q_t ---------------------------------------
    # Cholesky-factor Sigma_u once, invert, symmetrise for numerical safety.
    cho, low = cho_factor(Sigma_u, lower=True, check_finite=False)
    Sigma_u_inv = cho_solve((cho, low), np.eye(ny), check_finite=False)
    Sigma_u_inv = 0.5 * (Sigma_u_inv + Sigma_u_inv.T)

    U = compute_residuals(state)                           # (T, ny)
    US = U @ Sigma_u_inv                                   # (T, ny)
    q = np.einsum('ti,ti->t', US, U)                       # (T,)
    # Numerical floor: q_t cannot be exactly zero or negative due to roundoff.
    q = np.maximum(q, 1e-12)

    # ---- 2. Single-site MH sweep over h --------------------------------
    h_new, accept_rate = _sample_h_path(h, q, sigma_h2, V_h0,
                                        ny, proposal_sd, rng)

    # ---- 3. Adaptive scaling (burn-in only, then frozen) ---------------
    # Move tau toward the value that yields ~44% acceptance.
    # If accept too high  -> proposals too small -> increase tau.
    # If accept too low   -> proposals too large -> decrease tau.
    if sv_iter < burnin:
        if accept_rate > _MH_TARGET_ACCEPT:
            proposal_sd *= np.exp(_ADAPT_STEP)
        else:
            proposal_sd *= np.exp(-_ADAPT_STEP)
        # Safety bounds (avoid pathological collapse or runaway).
        proposal_sd = float(np.clip(proposal_sd, 1e-3, 5.0))

    # ---- 4. Conjugate IG update for sigma_h^2 --------------------------
    sigma_h2_new = _sample_sigma_h2(h_new, sigma_prior, rng)

    # ---- 5. Write back into state --------------------------------------
    state['h']         = h_new
    state['sigma_h2']  = sigma_h2_new
    state['sv_propsd'] = proposal_sd
    state['sv_iter']   = sv_iter + 1
    # lambda_t kept at 1 so that step3 / step4's   sqrt(exp(h) * lam)
    # rescaling reduces cleanly to   exp(h/2)   in this SV-only version.
    state['lambda_t']  = np.ones_like(h_new)

    return {
        'h_mean':      float(h_new.mean()),
        'h_std':       float(h_new.std()),
        'sigma_h2':    float(sigma_h2_new),
        'accept_rate': float(accept_rate),
        'proposal_sd': float(proposal_sd),
    }