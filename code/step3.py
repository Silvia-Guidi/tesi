from __future__ import annotations
import numpy as np
from scipy.linalg import lu_factor, lu_solve, LinAlgError
from scipy.stats import invwishart

from priors import inverse_wishart_prior

# ==============================================
# STEP 3: SAMPLE Sigma_u FROM ITS FULL CONDITIONAL
# ==============================================


# ---
# Inverse-Wishart sampler using scipy.stats.invwishart.
#
# Convention (matches scipy and the standard Bayesian-econometrics literature):
#   Sigma ~ IW(df, scale)     E[Sigma] = scale / (df - p - 1)
#
# Historical note: an earlier in-house Bartlett implementation sampled
# Sigma ~ IW(df, scale^{-1}) instead of IW(df, scale) because the matrix
# returned by `solve(W, I)` inverts the scale matrix implicitly. That bug
# caused Sigma_u to shrink by a factor ~ trace(scale)^{-2}, which in the
# BGVAR with ny=56 produced trace(Sigma_u) ~ 0.017 instead of ~ 1.
# ---
def invwishart_sample(df: float,
                      scale: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Draw Sigma ~ IW(df, scale) with scipy.

    Convention: E[Sigma] = scale / (df - p - 1).
    """
    seed = int(rng.integers(0, 2**31 - 1))
    return invwishart.rvs(df=df, scale=scale, random_state=seed)


# ---
# Map structural coefficients (Phi, Gamma) to reduced-form (A_endo, A_exo)
# via A_0 = I - (G_0 * Phi_0), A_i = A_0^{-1} (G_i * Phi_i).
# Uses LU factorisation once and triangular-solve per lag block:
#   cheaper than computing A_0^{-1} explicitly and multiplying.
# ---
def compute_reduced_form_coeff(state: dict) -> tuple[np.ndarray, np.ndarray]:
    ny = state['ny']

    Phi_list = state.get('Phi',   None)
    G_Phi    = state.get('G_Phi', None)

    Gamma_list = state.get('Gamma',   None)
    G_Gamma    = state.get('G_Gamma', None)

    # Early-out when steps 4-5 are not implemented yet: zero coefficients
    p_endo = len(Phi_list) if Phi_list is not None else state.get('n_lags', 0)
    q_exo  = len(Gamma_list) if Gamma_list is not None else state.get('n_lags_exo', 0)
    nz     = state.get('nz', 0)

    need_endo = (Phi_list is not None and G_Phi is not None)
    need_exo  = (Gamma_list is not None and G_Gamma is not None and nz > 0)

    if not need_endo:
        A_endo = np.zeros((ny, ny * p_endo)) if p_endo > 0 else np.zeros((ny, 0))
    if not need_exo:
        A_exo  = np.zeros((ny, nz * q_exo))  if (nz > 0 and q_exo > 0) else np.zeros((ny, 0))

    if not (need_endo or need_exo):
        return A_endo, A_exo

    # --- LU factorisation of A_0 (done once, reused for every lag block) ---
    Phi_0 = state.get('Phi_0', np.zeros((ny, ny)))
    G_0   = state.get('G0',    np.zeros((ny, ny), dtype=int))
    A0    = np.eye(ny) - (G_0 * Phi_0)

    try:
        lu, piv = lu_factor(A0)
    except (LinAlgError, ValueError):
        # Rare fallback if A_0 is numerically singular: treat reduced form as
        # identity (no contemporaneous rotation) so the sampler can continue.
        lu, piv = None, None

    def _apply_A0inv(M: np.ndarray) -> np.ndarray:
        if lu is None:
            return M
        return lu_solve((lu, piv), M)

    if need_endo:
        blocks = [_apply_A0inv(G_Phi[i] * Phi_list[i]) for i in range(len(Phi_list))]
        A_endo = np.hstack(blocks)

    if need_exo:
        blocks = [_apply_A0inv(G_Gamma[j] * Gamma_list[j]) for j in range(len(Gamma_list))]
        A_exo  = np.hstack(blocks)

    return A_endo, A_exo


# ---
# Compute the reduced-form residuals U = Y - X_endo A_endo' - X_exo A_exo'.
# ---
def compute_residuals(state: dict) -> np.ndarray:
    Y      = state['Y']
    X_endo = state.get('X_endo', None)
    X_exo  = state.get('X_exo',  None)

    A_endo, A_exo = compute_reduced_form_coeff(state)

    # Start from a view, copy only when we actually subtract something
    U = Y
    copied = False
    if X_endo is not None and A_endo.size > 0:
        U = Y - X_endo @ A_endo.T
        copied = True
    if X_exo is not None and A_exo.size > 0:
        if copied:
            U -= X_exo @ A_exo.T
        else:
            U = Y - X_exo @ A_exo.T
            copied = True

    # Always return an independent array (downstream code may modify it)
    return U if copied else Y.copy()


# ---
# Standardise residuals by stochastic-volatility scaling factors:
#     u_tilde_t = u_t / sqrt(exp(h_t) * lambda_t)
# Fast path when SV (step 6) is not yet active: scale is identically 1,
# so we skip the exp/sqrt/division entirely.
# ---
def standardise_residuals(U: np.ndarray, state: dict) -> np.ndarray:
    h   = state.get('h')
    lam = state.get('lambda_t')

    # Fast path: step 6 not yet implemented -> identity scaling
    if h is None and lam is None:
        return U

    T = U.shape[0]
    if h is None:
        h = np.zeros(T)
    if lam is None:
        lam = np.ones(T)

    # sqrt(exp(h) * lam) = exp(h/2) * sqrt(lam)
    # Two cheap elementwise ops instead of one exp + one multiply + one sqrt.
    scale = np.exp(0.5 * h) * np.sqrt(lam)
    return U / scale[:, None]


# ---
# MAIN SAMPLER
# ---
def step3_sample(state: dict, rng: np.random.Generator) -> dict:
    ny      = state['ny']
    T       = state['T']
    hparams = state['hparams']

    # --- Prior hyperparameters ---
    S_0, alpha_0 = inverse_wishart_prior(ny, hparams)

    # --- Residuals and scaling ---
    U       = compute_residuals(state)
    U_tilde = standardise_residuals(U, state)

    # --- Posterior parameters ---
    S_post     = S_0 + U_tilde.T @ U_tilde
    alpha_post = alpha_0 + T

    # Symmetrise for numerical safety (prevents Cholesky failures downstream)
    S_post = 0.5 * (S_post + S_post.T)

    # --- Sample Sigma_u from IW(alpha_post, S_post) ---
    Sigma_u = invwishart_sample(alpha_post, S_post, rng)
    Sigma_u = 0.5 * (Sigma_u + Sigma_u.T)

    # --- Write back into state ---
    state['Sigma_u'] = Sigma_u

    # --- Diagnostics ---
    sign, logdet = np.linalg.slogdet(Sigma_u)
    return {
        'logdet_Sigma': float(logdet) if sign > 0 else float('nan'),
        'trace_Sigma':  float(np.trace(Sigma_u)),
        'alpha_post':   int(alpha_post),
    }