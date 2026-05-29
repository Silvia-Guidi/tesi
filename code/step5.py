from __future__ import annotations
import numpy as np
from numpy.random import Generator
from scipy.linalg import cho_factor, cho_solve, solve_triangular

def _stack_G_Gamma(G_Gamma_list: list,
                   ny: int,
                   nz: int,
                   n_lags_exo: int) -> np.ndarray:
    """Stack [G_Gamma_1 | G_Gamma_2 | ... | G_Gamma_q] horizontally."""
    G_stack = np.empty((ny, nz * n_lags_exo), dtype=np.int8)
    for s in range(n_lags_exo):
        G_stack[:, s * nz:(s + 1) * nz] = G_Gamma_list[s]
    return G_stack

# ------------------------------------------------------------------
# build the endogenous fitted part X_endo @ Phi_stack'.
# Phi_stack has shape (ny, ny*p) with column blocks ordered by lag,
# matching X_endo's column layout.
# ------------------------------------------------------------------
def _endo_fitted(X_endo: np.ndarray,
                 Phi_list: list,
                 ny: int,
                 n_lags: int) -> np.ndarray:
    """Return X_endo @ Phi_stack', i.e. the part of Y explained by lags of Y."""
    Phi_stack = np.empty((ny, ny * n_lags))
    for s in range(n_lags):
        Phi_stack[:, s * ny:(s + 1) * ny] = Phi_list[s]
    return X_endo @ Phi_stack.T          # (T, ny)

# ------------------------------------------------------------------
# MAIN SAMPLER
# ------------------------------------------------------------------
def step5_sample_Gamma(state: dict, rng: Generator) -> dict:
    """
    Gibbs update for the exogenous coefficients Gamma_j, j=1..q.
    Mirrors step4_sample_Phi but uses partial residuals tildeY and the
    exogenous design X_exo.
    """
    # --- Pull required state -------------------------------------------------
    Y       = state['Y']             # (T, ny)
    X_endo  = state['X_endo']        # (T, ny * p)
    X_exo   = state['X_exo']         # (T, nz * q)
    ny      = state['ny']
    nz      = state['nz']
    n_lags     = state['n_lags']         # p
    n_lags_exo = state['n_lags_exo']     # q
    Sigma_u = state['Sigma_u']       # (ny, ny)

    omega_diag = np.diag(state['Omega_gamma'])    # (nz * q,)

    G_stack = _stack_G_Gamma(state['G_Gamma'], ny, nz, n_lags_exo)  # (ny, nz*q)
    
    # --- Partial residuals: subtract the endogenous fitted part --------------
    tildeY = Y - _endo_fitted(X_endo, state['Phi'], ny, n_lags)     # (T, ny)
    
    # ---  SV rescaling -----------------------------------
    h   = state.get('h', None)
    if h is not None:
        T = state['T']
        h_arr   = h   if h   is not None else np.zeros(T)
        scale = np.sqrt(np.exp(h_arr))[:, None]   # (T, 1)
        Yt = tildeY / scale
        Xt = X_exo  / scale
    else:
        Yt = tildeY
        Xt = X_exo
        
    # --- Precompute Gram matrices ONCE per iteration -------------------------
    XtX = Xt.T @ Xt          # (nz*q, nz*q)
    XtY = Xt.T @ Yt          # (nz*q, ny)

    # Diagonal variances of the residuals (equation-by-equation, from Step 3)
    sigma2_diag = np.diag(Sigma_u)                # (ny,)

    # --- Accumulators for diagnostics ----------------------------------------
    gamma_sq_sum    = 0.0
    gamma_max_abs   = 0.0
    n_active_total  = 0
    
    # --- Loop over equations (rows of Gamma_stack) ---------------------------
    for i in range(ny):
        # Active regressor indices for equation i (columns in X_exo)
        active = np.flatnonzero(G_stack[i])
        k_i = active.size

        if k_i == 0:
            # No active exogenous regressor for this equation: row stays zero.
            for s in range(n_lags_exo):
                state['Gamma'][s][i, :] = 0.0
            continue

        n_active_total += k_i
        inv_sig2 = 1.0 / sigma2_diag[i]

        # --- Posterior precision: V_bar_inv = diag(1/omega) + W'W / sigma_i^2
        WtW = XtX[np.ix_(active, active)]                  # (k_i, k_i)

        V_bar_inv = WtW * inv_sig2
        V_bar_inv[np.arange(k_i), np.arange(k_i)] += 1.0 / omega_diag[active]
        
        # --- Posterior mean numerator:  W' tildeY_i / sigma_i^2 ---------------
        # Note: prior mean for Gamma is zero, so it drops out of the formula.
        rhs = XtY[active, i] * inv_sig2                    # (k_i,)

        # --- Single Cholesky factorisation: V_bar_inv = L L' -----------------
        V_bar_inv = 0.5 * (V_bar_inv + V_bar_inv.T)        # enforce symmetry
        try:
            c, low = cho_factor(V_bar_inv, lower=True, overwrite_a=True,
                                check_finite=False)
        except np.linalg.LinAlgError:
            # Tiny ridge + retry (same pattern as Step 4)
            V_bar_inv[np.arange(k_i), np.arange(k_i)] += 1e-10
            c, low = cho_factor(V_bar_inv, lower=True, overwrite_a=True,
                                check_finite=False)

        # Posterior mean: mu_bar = V_bar @ rhs = (L L')^{-1} rhs
        mu_bar = cho_solve((c, low), rhs, check_finite=False)
        
        # --- Draw gamma ~ N(mu_bar, V_bar) reusing the same Cholesky ---------
        # If V_bar = (L L')^{-1}, then a draw with that covariance is
        #   mu_bar + L^{-T} z,   z ~ N(0, I_{k_i}).
        z = rng.standard_normal(k_i)
        noise = solve_triangular(c, z, lower=low, trans='T',
                                 check_finite=False)
        gamma_active = mu_bar + noise

        # --- Write back into the per-lag Gamma matrices ----------------------
        # Scatter gamma_active into row i, split by lag block of size nz.
        row = np.zeros(nz * n_lags_exo)
        row[active] = gamma_active
        for s in range(n_lags_exo):
            state['Gamma'][s][i, :] = row[s * nz:(s + 1) * nz]

        # --- Accumulate diagnostics ------------------------------------------
        gamma_sq_sum += float(gamma_active @ gamma_active)
        m = float(np.max(np.abs(gamma_active)))
        if m > gamma_max_abs:
            gamma_max_abs = m
        
    # --- Final diagnostics ---------------------------------------------------
    total_coefs = ny * nz * n_lags_exo
    return {
        'gamma_norm':       float(np.sqrt(gamma_sq_sum)),
        'gamma_max_abs':    gamma_max_abs,
        'active_fraction':  n_active_total / total_coefs if total_coefs else 0.0,
        'n_active':         n_active_total,
    }
