from __future__ import annotations
import numpy as np
from numpy.random import Generator
from scipy.linalg import cho_factor, cho_solve, solve_triangular


# STEP 4 : sample Phi_i | G_Phi, Sigma_u, h, lambda_t

# ------------------------------------------------------------------
# Helper: build the stacked G_Phi once per iteration (ny, ny*p)
# ------------------------------------------------------------------

def _stack_G_Phi(G_Phi_list: list, ny: int, n_lags: int) -> np.ndarray:
    G_stack = np.empty((ny, ny * n_lags), dtype=np.int8)
    for s in range(n_lags):
        G_stack[:, s * ny:(s + 1) * ny] = G_Phi_list[s]
    return G_stack


# ------------------------------------------------------------------
# MAIN SAMPLER
# ------------------------------------------------------------------

def step4_sample_Phi(state: dict, rng: Generator) -> dict:
    
    Y = state['Y']                  # (T, ny)
    X = state['X_endo']             # (T, ny * n_lags)
    ny = state['ny']
    n_lags = state['n_lags']
    Sigma_u = state['Sigma_u']      # (ny, ny)

    # Diagonal of the Minnesota prior covariance
    omega_diag = np.diag(state['Omega_phi'])    # (ny * n_lags,)

    # Stack G_Phi once for fast row slicing below
    G_stack = _stack_G_Phi(state['G_Phi'], ny, n_lags)    # (ny, ny*p)

    # --- Optional SV / Student-t rescaling ---------------------------.
    h = state.get('h', None)
    lam = state.get('lambda_t', None)
    if h is not None or lam is not None:
        T = state['T']
        h_arr = h if h is not None else np.zeros(T)
        lam_arr = lam if lam is not None else np.ones(T)
        scale = np.sqrt(np.exp(h_arr) * lam_arr)[:, None]    # (T, 1)
        Y_t = Y / scale
        X_t = X / scale
    else:
        Y_t = Y
        X_t = X

    # --- Precompute Gram matrices ONCE per iteration
    XtX = X_t.T @ X_t
    XtY = X_t.T @ Y_t

    # Diagonal variances of the residuals (eq.-by-eq., from Step 3)
    sigma2_diag = np.diag(Sigma_u)              # (ny,)

    # --- Accumulators for diagnostics --------------------------------
    phi_sq_sum = 0.0        
    phi_max_abs = 0.0
    n_active_total = 0

    # --- Loop over equations (rows of Phi_stack) ---------------------
    for i in range(ny):
        # Active regressor indices for equation i
        active = np.flatnonzero(G_stack[i])
        k_i = active.size

        if k_i == 0:
            # Nothing active -> row stays zero.
            for s in range(n_lags):
                state['Phi'][s][i, :] = 0.0
            continue

        n_active_total += k_i
        inv_sig2 = 1.0 / sigma2_diag[i]

        # --- Posterior precision: V_bar_inv = diag(1/omega) + W'W/sig2
        WtW = XtX[np.ix_(active, active)]                # (k_i, k_i)

        V_bar_inv = WtW * inv_sig2
        V_bar_inv[np.arange(k_i), np.arange(k_i)] += 1.0 / omega_diag[active]

        # --- Posterior mean numerator:  W' y / sigma_i^2
        rhs = XtY[active, i] * inv_sig2                   # (k_i,)

        # --- Single Cholesky: V_bar_inv = L L' ----------------------
        V_bar_inv = 0.5 * (V_bar_inv + V_bar_inv.T)
        try:
            c, low = cho_factor(V_bar_inv, lower=True, overwrite_a=True,
                                check_finite=False)
        except np.linalg.LinAlgError:
            # Tiny ridge + retry
            V_bar_inv[np.arange(k_i), np.arange(k_i)] += 1e-10
            c, low = cho_factor(V_bar_inv, lower=True, overwrite_a=True,
                                check_finite=False)

        # Posterior mean: mu_bar = V_bar @ rhs = (L L')^{-1} rhs
        mu_bar = cho_solve((c, low), rhs, check_finite=False)

        # --- Draw phi ~ N(mu_bar, V_bar) using the SAME L -----------
        z = rng.standard_normal(k_i)
        noise = solve_triangular(c, z, lower=low, trans='T',
                                 check_finite=False)
        phi_active = mu_bar + noise

        # --- Write back directly into the per-lag matrices ----------
        # Scatter phi_active into row i of Phi_stack, split by lag.
        # We build the full row once (mostly zeros) then slice it into
        # the n_lags matrices without another allocation.
        row = np.zeros(ny * n_lags)
        row[active] = phi_active
        for s in range(n_lags):
            state['Phi'][s][i, :] = row[s * ny:(s + 1) * ny]

        # --- Accumulate diagnostics ---------------------------------
        phi_sq_sum += float(phi_active @ phi_active)
        m = float(np.max(np.abs(phi_active)))
        if m > phi_max_abs:
            phi_max_abs = m

    # --- Final diagnostics -------------------------------------------
    total_coefs = ny * ny * n_lags
    return {
        'phi_norm': float(np.sqrt(phi_sq_sum)),
        'phi_max_abs': phi_max_abs,
        'active_fraction': n_active_total / total_coefs if total_coefs else 0.0,
        'n_active': n_active_total,
    }