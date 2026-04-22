from __future__ import annotations
import numpy as np
from math import lgamma
from numpy.random import Generator
from step1 import log_BGe

# =============================================
# STEP 2: SAMPLING THE TEMPORAL GRAPH STRUCTURES G_Phi (and G_Gamma)
# =============================================

# ---
# 1. CACHE: per-equation BGe scores for the current graph
#    Response i has global index i in the full design [Y | X_reg];
#    its lagged parents have global indices ny + parents.
# ---
def all_eq_scores(G_C: np.ndarray,
                  S_post_full: np.ndarray,
                  ny: int,
                  alpha_post_half: float,
                  bge_const: np.ndarray) -> np.ndarray:
    scores = np.zeros(ny)
    for i in range(ny):
        parents = np.where(G_C[i, :] == 1)[0]
        cols = (
            np.concatenate(([i], ny + parents))
            if parents.size > 0
            else np.array([i])
        )
        S_block     = S_post_full[np.ix_(cols, cols)]
        nd          = S_block.shape[0]
        scores[i]   = log_BGe(S_block, alpha_post_half, bge_const[nd])
    return scores


# ---
# 2. (Un)stacking design matrix and graph
#
#    Column layout of the stacked objects (X_reg and G_C):
#        [lag1_endo | lag1_exo | lag2_endo | lag2_exo | ... ]
# ---
def stack_design(X_endo: np.ndarray,
                 X_exo:  np.ndarray | None,
                 ny: int, nz: int, n_lags: int) -> np.ndarray:
    if nz == 0 or X_exo is None:
        return X_endo

    T = X_endo.shape[0]
    n = ny + nz
    X_reg = np.zeros((T, n * n_lags))
    for s in range(n_lags):
        X_reg[:, s*n      : s*n + ny    ] = X_endo[:, s*ny : (s+1)*ny]
        X_reg[:, s*n + ny : (s+1)*n     ] = X_exo[:,  s*nz : (s+1)*nz]
    return X_reg


def stack_GC(G_Phi:   list,
             G_Gamma: list | None,
             ny: int, nz: int, n_lags: int) -> np.ndarray:
    n   = ny + nz
    G_C = np.zeros((ny, n * n_lags), dtype=int)
    for s in range(n_lags):
        G_C[:, s*n : s*n + ny] = G_Phi[s]
        if nz > 0 and G_Gamma is not None:
            G_C[:, s*n + ny : (s+1)*n] = G_Gamma[s]
    return G_C


def unstack_GC(G_C: np.ndarray, ny: int, nz: int, n_lags: int):
    n       = ny + nz
    G_Phi   = [np.zeros((ny, ny), dtype=int) for _ in range(n_lags)]
    G_Gamma = [np.zeros((ny, nz), dtype=int) for _ in range(n_lags)]
    for s in range(n_lags):
        G_Phi[s] = G_C[:, s*n : s*n + ny].copy()
        if nz > 0:
            G_Gamma[s] = G_C[:, s*n + ny : (s+1)*n].copy()
    return G_Phi, G_Gamma


# ---
# 3. MAIN FUNCTION
# ---
def step2_sample(state: dict,
                 rng: Generator,
                 alpha_BGe: float = None) -> dict:

    Y      = state['Y']
    ny     = state['ny']
    nz     = state.get('nz', 0)
    n_lags = state['n_lags']
    pi     = state['pi_bernoulli']

    T_obs = Y.shape[0]

    # --- Build the stacked design matrix X_reg and the unified graph G_C ---
    X_reg = stack_design(state['X_endo'], state.get('X_exo'), ny, nz, n_lags)
    G_C   = stack_GC(state['G_Phi'], state.get('G_Gamma'), ny, nz, n_lags)

    npar = G_C.shape[1]   # = (ny + nz) * n_lags

    if alpha_BGe is None:
        alpha_BGe = float(ny + 2)

    # --- Pre-compute BGe constants that depend only on nd ---
    # Max parent-set size for equation i is all lagged regressors: nd up to
    # 1 + npar = 1 + (ny + nz) * n_lags.
    log_pi          = np.log(np.pi)
    alpha_half      = alpha_BGe / 2.0
    alpha_post_half = (alpha_BGe + T_obs) / 2.0

    def _log_mgamma(a: float, d: int) -> float:
        return (d * (d - 1) / 4.0) * log_pi + sum(
            lgamma(a - k / 2.0) for k in range(d)
        )

    max_nd = 1 + npar
    bge_const = np.empty(max_nd + 1)
    bge_const[0] = 0.0   # placeholder (nd >= 1 always)
    for nd in range(1, max_nd + 1):
        bge_const[nd] = (
            -(nd * T_obs / 2.0) * log_pi
            + _log_mgamma(alpha_post_half, nd)
            - _log_mgamma(alpha_half, nd)
        )

    # --- Pre-compute Gram matrix once for fast BGe scoring ---
    # Any sub-block S = I + D.T @ D for a candidate (response + parents) set
    # is obtained by plain index slicing into S_post_full.
    full_design = np.hstack([Y, X_reg])                         # (T, ny + npar)
    S_post_full = np.eye(ny + npar) + full_design.T @ full_design

    # Cache per-equation BGe scores for the current graph
    scores_curr = all_eq_scores(G_C, S_post_full, ny, alpha_post_half, bge_const)

    n_accept    = 0
    n_proposals = 0

    for i in rng.permutation(ny):

        # --- Propose a single edge toggle on equation i ---
        # Unlike G0, the temporal graph G_C has no acyclicity constraint
        # (edges always flow from t-s to t), so any column j is legal.
        j = int(rng.integers(0, npar))

        # Flip in place, revert on reject
        old_ij     = G_C[i, j]
        G_C[i, j]  = 1 - old_ij
        n_proposals += 1

        # --- Local Bayes factor on equation i (fast path) ---
        parents_prop = np.where(G_C[i, :] == 1)[0]
        cols = (
            np.concatenate(([i], ny + parents_prop))
            if parents_prop.size > 0
            else np.array([i])
        )
        S_block     = S_post_full[np.ix_(cols, cols)]
        nd          = S_block.shape[0]
        new_score_i = log_BGe(S_block, alpha_post_half, bge_const[nd])

        # Prior ratio depends on the direction of the move
        delta           = G_C[i, j] - old_ij            # +1 add, -1 remove
        log_prior_ratio = delta * (np.log(pi) - np.log(1 - pi))

        log_BF = (new_score_i - scores_curr[i]) + log_prior_ratio

        # --- Accept / reject ---
        if np.log(rng.uniform()) < log_BF:
            scores_curr[i] = new_score_i
            n_accept      += 1
        else:
            # revert
            G_C[i, j] = old_ij

    # --- Write back the new graph into the state ---
    G_Phi_new, G_Gamma_new = unstack_GC(G_C, ny, nz, n_lags)
    state['G_Phi'] = G_Phi_new
    if nz > 0:
        state['G_Gamma'] = G_Gamma_new

    return {
        'accept_rate': n_accept / n_proposals if n_proposals > 0 else float('nan'),
        'log_score':   float(scores_curr.sum()),
        'n_active':    int(G_C.sum()),
        'n_proposed':  n_proposals,
        'n_accepted':  n_accept,
    }