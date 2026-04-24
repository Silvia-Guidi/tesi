from __future__ import annotations
import numpy as np
from math import lgamma
from numpy.random import Generator
from step1 import log_marginal_block

# =============================================
# STEP 2: SAMPLING THE TEMPORAL GRAPH STRUCTURES G_Phi (and G_Gamma)
# =============================================

# ---
# Local BGe score (ratio form) for step 2.
#
# Same concept as step1.local_score, but the column indexing is different:
# the response has global index i (in the Y block), while parents have
# indices ny + j (in the X_reg block).
# ---
def local_score_step2(response: int,
                      parents: np.ndarray,
                      ny: int,
                      S_post_full: np.ndarray,
                      alpha_half: float,
                      alpha_post_half: float,
                      bge_const: np.ndarray,
                      c_0: float) -> float:
    """
    log P(X_{response, parents}) - log P(X_{parents})

    where `response` refers to column `i` of Y and `parents` refer to
    columns `ny + j` of the stacked [Y | X_reg] design.
    """
    # Numerator
    if parents.size > 0:
        num_cols = np.concatenate(([response], ny + parents))
    else:
        num_cols = np.array([response])
    S_num   = S_post_full[np.ix_(num_cols, num_cols)]
    nd_num  = S_num.shape[0]
    log_num = log_marginal_block(S_num, alpha_half, alpha_post_half,
                                 bge_const[nd_num], c_0)

    # Denominator
    if parents.size == 0:
        return log_num
    den_cols = ny + parents
    S_den    = S_post_full[np.ix_(den_cols, den_cols)]
    nd_den   = S_den.shape[0]
    log_den  = log_marginal_block(S_den, alpha_half, alpha_post_half,
                                  bge_const[nd_den], c_0)

    return log_num - log_den


def all_eq_scores(G_C: np.ndarray,
                  S_post_full: np.ndarray,
                  ny: int,
                  alpha_half: float,
                  alpha_post_half: float,
                  bge_const: np.ndarray,
                  c_0: float) -> np.ndarray:
    scores = np.zeros(ny)
    for i in range(ny):
        parents = np.where(G_C[i, :] == 1)[0]
        scores[i] = local_score_step2(i, parents, ny, S_post_full,
                                      alpha_half, alpha_post_half,
                                      bge_const, c_0)
    return scores


# ---
# (Un)stacking design matrix and graph
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
# MAIN FUNCTION
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

    # --- Build stacked design matrix X_reg and unified graph G_C ---
    X_reg = stack_design(state['X_endo'], state.get('X_exo'), ny, nz, n_lags)
    G_C   = stack_GC(state['G_Phi'], state.get('G_Gamma'), ny, nz, n_lags)

    npar = G_C.shape[1]   # = (ny + nz) * n_lags

    if alpha_BGe is None:
        # For step 2 the max block size is max_nd = 1 + npar. BGe requires
        # alpha_BGe > max_nd + 1 for the prior to be proper. Use T/4 lower
        # bound for scaling consistency with step1.
        alpha_BGe = float(max(1 + npar + 2, T_obs // 4))

    dim_full = ny + npar

    # Prior scale c_0 must keep the prior proper for blocks up to dim_full.
    c_0 = alpha_BGe - dim_full - 1
    if c_0 <= 0:
        c_0 = 1.0

    # --- Pre-compute BGe constants that depend only on nd ---
    log_pi          = np.log(np.pi)
    alpha_half      = alpha_BGe / 2.0
    alpha_post_half = (alpha_BGe + T_obs) / 2.0

    def _log_mgamma(a: float, d: int) -> float:
        return (d * (d - 1) / 4.0) * log_pi + sum(
            lgamma(a - k / 2.0) for k in range(d)
        )

    max_nd = 1 + npar
    bge_const = np.empty(max_nd + 1)
    bge_const[0] = 0.0
    for nd in range(1, max_nd + 1):
        bge_const[nd] = (
            -(nd * T_obs / 2.0) * log_pi
            + _log_mgamma(alpha_post_half, nd)
            - _log_mgamma(alpha_half, nd)
        )

    # --- Pre-compute Gram matrix once for fast BGe scoring ---
    full_design = np.hstack([Y, X_reg])                             # (T, dim_full)
    S_post_full = c_0 * np.eye(dim_full) + full_design.T @ full_design

    # Cache per-equation local scores for the current graph
    scores_curr = all_eq_scores(G_C, S_post_full, ny,
                                alpha_half, alpha_post_half,
                                bge_const, c_0)

    n_accept    = 0
    n_proposals = 0

    for i in rng.permutation(ny):

        # --- Propose a single edge toggle on equation i ---
        # Unlike G0, the temporal graph G_C has no acyclicity constraint.
        j = int(rng.integers(0, npar))

        old_ij       = G_C[i, j]
        G_C[i, j]    = 1 - old_ij
        n_proposals += 1

        # --- Local Bayes factor on equation i (ratio form) ---
        parents_prop = np.where(G_C[i, :] == 1)[0]
        new_score_i = local_score_step2(i, parents_prop, ny, S_post_full,
                                        alpha_half, alpha_post_half,
                                        bge_const, c_0)

        # Prior ratio
        delta           = G_C[i, j] - old_ij            # +1 add, -1 remove
        log_prior_ratio = delta * (np.log(pi) - np.log(1 - pi))

        log_BF = (new_score_i - scores_curr[i]) + log_prior_ratio

        # --- Accept / reject ---
        if np.log(rng.uniform()) < log_BF:
            scores_curr[i] = new_score_i
            n_accept      += 1
        else:
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