import numpy as np
from math import lgamma

# =============================================
# STEP 1: SAMPLING THE CONTEMPORANEOUS GRAPH G0
# =============================================

# ---
# BGe LOCAL SCORE (ratio form)
#
# Ahelegbey-Billio-Casarin Eq A.3: the log marginal likelihood of a graph
# factorises as a product of ratios across equations:
#
#    P(X | G) = prod_i [ P(X_{Y_i, pi_i}) / P(X_{pi_i}) ]
#
# Therefore the local score for equation i with parent set pi_i is:
#
#    score(i | pi_i) = log P(X_{Y_i, pi_i}) - log P(X_{pi_i})
#
# Computing only the numerator (as an earlier version did) adds a spurious
# -T/2 * log(pi) term per included parent (~209 per edge for T=365), which
# made every edge rejected regardless of data correlation. The denominator
# cancels those terms and leaves a well-behaved log-Bayes factor.
#
# Marginal likelihood of a block D of size nd (Heckerman-Geiger 1995):
#
#    log P(X_D) = c(nd, alpha, T)
#               + (alpha/2)        * log|S_0^{(D)}|
#               - ((alpha+T)/2)    * log|S_post^{(D)}|
#
# with S_0 = c_0 * I (scaled identity prior, c_0 = alpha - ny - 1).
# ---
def log_marginal_block(S_post_D: np.ndarray,
                       alpha_half: float,
                       alpha_post_half: float,
                       const_term: float,
                       c_0: float) -> float:
    """log P(X_D) for a single block D."""
    nd = S_post_D.shape[0]
    _, logdet_post = np.linalg.slogdet(S_post_D)
    logdet_prior   = nd * np.log(c_0)
    return const_term + alpha_half * logdet_prior - alpha_post_half * logdet_post


def local_score(response: int,
                parents: np.ndarray,
                S_post_full: np.ndarray,
                alpha_half: float,
                alpha_post_half: float,
                bge_const: np.ndarray,
                c_0: float) -> float:
    """
    Local BGe score for equation `response` given `parents`:

        log P(X_{response, parents}) - log P(X_{parents})
    """
    # Numerator: marginal over {response} U parents
    if parents.size > 0:
        num_cols = np.concatenate(([response], parents))
    else:
        num_cols = np.array([response])
    S_num   = S_post_full[np.ix_(num_cols, num_cols)]
    nd_num  = S_num.shape[0]
    log_num = log_marginal_block(S_num, alpha_half, alpha_post_half,
                                 bge_const[nd_num], c_0)

    # Denominator: marginal over parents only (0 if no parents)
    if parents.size == 0:
        return log_num
    S_den   = S_post_full[np.ix_(parents, parents)]
    nd_den  = S_den.shape[0]
    log_den = log_marginal_block(S_den, alpha_half, alpha_post_half,
                                 bge_const[nd_den], c_0)

    return log_num - log_den


# ---
# ACYCLICITY CHECK
#    Returns True if G is a DAG, False if any directed cycle exists.
# ---
def is_DAG(G: np.ndarray) -> bool:
    ny = G.shape[0]

    if np.any(np.diag(G)):
        return False

    G_bool = G.astype(bool)
    R = G_bool.copy()
    for _ in range(ny - 1):
        R_new = R | (R @ G_bool)
        if np.any(np.diag(R_new)):
            return False
        if np.array_equal(R_new, R):
            break
        R = R_new
    return True


# ---
# CACHE: compute all per-equation local scores for a given graph
# ---
def all_node_scores(G: np.ndarray,
                    S_post_full: np.ndarray,
                    alpha_half: float,
                    alpha_post_half: float,
                    bge_const: np.ndarray,
                    c_0: float) -> np.ndarray:
    ny = G.shape[0]
    scores = np.zeros(ny)
    for i in range(ny):
        parents = np.where(G[i, :] == 1)[0]
        scores[i] = local_score(i, parents, S_post_full,
                                alpha_half, alpha_post_half,
                                bge_const, c_0)
    return scores


# ---
# MAIN FUNCTION
# ---
def step1_sample_G0(state: dict,
                    rng: np.random.Generator,
                    alpha_BGe: float = None) -> dict:

    Y           = state['Y']               # (T, ny) endogenous data
    G_curr      = state['G0'].copy()       # current graph (defensive copy)
    G0_expanded = state['G0_expanded']     # physical admissibility mask (ny x ny)
    pi          = state['pi_bernoulli']    # Bernoulli prior prob per allowed arc

    T_obs, ny = Y.shape

    if alpha_BGe is None:
        # BGe prior strength. max(ny+2, T/4) ensures alpha > ny+1 (so c_0 > 0)
        # and scales with T to avoid the posterior-too-concentrated regime
        # when T ~ ny.
        alpha_BGe = float(max(ny + 2, T_obs // 4))

    # Prior scale: S_0 = c_0 * I with c_0 = alpha - ny - 1.
    c_0 = alpha_BGe - ny - 1
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

    bge_const = np.empty(ny + 1)
    bge_const[0] = 0.0   # unused (nd >= 1)
    for nd in range(1, ny + 1):
        bge_const[nd] = (
            -(nd * T_obs / 2.0) * log_pi
            + _log_mgamma(alpha_post_half, nd)
            - _log_mgamma(alpha_half, nd)
        )

    # --- Pre-compute Gram matrix once for fast BGe scoring ---
    S_post_full = c_0 * np.eye(ny) + Y.T @ Y

    # Cache per-equation local scores for the current graph
    scores_curr = all_node_scores(G_curr, S_post_full,
                                  alpha_half, alpha_post_half,
                                  bge_const, c_0)

    n_accept    = 0
    n_proposals = 0

    for i in rng.permutation(ny):

        # --- Physical admissibility mask ---
        allowed_j = np.where(G0_expanded[i, :] == 1)[0]
        if allowed_j.size == 0:
            continue

        j = int(rng.choice(allowed_j))

        # --- Flip in place, revert on reject ---
        old_ij = G_curr[i, j]
        old_ji = G_curr[j, i]

        if old_ji == 1:
            G_curr[j, i] = 0
        G_curr[i, j] = 1 - old_ij

        n_proposals += 1

        if not is_DAG(G_curr):
            G_curr[i, j] = old_ij
            G_curr[j, i] = old_ji
            continue

        # --- Local Bayes factor on equation i (ratio form) ---
        parents_prop = np.where(G_curr[i, :] == 1)[0]
        new_score_i = local_score(i, parents_prop, S_post_full,
                                  alpha_half, alpha_post_half,
                                  bge_const, c_0)

        # Prior ratio depends on direction of the move
        delta           = G_curr[i, j] - old_ij         # +1 add, -1 remove
        log_prior_ratio = delta * (np.log(pi) - np.log(1 - pi))

        log_BF = (new_score_i - scores_curr[i]) + log_prior_ratio

        # --- Accept / reject ---
        if np.log(rng.uniform()) < log_BF:
            scores_curr[i] = new_score_i
            n_accept      += 1
        else:
            G_curr[i, j] = old_ij
            G_curr[j, i] = old_ji

    state['G0'] = G_curr

    return {
        'accept_rate': n_accept / n_proposals if n_proposals > 0 else float('nan'),
        'log_score':   float(scores_curr.sum()),
    }