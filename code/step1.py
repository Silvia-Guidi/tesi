import numpy as np
from math import lgamma

# =============================================
# STEP 1: SAMPLING THE CONTEMPORANEOUS GRAPH G0
# =============================================

# ---
# 1. BGe LOCAL SCORE
# ---
def log_BGe(S_post_D: np.ndarray,
            alpha_post_half: float,
            const_term: float) -> float:
    _, logdet_post = np.linalg.slogdet(S_post_D)
    return const_term - alpha_post_half * logdet_post


# ---
# 2. ACYCLICITY CHECK
#    Returns True if G is a DAG, False if any directed cycle exists.
#    Uses boolean reachability closure with early termination.
# ---
def is_DAG(G: np.ndarray) -> bool:
    ny = G.shape[0]

    # Self-loop -> immediate cycle
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
# 3. CACHE: compute all per-variable BGe scores for a given graph
#    Uses the pre-computed Gram matrix S_post_full and the tabulated
#    BGe constants bge_const[nd].
# ---
def all_node_scores(G: np.ndarray,
                    S_post_full: np.ndarray,
                    alpha_post_half: float,
                    bge_const: np.ndarray) -> np.ndarray:
    ny = G.shape[0]
    scores = np.zeros(ny)
    for i in range(ny):
        parents = np.where(G[i, :] == 1)[0]
        cols = (
            np.concatenate(([i], parents))
            if parents.size > 0
            else np.array([i])
        )
        S_block = S_post_full[np.ix_(cols, cols)]
        nd = S_block.shape[0]
        scores[i] = log_BGe(S_block, alpha_post_half, bge_const[nd])
    return scores


# ---
# 4. MAIN FUNCTION
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
        alpha_BGe = float(ny + 2)

    # --- Pre-compute BGe constants that depend only on nd ---
    # nd ranges from 1 (response only) to ny (response + all others as parents).
    log_pi          = np.log(np.pi)
    alpha_half      = alpha_BGe / 2.0
    alpha_post_half = (alpha_BGe + T_obs) / 2.0

    def _log_mgamma(a: float, d: int) -> float:
        return (d * (d - 1) / 4.0) * log_pi + sum(
            lgamma(a - k / 2.0) for k in range(d)
        )

    bge_const = np.empty(ny + 1)
    bge_const[0] = 0.0   # unused placeholder (nd >= 1 always)
    for nd in range(1, ny + 1):
        bge_const[nd] = (
            -(nd * T_obs / 2.0) * log_pi
            + _log_mgamma(alpha_post_half, nd)
            - _log_mgamma(alpha_half, nd)
        )

    # --- Pre-compute Gram matrix once for fast BGe scoring ---
    # Any sub-block S = I + D.T @ D for a candidate parent set is obtained
    # by plain index slicing into S_post_full, avoiding T-sized matmuls.
    S_post_full = np.eye(ny) + Y.T @ Y

    # Cache per-variable BGe scores for the current graph
    scores_curr = all_node_scores(G_curr, S_post_full, alpha_post_half, bge_const)

    n_accept    = 0
    n_proposals = 0

    # Random permutation avoids systematic update bias across variables
    for i in rng.permutation(ny):

        # --- Physical admissibility mask ---
        # Only arcs allowed by the physical network topology are proposable.
        # Arcs outside the mask stay zero for the whole chain by construction.
        allowed_j = np.where(G0_expanded[i, :] == 1)[0]
        if allowed_j.size == 0:
            continue

        j = int(rng.choice(allowed_j))

        # --- Flip in place, revert on reject ---
        old_ij = G_curr[i, j]
        old_ji = G_curr[j, i]

        # 1) kill the reverse edge (paper, Algo 1, step 6) to avoid 2-cycles
        # 2) toggle the candidate arc (add if absent, remove if present)
        if old_ji == 1:
            G_curr[j, i] = 0
        G_curr[i, j] = 1 - old_ij

        n_proposals += 1

        # Acyclicity safeguard for longer cycles i -> ... -> j -> i
        if not is_DAG(G_curr):
            G_curr[i, j] = old_ij
            G_curr[j, i] = old_ji
            continue

        # --- Local Bayes factor on equation i (fast path) ---
        parents_prop = np.where(G_curr[i, :] == 1)[0]
        cols = (
            np.concatenate(([i], parents_prop))
            if parents_prop.size > 0
            else np.array([i])
        )
        S_block     = S_post_full[np.ix_(cols, cols)]
        nd          = S_block.shape[0]
        new_score_i = log_BGe(S_block, alpha_post_half, bge_const[nd])

        # Prior ratio depends on direction of the move
        delta           = G_curr[i, j] - old_ij         # +1 add, -1 remove
        log_prior_ratio = delta * (np.log(pi) - np.log(1 - pi))

        log_BF = (new_score_i - scores_curr[i]) + log_prior_ratio

        # --- Accept / reject ---
        if np.log(rng.uniform()) < log_BF:
            scores_curr[i] = new_score_i
            n_accept      += 1
        else:
            # revert to the original state
            G_curr[i, j] = old_ij
            G_curr[j, i] = old_ji

    state['G0'] = G_curr

    return {
        'accept_rate': n_accept / n_proposals if n_proposals > 0 else float('nan'),
        'log_score':   float(scores_curr.sum()),
    }