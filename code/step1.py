import numpy as np
from math import lgamma

# =============================================
# STEP 1: SAMPLING THE CONTEMPORANEOUS GRAPH G0
# =============================================

# ---
# 1. BGe LOCAL SCORE 
#   Given that variable i has these specific parents in the graph,
#       how well does this parent set explain the data?
#      The output is a single number (log-score): higher = better fit.

#   Args:
#      Y_i      : (T,)        — response variable for node i
#      Pa_i     : (T, k) or None — parent columns for node i
#      alpha_BGe: float       — BGe prior d.o.f. (ny + 2 by default)
#
#    Returns:
#      float — log BGe score for node i
# ---

def log_BGe (Y_i : np.ndarray,
             Pa_i : np.ndarray | None,
             alpha_BGe : float) -> float:
    
    # Local data block D = [Y_i | Pa_i],  shape (T, nd)
    D = (Y_i.reshape(-1,1) 
         if (Pa_i is None or Pa_i.size == 0) 
         else np.column_stack([Y_i, Pa_i]) 
    )
    
    T_obs, nd = D.shape
    S0_D = np.eye(nd)       # diffues identity prior scale
    S_post_D = S0_D + D.T @ D      # updated posterior scale (conjugate)
    
    # Multivariate log-gamma function
    def log_mgamma (a : float, d : int) -> float:
        return (d * (d-1) / 4) * np.log(np.pi) + sum(
            lgamma(a - i / 2 ) for i in range (d)
        )
        
    return float (
        -(nd * T_obs / 2) * np.log(np.pi)
        + log_mgamma((alpha_BGe + T_obs) / 2, nd)
        - log_mgamma(alpha_BGe / 2, nd)
        + (alpha_BGe / 2)           * np.linalg.slogdet(S0_D)[1]
        - ((alpha_BGe + T_obs) / 2) * np.linalg.slogdet(S_post_D)[1]
    )
    
    
# ---
# 2. ACYCLICITY CHECK
#    Returns True if G is a DAG, False if any directed cycle exists.
#---

def is_DAG (G: np.ndarray) -> bool:
    ny = G.shape[0]
    R = G.astype(float)
    for _ in range (ny -1):
        R = np.clip(R @ G + R, 0, 1)
    return bool(np.all(np.diag(R)==0))


# ---
# 3. CACHE: compute all per-variable BGe scores for a given graph
# ---

def all_node_scores (G: np.ndarray, Y: np.ndarray, alpha_BGe: float) -> np.ndarray:
    ny = Y.shape[1]
    scores = np.zeros(ny)
    for i in range(ny):
        parents    = np.where(G[i, :] == 1)[0]
        Pa_i       = Y[:, parents] if len(parents) > 0 else None
        scores[i]  = log_BGe(Y[:, i], Pa_i, alpha_BGe)
    return scores

# --- 
# 4. MAIN FUNCTION
# ---
def step1_sample_G0(state:     dict,
                    rng:       np.random.Generator,
                    alpha_BGe: float = None) -> dict:
 
    Y            = state['Y']               # (T, ny) — endogenous data
    G_curr       = state['G0'].copy()       # current graph (copy to avoid aliasing)
    G0_expanded  = state['G0_expanded']     # fixed structural prior (ny x ny)
    pi           = state['pi_bernoulli']    # Bernoulli prior prob per allowed arc
 
    ny = Y.shape[1]
 
    if alpha_BGe is None:
        alpha_BGe = float(ny + 2)   
 
    # Cache per-variable BGe scores for current graph
    scores_curr = all_node_scores(G_curr, Y, alpha_BGe)
 
    n_accept    = 0
    n_proposals = 0
 
    # Random permutation avoids systematic update bias across variables.
    for i in rng.permutation(ny):
 
        # topology constraint
        active_allowed = np.where(
            (G_curr[i, :] == 1) & (G0_expanded[i, :] == 1)
        )[0]
 
        if len(active_allowed) == 0:
            continue
        
        # Proposal 
        j  = int(rng.choice(active_allowed))
        G_prop  = G_curr.copy()
        G_prop[i, j]  = 0          # turn off arc j -> i
 
        n_proposals  += 1
 
        # Acyclicity safeguard
        if not is_DAG(G_prop):
            continue
 
        # Bayes Factor
        parents_prop = np.where(G_prop[i, :] == 1)[0]
        Pa_prop      = Y[:, parents_prop] if len(parents_prop) > 0 else None
        new_score_i  = log_BGe(Y[:, i], Pa_prop, alpha_BGe)
 
        # log BF = likelihood ratio + prior ratio
        log_BF = (new_score_i - scores_curr[i]) + np.log(1 - pi) - np.log(pi)
 
        # acceot / reject
        if np.log(rng.uniform()) < log_BF:
            G_curr         = G_prop
            scores_curr[i] = new_score_i
            n_accept      += 1
 
    state['G0'] = G_curr
 
    return {
        'accept_rate': n_accept / n_proposals if n_proposals > 0 else float('nan'),
        'log_score':   float(scores_curr.sum()),
    }