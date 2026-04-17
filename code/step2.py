from __future__ import annotations
import numpy as np
from numpy.random import Generator
from step1 import log_BGe

# =============================================
# STEP 2: SAMPLING THE TEMPORAL GRAPH STRUCTURE G_Phi and G_Gamma
# =============================================

# --- Per-equation BGe scores for current graph ---

def all_eq_scores(G_C:       np.ndarray,   # (ny, (ny+nz)*n_lags)
                  Y:         np.ndarray,   # (T, ny)
                  X_reg:     np.ndarray,   # (T, (ny+nz)*n_lags)
                  alpha_BGe: float) -> np.ndarray:
    ny = Y.shape[1]
    scores = np.zeros(ny)
    for i in range(ny):
        parents   = np.where(G_C[i, :] == 1)[0]
        Pa_i      = X_reg[:, parents] if parents.size > 0 else None
        scores[i] = log_BGe(Y[:, i], Pa_i, alpha_BGe)
    return scores

 
# ---
# 2. (un)stacking design matrix and graph
#
#    Column layout of the stacked objects (X_reg and G_C):
#        [lag1_endo | lag1_exo | lag2_endo | lag2_exo | ... ]

# ---

def stack_design(Xendo: np.ndarray,
                  Xexo:  np.ndarray | None,
                  ny: int, nz: int, n_lags: int) -> np.ndarray:
    if nz == 0 or Xexo is None:
        return Xendo
 
    T = Xendo.shape[0]
    n = ny + nz
    X_reg = np.zeros((T, n * n_lags))
    for s in range(n_lags):
        X_reg[:, s*n : s*n + ny]       = Xendo[:, s*ny : (s+1)*ny]
        X_reg[:, s*n + ny : (s+1)*n]   = Xexo[:,  s*nz : (s+1)*nz]
    return X_reg

def stack_GC(G_Phi:   list,                 # list of n_lags matrices (ny, ny)
              G_Gamma: list | None,          # list of n_lags matrices (ny, nz) or None
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



# --- MAIN FUNCTION ---
def step2_sample (state: dict,
                  rng: Generator,
                  alpha_BGe: float = None) -> dict:
    Y       = state['Y']
    ny      = state['ny']
    nz      = state.get('nz', 0)
    n_lags  = state['n_lags']
    pi      = state['pi_bernoulli']
    
    # Build the stacked design matrix  X_reg  and the unified graph  G_C.
    X_reg = stack_design(state['Xendo'], state.get('Xexo'), ny, nz, n_lags)
    G_C   = stack_GC(state['G_Phi'], state.get('G_Gamma'), ny, nz, n_lags)
 
    npar = G_C.shape[1]   # = (ny + nz) * n_lags
    
    if alpha_BGe is None:
        alpha_BGe = float(ny + 2)
        
    scores_curr = all_eq_scores(G_C, Y, X_reg, alpha_BGe)
    
    n_accept = 0
    n_proposals =0
    
    for i in rng.permutation(ny):
        # get the proposal
        j = int(rng.integers(0, npar))
        
        G_prop = G_C.copy()
        G_prop[i, j] = 1- G_prop[i, j]
        
        n_proposals += 1
        
        # score the proposal 
        parents_prop = np.where(G_prop[i, :]==1)[0]
        Pa_prop      = X_reg[:, parents_prop] if parents_prop.size > 0 else None
        new_score_i  = log_BGe(Y[:, i], Pa_prop, alpha_BGe)
        
         
        delta = G_prop[i, j] - G_C[i, j]   # +1 if add, -1 if remove
        log_prior_ratio = delta * (np.log(pi) - np.log(1 - pi))
 
        log_BF = (new_score_i - scores_curr[i]) + log_prior_ratio
 
        # accept / reject
        if np.log(rng.uniform()) < log_BF:
            G_C             = G_prop
            scores_curr[i]  = new_score_i
            n_accept       += 1
            
            
    G_Phi_new, G_Gamma_new = unstack_GC(G_C, ny, nz, n_lags)
    state['G_Phi']   = G_Phi_new
    if nz > 0:
        state['G_Gamma'] = G_Gamma_new
 
    return {
        'accept_rate': n_accept / n_proposals if n_proposals > 0 else float('nan'),
        'log_score':   float(scores_curr.sum()),
        'n_active':    int(G_C.sum()),
        'n_proposed':  n_proposals,
        'n_accepted':  n_accept,
    }
    
    