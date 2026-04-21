from __future__ import annotations
import numpy as np
from scipy.stats import invwishart

from priors import inverse_wishart_prior

# ==============================================
# Step 3: sample Sigma_u from its full conditional.
# ==============================================

# FUNCTIONS

def compute_reduced_form_coeff (state : dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Map the structural coefficients (Phi, Gamma) to the reduced-form
    coefficients (A_endo, A_eso) using A_0 = I - (G_0 * Phi_0).
    """
    
    ny = state ['ny']
    
    # --- Endo block ---
    Phi_list = state.get('Phi', None)
    G_Phi = state.get ('G_Phi', None)
    
    if Phi_list is None or G_Phi is None:
        # Steps 4-5 not implemented yet -> zero coefficients
        p = state.get('n_lags', 0)
        A_endo = np.zeros ((ny, ny * p)) if p > 0 else np.zeros ((ny, 0))
    else:
        p = len (Phi_list)
        # A_0 = I - (G_0 * Phi_0); if Phi_0 is missing we treat it as zero
        Phi_0 = state.get('Phi_0', np.zeros ((ny, ny)))
        G_0   = state.get('G0', np.zeros((ny, ny), dtype=int))
        A0    = np.eye(ny) - (G_0 * Phi_0)
        A0_inv = np.linalg.solve(A0, np.eye(ny))
        
        blocks = []
        for i in range (p):
            B_i = G_Phi[i] * Phi_list[i]          # structural lag-i matrix
            blocks.append(A0_inv @ B_i)            # reduced-form lag-i matrix
        A_endo = np.hstack(blocks)
        
    # --- Exo block ---
    Gamma_list = state.get('Gamma', None)
    G_Gamma    = state.get('G_Gamma', None)
 
    if Gamma_list is None or G_Gamma is None:
        n_eso = state.get('n_eso', 0)
        q     = state.get('n_lags_eso', 0)
        A_eso = np.zeros((ny, n_eso * q)) if (n_eso > 0 and q > 0) else np.zeros((ny, 0))
    else:
        # Same construction as above, with A_0_inv already computed
        if 'A0_inv' not in locals():
            Phi_0  = state.get('Phi_0', np.zeros((ny, ny)))
            G_0    = state.get('G0', np.zeros((ny, ny), dtype=int))
            A0     = np.eye(ny) - (G_0 * Phi_0)
            A0_inv = np.linalg.solve(A0, np.eye(ny))
 
        blocks = []
        for j in range(len(Gamma_list)):
            C_j = G_Gamma[j] * Gamma_list[j]
            blocks.append(A0_inv @ C_j)
        A_eso = np.hstack(blocks)
 
    return A_endo, A_eso

def compute_residuals (state : dict)-> np.ndarray:
    """
    Compute the reduced-form residuals U = Y - X_endo A_endo' - X_eso A_eso'
    """
    Y       = state['Y']            # (T, ny)
    X_endo  = state.get('X_endo', None)
    X_eso   = state.get('X_eso', None)
 
    A_endo, A_eso = compute_reduced_form_coeff(state)
 
    U = Y.copy()
    if X_endo is not None and A_endo.size > 0:
        U = U - X_endo @ A_endo.T
    if X_eso is not None and A_eso.size > 0:
        U = U - X_eso @ A_eso.T
 
    return U

def standardise_residuals (U: np.ndarray, state: dict) -> np.ndarray:
    """
    Standardise residuals by the stochastic-volatility scaling factors:
        u_tilde_t = u_t / sqrt(exp(h_t) * lambda_t)
 
    If h_t / lambda_t are not present in the state (step 6 not yet
    implemented), no standardisation is performed.
    """
    T  = U.shape[0]
    h  = state.get('h', np.zeros(T))     # log-variance series
    lam = state.get('lambda_t', np.ones(T))     # mixing weights
 
    # Scale factor per observation
    scale = np.sqrt(np.exp(h) * lam)           
    return U / scale[:, None]
    

# MAIN SAMPLER

def step3_sample (state: dict, rng: np.random.Generator) -> dict:
    
    ny = state['ny']
    T = state['T']
    hparams = state['hparams']
    
    # --- Prior hyperparams ---
    S_0, alpha_0 = inverse_wishart_prior(ny, hparams)
    
    # --- Residuals ---
    U = compute_residuals(state)       
    U_tilde = standardise_residuals(U, state)
    
    # --- Posterior parameters ---
    S_post = S_0 + U_tilde.T @ U_tilde        
    alpha_post = alpha_0 + T
    
     # Symmetrise for numerical safety
    S_post = 0.5 * (S_post + S_post.T)
    
    # --- Sample from IW
    seed = int(rng.integers(0, 2**31 - 1))
    Sigma_u = invwishart.rvs(df=alpha_post, scale=S_post, random_state=seed)
    
    # Symmetrise the draw too 
    Sigma_u = 0.5 * (Sigma_u + Sigma_u.T)
    
    # ---- Write back into state -------------------------------------------
    state['Sigma_u'] = Sigma_u
 
    # ---- Diagnostics ------------------------------------------------------
    sign, logdet = np.linalg.slogdet(Sigma_u)
    diag = {
        'logdet_Sigma': float(logdet) if sign > 0 else np.nan,
        'trace_Sigma':  float(np.trace(Sigma_u)),
        'alpha_post':   int(alpha_post),
    }
    return diag