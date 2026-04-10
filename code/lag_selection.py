import numpy as np
from numpy.linalg import lstsq

Y = np.load('data/Y.npy')
X = np.load('data/X.npy')

# ---------------------------------
# LAG SELECTION
# ---------------------------------

def build_regressors(Y, X, p, q):
    """
    Build the regressor matrix for a reduced-form VAR (p, q) model.
    
    Returns:
    Y_dep : dependent variable matrix
    Z : regressor matrix with all lagged variables
    """
    T = Y.shape[0]
    
    # ensure you have existing observations considering the lag
    start = max(p, q)
    T_eff = T -start
    
    Y_dep = Y[start:, :]
    endo_lags = []
    for i in range(1, p + 1):
        Y_lagged = Y[start - i : T - i, : ]
        endo_lags.append(Y_lagged)
        
    exo_lags = []
    for j in range(1, q + 1):
        X_lagged = X[start - j : T -j, :]
        exo_lags.append(X_lagged)
        
    all_lags = endo_lags + exo_lags
    Z = np.hstack(all_lags)
    
    return Y_dep, Z


def compute_residuals (Y_dep, Z):
    """
    Estimate the reduced-for VAR by OLS equation by equation
    
    Returns:
    U : matrix of OLS residuals (one col per endo var)
    Beta: Matrix of OLS coeff estimates
    """
    ny = Y_dep.shape[1]
    n_reg = Z.shape[1]
    
    Beta = np.zeros((n_reg, ny))
    U = np.zeros_like(Y_dep)
    
    for i in range(ny):
        yi = Y_dep[:, i]
        beta_i, _, _, _ = lstsq(Z, yi, rcond=None)
        Beta[:, i] = beta_i
        U[:, i] = yi - Z @ beta_i
    
    return U, Beta


def comput_BIC (U, p, q, ny, nz):
    """
    Compute the BIC:
    BIC = -2 * log_likelihood + n_params * log(T_eff)
    
    with: log L ≈ -(T_eff/2) * log(det(Sigma_u))
    
    Returns BIC value
    """
    T_eff = U.shape[0]
    
    # Residual covariance matrix
    Sigma_u = (U.T @ U) / T_eff
    
    # log likelihood 
    sign, log_det = np.linalg.slogdet(Sigma_u)
    if sign <= 0: 
        return np.inf
    
    log_like = -0.5 * T_eff * log_det
    
    # number of free parameters
    n_params = ny * (p * ny + q * nz)
    
    # BIC
    BIC = -2 * log_like + n_params * np.log(T_eff)
    
    return BIC

# --- IMPLEMENTATION ---

ny = Y.shape[1]
nz = X.shape[1]
results = {}
p_max = 5
q_max = 5

for p in range (1, p_max + 1):
    for q in range(0, q_max + 1):
        
        Y_dep, Z = build_regressors(Y, X, p, q)
        
        U, Beta = compute_residuals(Y_dep, Z)
        
        BIC = comput_BIC(U, p, q, ny, nz)
        
        results[(p, q)] = BIC
        print(f"p={p}, q={q} → BIC={BIC:.2f}")
        
p_star, q_star = min(results, key=results.get)
print(f"\nOptimal lag order: p*={p_star}, q*={q_star}")