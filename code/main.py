import numpy as np
from pathlib import Path
 
from step0 import initialize_model
from step1 import step1_sample_G0

# --------------------------------------------------------
# SETTINGS
# --------------------------------------------------------
DATA_DIR    = Path("data") 
NETWORK_DIR = Path("data/network_data")
OUTPUT_DIR  = Path("outputs")
DIAGNOSTICS_DIR = Path("notebooks")
OUTPUT_DIR.mkdir(exist_ok=True)
 
# Gibbs sampler settings
N_ITER  = 5_000    # total iterations
BURNIN  = 2_000    # burn-in iterations to discard
N_KEEP  = N_ITER - BURNIN
 
SEED    = 42
 
# Model settings
SELECTED_LAGS = [1, 2, 3, 7]     
 
HPARAMS = {
    # Minnesota prior
    'pi_shrink':    0.5,
    'lag_decay':    1.0,
    'exog_weight':  0.5,
 
    # Inverse-Wishart prior for Sigma_u
    'alpha_offset': 2,      # alpha = ny + alpha_offset
 
    # Bernoulli prior for graph edges
    'pi_bernoulli': 0.5,
 
    # Stochastic volatility prior
    'stochastic_volatility': {
        'phi_a':   20,
        'phi_b':   1.5,
        'mu_0':    0.0,
        'mu_var':  10.0,
        'sigma_v': 0.01,
        'sigma_s': 0.01,
    },
 
    # Degrees-of-freedom prior for lambda_t
    'degrees_of_freedom': {
        'min_nu':     2,
        'max_nu':     40,
        'initial_nu': 10,
    },
}

# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------
 
def load_data() -> np.ndarray:
    
    path  = DATA_DIR / "Y.npy"
    y_raw = np.load(path)
    print(f"[Data] Loaded Y: {y_raw.shape[0]} observations, {y_raw.shape[1]} variables")
    return y_raw.astype(float)


# --------------------------------------------------------
# STORAGE
# --------------------------------------------------------

def allocate_storage(ny: int) -> dict:
    """
    Pre-allocate arrays for all quantities to store after burn-in.
    """
    return {
        # Step 1
        'G0':           np.zeros((ny, ny, N_KEEP), dtype=int),
 
        # Step 2 — to be added
        # 'G_Phi':      np.zeros((ny, ny, N_KEEP), dtype=int),
 
        # Step 3 — to be added
        # 'Sigma_u':    np.zeros((ny, ny, N_KEEP)),
 
        # Step 4 — to be added
        # 'Phi':        np.zeros((ny, ny, N_KEEP)),
 
        # Step 6 — to be added
        # 'h':          np.zeros((T, N_KEEP)),
        # 'lambda_t':   np.zeros((T, N_KEEP)),
    }
 
 
# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main ():
    print("=" * 60)
    print("BGVAR MODEL — GIBBS SAMPLER")
    print("=" * 60)
 
    rng = np.random.default_rng(SEED)
 
    # Load data 
    y_raw = load_data()
 
    # Init model
    state = initialize_model(
        y_raw         = y_raw,
        selected_lags = SELECTED_LAGS,
        hparams       = HPARAMS,
    )
 
    ny = state['ny']
    T  = state['T']
    print(f"[Init] ny={ny} variables, T={T} effective observations")
    print(f"[Init] G0_expanded active arcs: {state['G0_expanded'].sum()}")
    print()
 
    # Pre-allocate storage 
    samples = allocate_storage(ny)
    
    diagnostics = {
        'step1_accept_rate': np.full(N_ITER, np.nan),
        'step1_log_score':   np.full(N_ITER, np.nan)
    }
 
    #  GIBBS LOOP
    print(f"Running {N_ITER} iterations ({BURNIN} burn-in + {N_KEEP} kept)...")
 
    for t in range(N_ITER):
 
        #  STEP 1: sample G0 (contemporaneous graph) 
        diag1 = step1_sample_G0(state, rng)
        diagnostics['step1_accept_rate'][t] = diag1['accept_rate']
        diagnostics['step1_log_score'][t]   = diag1['log_score']
 
        # ── STEP 2: sample G_Phi (lagged graph) ── TO BE ADDED
        # diag2 = step2_sample_GPhi(state, rng)
 
        # ── STEP 3: sample Sigma_u ── TO BE ADDED
        # step3_sample_Sigma(state, rng)
 
        # ── STEP 4: sample Phi ── TO BE ADDED
        # step4_sample_Phi(state, rng)
 
        # ── STEP 5: sample Gamma ── TO BE ADDED
        # step5_sample_Gamma(state, rng)
 
        # ── STEP 6: sample h_t, lambda_t ── TO BE ADDED
        # step6_sample_SV(state, rng)
 
        # STORE POST-BURNIN SAMPLES 
        if t >= BURNIN:
            k = t - BURNIN
            samples['G0'][:, :, k] = state['G0']
            # samples['Sigma_u'][:, :, k] = state['Sigma_u']   # uncomment when ready
            # samples['Phi'][:, :, k]     = state['Phi']        # uncomment when ready
 
        #  PROGRESS REPORT every 500 iterations 
        if (t + 1) % 500 == 0:
            phase = "burn-in" if t < BURNIN else "sampling"
            print(
                f"  Iter {t+1:>5}/{N_ITER}  [{phase}]  "
                f"Step1 accept: {diag1['accept_rate']:.2f}  "
                f"log-score: {diag1['log_score']:.1f}"
            )
 
    print("\nGibbs sampler complete.")
 
 
    # ── SAVE OUTPUTS ──────────────────────────────────────────────────────────
    np.save(OUTPUT_DIR / "G0_samples.npy",  samples['G0'])
    np.save(DIAGNOSTICS_DIR / "diagnostics.npy", diagnostics)
 
 
    print(f"\n[Output] Saved to {OUTPUT_DIR}/")
 
 
if __name__ == "__main__":
    main()