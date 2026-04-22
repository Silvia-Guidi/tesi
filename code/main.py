import numpy as np
from pathlib import Path
import time

from step0 import initialize_model
from step1 import step1_sample_G0
from step2 import step2_sample
from step3 import step3_sample
from step4 import step4_sample_Phi

# --------------------------------------------------------
# SETTINGS
# --------------------------------------------------------
DATA_DIR    = Path("data")
NETWORK_DIR = Path("data/network_data")
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gibbs sampler settings
N_ITER           = 5_000        # total iterations
BURNIN           = 2_000        # burn-in iterations to discard
N_KEEP           = N_ITER - BURNIN
CHECKPOINT_EVERY = 1_000        # flush partial samples to disk every N iterations

SEED = 42

# Model settings
SELECTED_LAGS = [1, 2, 3, 7]

HPARAMS = {
    # Minnesota prior
    'pi_shrink':   0.5,
    'lag_decay':   1.0,
    'exog_weight': 0.5,

    # Inverse-Wishart prior for Sigma_u
    'alpha_offset': 2,          # alpha = ny + alpha_offset
    'S0_scale':     1.0,

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
# DATA LOADING
# --------------------------------------------------------
def load_data() -> np.ndarray:
    path  = DATA_DIR / "Y.npy"
    y_raw = np.load(path)
    print(f"[Data] Loaded Y: {y_raw.shape[0]} observations, {y_raw.shape[1]} variables")
    return y_raw.astype(np.float64)


# --------------------------------------------------------
# STORAGE
# --------------------------------------------------------
def allocate_storage(ny: int, n_lags: int, T_eff: int) -> dict:
    """
    Pre-allocate arrays for all post-burn-in samples.

    Memory layout choices:
      - graphs stored as uint8 (0/1 values): 8x less RAM than default int64
      - Sigma_u stored as float32: 2x less RAM, precision still sufficient
        for posterior summaries (mean, logdet, eigenvalues)
      - G_Phi is 4D: (ny, ny, n_lags, N_KEEP) to match how step2 writes it
    """
    return {
        # Contemporaneous graph
        'G0':           np.zeros((ny, ny, N_KEEP),         dtype=np.uint8),

        # Temporal graph for all lags (4D)
        'G_Phi':        np.zeros((ny, ny, n_lags, N_KEEP), dtype=np.uint8),

        # Reduced-form residual covariance
        'Sigma_u':      np.zeros((ny, ny, N_KEEP),         dtype=np.float32),
        'logdet_Sigma': np.zeros(N_KEEP,                   dtype=np.float32),

        # Phi
        'Phi': np.zeros((ny, ny, n_lags, N_KEEP), dtype=np.float32),

        # Step 6 - to be added
        # 'h':        np.zeros((T_eff, N_KEEP), dtype=np.float32),
        # 'lambda_t': np.zeros((T_eff, N_KEEP), dtype=np.float32),
    }


def save_samples(samples: dict, out_dir: Path, tag: str = "") -> None:
    """Flush current sample arrays to disk. tag='' for final, 'partial' for checkpoints."""
    suffix = f"_{tag}" if tag else ""
    for name, arr in samples.items():
        np.save(out_dir / f"{name}_samples{suffix}.npy", arr)


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    print("=" * 60)
    print("BGVAR MODEL — GIBBS SAMPLER")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    # --- Data and initial state ---
    y_raw = load_data()
    state = initialize_model(
        y_raw         = y_raw,
        selected_lags = SELECTED_LAGS,
        hparams       = HPARAMS,
    )

    # Make hparams accessible to step3 (it reads state['hparams'])
    state['hparams'] = HPARAMS

    ny     = state['ny']
    T      = state['T']
    n_lags = state['n_lags']

    print(f"[Init] ny={ny} variables, T={T} effective observations, n_lags={n_lags}")
    print(f"[Init] G0_expanded active arcs: {int(state['G0_expanded'].sum())}")
    print()

    # Save the physical-network mask once (it never changes)
    np.save(OUTPUT_DIR / "G0_expanded.npy", state['G0_expanded'].astype(np.uint8))

    # --- Allocate sample storage ---
    samples = allocate_storage(ny, n_lags, T)

    # --- Gibbs loop ---
    print(f"Running {N_ITER} iterations ({BURNIN} burn-in + {N_KEEP} kept)...\n")
    t_start = time.perf_counter()

    for t in range(N_ITER):

        # STEP 1: contemporaneous graph G0
        diag1 = step1_sample_G0(state, rng)

        # STEP 2: temporal graph G_Phi
        diag2 = step2_sample(state, rng)

        # STEP 3: reduced-form residual covariance Sigma_u
        diag3 = step3_sample(state, rng)

        # STEP 4: sample Phi 
        diag4 = step4_sample_Phi(state, rng)

        # STEP 5: sample Gamma - TO BE ADDED
        # step5_sample_Gamma(state, rng)

        # STEP 6: sample h_t, lambda_t - TO BE ADDED
        # step6_sample_SV(state, rng)

        # --- Store post-burn-in samples ---
        if t >= BURNIN:
            k = t - BURNIN
            samples['G0'][:, :, k]           = state['G0']
            # Vectorised assignment: stack the list into a single (ny, ny, n_lags) array
            samples['G_Phi'][:, :, :, k]     = np.stack(state['G_Phi'], axis=-1)
            samples['Sigma_u'][:, :, k]      = state['Sigma_u']
            samples['logdet_Sigma'][k]       = diag3['logdet_Sigma']
            

        # --- Progress report every 500 iterations ---
        if (t + 1) % 500 == 0:
            phase   = "burn-in " if t < BURNIN else "sampling"
            elapsed = time.perf_counter() - t_start
            print(
                f"  Iter {t+1:>5}/{N_ITER}  [{phase}]  "
                f"S1 acc={diag1['accept_rate']:.2%}  "
                f"S2 acc={diag2['accept_rate']:.2%}  "
                f"n_active={diag2['n_active']:>4}  "
                f"logdet(Σ)={diag3['logdet_Sigma']:+.2f}  "
                f"S4 |Φ|_F={diag4['phi_norm']:.3f}  "
                f"S4 max|Φ|={diag4['phi_max_abs']:.3f}"
            )

        # --- Periodic checkpoint (only after burn-in, while we have real samples) ---
        if (t >= BURNIN) and ((t + 1) % CHECKPOINT_EVERY == 0):
            save_samples(samples, OUTPUT_DIR, tag="partial")

    # --- Final save ---
    save_samples(samples, OUTPUT_DIR)
    total = time.perf_counter() - t_start
    print(f"\nGibbs sampler complete in {total:.1f}s "
          f"({total / N_ITER * 1000:.1f} ms/iter).")
    print(f"[Output] Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
