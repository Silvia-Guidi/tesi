import numpy as np
from pathlib import Path
import time
import pandas as pd

from step0 import initialize_model
from step1 import step1_sample_G0
from step2 import step2_sample
from step3 import step3_sample
from step4 import step4_sample_Phi
from step5 import step5_sample_Gamma
from step6 import step6_sample_SV

# --------------------------------------------------------
# SETTINGS
# --------------------------------------------------------
DATA_DIR    = Path("data")
NETWORK_DIR = Path("data/network_data")
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_CHAINS = 4          
BASE_SEED = 42

# Gibbs sampler settings
N_ITER           = 20000        # total iterations
BURNIN           = 15000        # burn-in iterations to discard
N_KEEP           = N_ITER - BURNIN
CHECKPOINT_EVERY = 2000        # flush partial samples to disk every N iterations

SEED = 42

# Model settings
SELECTED_LAGS = [1, 7]

HPARAMS = {
    # Minnesota prior
    'pi_shrink':   0.001,
    'lag_decay':   2.0,
    'exog_weight': 0.5,

    # Inverse-Wishart prior for Sigma_u
    'alpha_offset': 2,          # alpha = ny + alpha_offset
    'S0_scale':     1.0,

    # Bernoulli prior for graph edges
    'pi_bernoulli': 0.5,

    # Stochastic volatility prior
    'stochastic_volatility': {
        'shape': 10,        # nu_h prior shape
        'scale': 0.01,      # S_h   prior scale  
        'sv_burnin_adapt': BURNIN,
    },

    # Degrees-of-freedom prior for lambda_t
    'degrees_of_freedom': {
        'min_nu':     2,
        'max_nu':     40,
        'initial_nu': 10,
    },
}

DATA_MODE = "prices_only"            # endogenous: "prices_only" | "all"
EXO_SUFFIXES: list[str] = ["wind", "solar"]   # empty list = no exogenous

def load_data() -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    """
    Load Y.parquet and X.parquet. Column names are preserved by parquet,
    so we can filter exogenous/endogenous variables by substring match on
    the column names (e.g. 'price', 'solar', 'wind').
    """
    # --- Endogenous ---
    Y_df = pd.read_parquet(DATA_DIR / "Y.parquet")
    print(f"[Data] Loaded Y: {Y_df.shape[0]} obs, {Y_df.shape[1]} vars")

    if DATA_MODE == "prices_only":
        endo_cols = [c for c in Y_df.columns if "price" in c.lower()]
    elif DATA_MODE == "all":
        endo_cols = list(Y_df.columns)
    else:
        raise ValueError(f"Unknown DATA_MODE: {DATA_MODE!r}")

    Y_df = Y_df[endo_cols]
    print(f"[Data] Endo selection: {len(endo_cols)} variables")

    # --- Exogenous (optional) ---
    X_path = DATA_DIR / "X.parquet"
    if X_path.exists() and EXO_SUFFIXES:
        X_df = pd.read_parquet(X_path)
        print(f"[Data] Loaded X: {X_df.shape[0]} obs, {X_df.shape[1]} vars")

        if X_df.shape[0] != Y_df.shape[0]:
            raise ValueError(
                f"X has {X_df.shape[0]} rows but Y has {Y_df.shape[0]}."
            )

        # Pick columns whose name contains any of the requested suffixes,
        # preserving the order in EXO_SUFFIXES (all 'winds' first, then
        # 'solars', etc.) so downstream lag stacking sees a clean block layout.
        exo_cols = []
        for sfx in EXO_SUFFIXES:
            matches = [c for c in X_df.columns if sfx.lower() in c.lower()]
            if not matches:
                raise ValueError(
                    f"No X columns match suffix {sfx!r}. "
                    f"Available: {list(X_df.columns)}"
                )
            exo_cols.extend(matches)
            print(f"[Data] Exo suffix {sfx!r}: {len(matches)} columns")

        X_df = X_df[exo_cols]
        X_exo_raw = X_df.values.astype(np.float64)
        exo_names = list(X_df.columns)
    else:
        X_exo_raw = None
        exo_names = []

    return Y_df.values.astype(np.float64), X_exo_raw, list(Y_df.columns), exo_names

# --------------------------------
# STORAGE
# --------------------------------------------------------
def allocate_storage(ny: int, nz: int, n_lags: int, n_lags_exo: int, T_eff: int) -> dict:
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
        'G_Gamma':      np.zeros((ny, nz, n_lags, N_KEEP), dtype=np.uint8),

        # Reduced-form residual covariance
        'Sigma_u':      np.zeros((ny, ny, N_KEEP),         dtype=np.float32),
        'logdet_Sigma': np.zeros(N_KEEP,                   dtype=np.float32),

        # Phi
        'Phi': np.zeros((ny, ny, n_lags, N_KEEP), dtype=np.float32),
        'phi_norm':    np.zeros(N_KEEP),
        
        # Gamma
        'Gamma' :    np.zeros((ny, nz, n_lags_exo, N_KEEP)),
        'gamma_norm':  np.zeros(N_KEEP), 

        # Step 6: stochastic volatility + Student-t mixing
        'h':            np.zeros((T_eff, N_KEEP), dtype=np.float32),
        'lambda_t':     np.zeros((T_eff, N_KEEP), dtype=np.float32),
        'sigma_h2':     np.zeros(N_KEEP, dtype=np.float32),
    }


def save_samples(samples: dict, out_dir: Path, tag: str = "") -> None:
    """Flush current sample arrays to disk. tag='' for final, 'partial' for checkpoints."""
    suffix = f"_{tag}" if tag else ""
    for name, arr in samples.items():
        np.save(out_dir / f"{name}_samples{suffix}.npy", arr)


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def run_one_chain(chain_id: int, seed: int,
                  y_raw, X_exo_raw, endo_names, exo_names):
    """Run one independent Gibbs chain and save its samples to a
    dedicated sub-directory outputs/chain_<id>/."""

    print(f"\n{'='*60}\nCHAIN {chain_id}  (seed={seed})\n{'='*60}")

    rng = np.random.default_rng(seed)          # <-- per-chain seed

    # --- Initial state (re-initialised independently per chain) ---
    state = initialize_model(
        y_raw         = y_raw,
        X_exo_raw     = X_exo_raw,
        selected_lags = SELECTED_LAGS,
        hparams       = HPARAMS,
        rng           = rng,
    )
    state['endo_names'] = endo_names
    state['exo_names']  = exo_names
    state['hparams']    = HPARAMS

    ny, nz       = state['ny'], state['nz']
    T            = state['T']
    n_lags       = state['n_lags']
    n_lags_exo   = state['n_lags_exo']

    # --- Per-chain output directory ---
    chain_dir = OUTPUT_DIR / f"chain_{chain_id}"
    chain_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Init] ny={ny} variables, T={T} effective observations, n_lags={n_lags}")
    print(f"[Init] G0_expanded active arcs: {int(state['G0_expanded'].sum())}\n")

    np.save(chain_dir / "G0_expanded.npy", state['G0_expanded'].astype(np.uint8))

    samples = allocate_storage(ny, nz, n_lags, n_lags_exo, T)

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

        # STEP 5: sample Gamma 
        diag5 = step5_sample_Gamma(state, rng)

        # STEP 6: sample h_t, lambda_t 
        diag6 = step6_sample_SV(state, rng)

        # --- Store post-burn-in samples ---
        if t >= BURNIN:
            k = t - BURNIN
            samples['G0'][:, :, k]           = state['G0']
            # Vectorised assignment: stack the list into a single (ny, ny, n_lags) array
            samples['G_Phi'][:, :, :, k]     = np.stack(state['G_Phi'], axis=-1)
            if nz > 0:
                samples['G_Gamma'][:, :, :, k] = np.stack(state['G_Gamma'], axis=-1)
            samples['Sigma_u'][:, :, k]      = state['Sigma_u']
            samples['logdet_Sigma'][k]       = diag3['logdet_Sigma']
            samples['Phi'][:, :, :, k] = np.stack(state['Phi'], axis=-1)
            samples['phi_norm'][k]     = diag4['phi_norm']
            samples['Gamma'][:, :, :, k] = np.stack(state['Gamma'], axis=-1)
            samples['gamma_norm'][k]     = diag5['gamma_norm']
            samples['h'][:, k]               = state['h']
            #samples['lambda_t'][:, k]        = state['lambda_t']
            #samples['mu_h'][k]               = state['mu_h']
            #samples['phi_h'][k]              = state['phi_h']
            samples['sigma_h2'][k]           = state['sigma_h2']
            

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
                f"S4 max|Φ|={diag4['phi_max_abs']:.3f}  "
                f"⟨h⟩={diag6['h_mean']:+.2f}  "
                f"σ_h²={diag6['sigma_h2']:.4f}  "
                f"acc6={diag6['accept_rate']:.2%}  "
                f"τ={diag6['proposal_sd']:.3f}"
            )

        # --- Periodic checkpoint (only after burn-in, while we have real samples) ---
        if (t >= BURNIN) and ((t + 1) % CHECKPOINT_EVERY == 0):
            save_samples(samples, OUTPUT_DIR, tag="partial")

    save_samples(samples, chain_dir)
    total = time.perf_counter() - t_start
    print(f"\n[chain {chain_id}] complete in {total:.1f}s "
          f"({total / N_ITER * 1000:.1f} ms/iter). Saved to {chain_dir}/")

def main():
    print("=" * 60)
    print(f"BGVAR MODEL — GIBBS SAMPLER ({N_CHAINS} chains)")
    print("=" * 60)

    # Load the data ONCE — it is identical across chains, so there is
    # no reason to read the parquet files four times.
    y_raw, X_exo_raw, endo_names, exo_names = load_data()

    for c in range(N_CHAINS):
        run_one_chain(
            chain_id   = c,
            seed       = BASE_SEED + c,
            y_raw      = y_raw,
            X_exo_raw  = X_exo_raw,
            endo_names = endo_names,
            exo_names  = exo_names,
        )

    print(f"\nAll {N_CHAINS} chains complete. "
          f"Outputs in {OUTPUT_DIR}/chain_0 … chain_{N_CHAINS-1}/")


if __name__ == "__main__":
    main()