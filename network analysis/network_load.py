from __future__ import annotations

from pathlib import Path
import tempfile
import shutil

import numpy as np

from network_load_old import load_chain_output_old, ChainOutput, COUNTRIES, SOLAR_EXCLUDED


# Sample files that carry a draw axis and therefore must be pooled.
# The draw axis is ALWAYS the last axis on disk.
_POOLED_FILES = [
    "G0_samples",
    "G_Phi_samples",
    "Phi_samples",
    "Sigma_u_samples",
    "G_Gamma_samples",
    "Gamma_samples",
]

# Files that are constant across chains: copied once, never pooled.
_CONSTANT_FILES = ["G0_expanded"]


def _pool_draws(arrays: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-chain arrays along the (last) draw axis.

    Each input has shape (..., N_KEEP); output has shape
    (..., n_chains * N_KEEP). Equivalent to stacking on a new chain axis
    and merging it with the draw axis, but done directly with concatenate
    since the draw axis is already last.
    """
    return np.concatenate(arrays, axis=-1)


def load_chain_output(
    sample_dir: str | Path,
    selected_lags: list[int],
    n_chains: int = 4,
    countries: list[str] = COUNTRIES,
    solar_excluded: list[str] = SOLAR_EXCLUDED,
    load_partial: bool = False,
    G_Gamma_admissibility_mode: str = "free",
) -> ChainOutput:
    """Load N chains, pool their draws, and return a single ChainOutput.

    Parameters
    ----------
    sample_dir : str or Path
        Parent directory containing chain_0/, chain_1/, ... sub-folders.
    n_chains : int
        Number of chains to pool.
    (all other parameters: identical to load_chain_output)

    Returns
    -------
    ChainOutput
        Same object as load_chain_output, but n_keep == n_chains * N_KEEP.
        Every downstream module consumes it without modification.

    Notes
    -----
    Pooling is only meaningful once the chains are known to agree
    (R-hat close to 1). This function does NOT check convergence — run
    the diagnostics notebook first.
    """
    sample_dir = Path(sample_dir)
    suffix = "_partial" if load_partial else ""

    chain_dirs = [sample_dir / f"chain_{c}" for c in range(n_chains)]
    for d in chain_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Chain directory not found: {d}")

    # Pool every draw-bearing file into a temp directory shaped exactly
    # like a single-chain output folder, then hand it to the original
    # loader so all its validation logic still runs.
    tmp_dir = Path(tempfile.mkdtemp(prefix="bgvar_pooled_"))
    try:
        for name in _POOLED_FILES:
            fname = f"{name}{suffix}.npy"
            paths = [d / fname for d in chain_dirs]

            # G_Gamma / Gamma are optional: if absent in chain_0, assume
            # absent everywhere and skip (matches load_chain_output).
            if not paths[0].exists():
                continue
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{fname} present in chain_0 but missing in: "
                    f"{[str(p.parent.name) for p in missing]}"
                )

            per_chain = [np.load(p) for p in paths]
            pooled = _pool_draws(per_chain)
            np.save(tmp_dir / fname, pooled)

        # Constant files: copy from chain_0 (identical across chains).
        for name in _CONSTANT_FILES:
            src = chain_dirs[0] / f"{name}.npy"
            if src.exists():
                shutil.copy(src, tmp_dir / f"{name}.npy")

        # Delegate to the original loader — every shape check, mask check
        # and admissibility check now runs against the pooled arrays.
        chain = load_chain_output_old(
            sample_dir                 = tmp_dir,
            selected_lags              = selected_lags,
            countries                  = countries,
            solar_excluded             = solar_excluded,
            load_partial               = load_partial,
            G_Gamma_admissibility_mode = G_Gamma_admissibility_mode,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return chain