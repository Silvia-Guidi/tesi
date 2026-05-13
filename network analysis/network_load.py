"""
network_loader.py
=================

Foundation module for BGVAR network analysis.

Loads the MCMC chain output produced by the Gibbs sampler (main.py) and
prepares clean tensors ready for downstream analysis: edge probabilities,
centrality metrics, spillover analysis, etc.

Design principles
-----------------
- Single point of entry: ChainOutput dataclass holds everything downstream
  modules need (samples, labels, masks, metadata).
- Defensive checks: shapes, dtypes, mask consistency are validated up front
  so that bugs surface here, not three modules later in a centrality plot.
- Mask-aware: G0 has a physical admissibility mask (G0_expanded), and
  G_Gamma has a country-specific mask (each country sees only its own
  wind/solar). Densities and edge counts must use the correct denominator
  (admissible edges, not n*n).
- Country-only granularity: this loader assumes DATA_MODE='prices_only',
  so ny == n_countries.

Conventions
-----------
- Graph orientation: G[i, j, ...] == 1  means  "node j influences node i"
  (i is the response/equation index, j is the parent/predictor).
  In-degree of node i  = sum over j of G[i, j]   ("how many parents has i")
  Out-degree of node j = sum over i of G[i, j]   ("how many children has j")
- Endogenous nodes: 28 countries (COUNTRIES list).
- Exogenous predictors: wind (28) + solar (25, IE/LV/NO excluded) = 53 per lag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
# COUNTRY / EXOGENOUS DEFINITIONS
# ============================================================
# Kept here as constants so any downstream module gets the same ordering.
# IMPORTANT: this ordering MUST match the column ordering of Y.npy and of
# the exogenous design matrix X_exo used in the Gibbs sampler.

COUNTRIES = [
    "AT", "BE", "BG", "CH", "CZ", "DE", "DK", "EE", "ES", "FI",
    "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LV", "ME", "NL",
    "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK",
]

# Countries that do NOT have a solar exogenous variable.
SOLAR_EXCLUDED = ["IE", "LV", "NO"]


def build_exogenous_labels(
    countries: list[str] = COUNTRIES,
    solar_excluded: list[str] = SOLAR_EXCLUDED,
    admissibility_mode: str = "free",
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Build the list of exogenous predictor labels and the admissibility mask
    for G_Gamma.

    Layout assumption
    -----------------
    Within a single lag, exogenous columns are ordered as:
        [AT_wind, BE_wind, ..., SK_wind,  AT_solar, BE_solar, ..., SK_solar]
    i.e. ALL wind countries first, then ALL solar countries (with the
    solar block missing IE/LV/NO).

    Parameters
    ----------
    countries : list[str]
        Country codes in the same order as the columns of Y.
    solar_excluded : list[str]
        Countries that do NOT have a solar exogenous variable.
    admissibility_mode : str
        How to build the admissibility mask:
          - "free"             : all (country, exog) pairs are admissible.
                                 Use this when the sampler imposes no
                                 country-specific mask on G_Gamma
                                 (this is the case in the current BGVAR
                                 implementation: each country can a priori
                                 depend on any country's wind/solar).
          - "country_specific" : country i is allowed only to depend on
                                 its own wind/solar (53 admissible cells).
                                 Use this only if the sampler enforces
                                 the corresponding mask.

    Returns
    -------
    exog_labels : list[str]
        Length = nx_per_lag = len(countries) + len(countries) - len(solar_excluded).
        For 28 countries with 3 solar exclusions: 28 + 25 = 53.
    exog_country_idx : np.ndarray, shape (nx_per_lag,)
        For each exogenous column, the index (into `countries`) of the
        country it belongs to. Always built regardless of mode: it is used
        downstream to identify "own" vs "cross-country" exogenous edges,
        independently of the admissibility mask.
    admissibility : np.ndarray, shape (ny, nx_per_lag), dtype=bool
        Mask defining which cells the sampler is allowed to activate.
    """
    if admissibility_mode not in ("free", "country_specific"):
        raise ValueError(
            f"admissibility_mode must be 'free' or 'country_specific', "
            f"got {admissibility_mode!r}"
        )

    ny = len(countries)
    country_to_idx = {c: i for i, c in enumerate(countries)}

    exog_labels: list[str] = []
    exog_country_idx: list[int] = []

    # Wind block: all countries
    for c in countries:
        exog_labels.append(f"{c}_wind")
        exog_country_idx.append(country_to_idx[c])

    # Solar block: all countries EXCEPT those in solar_excluded
    for c in countries:
        if c in solar_excluded:
            continue
        exog_labels.append(f"{c}_solar")
        exog_country_idx.append(country_to_idx[c])

    exog_country_idx_arr = np.asarray(exog_country_idx, dtype=np.int64)
    nx = len(exog_labels)

    # Build the (ny, nx) boolean admissibility mask.
    if admissibility_mode == "free":
        # All cells are admissible: the sampler can activate any
        # (country, exogenous) pair.
        admissibility = np.ones((ny, nx), dtype=bool)
    else:
        # Country-specific: only "own wind" and "own solar" edges are admissible.
        admissibility = np.zeros((ny, nx), dtype=bool)
        for j in range(nx):
            admissibility[exog_country_idx_arr[j], j] = True

    return exog_labels, exog_country_idx_arr, admissibility


# ============================================================
# DATACLASS: container for chain output
# ============================================================

@dataclass
class ChainOutput:
    """
    All chain samples and metadata needed by downstream network-analysis modules.

    Tensors are kept in their on-disk dtypes (uint8 for graphs, float32 for
    coefficients) to save memory. Convert on the fly inside compute-heavy
    code if numerical precision matters.
    """
    # --- Endogenous-block samples ------------------------------------
    G0_samples:    np.ndarray   # (ny, ny, N_KEEP)            uint8
    G_Phi_samples: np.ndarray   # (ny, ny, n_lags, N_KEEP)    uint8
    Phi_samples:   np.ndarray   # (ny, ny, n_lags, N_KEEP)    float32
    Sigma_u_samples: np.ndarray # (ny, ny, N_KEEP)            float32

    # --- Exogenous-block samples (optional, may be None until step5 is wired) -
    G_Gamma_samples: Optional[np.ndarray] = None  # (ny, nx, n_lags, N_KEEP) uint8
    Gamma_samples:   Optional[np.ndarray] = None  # (ny, nx, n_lags, N_KEEP) float32

    # --- Physical / structural masks ---------------------------------
    G0_admissible:      np.ndarray = field(default=None)  # (ny, ny) bool
    G_Gamma_admissible: np.ndarray = field(default=None)  # (ny, nx) bool

    # --- Labels and indexing -----------------------------------------
    country_labels: list[str] = field(default_factory=list)         # length ny
    exog_labels:    list[str] = field(default_factory=list)         # length nx
    exog_country_idx: np.ndarray = field(default=None)              # (nx,) int64
    selected_lags:  list[int] = field(default_factory=list)         # actual lag values

    # --- Shape metadata (denormalised for convenience) ---------------
    ny: int = 0       # number of endogenous (countries)
    nx: int = 0       # number of exogenous columns per lag
    n_lags: int = 0   # number of lags
    n_keep: int = 0   # number of kept posterior samples

    # =========================================================
    # Convenience properties
    # =========================================================
    @property
    def has_exogenous(self) -> bool:
        """True if exogenous samples are loaded (step5 was run)."""
        return self.G_Gamma_samples is not None

    @property
    def n_admissible_G0(self) -> int:
        """Number of admissible directed edges in G0 (denominator for densities)."""
        return int(self.G0_admissible.sum())

    @property
    def n_admissible_G_Gamma(self) -> int:
        """Number of admissible edges in G_Gamma per lag."""
        return int(self.G_Gamma_admissible.sum())

    @property
    def n_possible_G_Phi(self) -> int:
        """
        Number of possible edges in G_Phi PER LAG.
        G_Phi has no admissibility constraint (autoregressive structure allows
        any country i to depend on any country j's past, including self).
        """
        return self.ny * self.ny

    def __repr__(self) -> str:
        exo_str = "yes" if self.has_exogenous else "no (step5 not run)"
        return (
            f"ChainOutput(ny={self.ny}, nx={self.nx}, n_lags={self.n_lags}, "
            f"n_keep={self.n_keep}, exogenous={exo_str})"
        )


# ============================================================
# LOADER FUNCTIONS
# ============================================================

def _load_npy(path: Path, required: bool = True) -> Optional[np.ndarray]:
    """
    Thin wrapper around np.load with a clear error message.
    If required=False and the file does not exist, returns None.
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required sample file not found: {path}\n"
                f"Make sure main.py has finished and save_samples() was called."
            )
        return None
    return np.load(path)


def load_chain_output(
    sample_dir: str | Path,
    selected_lags: list[int],
    countries: list[str] = COUNTRIES,
    solar_excluded: list[str] = SOLAR_EXCLUDED,
    load_partial: bool = False,
    G_Gamma_admissibility_mode: str = "free",
) -> ChainOutput:
    """
    Load all chain samples from disk and return a ChainOutput object.

    Parameters
    ----------
    sample_dir : str or Path
        Directory where main.py saved the *_samples.npy files.
    selected_lags : list[int]
        The actual lag values used in the Gibbs sampler (e.g. [1, 2, 7]).
        Needed because the order of the lag axis in G_Phi_samples must be
        meaningful for downstream plotting / interpretation.
    countries : list[str]
        Country code list, in the SAME order as the columns of Y.
    solar_excluded : list[str]
        Countries without a solar exogenous variable.
    load_partial : bool
        If True, look for *_samples_partial.npy files (checkpoint dumps)
        instead of the final ones. Useful for inspecting a still-running chain.
    G_Gamma_admissibility_mode : str
        - "free" (default, matches the current sampler): all (country, exog)
          edges are admissible. The loader does NOT check the mask on
          G_Gamma samples.
        - "country_specific": only "own wind / own solar" edges are
          admissible. The loader enforces the mask and raises if any
          non-admissible edge is found in the samples.

    Returns
    -------
    ChainOutput
    """
    sample_dir = Path(sample_dir)
    suffix = "_partial" if load_partial else ""

    # --- Mandatory endogenous samples ---------------------------------
    G0          = _load_npy(sample_dir / f"G0_samples{suffix}.npy",      required=True)
    G_Phi       = _load_npy(sample_dir / f"G_Phi_samples{suffix}.npy",   required=True)
    Phi         = _load_npy(sample_dir / f"Phi_samples{suffix}.npy",     required=True)
    Sigma_u     = _load_npy(sample_dir / f"Sigma_u_samples{suffix}.npy", required=True)

    # --- Optional exogenous samples (may not exist yet) ---------------
    G_Gamma     = _load_npy(sample_dir / f"G_Gamma_samples{suffix}.npy", required=False)
    Gamma       = _load_npy(sample_dir / f"Gamma_samples{suffix}.npy",   required=False)

    # --- Physical-network mask for G0 ---------------------------------
    G0_expanded = _load_npy(sample_dir / "G0_expanded.npy", required=True)
    # In prices_only mode, the expanded mask equals the original country
    # adjacency matrix. Convert to bool for clean downstream usage.
    G0_admissible = G0_expanded.astype(bool)

    # --- Shape consistency checks -------------------------------------
    ny_g0,  ny_g0_b,  n_keep_g0      = G0.shape
    ny_phi, ny_phi_b, n_lags, n_keep = G_Phi.shape

    assert ny_g0 == ny_g0_b == ny_phi == ny_phi_b, (
        f"Endogenous dim mismatch: G0 has {ny_g0}x{ny_g0_b}, "
        f"G_Phi has {ny_phi}x{ny_phi_b}"
    )
    assert n_keep_g0 == n_keep, (
        f"n_keep mismatch: G0 has {n_keep_g0}, G_Phi has {n_keep}"
    )
    assert n_lags == len(selected_lags), (
        f"n_lags mismatch: G_Phi has {n_lags} lag slices, "
        f"selected_lags has {len(selected_lags)} entries"
    )
    assert ny_g0 == len(countries), (
        f"Country count mismatch: G0 has {ny_g0} rows, "
        f"COUNTRIES list has {len(countries)} entries. "
        f"Are you sure DATA_MODE='prices_only' was used?"
    )

    ny = ny_g0

    # --- Country and exogenous labels ---------------------------------
    country_labels = list(countries)  # defensive copy
    exog_labels, exog_country_idx, G_Gamma_admissible = build_exogenous_labels(
        countries=countries,
        solar_excluded=solar_excluded,
        admissibility_mode=G_Gamma_admissibility_mode,
    )
    nx = len(exog_labels)

    # --- Exogenous shape checks (only if loaded) ----------------------
    if G_Gamma is not None:
        ny_g, nx_g, n_lags_g, n_keep_g = G_Gamma.shape
        assert ny_g == ny, f"G_Gamma ny mismatch: {ny_g} vs {ny}"
        assert nx_g == nx, (
            f"G_Gamma nx mismatch: file has {nx_g} exogenous cols, "
            f"but COUNTRIES + solar exclusions imply {nx}. "
            f"Check the column ordering of X_exo in the sampler."
        )
        assert n_lags_g == n_lags, f"G_Gamma n_lags mismatch: {n_lags_g} vs {n_lags}"
        assert n_keep_g == n_keep, f"G_Gamma n_keep mismatch: {n_keep_g} vs {n_keep}"

        # --- Sanity check on the mask (only if the sampler claims to enforce it) ---
        # In "free" mode this check is skipped: every edge is admissible by
        # construction, so the chain cannot violate anything.
        if G_Gamma_admissibility_mode == "country_specific":
            ever_active = G_Gamma.any(axis=(2, 3))   # (ny, nx) bool
            violations = ever_active & (~G_Gamma_admissible)
            n_viol = int(violations.sum())
            if n_viol > 0:
                i_v, j_v = np.where(violations)
                example = (country_labels[i_v[0]], exog_labels[j_v[0]])
                raise ValueError(
                    f"Found {n_viol} non-admissible edges activated in G_Gamma "
                    f"samples while in 'country_specific' mode. "
                    f"Example: country={example[0]} <- exog={example[1]}. "
                    f"Either the sampler does not enforce the mask, or you "
                    f"meant to use G_Gamma_admissibility_mode='free'."
                )

    # --- G0 admissibility check ---------------------------------------
    # Same logic: any cell ever activated in G0 samples must be admissible.
    ever_active_G0 = G0.any(axis=2)  # (ny, ny) bool
    violations_G0 = ever_active_G0 & (~G0_admissible)
    n_viol_G0 = int(violations_G0.sum())
    if n_viol_G0 > 0:
        i_v, j_v = np.where(violations_G0)
        example = (country_labels[i_v[0]], country_labels[j_v[0]])
        raise ValueError(
            f"Found {n_viol_G0} non-admissible edges activated in G0 samples. "
            f"Example: {example[0]} <- {example[1]}. "
            f"Check step1 / G0_expanded usage."
        )

    return ChainOutput(
        G0_samples         = G0,
        G_Phi_samples      = G_Phi,
        Phi_samples        = Phi,
        Sigma_u_samples    = Sigma_u,
        G_Gamma_samples    = G_Gamma,
        Gamma_samples      = Gamma,
        G0_admissible      = G0_admissible,
        G_Gamma_admissible = G_Gamma_admissible,
        country_labels     = country_labels,
        exog_labels        = exog_labels,
        exog_country_idx   = exog_country_idx,
        selected_lags      = list(selected_lags),
        ny     = ny,
        nx     = nx,
        n_lags = n_lags,
        n_keep = n_keep,
    )


# ============================================================
# DIAGNOSTIC: print a human-readable summary
# ============================================================

def print_summary(chain: ChainOutput) -> None:
    """Print a quick summary of what's been loaded. Useful as a sanity-check."""
    n_g_gamma_possible = chain.ny * chain.nx
    g_gamma_mode = (
        "free (no mask)"
        if chain.n_admissible_G_Gamma == n_g_gamma_possible
        else "country-specific mask"
    )

    print("=" * 60)
    print("CHAIN OUTPUT SUMMARY")
    print("=" * 60)
    print(f"Endogenous nodes (countries) : {chain.ny}")
    print(f"Exogenous predictors per lag : {chain.nx}")
    print(f"Number of lags               : {chain.n_lags}   (values: {chain.selected_lags})")
    print(f"Posterior samples kept       : {chain.n_keep}")
    print()
    print(f"G0 admissible edges          : {chain.n_admissible_G0} "
          f"/ {chain.ny * (chain.ny - 1)} possible directed off-diagonal")
    print(f"G_Phi possible edges per lag : {chain.n_possible_G_Phi}")
    print(f"G_Gamma mode                 : {g_gamma_mode}")
    print(f"G_Gamma admissible per lag   : {chain.n_admissible_G_Gamma} "
          f"/ {n_g_gamma_possible} possible")
    print()
    print(f"Exogenous block loaded       : {chain.has_exogenous}")
    print()
    print("First 5 countries :", chain.country_labels[:5])
    print("First 5 exog cols :", chain.exog_labels[:5])
    print("Last  5 exog cols :", chain.exog_labels[-5:])
    print("=" * 60)


# ============================================================
# QUICK SELF-TEST (run only when executed as a script)
# ============================================================

if __name__ == "__main__":
    # Quick smoke test of build_exogenous_labels in both modes.
    print("--- Mode: free ---")
    labels, country_idx, mask_free = build_exogenous_labels(admissibility_mode="free")
    print(f"Built {len(labels)} exogenous labels.")
    print(f"First 3 wind  : {labels[:3]}")
    print(f"First 3 solar : {labels[28:31]}")
    print(f"Admissibility mask shape : {mask_free.shape}")
    print(f"Admissible cells (free)  : {int(mask_free.sum())}  "
          f"(expected: 28 * 53 = 1484)")
    assert int(mask_free.sum()) == 28 * 53

    print()
    print("--- Mode: country_specific ---")
    _, _, mask_cs = build_exogenous_labels(admissibility_mode="country_specific")
    print(f"Admissible cells (cs)    : {int(mask_cs.sum())}  "
          f"(expected: 28 + 25 = 53)")
    row_sums = mask_cs.sum(axis=1)
    print(f"Row sums (admissible exogenous per country): "
          f"min={row_sums.min()}, max={row_sums.max()}")
    for c in SOLAR_EXCLUDED:
        i = COUNTRIES.index(c)
        assert row_sums[i] == 1, f"{c} should have 1 admissible exog, got {row_sums[i]}"
    print("Solar-excluded countries correctly have only wind: OK")