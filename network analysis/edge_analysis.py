"""
network_edges.py
================

Posterior edge analysis for BGVAR networks.

Given the MCMC samples loaded by network_loader.py, this module computes:

  1. Posterior edge probabilities  Ê_ij = mean over chain of G^(k)_ij
  2. Effective sample size per edge (autocorrelation-corrected)
  3. Credibility-interval-based edge selection (Ahelegbey et al. 2016, eq. C.1)
  4. Three "edge sets":
       - probability matrix      (soft, weighted)
       - naive 0.5-threshold     (Ê > 0.5)
       - credibility threshold   (lower CI bound > 0.5) — the paper's choice
  5. Same outputs for G0, each lag of G_Phi, and each lag of G_Gamma.

Why this module is the first interpretive step
----------------------------------------------
Every downstream metric (centrality, density, modularity, spillover) needs
to know "which edges exist". The answer in a Bayesian setting is not a
single matrix but a posterior distribution over edges. This module turns
that distribution into three usable summaries, each with its own use case:

  - Probability matrix : best for heatmaps and weighted-network metrics.
  - Naive threshold    : easy to communicate but ignores Monte Carlo error.
  - Credibility set    : conservative, matches the paper's methodology.

Output containers
-----------------
- EdgeAnalysis      : per-graph result (G0, G_Phi at lag k, G_Gamma at lag k).
- EdgeAnalysisBundle: holds all per-graph results plus convenience accessors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from network_load import ChainOutput


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class EdgeAnalysis:
    """
    Posterior edge analysis for a single graph slice (G0, or G_Phi at one
    lag, or G_Gamma at one lag).

    Attributes
    ----------
    edge_prob : np.ndarray, shape (n_rows, n_cols), float
        Posterior probability of each directed edge.
        edge_prob[i, j] = P(edge j -> i | data).
    n_eff : np.ndarray, shape (n_rows, n_cols), float
        Effective sample size per edge, corrected for autocorrelation.
    ci_lower : np.ndarray, shape (n_rows, n_cols), float
        Lower bound of the one-sided (1-alpha) credibility interval, eq. C.1.
    selected_naive : np.ndarray, shape (n_rows, n_cols), bool
        edge_prob > 0.5.
    selected_credible : np.ndarray, shape (n_rows, n_cols), bool
        ci_lower > 0.5. The paper's preferred edge set.
    admissibility : np.ndarray, shape (n_rows, n_cols), bool
        Physical / structural admissibility mask. Cells outside the mask
        are forced to 0 across all summaries.
    alpha : float
        Significance level used for the CI (default 0.10 -> 90% CI).
    row_labels, col_labels : list[str]
        Labels for rows (response) and columns (parents).
    name : str
        Human-readable identifier, e.g. "G0", "G_Phi[lag=1]".
    """
    edge_prob:         np.ndarray
    n_eff:             np.ndarray
    ci_lower:          np.ndarray
    selected_naive:    np.ndarray
    selected_credible: np.ndarray
    admissibility:     np.ndarray
    alpha:             float
    row_labels:        list[str]
    col_labels:        list[str]
    name:              str

    # ---- Convenience methods --------------------------------------------
    def n_selected(self, kind: str = "credible") -> int:
        """Number of selected edges under a given criterion."""
        if kind == "credible":
            return int(self.selected_credible.sum())
        if kind == "naive":
            return int(self.selected_naive.sum())
        raise ValueError(f"Unknown kind: {kind}. Use 'credible' or 'naive'.")

    def density(self, kind: str = "credible") -> float:
        """
        Network density: selected edges / admissible edges.
        Denominator excludes physically forbidden cells (e.g. G0 self-loops
        between non-connected countries, or G_Gamma cross-country exogenous).
        """
        denom = int(self.admissibility.sum())
        if denom == 0:
            return float("nan")
        return self.n_selected(kind) / denom

    def __repr__(self) -> str:
        return (
            f"EdgeAnalysis({self.name}, "
            f"shape={self.edge_prob.shape}, "
            f"density_credible={self.density('credible'):.3f}, "
            f"n_selected_credible={self.n_selected('credible')})"
        )


@dataclass
class EdgeAnalysisBundle:
    """
    Container holding the EdgeAnalysis for every graph slice in the model.

    Layout:
      - G0          : single EdgeAnalysis
      - G_Phi[lag]  : list of EdgeAnalysis, indexed by lag position
                      (NOT by lag value: G_Phi[0] is selected_lags[0])
      - G_Gamma[lag]: list of EdgeAnalysis, may be empty if step5 not run
    """
    G0:      EdgeAnalysis
    G_Phi:   list[EdgeAnalysis]
    G_Gamma: list[EdgeAnalysis] = field(default_factory=list)
    selected_lags: list[int] = field(default_factory=list)
    alpha:   float = 0.10

    def lag_value(self, lag_idx: int) -> int:
        """Convert a lag-axis index (0, 1, ...) to the actual lag value (e.g. 7)."""
        return self.selected_lags[lag_idx]

    def summary_table(self) -> str:
        """Plain-text summary of densities across all graph slices."""
        lines = []
        lines.append(f"{'Graph':<22} {'n_sel_naive':>12} {'n_sel_cred':>12} "
                     f"{'dens_naive':>12} {'dens_cred':>12}")
        lines.append("-" * 75)

        def row(ea: EdgeAnalysis) -> str:
            return (f"{ea.name:<22} "
                    f"{ea.n_selected('naive'):>12} "
                    f"{ea.n_selected('credible'):>12} "
                    f"{ea.density('naive'):>12.4f} "
                    f"{ea.density('credible'):>12.4f}")

        lines.append(row(self.G0))
        for ea in self.G_Phi:
            lines.append(row(ea))
        for ea in self.G_Gamma:
            lines.append(row(ea))
        return "\n".join(lines)


# ============================================================
# EFFECTIVE SAMPLE SIZE
# ============================================================

def _autocorr_at_lag(x: np.ndarray, lag: int) -> float:
    """
    Sample autocorrelation of a 1D binary chain at a given lag.
    Uses the standard (biased) estimator, which is the convention for ESS.
    """
    n = x.size
    if lag >= n:
        return 0.0
    mean = x.mean()
    var = x.var()
    if var < 1e-12:
        # Constant chain (always 0 or always 1) -> no autocorrelation to estimate
        return 0.0
    c = np.mean((x[: n - lag] - mean) * (x[lag:] - mean))
    return c / var


def _ess_one_edge(chain_1d: np.ndarray, max_lag: int = 100) -> float:
    """
    Effective sample size for a single edge's binary chain.

    Uses the initial-positive-sequence estimator (Geyer 1992): sum
    autocorrelations until they become non-positive, then truncate.
    This is the standard rho-hat-based ESS used in MCMC diagnostics.

    Notes on edge cases
    -------------------
    - If the chain is constant (always 0 or always 1), ESS is set to N
      (effectively a degenerate distribution, no Monte Carlo error to
      worry about). The credibility interval will then collapse to 0.
    - max_lag caps how far we look; for short chains it is auto-reduced.
    """
    n = chain_1d.size

    # Constant chain -> nothing to estimate
    if chain_1d.min() == chain_1d.max():
        return float(n)

    max_lag = min(max_lag, n // 4)
    if max_lag < 1:
        return float(n)

    # Initial positive sequence: stop summing when autocorrelation goes
    # non-positive (Geyer 1992).
    sum_rho = 0.0
    for lag in range(1, max_lag + 1):
        rho = _autocorr_at_lag(chain_1d, lag)
        if rho <= 0:
            break
        sum_rho += rho

    # ESS = N / (1 + 2 * sum_{k=1}^{K} rho_k)
    ess = n / (1.0 + 2.0 * sum_rho)
    # Numerical guard: never report ESS larger than N
    return float(min(ess, n))


def compute_n_eff_matrix(
    samples_3d: np.ndarray,
    admissibility: np.ndarray,
    max_lag: int = 100,
) -> np.ndarray:
    """
    Compute ESS for each (i, j) edge in a 3D binary sample tensor.

    Parameters
    ----------
    samples_3d : np.ndarray, shape (n_rows, n_cols, N_KEEP), dtype uint8 or bool
        Binary indicator samples for one graph slice.
    admissibility : np.ndarray, shape (n_rows, n_cols), dtype bool
        Cells outside the mask are skipped (ESS set to NaN; they don't matter
        anyway since edge_prob is 0 there).
    max_lag : int
        Maximum autocorrelation lag to consider.

    Returns
    -------
    n_eff : np.ndarray, shape (n_rows, n_cols), float
        Effective sample size per edge. NaN on non-admissible cells.
    """
    n_rows, n_cols, n_keep = samples_3d.shape
    n_eff = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    # Cast once: ESS routines need a numeric (not uint8) array
    for i in range(n_rows):
        for j in range(n_cols):
            if not admissibility[i, j]:
                continue
            chain_ij = samples_3d[i, j, :].astype(np.float64)
            n_eff[i, j] = _ess_one_edge(chain_ij, max_lag=max_lag)

    return n_eff


# ============================================================
# CORE EDGE ANALYSIS
# ============================================================

def _analyse_edges(
    samples_3d: np.ndarray,
    admissibility: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    name: str,
    alpha: float = 0.10,
    max_lag: int = 100,
) -> EdgeAnalysis:
    """
    Run the full edge analysis on a single 3D sample tensor.

    Parameters
    ----------
    samples_3d : (n_rows, n_cols, N_KEEP) uint8/bool
    admissibility : (n_rows, n_cols) bool
    row_labels, col_labels : node label lists
    name : identifier for this graph slice
    alpha : 1 - confidence level. alpha=0.10 -> 90% one-sided CI.
    max_lag : max autocorrelation lag for ESS

    Returns
    -------
    EdgeAnalysis
    """
    n_rows, n_cols, n_keep = samples_3d.shape

    # --- 1. Posterior edge probability: mean over chain dimension ---
    # Cast to float64 for the mean to avoid uint8 overflow.
    edge_prob = samples_3d.astype(np.float64).mean(axis=2)

    # Force non-admissible cells to 0. Defensive: they should already be 0
    # because the sampler respects the mask, but we re-zero in case of
    # numerical noise or future changes.
    edge_prob = np.where(admissibility, edge_prob, 0.0)

    # --- 2. Effective sample size per edge ---
    n_eff = compute_n_eff_matrix(samples_3d, admissibility, max_lag=max_lag)

    # --- 3. Credibility-interval lower bound (eq. C.1 of the paper) ---
    # q_(1-alpha) = ê - z_(1-alpha) * sqrt(ê(1-ê) / n_eff)
    # where z is the (1-alpha)-quantile of the standard normal.
    # For alpha=0.10, z_(0.90) ≈ 1.2816.
    z_score = stats.norm.ppf(1.0 - alpha)

    # Handle n_eff = NaN (non-admissible) and n_eff = 0 safely.
    # Use np.errstate to suppress divide warnings in masked cells.
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(edge_prob * (1.0 - edge_prob) / n_eff)
    ci_lower = edge_prob - z_score * se

    # Replace NaN (from non-admissible cells) with 0.
    ci_lower = np.where(np.isnan(ci_lower), 0.0, ci_lower)
    # Clamp into [0, 1] for cleanliness (the formula can dip slightly negative
    # for edges with ê near 0 and large variance).
    ci_lower = np.clip(ci_lower, 0.0, 1.0)
    # Mask out non-admissible cells.
    ci_lower = np.where(admissibility, ci_lower, 0.0)

    # --- 4. Edge selection (two criteria) ---
    selected_naive    = (edge_prob > 0.5) & admissibility
    selected_credible = (ci_lower  > 0.5) & admissibility

    return EdgeAnalysis(
        edge_prob         = edge_prob,
        n_eff             = n_eff,
        ci_lower          = ci_lower,
        selected_naive    = selected_naive,
        selected_credible = selected_credible,
        admissibility     = admissibility,
        alpha             = alpha,
        row_labels        = row_labels,
        col_labels        = col_labels,
        name              = name,
    )


# ============================================================
# TOP-LEVEL: ANALYSE ALL GRAPHS IN A CHAIN
# ============================================================

def analyse_all_edges(
    chain: ChainOutput,
    alpha: float = 0.10,
    max_lag: int = 100,
    verbose: bool = True,
) -> EdgeAnalysisBundle:
    """
    Run edge analysis on every graph slice in the chain: G0, each lag of
    G_Phi, and each lag of G_Gamma (if present).

    Parameters
    ----------
    chain : ChainOutput
        Loaded chain output (from network_loader.load_chain_output).
    alpha : float
        1 - one-sided credibility level. 0.10 -> 90% CI (Ahelegbey et al.
        use this level).
    max_lag : int
        Maximum autocorrelation lag for ESS computation.
    verbose : bool
        Print progress as each graph slice is processed (useful because
        ESS is the slow part).

    Returns
    -------
    EdgeAnalysisBundle
    """

    # --- G0 (contemporaneous, endogenous-endogenous) ---
    if verbose:
        print(f"[edges] Analysing G0 ({chain.ny}x{chain.ny}, "
              f"{chain.n_admissible_G0} admissible) ...")
    ea_G0 = _analyse_edges(
        samples_3d    = chain.G0_samples,
        admissibility = chain.G0_admissible,
        row_labels    = chain.country_labels,
        col_labels    = chain.country_labels,
        name          = "G0",
        alpha         = alpha,
        max_lag       = max_lag,
    )

    # --- G_Phi (temporal, endogenous-endogenous, one slice per lag) ---
    # G_Phi has NO admissibility constraint: any country can a priori
    # depend on any country's past, including its own (autoregressive
    # diagonal). The "admissibility" mask is all True.
    full_mask_phi = np.ones((chain.ny, chain.ny), dtype=bool)

    ea_G_Phi: list[EdgeAnalysis] = []
    for lag_idx, lag_val in enumerate(chain.selected_lags):
        if verbose:
            print(f"[edges] Analysing G_Phi[lag={lag_val}] "
                  f"({chain.ny}x{chain.ny}) ...")
        # Extract the (ny, ny, N_KEEP) slice for this lag
        slice_3d = chain.G_Phi_samples[:, :, lag_idx, :]
        ea = _analyse_edges(
            samples_3d    = slice_3d,
            admissibility = full_mask_phi,
            row_labels    = chain.country_labels,
            col_labels    = chain.country_labels,
            name          = f"G_Phi[lag={lag_val}]",
            alpha         = alpha,
            max_lag       = max_lag,
        )
        ea_G_Phi.append(ea)

    # --- G_Gamma (temporal, endogenous-exogenous, one slice per lag) ---
    ea_G_Gamma: list[EdgeAnalysis] = []
    if chain.has_exogenous:
        for lag_idx, lag_val in enumerate(chain.selected_lags):
            if verbose:
                print(f"[edges] Analysing G_Gamma[lag={lag_val}] "
                      f"({chain.ny}x{chain.nx}, "
                      f"{chain.n_admissible_G_Gamma} admissible) ...")
            slice_3d = chain.G_Gamma_samples[:, :, lag_idx, :]
            ea = _analyse_edges(
                samples_3d    = slice_3d,
                admissibility = chain.G_Gamma_admissible,
                row_labels    = chain.country_labels,
                col_labels    = chain.exog_labels,
                name          = f"G_Gamma[lag={lag_val}]",
                alpha         = alpha,
                max_lag       = max_lag,
            )
            ea_G_Gamma.append(ea)
    else:
        if verbose:
            print("[edges] G_Gamma samples not available (step5 not run). Skipping.")

    return EdgeAnalysisBundle(
        G0           = ea_G0,
        G_Phi        = ea_G_Phi,
        G_Gamma      = ea_G_Gamma,
        selected_lags= list(chain.selected_lags),
        alpha        = alpha,
    )


