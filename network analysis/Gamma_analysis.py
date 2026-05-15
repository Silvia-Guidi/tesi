"""
network_metrics_gamma.py
========================

Weighted network analysis on the exogenous coefficient matrices Gamma.

Rationale
---------
Same as network_metrics_phi: G_Gamma binary samples have ESS ~50 on
individual edges, but Gamma posterior means are stable (ESS > 300 on
aggregate quantities). We work on the continuous coefficients.

Three complementary views
-------------------------
G_Gamma is bipartite (28 countries x 53 exogenous predictors), not square,
so standard centrality measures don't apply directly. We build THREE views:

  1. BIPARTITE VIEW (28 x 53):
     - Per-country exposure (row sums): total reliance on renewables.
     - Per-exogenous pervasiveness (column sums): how many countries does
       this renewable shock reach.
     - Separated by wind / solar AND by own / cross-country.

  2. COUNTRY-LEVEL EXPOSURE (28 x 2):
     - For each country, aggregate weight from its OWN wind + solar vs from
       OTHER countries' wind + solar. Tells you who is "self-driven" vs
       "spillover-driven".

  3. COUNTRY-TO-COUNTRY COLLAPSED NETWORK (28 x 28):
     - Edge (j, i) = sum of |Gamma[i, j_wind]| + |Gamma[i, j_solar]|
       over both lags, where j_wind and j_solar are the renewables of country j.
     - This is a 28x28 directed network "country -> country via renewables",
       directly comparable to Phi-based country networks.

Edge weight definition
----------------------
W^(l)_{ij} = | mean_k Gamma^(l, k)_{ij} |  (absolute posterior mean)
W_agg_{ij} = sum_l W^(l)_{ij}              (aggregate across lags)

Convention: W[i, j] = weight of  exog_j -> country_i  influence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from network_load import ChainOutput
from Phi_analysis import (
    WeightedNodeMetric,
    WeightedGraphMetric,
    WeightedMetricsBundle,
    compute_weighted_metrics,
    print_top_k,
)


# ============================================================
# DATACLASS FOR THE BIPARTITE / COUNTRY-LEVEL VIEWS
# ============================================================

@dataclass
class GammaExposureSummary:
    """
    Per-country exposure to wind and solar, broken down by own vs cross-country.

    All fields are arrays of shape (ny,) -- one value per country.

    Fields
    ------
    own_wind, own_solar :
        |Gamma[i, i_wind]|  and  |Gamma[i, i_solar]|, summed across lags.
        Zero for countries without solar (IE, LV, NO).
    cross_wind, cross_solar :
        Total weight of edges from OTHER countries' wind / solar to country i.
    total_wind, total_solar :
        own + cross.
    total :
        total_wind + total_solar (total renewable exposure).
    share_own :
        (own_wind + own_solar) / total : how much of country i's renewable
        exposure is "self-driven".
    share_wind :
        total_wind / total : wind dominance.
    """
    own_wind:    np.ndarray
    own_solar:   np.ndarray
    cross_wind:  np.ndarray
    cross_solar: np.ndarray
    total_wind:  np.ndarray
    total_solar: np.ndarray
    total:       np.ndarray
    share_own:   np.ndarray
    share_wind:  np.ndarray
    labels:      list[str]


@dataclass
class GammaPervasiveness:
    """
    Per-exogenous "pervasiveness": how many countries does each renewable
    shock reach, weighted by influence magnitude.

    Each field is an array of length nx (53 in this study).
    """
    out_strength: np.ndarray   # sum over countries i of |Gamma[i, j]|
    n_targets:    np.ndarray   # number of countries with non-zero weight
    is_wind:      np.ndarray   # boolean: True if exog j is a wind variable
    source_country: np.ndarray # int index: which country exog j belongs to
    labels:       list[str]


@dataclass
class GammaCountryNetwork:
    """
    The 28x28 country-to-country network obtained by collapsing the bipartite
    structure. W[i, j] = total |Gamma| from country j's renewables to country i.

    Separate matrices for wind, solar, and combined.
    Each is then wrapped in a WeightedMetricsBundle so the same metrics from
    network_metrics_phi apply.
    """
    wind_bundle:     WeightedMetricsBundle
    solar_bundle:    WeightedMetricsBundle
    combined_bundle: WeightedMetricsBundle


# ============================================================
# WEIGHT MATRIX CONSTRUCTION FOR GAMMA
# ============================================================

def build_gamma_weight_matrices(chain: ChainOutput,
                                threshold: float = 0.0
                               ) -> dict[str, np.ndarray]:
    """
    Build aggregate (across lags) weight matrices from Gamma posterior samples.

    Returns
    -------
    dict with keys:
      - 'lag1', 'lag7', ... : per-lag weight matrices, shape (ny, nx)
      - 'agg'               : aggregate across lags, shape (ny, nx)
    """
    if not chain.has_exogenous:
        raise ValueError(
            "Chain has no Gamma samples. Make sure step5 was run and "
            "Gamma_samples.npy / G_Gamma_samples.npy are present."
        )
    gamma_mean = chain.Gamma_eff.mean(axis=3)   # (ny, nx, n_lags)
    weights_abs = np.abs(gamma_mean)

    if threshold > 0:
        weights_abs = np.where(weights_abs > threshold, weights_abs, 0.0)

    out: dict[str, np.ndarray] = {}
    for lag_idx, lag_val in enumerate(chain.selected_lags):
        out[f"lag{lag_val}"] = weights_abs[:, :, lag_idx].copy()

    out["agg"] = weights_abs.sum(axis=2)

    return out


# ============================================================
# VIEW 1: BIPARTITE — PER-COUNTRY EXPOSURE
# ============================================================

def compute_exposure_summary(chain: ChainOutput,
                             W_bipartite: np.ndarray
                            ) -> GammaExposureSummary:
    """
    Compute per-country exposure decomposed by own/cross and wind/solar.

    Parameters
    ----------
    chain : ChainOutput
    W_bipartite : np.ndarray, shape (ny, nx)
        Bipartite weight matrix (typically the aggregated one).
    """
    ny, nx = W_bipartite.shape

    # Identify wind vs solar columns: in our layout, first 28 are wind,
    # last 25 are solar. We use exog_labels to be robust to layout changes.
    is_wind  = np.array([lbl.endswith("_wind")  for lbl in chain.exog_labels])
    is_solar = np.array([lbl.endswith("_solar") for lbl in chain.exog_labels])

    # For each country i and each exog j, decide if j is "own" (belongs to i)
    # exog_country_idx[j] gives the country index of exogenous j.
    # own_mask[i, j] = True iff exog j belongs to country i.
    own_mask = chain.exog_country_idx[None, :] == np.arange(ny)[:, None]
    # cross_mask = not own (within admissible cells; here all cells are
    # admissible in 'free' mode so we just use ~own_mask)
    cross_mask = ~own_mask

    own_wind   = (W_bipartite * own_mask   * is_wind[None,  :]).sum(axis=1)
    own_solar  = (W_bipartite * own_mask   * is_solar[None, :]).sum(axis=1)
    cross_wind = (W_bipartite * cross_mask * is_wind[None,  :]).sum(axis=1)
    cross_solar= (W_bipartite * cross_mask * is_solar[None, :]).sum(axis=1)

    total_wind  = own_wind  + cross_wind
    total_solar = own_solar + cross_solar
    total       = total_wind + total_solar

    # Shares: handle zeros gracefully
    with np.errstate(divide="ignore", invalid="ignore"):
        share_own  = np.where(total > 0, (own_wind + own_solar) / total, 0.0)
        share_wind = np.where(total > 0, total_wind / total,             0.0)

    return GammaExposureSummary(
        own_wind=own_wind,       own_solar=own_solar,
        cross_wind=cross_wind,   cross_solar=cross_solar,
        total_wind=total_wind,   total_solar=total_solar,
        total=total,
        share_own=share_own,     share_wind=share_wind,
        labels=list(chain.country_labels),
    )


# ============================================================
# VIEW 2: BIPARTITE — PER-EXOGENOUS PERVASIVENESS
# ============================================================

def compute_pervasiveness(chain: ChainOutput,
                          W_bipartite: np.ndarray
                         ) -> GammaPervasiveness:
    """
    For each exogenous variable, compute how widely it influences the system.

    Out-strength = column sum (total weight propagating from this renewable).
    n_targets    = number of countries it has non-zero weight on.
    """
    out_strength = W_bipartite.sum(axis=0)            # (nx,)
    n_targets    = (W_bipartite > 0).sum(axis=0)      # (nx,)
    is_wind      = np.array([lbl.endswith("_wind")  for lbl in chain.exog_labels])

    return GammaPervasiveness(
        out_strength=out_strength,
        n_targets=n_targets,
        is_wind=is_wind,
        source_country=chain.exog_country_idx.copy(),
        labels=list(chain.exog_labels),
    )


# ============================================================
# VIEW 3: COLLAPSED COUNTRY-TO-COUNTRY NETWORK
# ============================================================

def collapse_to_country_network(chain: ChainOutput,
                                W_bipartite: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse the bipartite (ny x nx) weight matrix into three (ny x ny)
    country-to-country directed networks.

    For countries i, j:
      W_country[i, j] = sum of |Gamma[i, e]| over exogenous e belonging to
                        country j of the relevant type (wind / solar / either).

    This produces a 28x28 network "country j's renewables influence country i".

    Returns
    -------
    (W_wind, W_solar, W_combined) : each of shape (ny, ny)
    """
    ny = chain.ny
    nx = chain.nx

    is_wind  = np.array([lbl.endswith("_wind")  for lbl in chain.exog_labels])
    is_solar = np.array([lbl.endswith("_solar") for lbl in chain.exog_labels])

    W_wind     = np.zeros((ny, ny))
    W_solar    = np.zeros((ny, ny))

    # For each exogenous column j, attribute its weight to the source country.
    # exog_country_idx[j] gives the source country of exog j.
    for j_exog in range(nx):
        src_country = chain.exog_country_idx[j_exog]
        col = W_bipartite[:, j_exog]    # weight from this exog to each country
        if is_wind[j_exog]:
            W_wind[:, src_country]  += col
        elif is_solar[j_exog]:
            W_solar[:, src_country] += col

    W_combined = W_wind + W_solar
    return W_wind, W_solar, W_combined


def compute_country_network(chain: ChainOutput,
                            W_bipartite: np.ndarray,
                            verbose: bool = True
                           ) -> GammaCountryNetwork:
    """
    Build the three collapsed 28x28 networks and wrap them in
    WeightedMetricsBundles (so the same metrics from network_metrics_phi
    apply, allowing direct comparison with Phi-based networks).
    """
    W_wind, W_solar, W_combined = collapse_to_country_network(chain, W_bipartite)

    return GammaCountryNetwork(
        wind_bundle    = compute_weighted_metrics(
            W_wind,     chain.country_labels, "Gamma[wind, country-collapsed]",
            verbose=verbose),
        solar_bundle   = compute_weighted_metrics(
            W_solar,    chain.country_labels, "Gamma[solar, country-collapsed]",
            verbose=verbose),
        combined_bundle= compute_weighted_metrics(
            W_combined, chain.country_labels, "Gamma[combined, country-collapsed]",
            verbose=verbose),
    )


# ============================================================
# TOP-LEVEL
# ============================================================

@dataclass
class GammaAnalysisResults:
    """All Gamma-based network analyses bundled together."""
    weight_matrices:    dict[str, np.ndarray]    # 'lag1', 'lag7', 'agg'  shape (ny, nx)
    exposure_summary:   GammaExposureSummary
    pervasiveness:      GammaPervasiveness
    country_network:    GammaCountryNetwork


def compute_all_gamma_metrics(chain: ChainOutput,
                              threshold: float = 0.0,
                              verbose: bool = True
                             ) -> GammaAnalysisResults:
    """
    Run the full Gamma analysis: build weights, compute the three views.

    Uses the AGGREGATE (across lags) weight matrix for the exposure /
    pervasiveness / country-network views. Per-lag weights are kept in
    `weight_matrices` for users who want to dissect further.
    """
    weight_matrices = build_gamma_weight_matrices(chain, threshold=threshold)
    W_agg = weight_matrices["agg"]

    if verbose:
        print(f"[metrics_gamma] Computing exposure summary ...")
    expo = compute_exposure_summary(chain, W_agg)

    if verbose:
        print(f"[metrics_gamma] Computing pervasiveness ...")
    perv = compute_pervasiveness(chain, W_agg)

    if verbose:
        print(f"[metrics_gamma] Building country-level collapsed network ...")
    country_net = compute_country_network(chain, W_agg, verbose=verbose)

    return GammaAnalysisResults(
        weight_matrices  = weight_matrices,
        exposure_summary = expo,
        pervasiveness    = perv,
        country_network  = country_net,
    )


# ============================================================
# PRETTY-PRINT HELPERS
# ============================================================

def print_exposure_summary(expo: GammaExposureSummary, k: int = 10) -> None:
    """Print the per-country exposure table, sorted by total exposure."""
    print("\n=== Per-country renewable exposure (aggregate across lags) ===")
    print(f"  {'country':<8}  {'total':>8}  {'wind':>8}  {'solar':>8}  "
          f"{'own%':>6}  {'wind%':>6}")
    print("  " + "-" * 60)
    order = np.argsort(expo.total)[::-1]
    for i in order[:k]:
        print(f"  {expo.labels[i]:<8}  "
              f"{expo.total[i]:>8.3f}  "
              f"{expo.total_wind[i]:>8.3f}  "
              f"{expo.total_solar[i]:>8.3f}  "
              f"{100*expo.share_own[i]:>5.1f}%  "
              f"{100*expo.share_wind[i]:>5.1f}%")


def print_pervasiveness(perv: GammaPervasiveness,
                        country_labels: list[str],
                        k: int = 10) -> None:
    """Print the top-k most pervasive exogenous variables."""
    print("\n=== Top renewable shocks by pervasiveness ===")
    print(f"  {'exog':<14}  {'src':<5}  {'type':<6}  "
          f"{'out_str':>8}  {'n_targets':>10}")
    print("  " + "-" * 55)
    order = np.argsort(perv.out_strength)[::-1]
    for j in order[:k]:
        kind = "wind" if perv.is_wind[j] else "solar"
        src  = country_labels[perv.source_country[j]]
        print(f"  {perv.labels[j]:<14}  {src:<5}  {kind:<6}  "
              f"{perv.out_strength[j]:>8.3f}  "
              f"{int(perv.n_targets[j]):>10d}")

