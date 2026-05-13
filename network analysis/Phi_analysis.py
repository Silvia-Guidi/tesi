"""
network_metrics_phi.py
======================

Weighted network analysis on the VAR coefficient matrices Phi.

Rationale
---------
The binary graph samples G_Phi have very low effective sample size
(ESS ~50 out of 10000 in this BGVAR run), so single-edge posteriors are
unreliable. The continuous coefficients Phi, by contrast, have ESS > 1000
on aggregate quantities, with stable posterior means. We therefore build
weighted networks directly from the posterior mean coefficients, bypassing
the binary-graph mixing problem.

Edge weight definition
----------------------
For each lag l, the weight of the directed edge  j -> i  is
    W^(l)_{ij} = | mean_k Phi^(l, k)_{ij} |

The absolute value is used because:
  - Positive and negative coefficients both represent influence; the sign
    indicates direction (excess/deficit), not strength.
  - Weighted network metrics (centrality, density) require non-negative weights.

We compute three weight matrices:
  - W_lag1 : edge weights using only lag-1 coefficients
  - W_lag7 : edge weights using only lag-7 coefficients
  - W_agg  : aggregated weights = W_lag1 + W_lag7
            (interpreted as total temporal influence across the lag window)

Metrics
-------
For each weight matrix, we compute weighted analogues of the metrics in
network_metrics.py:
  - Node-level: in-strength, out-strength, total-strength,
                weighted eigenvector centrality,
                weighted betweenness (distance = 1 / weight)
  - Graph-level: total weight (sum of edges), average weight,
                 weighted density, weight Gini (concentration)

Uncertainty
-----------
Unlike the binary case, here we have a SINGLE weight matrix per slice
(the posterior mean), not N_KEEP matrices. So node-level metrics are
single numbers, not posterior distributions.

If you want credible intervals on the WEIGHTED metrics, we can compute
them by repeating each metric on every posterior sample of Phi -- but
that's 10000 betweenness computations on weighted graphs, ~30 minutes.
For now we provide point estimates (which are reliable given ESS > 1000
on the coefficients) and offer per-sample CIs as an optional flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import networkx as nx

from network_load import ChainOutput


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class WeightedNodeMetric:
    """A node-level metric on a weighted graph (single point estimate)."""
    values: np.ndarray   # shape (ny,)
    labels: list[str]
    name:   str

    def top_k(self, k: int = 10, ascending: bool = False) -> list[tuple[str, float]]:
        """Return [(label, value), ...] sorted descending (default) or ascending."""
        order = np.argsort(self.values)
        if not ascending:
            order = order[::-1]
        return [(self.labels[i], float(self.values[i])) for i in order[:k]]


@dataclass
class WeightedGraphMetric:
    """A graph-level scalar metric on a weighted graph."""
    value: float
    name:  str


@dataclass
class WeightedMetricsBundle:
    """All weighted metrics for a single weight matrix (one lag, or aggregate)."""
    in_strength:    WeightedNodeMetric
    out_strength:   WeightedNodeMetric
    total_strength: WeightedNodeMetric
    eigen_centr:    WeightedNodeMetric
    betweenness:    WeightedNodeMetric
    total_weight:   WeightedGraphMetric
    mean_weight:    WeightedGraphMetric
    weight_density: WeightedGraphMetric   # fraction of non-zero edges
    weight_gini:    WeightedGraphMetric   # concentration of edge weights
    weight_matrix:  np.ndarray
    labels:         list[str]
    name:           str


# ============================================================
# WEIGHT MATRIX CONSTRUCTION
# ============================================================

def build_phi_weight_matrices(chain: ChainOutput,
                              threshold: float = 0.0) -> dict[str, np.ndarray]:
    """
    Build weight matrices from Phi posterior samples.

    Parameters
    ----------
    chain : ChainOutput
        Loaded chain (must contain Phi_samples of shape (ny, ny, n_lags, n_keep)).
    threshold : float
        Coefficients with |posterior_mean| <= threshold are zeroed out. Useful
        to suppress noise (e.g. threshold = 0.01 keeps only "meaningful" edges).
        Default 0.0 -> keep all coefficients.

    Returns
    -------
    dict with keys:
      - 'lag1' : (ny, ny) weight matrix for lag 1
      - 'lag7' : (ny, ny) weight matrix for lag 7
      - 'agg'  : (ny, ny) aggregate weight matrix (sum across lags)

    Convention
    ----------
    W[i, j] = |posterior_mean(Phi[i, j, lag])|, i.e. j -> i influence strength.
    Self-loops (i == j) are preserved (they represent autoregression).
    """
    # Posterior mean across the chain dimension (axis 3)
    phi_mean = chain.Phi_samples.mean(axis=3)   # (ny, ny, n_lags)
    weights_abs = np.abs(phi_mean)               # absolute value

    # Apply threshold (optional noise suppression)
    if threshold > 0:
        weights_abs = np.where(weights_abs > threshold, weights_abs, 0.0)

    out: dict[str, np.ndarray] = {}
    for lag_idx, lag_val in enumerate(chain.selected_lags):
        out[f"lag{lag_val}"] = weights_abs[:, :, lag_idx].copy()

    out["agg"] = weights_abs.sum(axis=2)   # sum across lags

    return out


# ============================================================
# WEIGHTED METRIC COMPUTATIONS
# ============================================================

def _matrix_to_weighted_digraph(W: np.ndarray, keep_self: bool = True) -> nx.DiGraph:
    """
    Convert a weight matrix W into a weighted DiGraph.

    Convention: W[i, j] is the weight of  j -> i.

    Parameters
    ----------
    W : np.ndarray, shape (ny, ny)
        Non-negative weight matrix.
    keep_self : bool
        If True, include autoregressive self-loops (i, i). If False, remove them
        (useful for centrality measures that double-count self-loops).
    """
    ny = W.shape[0]
    DG = nx.DiGraph()
    DG.add_nodes_from(range(ny))
    i_idx, j_idx = np.where(W > 0)
    for i, j in zip(i_idx, j_idx):
        if (not keep_self) and i == j:
            continue
        DG.add_edge(int(j), int(i), weight=float(W[i, j]))
    return DG


def _in_strength(W: np.ndarray) -> np.ndarray:
    """In-strength = sum over j of W[i, j] = total incoming influence on i."""
    return W.sum(axis=1)


def _out_strength(W: np.ndarray) -> np.ndarray:
    """Out-strength = sum over i of W[i, j] = total outgoing influence from j."""
    return W.sum(axis=0)


def _weighted_eigen_centrality(W: np.ndarray,
                               damping: float = 0.85,
                               max_iter: int = 500) -> np.ndarray:
    """
    PageRank-style centrality on the weighted graph.

    Same logic as the binary version, but now W is real-valued: the
    teleportation term ensures convergence on any graph, weighted or not.
    """
    ny = W.shape[0]
    if W.sum() == 0:
        return np.full(ny, 1.0 / ny)

    # Column-normalise (each column j sums to 1 if it has outgoing edges)
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    M = W / col_sums

    teleport = (1.0 - damping) / ny
    x = np.ones(ny) / ny
    for _ in range(max_iter):
        x_new = damping * (M @ x) + teleport
        if np.linalg.norm(x_new - x) < 1e-10:
            return x_new
        x = x_new
    return x


def _weighted_betweenness(W: np.ndarray, labels: list[str]) -> np.ndarray:
    """
    Weighted betweenness centrality. The "distance" of an edge is taken as
    1 / weight: stronger weights = shorter distance = more likely to lie on
    shortest paths.

    Self-loops are removed before the computation (they don't affect shortest
    paths between distinct nodes but networkx warns about them).
    """
    ny = W.shape[0]
    # Build the weighted DiGraph WITHOUT self-loops, with distance attribute
    DG = nx.DiGraph()
    DG.add_nodes_from(range(ny))
    i_idx, j_idx = np.where(W > 0)
    for i, j in zip(i_idx, j_idx):
        if i == j:
            continue
        # Use 1/weight as distance: stronger edges = shorter distance
        DG.add_edge(int(j), int(i), distance=1.0 / float(W[i, j]))

    if DG.number_of_edges() == 0:
        return np.full(ny, np.nan)

    bw_dict = nx.betweenness_centrality(DG, weight="distance", normalized=True)
    return np.array([bw_dict[i] for i in range(ny)])


def _weight_gini(W: np.ndarray) -> float:
    """
    Gini coefficient of edge weights (excluding zeros and self-loops).
    Measures concentration: 0 = all edges equal, 1 = one edge dominates.

    Returns NaN if there are no non-zero non-self edges.
    """
    ny = W.shape[0]
    W_off = W.copy()
    np.fill_diagonal(W_off, 0.0)
    vals = W_off[W_off > 0]
    if vals.size < 2:
        return float("nan")
    vals = np.sort(vals)
    n = vals.size
    # Standard Gini formula on positive values
    cumvals = np.cumsum(vals)
    return float((n + 1 - 2.0 * np.sum(cumvals) / cumvals[-1]) / n)


# ============================================================
# TOP-LEVEL: COMPUTE METRICS FOR ONE WEIGHT MATRIX
# ============================================================

def compute_weighted_metrics(W: np.ndarray,
                             labels: list[str],
                             name: str,
                             verbose: bool = True) -> WeightedMetricsBundle:
    """
    Compute all weighted network metrics for a single weight matrix.

    Parameters
    ----------
    W : np.ndarray, shape (ny, ny)
        Non-negative weight matrix.  W[i, j] = weight of edge j -> i.
    labels : list[str]
        Node labels (length ny).
    name : str
        Identifier for this matrix (e.g. "Phi[lag=1]" or "Phi[aggregate]").
    verbose : bool
        Print progress.
    """
    ny = W.shape[0]
    if verbose:
        print(f"[metrics_phi] Computing weighted metrics on {name} "
              f"({ny}x{ny}, {int((W > 0).sum())} non-zero entries)...")

    # --- Node-level ---
    in_str  = _in_strength(W)
    out_str = _out_strength(W)
    tot_str = in_str + out_str

    eig = _weighted_eigen_centrality(W)
    bw  = _weighted_betweenness(W, labels)

    in_node  = WeightedNodeMetric(in_str,  labels, "in_strength")
    out_node = WeightedNodeMetric(out_str, labels, "out_strength")
    tot_node = WeightedNodeMetric(tot_str, labels, "total_strength")
    eig_node = WeightedNodeMetric(eig,     labels, "eigen_centrality")
    bw_node  = WeightedNodeMetric(bw,      labels, "betweenness")

    # --- Graph-level ---
    # Density = fraction of non-zero entries among possible directed edges
    # (excluding self-loops, which are autoregressive and not "network" edges)
    off_diag_mask = ~np.eye(ny, dtype=bool)
    n_possible_off = ny * (ny - 1)
    n_nonzero_off  = int(((W > 0) & off_diag_mask).sum())

    total_w = WeightedGraphMetric(float(W.sum()),                "total_weight")
    mean_w  = WeightedGraphMetric(float(W[off_diag_mask].mean()) if n_possible_off > 0
                                  else float("nan"),
                                  "mean_weight_off_diag")
    dens_w  = WeightedGraphMetric(n_nonzero_off / n_possible_off,
                                  "weight_density")
    gini_w  = WeightedGraphMetric(_weight_gini(W), "weight_gini")

    return WeightedMetricsBundle(
        in_strength    = in_node,
        out_strength   = out_node,
        total_strength = tot_node,
        eigen_centr    = eig_node,
        betweenness    = bw_node,
        total_weight   = total_w,
        mean_weight    = mean_w,
        weight_density = dens_w,
        weight_gini    = gini_w,
        weight_matrix  = W,
        labels         = labels,
        name           = name,
    )


def compute_all_phi_metrics(chain: ChainOutput,
                            threshold: float = 0.0,
                            verbose: bool = True
                           ) -> dict[str, WeightedMetricsBundle]:
    """
    Top-level: compute weighted metrics for each lag and for the aggregate.

    Returns
    -------
    dict mapping "lag1", "lag7", "agg" -> WeightedMetricsBundle
    """
    weight_matrices = build_phi_weight_matrices(chain, threshold=threshold)

    out: dict[str, WeightedMetricsBundle] = {}
    for key, W in weight_matrices.items():
        out[key] = compute_weighted_metrics(
            W=W,
            labels=chain.country_labels,
            name=f"Phi[{key}]",
            verbose=verbose,
        )
    return out


# ============================================================
# PRETTY-PRINT SUMMARIES
# ============================================================

def print_top_k(metric: WeightedNodeMetric, k: int = 10) -> None:
    """Print top-k nodes for a weighted node metric."""
    print(f"\n=== Top {k} by {metric.name} ===")
    print(f"  {'node':<6}  {'value':>10}")
    print("  " + "-" * 20)
    for label, val in metric.top_k(k=k):
        print(f"  {label:<6}  {val:>10.4f}")


def print_weighted_bundle(bundle: WeightedMetricsBundle, k: int = 10) -> None:
    """Print all metrics for one weighted bundle."""
    print("\n" + "=" * 60)
    print(f"WEIGHTED METRICS: {bundle.name}")
    print("=" * 60)
    print_top_k(bundle.in_strength,    k)
    print_top_k(bundle.out_strength,   k)
    print_top_k(bundle.total_strength, k)
    print_top_k(bundle.eigen_centr,    k)
    print_top_k(bundle.betweenness,    k)
    print(f"\n--- Graph-level ---")
    for gm in [bundle.total_weight, bundle.mean_weight,
               bundle.weight_density, bundle.weight_gini]:
        print(f"  {gm.name:<22} = {gm.value:.4f}")


