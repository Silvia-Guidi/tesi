"""
network_metrics.py
==================

Per-sample network metrics with posterior credible intervals.

Computes node-level, pair-level, and graph-level metrics on EACH MCMC sample
of G0 (the contemporaneous binary graph), then aggregates across samples into
posterior summaries (mean / median + 90% CI).

Why per-sample, not on the posterior mean matrix
-------------------------------------------------
A common mistake is to compute metrics on the matrix of posterior edge
probabilities Ê_ij directly. This gives biased results because:

  - Centrality measures on the average graph != average of centralities.
  - The "average graph" mixes incompatible configurations
    (e.g. a cycle that exists only sometimes).

The Bayesian-correct approach is:
  1. For each MCMC sample G^(k), compute the metric m(G^(k)).
  2. Aggregate the N_KEEP values into mean / quantiles.

This way every quantity inherits a proper posterior distribution from the chain.

Why this module focuses on G0
-----------------------------
G0 has good mixing (ESS ~1366 out of 10000 -- about 14%), so single-edge
posteriors are reliable. G_Phi and G_Gamma have ESS ~50 on individual edges,
which makes node-level centrality unstable. For those we will use a
coefficient-weighted approach in a separate module.

Output container
----------------
- NodeMetricResult  : per-node metric, samples + posterior summary
- GraphMetricResult : graph-level scalar metric, samples + posterior summary
- MetricsBundle     : container holding all metrics for a single graph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import networkx as nx

from network_load import ChainOutput


# ============================================================
# DATACLASSES FOR METRIC RESULTS
# ============================================================

@dataclass
class NodeMetricResult:
    """
    Posterior summary of a node-level metric (e.g. in-degree per country).

    Attributes
    ----------
    samples : np.ndarray, shape (ny, N_KEEP)
        Per-iteration value for each node.
    mean : np.ndarray, shape (ny,)
        Posterior mean across iterations.
    median : np.ndarray, shape (ny,)
        Posterior median.
    ci_low, ci_high : np.ndarray, shape (ny,)
        90% credible interval bounds (5th and 95th percentile).
    labels : list[str]
        Node labels (countries).
    name : str
        Human-readable metric name.
    """
    samples: np.ndarray
    mean:    np.ndarray
    median:  np.ndarray
    ci_low:  np.ndarray
    ci_high: np.ndarray
    labels:  list[str]
    name:    str

    def to_table_rows(self, sort_by: str = "mean", descending: bool = True) -> list[dict]:
        """
        Return rows ready to be turned into a DataFrame or printed.
        Sorted by `mean` (default) or any other summary statistic.
        """
        keys = {"mean": self.mean, "median": self.median,
                "ci_low": self.ci_low, "ci_high": self.ci_high}
        if sort_by not in keys:
            raise ValueError(f"sort_by must be in {list(keys.keys())}")
        order = np.argsort(keys[sort_by])
        if descending:
            order = order[::-1]
        return [
            {
                "node":    self.labels[i],
                "mean":    float(self.mean[i]),
                "median":  float(self.median[i]),
                "ci_low":  float(self.ci_low[i]),
                "ci_high": float(self.ci_high[i]),
            }
            for i in order
        ]


@dataclass
class GraphMetricResult:
    """
    Posterior summary of a graph-level scalar metric (e.g. density).
    """
    samples: np.ndarray   # shape (N_KEEP,)
    mean:    float
    median:  float
    ci_low:  float
    ci_high: float
    name:    str


@dataclass
class MetricsBundle:
    """Container for all metrics computed on a single graph (G0)."""
    # Node-level
    in_degree:    NodeMetricResult
    out_degree:   NodeMetricResult
    total_degree: NodeMetricResult
    eigen_centr:  NodeMetricResult       # eigenvector centrality (in-mode)
    betweenness:  NodeMetricResult
    # Graph-level
    density:           GraphMetricResult
    reciprocity:       GraphMetricResult
    transitivity:      GraphMetricResult  # global clustering coefficient
    n_strong_cc:       GraphMetricResult  # # strongly connected components
    # Metadata
    graph_name:   str = ""
    labels:       list[str] = field(default_factory=list)
    n_samples:    int = 0


# ============================================================
# POSTERIOR SUMMARY HELPER
# ============================================================

def _posterior_summary_node(samples_2d: np.ndarray,
                            labels: list[str],
                            name: str,
                            alpha: float = 0.10) -> NodeMetricResult:
    """
    Build a NodeMetricResult from a (ny, N_KEEP) array of per-iteration values.

    Parameters
    ----------
    samples_2d : np.ndarray, shape (ny, N_KEEP)
        Per-iteration metric values, one row per node.
    labels : list[str]
        Node labels (length ny).
    name : str
        Metric name for the output container.
    alpha : float
        Significance level. alpha=0.10 -> 90% CI (5th, 95th percentile).
    """
    q_low  = 100.0 * (alpha / 2.0)             # 5.0 for alpha=0.10
    q_high = 100.0 * (1.0 - alpha / 2.0)       # 95.0 for alpha=0.10

    mean    = samples_2d.mean(axis=1)
    median  = np.median(samples_2d, axis=1)
    ci_low  = np.percentile(samples_2d, q_low,  axis=1)
    ci_high = np.percentile(samples_2d, q_high, axis=1)

    return NodeMetricResult(
        samples=samples_2d, mean=mean, median=median,
        ci_low=ci_low, ci_high=ci_high, labels=labels, name=name,
    )


def _posterior_summary_scalar(samples_1d: np.ndarray,
                              name: str,
                              alpha: float = 0.10) -> GraphMetricResult:
    """Build a GraphMetricResult from a 1D array of per-iteration values."""
    q_low  = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)

    # Handle NaN: NetworkX returns NaN for some metrics on degenerate graphs.
    # Use nanpercentile / nanmean so we don't propagate NaN into the summary.
    return GraphMetricResult(
        samples=samples_1d,
        mean   =float(np.nanmean(samples_1d)),
        median =float(np.nanmedian(samples_1d)),
        ci_low =float(np.nanpercentile(samples_1d, q_low)),
        ci_high=float(np.nanpercentile(samples_1d, q_high)),
        name   =name,
    )


# ============================================================
# PER-SAMPLE METRIC COMPUTATIONS
# ============================================================
# Convention: input matrix G has shape (ny, ny) with G[i, j] == 1 meaning
# "node j influences node i" (j is the parent of i).
#
# A *directed* networkx DiGraph stores edges as (source, target). Following
# the influence convention, edge (j, i) means "j -> i": j is source, i is
# target. So when we read G[i, j] == 1, we add edge (j, i) to the graph.
# This is critical: getting the direction wrong inverts in/out-degree.

def _matrix_to_digraph(G: np.ndarray) -> nx.DiGraph:
    """
    Convert an (ny, ny) binary matrix into a networkx DiGraph.

    G[i, j] == 1  =>  edge (j -> i)  in the DiGraph.

    Returns a graph with `ny` nodes (0..ny-1) and the corresponding edges.
    Self-loops on diagonal cells (if any) are included.
    """
    ny = G.shape[0]
    DG = nx.DiGraph()
    DG.add_nodes_from(range(ny))
    # Find (i, j) where G[i, j] == 1, then add (j, i) as a directed edge
    i_idx, j_idx = np.where(G == 1)
    edges = [(int(j), int(i)) for i, j in zip(i_idx, j_idx)]
    DG.add_edges_from(edges)
    return DG


# ----- Node-level helpers ----------------------------------------

def _in_degree_vec(G: np.ndarray) -> np.ndarray:
    """In-degree = number of parents (sum over j of G[i, j])."""
    return G.sum(axis=1).astype(np.float64)


def _out_degree_vec(G: np.ndarray) -> np.ndarray:
    """Out-degree = number of children (sum over i of G[i, j])."""
    return G.sum(axis=0).astype(np.float64)


def _eigen_centrality_vec(G: np.ndarray, max_iter: int = 200,
                          damping: float = 0.85) -> np.ndarray:
    """
    Eigenvector centrality (in-mode): centrality of node i is high if i is
    pointed to by other high-centrality nodes.

    For directed graphs, the pure power iteration on G^T can diverge to zero
    if the graph is acyclic (no recurrent structure). To make this robust
    on ANY graph, we use a PageRank-style damped iteration:

        x_{k+1} = damping * G^T x_k + (1 - damping) / ny * 1

    With damping=0.85 this matches the standard PageRank formulation and
    always converges to a positive stationary distribution. For graphs with
    strong cyclic structure (like G0 in this BGVAR analysis) the result is
    nearly identical to pure eigenvector centrality. For acyclic graphs it
    degrades gracefully instead of collapsing to NaN.
    """
    ny = G.shape[0]
    if G.sum() == 0:
        return np.full(ny, 1.0 / ny)

    # Convention: G[i,j]=1 means j -> i. For in-centrality we propagate
    # along incoming edges, which means multiplying by G (rows sum incoming
    # influence into node i).
    A = G.astype(np.float64)

    # Normalise rows so each node redistributes its centrality among its
    # out-neighbours (column-stochastic-like normalisation on G^T = A's columns)
    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1.0   # avoid division by zero on dangling nodes
    M = A / col_sums                 # column-normalised; M[i,j] = G[i,j] / out_j

    # Power iteration with teleportation (PageRank)
    teleport = (1.0 - damping) / ny
    x = np.ones(ny) / ny
    for _ in range(max_iter):
        x_new = damping * (M @ x) + teleport
        if np.linalg.norm(x_new - x) < 1e-10:
            return x_new
        x = x_new
    return x  # return last iterate


def _betweenness_vec(DG: nx.DiGraph) -> np.ndarray:
    """
    Betweenness centrality (Brandes 2001 algorithm via networkx).
    Returns NaN vector on empty graph.
    """
    if DG.number_of_edges() == 0:
        return np.full(DG.number_of_nodes(), np.nan)
    bw_dict = nx.betweenness_centrality(DG, normalized=True)
    return np.array([bw_dict[i] for i in range(DG.number_of_nodes())])


# ----- Graph-level helpers ----------------------------------------

def _density(G: np.ndarray, n_admissible: int) -> float:
    """
    Density = #edges / #admissible_edges.
    Uses the admissibility-aware denominator to avoid underestimation when
    G0 has a physical interconnection mask.
    """
    if n_admissible == 0:
        return float("nan")
    return float(G.sum()) / float(n_admissible)


def _reciprocity(G: np.ndarray) -> float:
    """
    Reciprocity = fraction of directed edges that are bidirectional.
    Defined as: |{(i,j) : G[i,j]=1 and G[j,i]=1}| / |{(i,j) : G[i,j]=1}|.

    Returns NaN if G has no edges.
    """
    total_edges = int(G.sum())
    if total_edges == 0:
        return float("nan")
    mutual = int(np.sum(G * G.T))  # element-wise AND
    return mutual / total_edges


def _transitivity(DG: nx.DiGraph) -> float:
    """
    Transitivity (global clustering coefficient on the directed graph).
    Uses networkx's directed transitivity.
    """
    if DG.number_of_edges() == 0:
        return float("nan")
    return nx.transitivity(DG)


def _n_strongly_connected(DG: nx.DiGraph) -> int:
    """Number of strongly connected components."""
    return nx.number_strongly_connected_components(DG)


# ============================================================
# TOP-LEVEL: COMPUTE METRICS ON G0 SAMPLES
# ============================================================

def compute_G0_metrics(
    chain: ChainOutput,
    alpha: float = 0.10,
    verbose: bool = True,
    skip_betweenness: bool = False,
) -> MetricsBundle:
    """
    Compute the full metrics bundle on G0 samples.

    Parameters
    ----------
    chain : ChainOutput
        Loaded chain (must contain G0_samples).
    alpha : float
        Significance level for credible intervals (0.10 -> 90% CI).
    verbose : bool
        Print progress (helpful: betweenness is the slow part).
    skip_betweenness : bool
        If True, do not compute betweenness centrality (saves ~half of
        the runtime). The slot in the bundle is filled with NaNs.

    Returns
    -------
    MetricsBundle
    """
    G0 = chain.G0_samples           # (ny, ny, N_KEEP) uint8
    ny, _, n_keep = G0.shape
    n_admissible = chain.n_admissible_G0

    if verbose:
        print(f"[metrics] Computing metrics on G0 ({ny}x{ny}, "
              f"{n_keep} samples, {n_admissible} admissible edges)...")

    # --- Pre-allocate storage for per-sample metric values ---
    in_deg_arr        = np.zeros((ny, n_keep))
    out_deg_arr       = np.zeros((ny, n_keep))
    eig_arr           = np.zeros((ny, n_keep))
    bw_arr            = np.zeros((ny, n_keep))

    density_arr       = np.zeros(n_keep)
    reciprocity_arr   = np.zeros(n_keep)
    transitivity_arr  = np.zeros(n_keep)
    n_scc_arr         = np.zeros(n_keep)

    # --- Loop over samples ---
    # We iterate once: for each iteration build the matrix and the DiGraph,
    # compute every metric, store. This is more cache-friendly than looping
    # ny x n_keep times.
    for k in range(n_keep):
        G_k = G0[:, :, k]                              # (ny, ny) uint8

        # Node-level: degree counts (vectorised, very fast)
        in_deg_arr[:, k]   = _in_degree_vec(G_k)
        out_deg_arr[:, k]  = _out_degree_vec(G_k)

        # Eigenvector centrality (custom, fast)
        eig_arr[:, k] = _eigen_centrality_vec(G_k)

        # Build DiGraph once for the betweenness + graph-level metrics
        DG = _matrix_to_digraph(G_k)

        # Betweenness (slow): O(V*E)
        if not skip_betweenness:
            bw_arr[:, k] = _betweenness_vec(DG)
        else:
            bw_arr[:, k] = np.nan

        # Graph-level metrics
        density_arr[k]      = _density(G_k, n_admissible)
        reciprocity_arr[k]  = _reciprocity(G_k)
        transitivity_arr[k] = _transitivity(DG)
        n_scc_arr[k]        = _n_strongly_connected(DG)

        # Progress every 1000 iterations
        if verbose and (k + 1) % 1000 == 0:
            print(f"  ... processed {k+1}/{n_keep} samples")

    # --- Aggregate posterior summaries ---
    in_deg_res = _posterior_summary_node(
        in_deg_arr,   chain.country_labels, "in_degree",          alpha)
    out_deg_res = _posterior_summary_node(
        out_deg_arr,  chain.country_labels, "out_degree",         alpha)
    total_deg_res = _posterior_summary_node(
        in_deg_arr + out_deg_arr,
        chain.country_labels, "total_degree",       alpha)
    eig_res = _posterior_summary_node(
        eig_arr,      chain.country_labels, "eigen_centrality",   alpha)
    bw_res = _posterior_summary_node(
        bw_arr,       chain.country_labels, "betweenness",        alpha)

    density_res      = _posterior_summary_scalar(density_arr,      "density",      alpha)
    reciprocity_res  = _posterior_summary_scalar(reciprocity_arr,  "reciprocity",  alpha)
    transitivity_res = _posterior_summary_scalar(transitivity_arr, "transitivity", alpha)
    n_scc_res        = _posterior_summary_scalar(n_scc_arr,        "n_strong_cc",  alpha)

    if verbose:
        print(f"[metrics] Done.")

    return MetricsBundle(
        in_degree   = in_deg_res,
        out_degree  = out_deg_res,
        total_degree= total_deg_res,
        eigen_centr = eig_res,
        betweenness = bw_res,
        density     = density_res,
        reciprocity = reciprocity_res,
        transitivity= transitivity_res,
        n_strong_cc = n_scc_res,
        graph_name  = "G0",
        labels      = list(chain.country_labels),
        n_samples   = n_keep,
    )


# ============================================================
# PRETTY-PRINT SUMMARIES
# ============================================================

def print_node_metric(result: NodeMetricResult,
                      top_k: int = 10,
                      ascending: bool = False) -> None:
    """Print top-k or bottom-k nodes for a node-level metric."""
    rows = result.to_table_rows(sort_by="mean", descending=not ascending)
    direction = "Bottom" if ascending else "Top"
    print(f"\n=== {direction} {top_k} by {result.name} ===")
    print(f"  {'node':<6}  {'mean':>8}  {'median':>8}  "
          f"{'ci_low':>8}  {'ci_high':>8}")
    print("  " + "-" * 50)
    for r in rows[:top_k]:
        print(f"  {r['node']:<6}  "
              f"{r['mean']:>8.3f}  {r['median']:>8.3f}  "
              f"{r['ci_low']:>8.3f}  {r['ci_high']:>8.3f}")


def print_graph_metrics(bundle: MetricsBundle) -> None:
    """Print all graph-level metrics as a compact table."""
    print(f"\n=== Graph-level metrics for {bundle.graph_name} ===")
    print(f"  {'metric':<14}  {'mean':>8}  {'median':>8}  "
          f"{'ci_low':>8}  {'ci_high':>8}")
    print("  " + "-" * 55)
    for gm in [bundle.density, bundle.reciprocity,
               bundle.transitivity, bundle.n_strong_cc]:
        print(f"  {gm.name:<14}  "
              f"{gm.mean:>8.3f}  {gm.median:>8.3f}  "
              f"{gm.ci_low:>8.3f}  {gm.ci_high:>8.3f}")


