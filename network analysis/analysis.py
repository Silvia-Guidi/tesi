import numpy as np
from network_load import load_chain_output
from edge_analysis import analyse_all_edges
from network_metrics import compute_G0_metrics, print_node_metric, print_graph_metrics
from Phi_analysis import compute_all_phi_metrics, print_weighted_bundle
from Gamma_analysis import compute_all_gamma_metrics
from excel_export import export_to_excel

# selected_lags devono essere gli stessi del Gibbs sampler
chain = load_chain_output(
    sample_dir    = "outputs",
    selected_lags = [1, 7],
    n_chains      = 4,
)

def threshold_diagnostic(chain, name="Phi"):
    """
    For each lag, show the distribution of |posterior_mean| coefficients
    and the impact of various thresholds on the implied network density.
    """
    if name == "Phi":
        samples = chain.Phi_samples
        n_cols  = chain.ny
    elif name == "Gamma":
        samples = chain.Gamma_samples
        n_cols  = chain.nx
    else:
        raise ValueError("name must be 'Phi' or 'Gamma'")

    mean_abs = np.abs(samples.mean(axis=3))   # (ny, n_cols, n_lags)

    print(f"\n{'='*70}")
    print(f"Threshold diagnostic on |{name}| posterior mean")
    print(f"{'='*70}")

    candidate_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

    for lag_idx, lag_val in enumerate(chain.selected_lags):
        W = mean_abs[:, :, lag_idx]
        n_total = W.size
        off_diag_mask = ~np.eye(W.shape[0], dtype=bool) if W.shape[0] == W.shape[1] else np.ones_like(W, dtype=bool)
        n_off = int(off_diag_mask.sum())

        # Distribution of |w| on off-diagonal cells
        w_off = W[off_diag_mask]

        print(f"\n--- {name}[lag={lag_val}] ---")
        print(f"  off-diagonal cells               : {n_off}")
        print(f"  max  |w|                         : {w_off.max():.5f}")
        print(f"  mean |w|                         : {w_off.mean():.5f}")
        print(f"  median |w|                       : {np.median(w_off):.5f}")
        print(f"  q90  |w|                         : {np.quantile(w_off, 0.90):.5f}")
        print(f"  q95  |w|                         : {np.quantile(w_off, 0.95):.5f}")
        print(f"  q99  |w|                         : {np.quantile(w_off, 0.99):.5f}")
        print()
        print(f"  Density retained at each threshold (off-diagonal cells):")
        print(f"  {'threshold':>10}  {'n_kept':>8}  {'density':>10}")
        for t in candidate_thresholds:
            n_kept = int(np.sum(w_off > t))
            density = n_kept / n_off
            print(f"  {t:>10.4f}  {n_kept:>8d}  {density:>10.3f}")


threshold_diagnostic(chain, name="Phi")
threshold_diagnostic(chain, name="Gamma")

bundle = analyse_all_edges(chain, alpha=0.10, verbose=True)
print()
print(bundle.summary_table())


def top_edges(ea, k=10):
    """
    Return the top-k edges by posterior probability for a single
    EdgeAnalysis object.
    """
    # Flatten the probability matrix and sort indices in descending order
    flat = ea.edge_prob.flatten()
    idx_flat = np.argsort(flat)[::-1]

    n_cols = ea.edge_prob.shape[1]
    out = []
    for f in idx_flat:
        i, j = divmod(f, n_cols)
        # Skip non-admissible cells (probability is forced to 0 there anyway)
        if not ea.admissibility[i, j]:
            continue
        out.append({
            "from":   ea.col_labels[j],            # parent (predictor)
            "to":     ea.row_labels[i],            # response
            "prob":   ea.edge_prob[i, j],
            "ci_low": ea.ci_lower[i, j],
            "n_eff":  ea.n_eff[i, j],
            "cred":   ea.selected_credible[i, j],
        })
        if len(out) >= k:
            break
    return out


# Loop over all graph slices and print the top 10
for ea in [bundle.G0] + bundle.G_Phi + bundle.G_Gamma:
    print(f"\n=== Top 10 edges for {ea.name} ===")
    print(f"  {'from':>14} -> {'to':<5}   prob   ci_low  n_eff   cred")
    print(f"  " + "-" * 55)
    for e in top_edges(ea, k=10):
        cred_flag = "*" if e["cred"] else " "
        print(f"  {cred_flag} {e['from']:>12} -> {e['to']:<5} "
              f"{e['prob']:.3f}  {e['ci_low']:.3f}  "
              f"{e['n_eff']:>5.0f}")
        


metrics = compute_G0_metrics(chain, alpha=0.10, verbose=True)

print_node_metric(metrics.in_degree,    top_k=10)
print_node_metric(metrics.out_degree,   top_k=10)
print_node_metric(metrics.total_degree, top_k=10)
print_node_metric(metrics.eigen_centr,  top_k=10)
print_node_metric(metrics.betweenness,  top_k=10)
print_graph_metrics(metrics)


phi_results = compute_all_phi_metrics(chain, threshold=0.01, verbose=True)

# Stampa tutto
for key in ["lag1", "lag7", "agg"]:
    print_weighted_bundle(phi_results[key], k=10)
    
gamma_res = compute_all_gamma_metrics(chain, threshold=0.0, verbose=True)

#export_to_excel(
    #chain        = chain,
    #edge_bundle  = bundle,
    #g0_metrics   = metrics,
   # phi_results  = phi_results,
  #  gamma_res    = gamma_res,
 #   output_path  = "network_metrics.xlsx",
#)