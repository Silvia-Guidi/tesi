import numpy as np
from network_load import load_chain_output, print_summary
from edge_analysis import analyse_all_edges
from network_metrics import compute_G0_metrics, print_node_metric, print_graph_metrics
from Phi_analysis import compute_all_phi_metrics, print_weighted_bundle
from Gamma_analysis import compute_all_gamma_metrics
from excel_export import export_to_excel

# selected_lags devono essere gli stessi del Gibbs sampler
chain = load_chain_output(
    sample_dir    = "outputs",
    selected_lags = [1, 7],
)

print_summary(chain)

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

export_to_excel(
    chain        = chain,
    edge_bundle  = bundle,
    g0_metrics   = metrics,
    phi_results  = phi_results,
    gamma_res    = gamma_res,
    output_path  = "network_metrics.xlsx",
)