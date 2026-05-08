"""
network_plot.py
================
Geographic visualization of the BGVAR network on a map of Europe.

Two complementary functions are exported:

    plot_G0_split_geographic(...)
        Renders the contemporaneous network G0 from a NetworkX DiGraph
        whose edge weights are PIPs (Posterior Inclusion Probabilities).
        Edge thickness encodes confidence; edge colour is neutral.

    plot_coef_split_geographic(...)
        Renders a coefficient-based network from a (ny, ny) matrix of
        posterior-mean coefficients (Phi at a given lag, or any signed
        weighted adjacency). Edge thickness encodes |coefficient|, edge
        colour is green/red for positive/negative effect.

Both functions split the graph into Demand and Price sub-networks and
plot them side by side on a map of Europe.

Requires cartopy for the Europe map background:
    pip install cartopy
Falls back to a plain plot if cartopy is unavailable.
"""

import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ----- Optional cartopy import -----
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# -----------------------------------------------------------------------------
# Country label positions (lon, lat). A few are manually adjusted because the
# centroid falls in the sea (long, curved countries) or in a less readable
# spot relative to the network layout.
# -----------------------------------------------------------------------------
COUNTRY_COORDS = {
    'AT': (14.55, 47.52), 'BE': (4.66,  50.64), 'BG': (25.21, 42.76),
    'CH': (8.23,  46.82), 'CZ': (15.47, 49.82), 'DE': (10.45, 51.16),
    'DK': (9.50,  56.26), 'EE': (25.50, 58.60), 'ES': (-3.75, 40.20),
    'FI': (25.75, 64.92), 'FR': (2.21,  46.23), 'GR': (22.50, 39.50),
    'HR': (16.00, 45.30), 'HU': (19.50, 47.16), 'IE': (-8.24, 53.41),
    'IT': (12.50, 42.50), 'LT': (23.88, 55.17), 'LV': (24.60, 56.88),
    'ME': (19.37, 42.71), 'NL': (5.29,  52.13), 'NO': (10.50, 61.00),
    'PL': (19.13, 51.92), 'PT': (-8.22, 39.40), 'RO': (24.97, 45.94),
    'RS': (21.01, 44.02), 'SE': (15.50, 60.50), 'SI': (14.99, 46.15),
    'SK': (19.70, 48.67),
}

EUROPE_BBOX = (-13, 32, 34, 71)


# =============================================================================
# Internal helpers
# =============================================================================

def _parse_var(name):
    """'DE_Demand' -> ('DE', 'Demand')"""
    cc, kind = name.split('_')
    return cc, kind


def _setup_map_background(ax):
    """Add Europe coastlines and country borders, very light fill."""
    lon_min, lon_max, lat_min, lat_max = EUROPE_BBOX
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#f3f7fb", zorder=0, alpha=0.7)
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#fbf8f3", zorder=0, alpha=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.4, edgecolor="#b8b8b8", zorder=1, alpha=0.7)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   linewidth=0.5, edgecolor="#999999", zorder=1, alpha=0.7)


def _saturate_color(rgba, min_saturation=0.85, max_lightness=0.45):
    """Boost saturation and cap lightness so labels remain readable."""
    r, g, b, a = rgba
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(s, min_saturation)
    l = min(l, max_lightness)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2, a)


def _make_axes(n_panels=2):
    """Create the figure and axes (with cartopy if available)."""
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(9 * n_panels, 10))
        proj = ccrs.PlateCarree()
        axes = [fig.add_subplot(1, n_panels, i + 1, projection=proj)
                for i in range(n_panels)]
        for ax in axes:
            _setup_map_background(ax)
    else:
        print("[INFO] cartopy not installed — plotting without map background.")
        print("       Install with:  pip install cartopy")
        fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 10))
        if n_panels == 1:
            axes = [axes]
    return fig, axes


def _draw_labels(ax, pos, fontsizes, colors, top_set, transform_kwargs):
    """Draw country-code labels with white halo for legibility."""
    for n, (x, y) in pos.items():
        cc = _parse_var(n)[0]
        is_top = n in top_set
        txt = ax.text(
            x, y, cc,
            ha="center", va="center",
            fontsize=fontsizes[n],
            fontweight="bold" if is_top else "semibold",
            color=colors[n],
            zorder=5,
            clip_on=False,
            **transform_kwargs,
        )
        txt.set_path_effects([
            pe.withStroke(linewidth=4.5, foreground="white", alpha=1.0),
            pe.Normal(),
        ])


def _format_axes(ax, use_cartopy):
    """Apply cosmetics for the non-cartopy fallback."""
    if not use_cartopy:
        lon_min, lon_max, lat_min, lat_max = EUROPE_BBOX
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect(1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")


# =============================================================================
# 1. PIP-based network (existing function, kept for backward compatibility)
# =============================================================================

def _build_subgraph_from_digraph(G0_graph, kind):
    nodes_kind = [n for n in G0_graph.nodes() if _parse_var(n)[1] == kind]
    H = G0_graph.subgraph(nodes_kind).copy()
    in_deg = dict(H.in_degree(weight="weight"))
    out_deg = dict(H.out_degree(weight="weight"))
    metrics = {n: {"in_degree":  in_deg.get(n, 0.0),
                   "out_degree": out_deg.get(n, 0.0),
                   "net_flow":   out_deg.get(n, 0.0) - in_deg.get(n, 0.0)}
               for n in nodes_kind}
    return H, metrics


def _draw_panel_pip(ax, H, metrics, kind, top_k, use_cartopy):
    pos = {n: COUNTRY_COORDS[_parse_var(n)[0]]
           for n in H.nodes() if _parse_var(n)[0] in COUNTRY_COORDS}
    transform_kwargs = ({"transform": ccrs.PlateCarree()} if use_cartopy
                        else {})

    nf_vals = np.array([metrics[n]["net_flow"] for n in pos])
    vmax = max(abs(nf_vals.min()), abs(nf_vals.max())) if len(nf_vals) else 1.0
    vmax = vmax if vmax > 0 else 1.0
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    od_vals = np.array([metrics[n]["out_degree"] for n in pos])
    od_max = od_vals.max() if od_vals.max() > 0 else 1.0
    fontsizes = {n: 11.0 + 11.0 * (metrics[n]["out_degree"] / od_max)
                 for n in pos}
    colors = {n: _saturate_color(cmap(norm(metrics[n]["net_flow"])))
              for n in pos}

    edges = [(u, v, d) for u, v, d in H.edges(data=True)
             if u in pos and v in pos]
    if edges:
        pip_vals = np.array([d["weight"] for _, _, d in edges])
        w_min, w_max = pip_vals.min(), pip_vals.max()
    else:
        w_min, w_max = 0.5, 1.0

    for u, v, d in edges:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        pip = d["weight"]
        lw = (0.7 + 2.6 * (pip - w_min) / (w_max - w_min)) if w_max > w_min else 1.8
        alpha = float(np.clip(0.45 + 0.45 * (pip - 0.5) / 0.5, 0.45, 0.90))
        _draw_arrow(ax, (x1, y1), (x2, y2), lw, alpha, "#3a3a3a",
                    use_cartopy)

    sorted_by_out = sorted(pos.keys(),
                           key=lambda n: metrics[n]["out_degree"],
                           reverse=True)
    top_set = set(sorted_by_out[:top_k])
    _draw_labels(ax, pos, fontsizes, colors, top_set, transform_kwargs)
    _format_axes(ax, use_cartopy)

    ax.set_title(
        f"{kind} network G$_0$  —  {len(pos)} nodes, {H.number_of_edges()} edges\n"
        f"label colour = net out-flow (red = transmitter, blue = receiver)\n"
        f"label size ∝ out-degree   ·   edge width ∝ PIP",
        fontsize=11, pad=10,
    )


def _draw_arrow(ax, p1, p2, lw, alpha, color, use_cartopy):
    """Draw one curved arrow."""
    x1, y1 = p1; x2, y2 = p2
    if use_cartopy:
        transform = ccrs.PlateCarree()
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            xycoords=transform._as_mpl_transform(ax),
            textcoords=transform._as_mpl_transform(ax),
            arrowprops=dict(
                arrowstyle="-|>", color=color,
                lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.12",
                mutation_scale=10,
            ),
            zorder=2,
            annotation_clip=True,
        )
    else:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.15",
            arrowstyle="-|>", mutation_scale=10,
            linewidth=lw, color=color, alpha=alpha, zorder=2,
        )
        ax.add_patch(arrow)


def plot_G0_split_geographic(G0_graph, G0_metrics=None, top_k=5,
                             savepath=None, show=True):
    """Plot the PIP-based contemporaneous network (Demand + Price panels)."""
    H_dem, met_dem = _build_subgraph_from_digraph(G0_graph, "Demand")
    H_pri, met_pri = _build_subgraph_from_digraph(G0_graph, "Price")

    n_cross = sum(1 for u, v in G0_graph.edges()
                  if _parse_var(u)[1] != _parse_var(v)[1])

    fig, (ax_dem, ax_pri) = _make_axes(n_panels=2)
    _draw_panel_pip(ax_dem, H_dem, met_dem, "Demand", top_k, HAS_CARTOPY)
    _draw_panel_pip(ax_pri, H_pri, met_pri, "Price",  top_k, HAS_CARTOPY)

    fig.suptitle(
        "Contemporaneous Bayesian network G$_0$  —  Demand vs Price markets "
        "(PIP weights)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.text(0.5, 0.015,
             f"Note: {n_cross} cross-market edges (Demand↔Price) omitted "
             "from these panels for clarity. Edges shown are the within-market "
             "structure only.",
             ha="center", fontsize=9, style="italic", color="#555555")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if savepath is not None:
        fig.savefig(str(savepath), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =============================================================================
# 2. Coefficient-based network (NEW)
# =============================================================================

def _build_subgraph_from_matrix(coef_matrix, mask, labels, kind, min_abs=0.0):
    """
    Extract a within-kind sub-network from a coefficient matrix.

    Parameters
    ----------
    coef_matrix : (ny, ny) array
        Posterior-mean coefficient matrix. Convention: entry (i, j) is
        the effect of driver j on response i (j -> i).
    mask : (ny, ny) array of 0/1 or None
        Optional MPM mask. If None, no mask is applied (all coefficients
        used). If provided, coefficients where mask == 0 are zeroed.
    labels : list[str]
        Variable names of length ny.
    kind : str
        'Demand' or 'Price'.
    min_abs : float
        Only edges with |coef| > min_abs are kept (after masking).

    Returns
    -------
    edges : list of tuples (driver, receiver, coef)
    metrics : dict mapping variable name -> dict with
              'out_strength', 'in_strength', 'net_flow_signed'.
    nodes : list of variable names of the chosen kind that have positions
    """
    coef = np.asarray(coef_matrix, dtype=float).copy()
    if mask is not None:
        coef = coef * np.asarray(mask, dtype=int)

    # Names of the chosen kind, restricted to ones we have coordinates for
    nodes_kind = [name for name in labels
                  if _parse_var(name)[1] == kind
                  and _parse_var(name)[0] in COUNTRY_COORDS]
    idx_of = {name: labels.index(name) for name in nodes_kind}

    edges = []
    for rec in nodes_kind:
        i = idx_of[rec]
        for drv in nodes_kind:
            if drv == rec:
                continue                     # skip self-loops
            j = idx_of[drv]
            c = coef[i, j]
            if abs(c) > min_abs:
                edges.append((drv, rec, c))

    # Signed strengths
    out_strength = {n: 0.0 for n in nodes_kind}
    in_strength = {n: 0.0 for n in nodes_kind}
    out_signed = {n: 0.0 for n in nodes_kind}
    in_signed = {n: 0.0 for n in nodes_kind}

    for drv, rec, c in edges:
        out_strength[drv] += abs(c)
        in_strength[rec] += abs(c)
        out_signed[drv] += c
        in_signed[rec] += c

    metrics = {
        n: {
            "out_strength":     out_strength[n],
            "in_strength":      in_strength[n],
            "net_flow_signed":  out_signed[n] - in_signed[n],
            "net_flow_abs":     out_strength[n] - in_strength[n],
        }
        for n in nodes_kind
    }
    return edges, metrics, nodes_kind


def _draw_panel_coef(ax, edges, metrics, nodes, kind, top_k, use_cartopy,
                    lag_label="t-1"):
    pos = {n: COUNTRY_COORDS[_parse_var(n)[0]] for n in nodes}
    transform_kwargs = ({"transform": ccrs.PlateCarree()} if use_cartopy
                        else {})

    # Node colour scale: based on signed net flow (positive = net outflow
    # of positive influence, negative = net outflow of negative influence)
    nf_vals = np.array([metrics[n]["net_flow_signed"] for n in pos])
    vmax = max(abs(nf_vals.min()), abs(nf_vals.max())) if len(nf_vals) else 1.0
    vmax = vmax if vmax > 0 else 1.0
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    # Label size proportional to total |out| strength
    od_vals = np.array([metrics[n]["out_strength"] for n in pos])
    od_max = od_vals.max() if od_vals.max() > 0 else 1.0
    fontsizes = {n: 11.0 + 11.0 * (metrics[n]["out_strength"] / od_max)
                 for n in pos}
    colors = {n: _saturate_color(cmap(norm(metrics[n]["net_flow_signed"])))
              for n in pos}

    # Edge widths proportional to |coef|; colour green/red on sign
    if edges:
        abs_vals = np.array([abs(c) for _, _, c in edges])
        a_min, a_max = abs_vals.min(), abs_vals.max()
    else:
        a_min, a_max = 0.0, 1.0

    for drv, rec, c in edges:
        x1, y1 = pos[drv]; x2, y2 = pos[rec]
        # Width: 0.6 to 3.5 pt
        if a_max > a_min:
            lw = 0.6 + 2.9 * (abs(c) - a_min) / (a_max - a_min)
        else:
            lw = 1.8
        # Alpha: stronger coefficients more opaque
        if a_max > 0:
            alpha = float(np.clip(0.40 + 0.50 * abs(c) / a_max, 0.40, 0.90))
        else:
            alpha = 0.7
        # Colour: green positive, red negative (paper style)
        color = "#1a8c1a" if c > 0 else "#c0392b"
        _draw_arrow(ax, (x1, y1), (x2, y2), lw, alpha, color, use_cartopy)

    # Top transmitters by |out_strength|
    sorted_by_out = sorted(pos.keys(),
                           key=lambda n: metrics[n]["out_strength"],
                           reverse=True)
    top_set = set(sorted_by_out[:top_k])
    _draw_labels(ax, pos, fontsizes, colors, top_set, transform_kwargs)
    _format_axes(ax, use_cartopy)

    n_pos = sum(1 for _, _, c in edges if c > 0)
    n_neg = sum(1 for _, _, c in edges if c < 0)
    ax.set_title(
        f"{kind} network — coefficients at lag ${lag_label}$\n"
        f"{len(pos)} nodes · {len(edges)} edges "
        f"(green: {n_pos} positive · red: {n_neg} negative)\n"
        f"label colour = signed net out-flow   ·   "
        f"edge width ∝ |coef|",
        fontsize=11, pad=10,
    )


def plot_coef_split_geographic(coef_matrix, labels, mask=None,
                               lag_label="t-1", top_k=5, min_abs=0.0,
                               savepath=None, show=True):
    """
    Plot a coefficient-based contemporaneous (or temporal) network as two
    side-by-side maps (Demand and Price markets).

    Parameters
    ----------
    coef_matrix : numpy array of shape (ny, ny)
        Posterior-mean coefficient matrix. Entry (i, j) is the effect of
        driver j on response i. For G0 pass the contemporaneous coefficient
        matrix Phi0; for a temporal network pass Phi[lag].
    labels : list of str
        Variable names of length ny in the order of coef_matrix's rows/cols.
    mask : numpy array of shape (ny, ny) or None
        Optional 0/1 mask (e.g. the MPM connectivity matrix). If provided,
        coefficients where mask == 0 are forced to zero (white in the paper).
    lag_label : str
        Used in the title. e.g. 't-1' or '0' for contemporaneous.
    top_k : int
        Number of top transmitters by total |out_strength| to render in bold.
    min_abs : float
        Only show edges with |coef| > min_abs. Useful to remove visual noise
        from very small but non-zero coefficients.
    savepath : str or pathlib.Path or None
        If given, saves the figure as PNG (300 dpi).
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    edges_dem, met_dem, nodes_dem = _build_subgraph_from_matrix(
        coef_matrix, mask, labels, "Demand", min_abs=min_abs)
    edges_pri, met_pri, nodes_pri = _build_subgraph_from_matrix(
        coef_matrix, mask, labels, "Price",  min_abs=min_abs)

    # Count cross-market edges that are not shown
    coef = np.asarray(coef_matrix, dtype=float)
    if mask is not None:
        coef = coef * np.asarray(mask, dtype=int)
    n_cross = 0
    for i, rec in enumerate(labels):
        if _parse_var(rec)[0] not in COUNTRY_COORDS:
            continue
        for j, drv in enumerate(labels):
            if _parse_var(drv)[0] not in COUNTRY_COORDS:
                continue
            if i == j:
                continue
            if (_parse_var(rec)[1] != _parse_var(drv)[1]
                    and abs(coef[i, j]) > min_abs):
                n_cross += 1

    fig, (ax_dem, ax_pri) = _make_axes(n_panels=2)
    _draw_panel_coef(ax_dem, edges_dem, met_dem, nodes_dem, "Demand",
                     top_k, HAS_CARTOPY, lag_label)
    _draw_panel_coef(ax_pri, edges_pri, met_pri, nodes_pri, "Price",
                     top_k, HAS_CARTOPY, lag_label)

    fig.suptitle(
        f"BGVAR posterior-mean coefficient network at lag ${lag_label}$  —  "
        "Demand vs Price markets",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.text(0.5, 0.015,
             f"Note: {n_cross} cross-market edges (Demand↔Price) omitted from "
             "these panels. Edges shown are the within-market structure only. "
             "Green = positive coefficient · red = negative coefficient.",
             ha="center", fontsize=9, style="italic", color="#555555")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if savepath is not None:
        fig.savefig(str(savepath), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig