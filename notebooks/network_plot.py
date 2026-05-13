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
# Country label positions (lon, lat). Manually adjusted where the centroid
# falls in the sea or in a less readable spot relative to the network layout.
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
# Naming / parsing
# =============================================================================

def _parse_var(name):
    """'DE_Price' -> ('DE', 'Price'). Strict: requires exactly one underscore."""
    if "_" not in name:
        raise ValueError(
            f"Variable name {name!r} does not match the required CC_KIND format "
            f"(e.g. 'DE_Price'). Rename your variables before plotting."
        )
    cc, kind = name.split("_", 1)
    return cc, kind


def _all_kinds(labels):
    """Return the list of unique kinds present in `labels`, in first-seen order."""
    seen = []
    for n in labels:
        _, k = _parse_var(n)
        if k not in seen:
            seen.append(k)
    return seen


# =============================================================================
# Map background and axis helpers
# =============================================================================

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


def _make_axes(n_panels):
    """Create one figure with `n_panels` map axes side by side."""
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(9 * n_panels, 10))
        proj = ccrs.PlateCarree()
        axes = [fig.add_subplot(1, n_panels, i + 1, projection=proj)
                for i in range(n_panels)]
        for ax in axes:
            _setup_map_background(ax)
    else:
        print("[INFO] cartopy not installed - plotting without map background.")
        print("       Install with:  pip install cartopy")
        fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 10))
        if n_panels == 1:
            axes = [axes]
        else:
            axes = list(axes)
    return fig, axes


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


def _draw_arrow(ax, p1, p2, lw, alpha, color, use_cartopy):
    """Draw one curved directed arrow."""
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


def _resolve_kinds(kinds, available):
    """
    Normalise the `kinds` argument.

    None / 'all'         -> use every kind in the data, in first-seen order
    str (single kind)    -> one-panel plot for that kind
    list[str]            -> one panel per requested kind, in given order

    Raises if a requested kind is not present in `available`.
    """
    if kinds is None or kinds == "all":
        return list(available)
    if isinstance(kinds, str):
        kinds = [kinds]
    kinds = list(kinds)
    missing = [k for k in kinds if k not in available]
    if missing:
        raise ValueError(
            f"Requested kind(s) {missing} not found in data. "
            f"Available: {list(available)}"
        )
    return kinds


# =============================================================================
# 1. PIP-based renderer
# =============================================================================

def _build_subgraph_pip(G_full, kind):
    """Restrict a DiGraph to nodes of a single kind; return (subgraph, metrics)."""
    nodes_kind = [n for n in G_full.nodes() if _parse_var(n)[1] == kind]
    H = G_full.subgraph(nodes_kind).copy()
    in_deg  = dict(H.in_degree(weight="weight"))
    out_deg = dict(H.out_degree(weight="weight"))
    metrics = {
        n: {
            "in_degree":  in_deg.get(n, 0.0),
            "out_degree": out_deg.get(n, 0.0),
            "net_flow":   out_deg.get(n, 0.0) - in_deg.get(n, 0.0),
        }
        for n in nodes_kind
    }
    return H, metrics


def _draw_panel_pip(ax, H, metrics, kind, top_k, use_cartopy):
    """Render one PIP-based panel."""
    pos = {n: COUNTRY_COORDS[_parse_var(n)[0]]
           for n in H.nodes() if _parse_var(n)[0] in COUNTRY_COORDS}

    # Empty-panel guard: render an explanatory placeholder rather than crash.
    if not pos:
        ax.set_title(f"{kind} network - no plottable nodes", fontsize=11)
        _format_axes(ax, use_cartopy)
        return

    transform_kwargs = ({"transform": ccrs.PlateCarree()} if use_cartopy
                        else {})

    # --- Node colouring by signed net flow (red = transmitter, blue = receiver)
    nf_vals = np.array([metrics[n]["net_flow"] for n in pos])
    vmax = max(abs(nf_vals.min()), abs(nf_vals.max())) if nf_vals.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    # --- Label size by out-degree (with safe fallback if all out-degrees are 0)
    od_vals = np.array([metrics[n]["out_degree"] for n in pos])
    od_max = od_vals.max() if (od_vals.size and od_vals.max() > 0) else 1.0
    fontsizes = {n: 11.0 + 11.0 * (metrics[n]["out_degree"] / od_max)
                 for n in pos}
    colors = {n: _saturate_color(cmap(norm(metrics[n]["net_flow"])))
              for n in pos}

    # --- Edges: width and alpha scaled to PIP
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
        _draw_arrow(ax, (x1, y1), (x2, y2), lw, alpha, "#3a3a3a", use_cartopy)

    # --- Top-k transmitters by out-degree are drawn in bold
    sorted_by_out = sorted(pos.keys(),
                           key=lambda n: metrics[n]["out_degree"],
                           reverse=True)
    top_set = set(sorted_by_out[:top_k])
    _draw_labels(ax, pos, fontsizes, colors, top_set, transform_kwargs)
    _format_axes(ax, use_cartopy)

    ax.set_title(
        f"{kind} network  -  {len(pos)} nodes, {H.number_of_edges()} edges\n"
        f"label colour = net out-flow (red = transmitter, blue = receiver)\n"
        f"label size proportional to out-degree   .   edge width proportional to PIP",
        fontsize=11, pad=10,
    )


def plot_pip_network(G, kinds=None, top_k=5, title=None,
                     savepath=None, show=True):
    """
    Plot a PIP-weighted DiGraph as one map panel per requested variable kind.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph with edge attribute 'weight' = PIP. Node names must follow the
        'CC_KIND' convention (e.g. 'DE_Price').
    kinds : None | str | list[str]
        Which kinds to render. None / 'all' uses every kind found in G's
        node labels. A single string renders one panel. A list renders one
        panel per kind, in the given order.
    top_k : int
        Number of top transmitters by out-degree to highlight in bold.
    title : str or None
        Figure suptitle. A sensible default is built if None.
    savepath, show : standard matplotlib I/O.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    available = _all_kinds(list(G.nodes()))
    kinds     = _resolve_kinds(kinds, available)

    # Count cross-kind edges that the panels won't show, so the user knows.
    chosen = set(kinds)
    n_cross = sum(
        1 for u, v in G.edges()
        if (_parse_var(u)[1] in chosen and _parse_var(v)[1] in chosen
            and _parse_var(u)[1] != _parse_var(v)[1])
    )

    fig, axes = _make_axes(n_panels=len(kinds))
    for ax, kind in zip(axes, kinds):
        H, metrics = _build_subgraph_pip(G, kind)
        _draw_panel_pip(ax, H, metrics, kind, top_k, HAS_CARTOPY)

    if title is None:
        kinds_str = " . ".join(kinds)
        title = f"Bayesian network - PIP weights ({kinds_str})"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    if len(kinds) > 1:
        fig.text(0.5, 0.015,
                 f"Note: {n_cross} cross-kind edges omitted from these panels "
                 "for clarity. Edges shown are within-kind only.",
                 ha="center", fontsize=9, style="italic", color="#555555")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if savepath is not None:
        fig.savefig(str(savepath), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =============================================================================
# 2. Coefficient-based renderer
# =============================================================================

def _build_subgraph_coef(coef_matrix, mask, labels, kind, min_abs=0.0):
    """
    Extract a within-kind sub-network from a (ny, ny) coefficient matrix.

    Convention: entry (i, j) of `coef_matrix` is the effect of driver j on
    response i (j -> i). If `mask` is provided, entries where mask == 0 are
    zeroed first; then only edges with |coef| > min_abs are kept.

    Returns
    -------
    edges : list of (driver, receiver, coef)
    metrics : dict[name -> {'out_strength', 'in_strength', 'net_flow_signed', ...}]
    nodes : list of variable names of this kind that have plottable coordinates
    """
    coef = np.asarray(coef_matrix, dtype=float).copy()
    if mask is not None:
        coef = coef * np.asarray(mask, dtype=int)

    nodes_kind = [name for name in labels
                  if _parse_var(name)[1] == kind
                  and _parse_var(name)[0] in COUNTRY_COORDS]
    idx_of = {name: labels.index(name) for name in nodes_kind}

    edges = []
    for rec in nodes_kind:
        i = idx_of[rec]
        for drv in nodes_kind:
            if drv == rec:
                continue
            j = idx_of[drv]
            c = coef[i, j]
            if abs(c) > min_abs:
                edges.append((drv, rec, c))

    out_strength = {n: 0.0 for n in nodes_kind}
    in_strength  = {n: 0.0 for n in nodes_kind}
    out_signed   = {n: 0.0 for n in nodes_kind}
    in_signed    = {n: 0.0 for n in nodes_kind}
    for drv, rec, c in edges:
        out_strength[drv] += abs(c)
        in_strength[rec]  += abs(c)
        out_signed[drv]   += c
        in_signed[rec]    += c

    metrics = {
        n: {
            "out_strength":    out_strength[n],
            "in_strength":     in_strength[n],
            "net_flow_signed": out_signed[n] - in_signed[n],
            "net_flow_abs":    out_strength[n] - in_strength[n],
        }
        for n in nodes_kind
    }
    return edges, metrics, nodes_kind


def _draw_panel_coef(ax, edges, metrics, nodes, kind, top_k, use_cartopy,
                     lag_label):
    """Render one coefficient-based panel (signed, green/red)."""
    pos = {n: COUNTRY_COORDS[_parse_var(n)[0]] for n in nodes}

    if not pos:
        ax.set_title(f"{kind} network - no plottable nodes", fontsize=11)
        _format_axes(ax, use_cartopy)
        return

    transform_kwargs = ({"transform": ccrs.PlateCarree()} if use_cartopy
                        else {})

    # Node colour from signed net flow
    nf_vals = np.array([metrics[n]["net_flow_signed"] for n in pos])
    vmax = max(abs(nf_vals.min()), abs(nf_vals.max())) if nf_vals.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    # Label size from |out-strength|
    os_vals = np.array([metrics[n]["out_strength"] for n in pos])
    os_max  = os_vals.max() if (os_vals.size and os_vals.max() > 0) else 1.0
    fontsizes = {n: 11.0 + 11.0 * (metrics[n]["out_strength"] / os_max)
                 for n in pos}
    colors = {n: _saturate_color(cmap(norm(metrics[n]["net_flow_signed"])))
              for n in pos}

    # Edge widths from |coef|
    if edges:
        a_vals = np.array([abs(c) for _, _, c in edges])
        a_min, a_max = a_vals.min(), a_vals.max()
    else:
        a_min, a_max = 0.0, 1.0

    for drv, rec, c in edges:
        if drv not in pos or rec not in pos:
            continue
        x1, y1 = pos[drv]; x2, y2 = pos[rec]
        if a_max > a_min:
            lw = 0.7 + 2.6 * (abs(c) - a_min) / (a_max - a_min)
        else:
            lw = 1.8
        if a_max > 0:
            alpha = float(np.clip(0.40 + 0.50 * abs(c) / a_max, 0.40, 0.90))
        else:
            alpha = 0.7
        color = "#1a8c1a" if c > 0 else "#c0392b"
        _draw_arrow(ax, (x1, y1), (x2, y2), lw, alpha, color, use_cartopy)

    sorted_by_out = sorted(pos.keys(),
                           key=lambda n: metrics[n]["out_strength"],
                           reverse=True)
    top_set = set(sorted_by_out[:top_k])
    _draw_labels(ax, pos, fontsizes, colors, top_set, transform_kwargs)
    _format_axes(ax, use_cartopy)

    n_pos = sum(1 for _, _, c in edges if c > 0)
    n_neg = sum(1 for _, _, c in edges if c < 0)
    ax.set_title(
        f"{kind} network - coefficients at lag ${lag_label}$\n"
        f"{len(pos)} nodes . {len(edges)} edges "
        f"(green: {n_pos} positive . red: {n_neg} negative)\n"
        f"label colour = signed net out-flow   .   "
        f"edge width proportional to |coef|",
        fontsize=11, pad=10,
    )


def plot_coef_network(coef_matrix, labels, kinds=None, mask=None,
                      lag_label="t-1", top_k=5, min_abs=0.0,
                      title=None, savepath=None, show=True):
    """
    Plot a coefficient-based network as one map panel per requested kind.

    Parameters
    ----------
    coef_matrix : (ny, ny) array
        Posterior-mean coefficient matrix. Entry (i, j) is the effect of
        driver j on response i.
    labels : list of str
        Variable names of length ny in the order of the matrix's rows/cols.
        Each must be 'CC_KIND'.
    kinds : None | str | list[str]
        Which kinds to render. See plot_pip_network for the convention.
    mask : (ny, ny) array of 0/1 or None
        Optional MPM mask. If given, coefficients where mask == 0 are zeroed.
    lag_label : str
        Used in panel titles, e.g. 't-1', 't-7', '0' for contemporaneous.
    top_k : int
        Number of top transmitters by total |out_strength| to render in bold.
    min_abs : float
        Only edges with |coef| > min_abs are drawn.
    title : str or None
        Figure suptitle. A sensible default is built if None.
    savepath, show : standard matplotlib I/O.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    available = _all_kinds(labels)
    kinds     = _resolve_kinds(kinds, available)

    # Count cross-kind edges among the chosen kinds (omitted from panels)
    coef = np.asarray(coef_matrix, dtype=float)
    if mask is not None:
        coef = coef * np.asarray(mask, dtype=int)
    chosen = set(kinds)
    n_cross = 0
    for i, rec in enumerate(labels):
        cc_r, kind_r = _parse_var(rec)
        if cc_r not in COUNTRY_COORDS or kind_r not in chosen:
            continue
        for j, drv in enumerate(labels):
            if i == j:
                continue
            cc_d, kind_d = _parse_var(drv)
            if cc_d not in COUNTRY_COORDS or kind_d not in chosen:
                continue
            if kind_r != kind_d and abs(coef[i, j]) > min_abs:
                n_cross += 1

    fig, axes = _make_axes(n_panels=len(kinds))
    for ax, kind in zip(axes, kinds):
        edges, metrics, nodes = _build_subgraph_coef(
            coef_matrix, mask, labels, kind, min_abs=min_abs
        )
        _draw_panel_coef(ax, edges, metrics, nodes, kind,
                         top_k, HAS_CARTOPY, lag_label)

    if title is None:
        kinds_str = " . ".join(kinds)
        title = (f"BGVAR posterior-mean coefficient network at lag "
                 f"${lag_label}$ ({kinds_str})")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    if len(kinds) > 1:
        fig.text(0.5, 0.015,
                 f"Note: {n_cross} cross-kind edges omitted from these panels. "
                 "Green = positive coefficient . red = negative coefficient.",
                 ha="center", fontsize=9, style="italic", color="#555555")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if savepath is not None:
        fig.savefig(str(savepath), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =============================================================================
# Backwards-compatible wrappers
# =============================================================================

def plot_G0_split_geographic(G0_graph, G0_metrics=None, top_k=5,
                             savepath=None, show=True):
    """
    Legacy wrapper: plot G0 with one panel per kind found in the graph.
    Equivalent to plot_pip_network(G0_graph, kinds='all', ...).
    """
    return plot_pip_network(
        G=G0_graph, kinds="all", top_k=top_k,
        title=r"Contemporaneous Bayesian network $G_0$  -  PIP weights",
        savepath=savepath, show=show,
    )


def plot_coef_split_geographic(coef_matrix, labels, mask=None,
                               lag_label="t-1", top_k=5, min_abs=0.0,
                               savepath=None, show=True):
    """
    Legacy wrapper: plot a coefficient network with one panel per kind found
    in `labels`. Equivalent to plot_coef_network(..., kinds='all').
    """
    return plot_coef_network(
        coef_matrix=coef_matrix, labels=labels, kinds="all",
        mask=mask, lag_label=lag_label, top_k=top_k, min_abs=min_abs,
        savepath=savepath, show=show,
    )