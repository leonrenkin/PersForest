"""
Shared plotting utilities for forest-like persistence objects.

This module is designed to be *forest-agnostic*: it only assumes that a
"forest" object provides

    - forest.barcode        : iterable of bar objects
    - each bar has .birth, .death, and preferably .lifespan()
    - (optionally) forest._build_color_map_forest()
                    forest._build_color_map_bars()
                    forest.color_map_forest
                    forest.color_map_bars
    - (for animations) forest.filtration : iterable of (simplex, filt_val)
      and a method
          forest.plot_at_filtration(filt_val: float, ax=None, **kwargs)

`PersistenceForest` satisfies these assumptions. Future forest classes can
reuse these utilities by exposing the same small interface.

Typical use inside a class:

    from forest_plotting import plot_barcode as _plot_barcode_generic
    from forest_plotting import animate_filtration as _animate_filtration_generic

    class PersistenceForest:
        ...
        def plot_barcode(self, *args, **kwargs):
            return _plot_barcode_generic(self, *args, **kwargs)

        def animate_filtration(self, *args, **kwargs):
            return _animate_filtration_generic(self, *args, **kwargs)

You are free to adapt the wrappers (defaults, docstrings, etc.) per class.
"""

from typing import Any, Literal, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _plot_barcode_generic(
        forest,
        *,
        ax=None,
        sort: str | None = "birth",   # "length" | "birth" | "death" | None
        title: str = "Barcode",
        xlabel: str = "filtration value",
        coloring: Literal["forest", "bars","none","grey"] = "forest",
        max_bars: int = 0,
        min_bar_length: float = 0.0,
        bar_width: float = 2.0,
        descending: bool = False,
        tight_layout: bool = True,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        ylabel: str | None = None,
    ):
    """
    Plot a 1D barcode from ``forest.barcode``.

    Each bar runs from birth to death along the filtration axis. Infinite
    intervals end in an arrow pointing right (horizontal) or up (vertical).

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        If given, draw on this axes. Otherwise a new figure/axes is created.
    sort : {"length","birth","death",None}
        Sort bars before plotting (None preserves current order).
        Default is "birth".
    title : str
        Plot title.
    orientation : {"horizontal", "vertical"}
        Direction of barcode intervals; horizontal by default. With vertical
        bars the filtration axis is y, suitable for sharing with a forest plot.
    xlabel : str
        Filtration-axis label (also used vertically unless ylabel is supplied).
    ylabel : str or None
        Override the filtration-axis label for vertical barcodes.
    coloring : {"forest","bars","none","grey"}
        Which color scheme to use:
        - "forest": use forest.color_map_forest (tree-structured colors).
        - "bars":   use forest.color_map_bars (ignores tree structure).
        - "none":   all bars share matplotlib defaults.
        - "grey":   draw all bars in black.
        If the chosen color map does not exist yet, it is built as in
        `plot_at_filtration`.
    max_bars : int
        If > 0, display at most this many bars, keeping the longest ones
        (by lifespan). 0 means show all bars.
    min_bar_length : float
        Filter out bars with lifespan < min_bar_length before plotting.
    bar_width : float
        Line width used for each barcode interval.
    descending : bool
        If True, reverse the selected sort order.
    tight_layout : bool
        If True, call ``fig.tight_layout()`` after drawing.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the barcode was drawn on.
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    vertical = orientation == "vertical"
    if not getattr(forest, "barcode", None):
        raise ValueError("No bars to plot: `forest.barcode` is empty.")

    # ---- Prepare color map (same logic as plot_at_filtration) ----
    if coloring == "forest":
        if not hasattr(forest, "color_map_forest"):
            forest._build_color_map_forest()
        color_map = forest.color_map_forest
    elif coloring == "bars":
        if not hasattr(forest, "color_map_bars"):
            forest._build_color_map_bars()
        color_map = forest.color_map_bars
    else:
        color_map = {}

    # ---- Work on a copy so we don't mutate original order ----
    bars = list(forest.barcode)

    # Filter by minimum length (Gudhi-like)
    if min_bar_length > 0.0:
        bars = [b for b in bars if b.lifespan() >= min_bar_length]

    if not bars:
        raise ValueError(
            "No bars to plot after applying min_bar_length filter "
            f"(min_bar_length = {min_bar_length})."
        )

    # Limit to longest `max_bars` bars if requested (Gudhi-like)
    if max_bars and max_bars > 0 and len(bars) > max_bars:
        bars = sorted(bars, key=lambda b: b.lifespan(), reverse=True)[:max_bars]

    # Optional sorting for display
    if sort == "birth":
        bars.sort(key=lambda b: (b.birth, b.death), reverse=descending)
    elif sort == "death":
        def dkey(b):
            d = b.death
            return (math.inf if not math.isfinite(d) else d, b.birth)
        bars.sort(key=dkey, reverse=descending)
    elif sort == "length":
        def length(b):
            d = b.death
            d_val = math.inf if not math.isfinite(d) else d
            return d_val - b.birth
        bars.sort(key=length, reverse=descending)
    elif sort is None:
        # Keep whatever order came out of filtering
        pass
    else:
        raise ValueError(
            f"Unknown sort option {sort!r}. "
            "Expected one of 'birth', 'death', 'length', or None."
        )

    n_bars = len(bars)

    # ---- Create axes if needed, with controlled figure height ----
    created_ax = False
    if ax is None:
        # Height grows sublinearly and is capped to avoid gigantic figures
        base_height = 2.5
        extra_height = 0.12 * min(n_bars, 80)   # at most ~9.1 total
        size = (base_height + extra_height, 7) if vertical else (7, base_height + extra_height)
        fig, ax = plt.subplots(figsize=size)
        created_ax = True
    else:
        fig = ax.figure

    # Include births as well as finite deaths, including all-infinite barcodes.
    births = np.array([b.birth for b in bars], dtype=float)
    deaths = np.array([b.death for b in bars], dtype=float)
    if not np.all(np.isfinite(births)) or np.any(np.isnan(deaths)) or np.any(np.isneginf(deaths)):
        raise ValueError("Barcode births must be finite; deaths must be finite or +inf")
    finite_deaths = deaths[np.isfinite(deaths)]
    endpoints = np.concatenate((births, finite_deaths))
    lower, upper = min(0., float(endpoints.min())), float(endpoints.max())
    if not len(finite_deaths) or (np.isposinf(deaths).any() and upper <= births.max()):
        upper += max(1., upper-lower)
    pad = (upper-lower)*0.05 if upper > lower else 1.
    limit = upper + pad
    if vertical:
        ax.set_ylim(lower, limit)
    else:
        ax.set_xlim(lower, limit)

    def point(value, index):
        return (index, value) if vertical else (value, index)

    for i, b in enumerate(bars):
        start, end = float(b.birth), float(b.death)
        if math.isfinite(end) and end < start:
            start, end = end, start
        color = "black" if coloring == "grey" else color_map.get(b, None)
        kwargs = {"linewidth": bar_width}
        if color is not None:
            kwargs["color"] = color
        stop = end if math.isfinite(end) else limit - 0.25*pad
        if vertical:
            ax.vlines(i, start, stop, **kwargs)
        else:
            ax.hlines(i, start, stop, **kwargs)
        if not math.isfinite(end):
            arrowprops = dict(arrowstyle="->", lw=bar_width)
            if color is not None:
                arrowprops["color"] = color
            ax.annotate("", xy=point(limit-0.15*pad, i),
                        xytext=point(start, i), arrowprops=arrowprops)

    ax.set_title(title)
    if vertical:
        ax.set_xticks([])
        ax.set_ylabel(xlabel if ylabel is None else ylabel)
        ax.set_xlim(-1, n_bars)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
    else:
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_ylim(-1, n_bars)
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    if tight_layout:
        fig.tight_layout()

    # If we created the axes, show it immediately (so this works in scripts)
    if created_ax:
        import matplotlib.pyplot as plt
        plt.show()

    return ax

from .forest_plotting_geometry import (
    route_x as _forest_route_x,
    routes_intersect as _forest_routes_intersect,
    edge_routes as _forest_edge_routes,
)


def _plot_persistence_forest_generic(
    forest, *, ax=None, show=True, min_tree_span=0.0, min_bar_length=0.0,
    min_branch_span=0.0, max_trees=None, nodes="none", collapse_degree2=True,
    coloring="forest", color="0.28", linewidth=1.2, alpha=1.0,
    node_size=12.0, node_color=None, annotate_ids=False,
    leaf_spacing=1.0, tree_gap=2.0, figsize=(10, 6),
    title=None, ylabel="Filtration value", grid=False, rasterized=False,
    return_layout=False, edge_style="curved", curvature=1.0, branch_angle=45.0,
    min_clearance=2.0, orientation="vertical",
):
    """Draw persistence trees with filtration on the vertical or horizontal axis.

    Filtration coordinates are exact; the perpendicular coordinates carry no
    metric meaning. Connections are smooth by default. Crossing
    avoidance shortens lateral transitions only where full-height connections
    would intersect another branch; filtration values never change.
    The barcode continuation defines the trunk where
    available, otherwise the highest descendant leaf does. Ties use node IDs.

    Parameters
    ----------
    edge_style : {'curved', 'straight', 'routed'}
        'curved' (default) uses cubic Bezier connections; 'straight' uses direct
        segments; 'routed' retains the early lateral move and vertical rise.
    curvature : float
        Curve handle length, from 0 (short) to 1 (generous). Departure direction
        remains fixed; only affects edge_style='curved'.
    branch_angle : float
        Outward departure angle in screen degrees from the vertical trunk,
        in (0, 90]. Default 45; 90 gives orthogonal departure. Curves arrive
        vertically. Geometry updates on draw after resizing or axis changes.
    min_clearance : float
        Desired stroke-to-stroke separation in physical points, default 2.
        Optimize separation away from shared junctions, without changing
        filtration heights. Dense plots may not achieve the target: inspect
        metadata['clearance_violations'] or enlarge/filter the plot. Intersection
        prevention is mandatory for every style. Unresolvable coincident-event
        intersections raise ValueError instead of producing a crossing plot.
    orientation : {'vertical', 'horizontal'}
        Filtration runs upward by default, or to the right when horizontal.
        The trunk and branches rotate with the filtration axis.
    min_tree_span : float
        Hide trees whose highest leaf minus root value is below this threshold.
        Evaluated on the original tree, before branch filtering.
    min_bar_length : float
        Keep paths owned by bars of at least this lifespan, plus all ancestors
        needed to connect them to their roots. Requires a computed barcode.
    min_branch_span : float
        After barcode filtering, remove a whole side offshoot when its highest
        descendant minus its attachment junction value is below this threshold.
        The dominant continuation is protected. Decisions use spans before this
        pruning pass; no repeated pruning or individual short-edge removal.
    max_trees : int or None
        Keep at most this many trees, ranked by original span (largest first).
    nodes : {'none', 'critical', 'all'} or iterable of str
        Marker selection, or a subset of ('roots', 'branches', 'leaves').
        Roles are determined after filtering. Roots are always critical.
    collapse_degree2 : bool
        Compress chains with one parent and one child. Original IDs, values,
        and edge colors remain available, including when nodes='all'.
    coloring : {'forest', 'bars', 'grey', 'none'}
        Reuse full-barcode colors without recoloring after filtering. 'grey'
        and 'none' use ``color``. Unowned edges also use ``color``.
    color, linewidth, alpha : color, float, float
        Fallback edge color, line width in points, and opacity.
    node_size, node_color : float, color or None
        Marker area in points squared and override color. By default markers
        inherit their incoming edge's color; roots use the trunk color.
    annotate_ids : bool
        Label selected markers with original node IDs.
    leaf_spacing, tree_gap : float
        Horizontal leaf spacing and additional inter-tree gap in leaf units.
    figsize : tuple
        Figure size when creating an axes. Supplied axes are reused.
    title, ylabel, grid, rasterized : optional
        Appearance controls. Rasterization applies only to edges and markers.
    return_layout : bool
        Return (ax, metadata) instead of ax. Metadata contains positions for
        all visible original nodes in plotted coordinates, visible children, roots, edge owners,
        compressed paths, marker IDs, and adjusted_paths. edge_routes contains
        ForestRoute objects with cubic controls and the endpoint of a possible
        continuation, in internal (branch, filtration) coordinates.
        clearance_violations holds
        (path_index, other_path_index, gap_in_points). Shared junction disks
        are excluded from clearance measurement; labels/markers are not packed.
        Metadata and marker positions update when the figure is redrawn.
        The input forest is never modified.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.artist import allow_rasterization
    from matplotlib.colors import to_rgba
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    for name, value in (("min_tree_span", min_tree_span),
                        ("min_bar_length", min_bar_length),
                        ("min_branch_span", min_branch_span),
                        ("tree_gap", tree_gap), ("node_size", node_size),
                        ("min_clearance", min_clearance)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (("leaf_spacing", leaf_spacing), ("linewidth", linewidth)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if edge_style not in ("straight", "curved", "routed"):
        raise ValueError("edge_style must be 'straight', 'curved', or 'routed'")
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if not np.isfinite(curvature) or not 0 <= curvature <= 1:
        raise ValueError("curvature must be between 0 and 1")
    if not np.isfinite(branch_angle) or not 0 < branch_angle <= 90:
        raise ValueError("branch_angle must be in (0, 90]")
    if max_trees is not None and (isinstance(max_trees, bool) or
            not isinstance(max_trees, (int, np.integer)) or max_trees < 1):
        raise ValueError("max_trees must be a positive integer or None")
    if coloring not in ("forest", "bars", "grey", "none"):
        raise ValueError("coloring must be 'forest', 'bars', 'grey', or 'none'")
    if isinstance(nodes, str):
        if nodes not in ("none", "critical", "all"):
            raise ValueError("nodes must be 'none', 'critical', 'all', or a collection of roles")
        roles = {"roots", "branches", "leaves"} if nodes == "critical" else set()
    else:
        roles = set(nodes)
        if not roles <= {"roots", "branches", "leaves"}:
            raise ValueError("Unknown node role")

    original = forest.nodes
    values = {i: float(n.filt_val) for i, n in original.items()}
    if any(not np.isfinite(v) for v in values.values()):
        raise ValueError("Forest node filtration values must be finite")
    children = {i: sorted(n.children) for i, n in original.items()}
    roots = sorted(i for i, n in original.items() if n.parent is None)
    for i, n in original.items():
        if n.parent is not None and (n.parent not in original or i not in children[n.parent]):
            raise ValueError("Forest has inconsistent parent/child links")
        for c in children[i]:
            if c not in original or original[c].parent != i:
                raise ValueError("Forest has inconsistent parent/child links")
            if values[c] < values[i]:
                raise ValueError("Filtration values must increase from roots to leaves")
    order, seen, stack = [], set(), list(reversed(roots))
    while stack:
        i = stack.pop()
        if i in seen:
            raise ValueError("Forest contains a cycle or duplicate child")
        seen.add(i)
        order.append(i)
        stack.extend(reversed(children[i]))
    if len(order) != len(original):
        raise ValueError("Forest contains a component without a root")

    def heights(child_map):
        result = {}
        for i in reversed(order):
            if i in child_map:
                result[i] = max((result[c] for c in child_map[i]), default=values[i])
        return result

    high = heights(children)
    roots.sort(key=lambda i: (-(high[i] - values[i]), i))
    roots = [i for i in roots if high[i] - values[i] >= min_tree_span]
    if max_trees is not None:
        roots = roots[:max_trees]
    bars = list(getattr(forest, "barcode", ()))
    if min_bar_length > 0 and not bars:
        raise ValueError("min_bar_length requires a computed barcode")
    owners = {}
    for bar in bars:
        for i in bar._node_progression:
            if i in original and original[i].parent is not None:
                if i in owners and owners[i] is not bar:
                    raise ValueError("Barcode paths assign multiple owners to an edge")
                owners[i] = bar
    keep = set(original)
    if min_bar_length > 0:
        keep = {i for i, b in owners.items() if b.lifespan() >= min_bar_length}
        # One reverse traversal closes the selection under ancestry in O(N).
        for i in reversed(order):
            if i in keep and original[i].parent is not None:
                keep.add(original[i].parent)
    children = {i: [c for c in children[i] if c in keep] for i in order if i in keep}
    roots = [i for i in roots if i in keep]
    high = heights(children)

    def continuation(i):
        cs = children[i]
        if not cs:
            return None
        owner = owners.get(i)
        matching = [c for c in cs if owner is not None and owners.get(c) is owner]
        return min(matching or cs, key=lambda c: (-high[c], c))

    visible, stack = set(), list(roots)
    while stack:
        i = stack.pop()
        visible.add(i)
        main = continuation(i)
        children[i] = [c for c in children[i] if c == main or
                       high[c] - values[i] >= min_branch_span]
        stack.extend(children[i])
    children = {i: children[i] for i in order if i in visible}
    high = heights(children)
    main = {i: continuation(i) for i in children}

    # Keep subtree intervals disjoint and stagger successive offshoots along
    # a trunk between left and right. Binary junctions otherwise put every
    # offshoot on one side, creating needless obstructions and a lopsided tree.
    arranged, phase = {}, {}
    for i, cs in children.items():
        parent = original[i].parent
        phase[i] = (phase[parent] + max(0, len(children[parent]) - 1)
                    if parent in children and main[parent] == i else 0)
        sides = sorted((c for c in cs if c != main[i]), key=lambda c: (-high[c], c))
        left, right = sides[::2], sides[1::2]
        if phase[i] % 2:
            left, right = right, left
        arranged[i] = list(reversed(left)) + ([main[i]] if cs else []) + right
    x, cursor = {}, 0.0
    for root in roots:
        stack = [(root, False)]
        while stack:
            i, done = stack.pop()
            if done:
                x[i] = x[main[i]]
            elif not arranged[i]:
                x[i] = cursor
                cursor += leaf_spacing
            else:
                stack.append((i, True))
                stack.extend((c, False) for c in reversed(arranged[i]))
        cursor += tree_gap * leaf_spacing

    # Keep full path provenance while interpolating degree-two nodes between
    # structural endpoints. Color changes never disappear during compression.
    paths = []
    for i in children:
        if original[i].parent in visible and len(children[i]) == 1 and collapse_degree2:
            continue
        for c in children[i]:
            path = [i, c]
            while collapse_degree2 and len(children[c]) == 1:
                c = children[c][0]
                path.append(c)
            paths.append(tuple(path))
    endpoint_x = dict(x)
    horizontal = orientation == "horizontal"

    def plotted(point):
        return tuple(point[::-1]) if horizontal else tuple(point)

    positions = {i: plotted((x[i], values[i])) for i in children}
    palette = forest._get_color_map(coloring) if bars and coloring in ("forest", "bars") else {}

    def edge_color(i):
        return palette.get(owners.get(i), color)

    color_runs, colors = [], []
    for index, path in enumerate(paths):
        runs = []
        for a, b in zip(path, path[1:]):
            rgba = to_rgba(edge_color(b))
            if runs and rgba == runs[-1][2]:
                runs[-1] = (runs[-1][0], b, rgba)
            else:
                runs.append((a, b, rgba))
        for a, b, rgba in runs:
            color_runs.append((index, a, b))
            colors.append(rgba)
    root_set = set(roots)
    markers = [i for i in children if (isinstance(nodes, str) and nodes == "all") or
               ("roots" in roles and i in root_set) or
               ("branches" in roles and len(children[i]) >= 2) or
               ("leaves" in roles and not children[i])]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if ax.get_xscale() != 'linear' or ax.get_yscale() != 'linear':
        raise ValueError("Persistence forest geometry requires linear axes")
    metadata = dict(positions=positions, children=children, roots=tuple(roots),
                    edge_owners={i: owners.get(i) for i in children
                                 if original[i].parent in visible},
                    paths=paths, marker_ids=tuple(markers), edge_style=edge_style,
                    orientation=orientation)
    marker_artist, annotations = None, []
    last_transform = None

    def refresh_geometry():
        nonlocal last_transform
        matrix = ax.transData.get_affine().get_matrix()
        scales = np.abs([matrix[0, 0], matrix[1, 1]]) * 72 / ax.figure.dpi
        if horizontal:
            scales = scales[::-1]
        key = tuple(round(float(scale), 10) for scale in scales)
        if key == last_transform:
            return
        if ax.get_xscale() != 'linear' or ax.get_yscale() != 'linear':
            raise ValueError("Persistence forest geometry requires linear axes")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("Cannot resolve forest geometry on these axes")
        routes, adjusted, deficits = _forest_edge_routes(
            paths, endpoint_x, values, edge_style, curvature, branch_angle,
            scales, min_clearance, linewidth)
        for path, route in zip(paths, routes):
            for k in path[1:-1]:
                positions[k] = plotted((route.at(values[k])[0], values[k]))
        patches = []
        for index, a, b in color_runs:
            path = routes[index].path(values[a], values[b])
            if horizontal:
                path = Path(path.vertices[:, ::-1], path.codes)
            patches.append(PathPatch(path))
        collection.set_paths(patches)
        if marker_artist is not None:
            marker_artist.set_offsets([positions[i] for i in markers])
        for i, annotation in annotations:
            annotation.xy = positions[i]
        metadata.update(edge_routes=tuple(routes), adjusted_paths=adjusted,
                        clearance_violations=deficits, branch_angle=branch_angle,
                        min_clearance=min_clearance)
        last_transform = key

    class ForestCollection(PatchCollection):
        # Draw-time refresh happens after constrained/tight layout and before
        # markers, on screen and during PDF/PNG export. No draw-event recursion.
        @allow_rasterization
        def draw(self, renderer):
            refresh_geometry()
            super().draw(renderer)

    collection = ForestCollection([], facecolors="none", edgecolors=colors,
                                  linewidths=linewidth, alpha=alpha,
                                  rasterized=rasterized, zorder=2)
    collection.set_capstyle("round")
    collection.set_joinstyle("round")
    ax.add_collection(collection)
    if markers:
        coords = np.array([positions[i] for i in markers])
        marker_colors = [node_color if node_color is not None else
                         edge_color(main[i] if i in root_set and main[i] is not None else i)
                         for i in markers]
        marker_artist = ax.scatter(coords[:, 0], coords[:, 1], s=node_size, c=marker_colors,
                   linewidths=0, alpha=alpha, rasterized=rasterized, zorder=3)
        if annotate_ids:
            for i in markers:
                annotation = ax.annotate(str(i), positions[i], xytext=(4, 4),
                                         textcoords="offset points", fontsize=7, zorder=4)
                annotations.append((i, annotation))
    if positions:
        ax.update_datalim(np.array(list(positions.values())))
        ax.autoscale_view()
        ax.margins(x=0.04, y=0.05)
    else:
        ax.text(0.5, 0.5, "No trees match the filters", transform=ax.transAxes,
                ha="center", va="center", color="0.45")
    if horizontal:
        ax.set_yticks([])
        ax.set_xlabel(ylabel)
    else:
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    for spine in (("top", "right", "left") if horizontal else ("top", "right", "bottom")):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom" if horizontal else "left"].set_color("0.75")
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid:
        ax.grid(axis="x" if horizontal else "y", color="0.9", linewidth=0.6)
    refresh_geometry()
    if show:
        plt.show()
    return (ax, metadata) if return_layout else ax
