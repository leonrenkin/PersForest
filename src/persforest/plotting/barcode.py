"""Matplotlib rendering of barcodes."""

from typing import Literal
from ._dependencies import require_plotting

def _bar_color_map(forest, coloring) -> dict:
    """Return the bar -> color map for ``coloring``; empty for "none" and "grey".

    Reads the ``color_map_*`` attributes directly so forest-like objects
    without ``PersistenceForest._get_color_map`` remain supported.
    """
    if coloring in ("forest", "bars"):
        name = f"color_map_{coloring}"
        if not hasattr(forest, name):
            getattr(forest, f"_build_{name}")()
        return getattr(forest, name)
    if coloring in ("none", "grey"):
        return {}
    raise ValueError("coloring must be 'forest', 'bars', 'none' or 'grey'")


def _select_bars(forest, min_bar_length: float = 0.0, max_bars: int = 0, bars=None) -> list:
    """Filter bars by minimum lifespan, then keep at most ``max_bars`` longest."""
    bars = list(forest.barcode if bars is None else bars)

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
    return bars


def _filtration_limits(bars) -> tuple[float, float, float]:
    """Return ``(lower, limit, pad)`` of the filtration axis for ``bars``.

    Infinite bars are drawn up to ``limit - 0.25*pad``.
    """
    import numpy as np

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
    return lower, upper + pad, pad


def _display_rows(forest, bars: list, sort, descending: bool) -> list:
    """Order bars for display; ``None`` entries are gaps between trees."""
    import math

    def finite(value):
        return math.inf if not math.isfinite(value) else value

    if sort == "birth":
        rows = sorted(bars, key=lambda b: (b.birth, finite(b.death)))
    elif sort == "death":
        rows = sorted(bars, key=lambda b: (finite(b.death), b.birth))
    elif sort == "length":
        rows = sorted(bars, key=lambda b: finite(b.death) - b.birth)
    elif sort == "tree":
        # Trees in order of their longest bar, each tree's bars by birth.
        trees: dict = {}
        for bar in sorted(bars, key=lambda b: finite(b.death) - b.birth, reverse=True):
            tree = bar.tree_id if getattr(bar, "tree_id", None) is not None else ("bar", id(bar))
            trees.setdefault(tree, []).append(bar)
        groups = [sorted(group, key=lambda b: (b.birth, finite(b.death))) for group in trees.values()]
        if descending:
            groups = [group[::-1] for group in groups[::-1]]
        rows = []
        for group in groups:
            if rows:
                rows.append(None)
            rows.extend(group)
        return rows
    elif sort is None:
        rows = list(bars)
    else:
        raise ValueError(f"Unknown sort option {sort!r}. "
                         "Expected one of 'birth', 'death', 'length', 'tree', or None.")
    return rows[::-1] if descending else rows


def _plot_barcode_generic(
        forest,
        *,
        ax=None,
        sort: str | None = "birth",
        title: str | None = None,
        xlabel: str = "filtration value",
        coloring: Literal["forest", "bars", "none", "grey"] = "forest",
        max_bars: int = 100,
        min_bar_length: float = 0.0,
        bar_width: float | None = None,
        descending: bool = False,
        tight_layout: bool = True,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        ylabel: str | None = None,
        show: bool = False,
    ):
    """
    Plot a 1D barcode from ``forest.barcode``.

    Each bar runs from birth to death along the filtration axis, the first
    bar at the top (or on the left when vertical). Infinite intervals end in
    an arrow pointing right (horizontal) or up (vertical). The title names
    the homology degree on the left and the number of bars shown on the right.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        If given, draw on this axes. Otherwise a new 6×4 figure is created
        (4×6 when vertical).
    sort : {"birth", "death", "length", "tree", None}
        Display order (None keeps the barcode's order). Default "birth".
        ``"tree"`` groups the bars of each tree, trees in order of their
        longest bar and each tree's bars by birth, with a gap between trees.
    title : str | None
        Plot title. Defaults to "$H_d$ barcode", where d is the ambient
        dimension minus one.
    orientation : {"horizontal", "vertical"}
        Direction of barcode intervals; horizontal by default. With vertical
        bars the filtration axis is y, suitable for sharing with a forest plot.
    xlabel : str
        Filtration-axis label (also used vertically unless ylabel is supplied).
    ylabel : str or None
        Override the filtration-axis label for vertical barcodes.
    coloring : {"forest","bars","none","grey"}
        Which color scheme to use:
        - "forest": use forest.color_map_forest (one color per tree).
        - "bars":   use forest.color_map_bars (by lifespan rank).
        - "none":   Matplotlib's color cycle.
        - "grey":   draw all bars in black.
        If the chosen color map does not exist yet, it is built as in
        `plot_at_filtration`.
    max_bars : int
        Display at most this many bars, keeping the longest ones (by
        lifespan); default 100. 0 shows all bars.
    min_bar_length : float
        Filter out bars with lifespan < min_bar_length before plotting.
    bar_width : float or None
        Line width of each bar in points. None (default) scales it with the
        number of rows, from 0.6 to 6 points.
    descending : bool
        If True, reverse the selected sort order.
    tight_layout : bool
        If True, call ``fig.tight_layout()`` after drawing.
    show : bool
        If True, call ``matplotlib.pyplot.show`` after drawing.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the barcode was drawn on.
    """
    require_plotting()
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    from persforest.plotting._style import AXIS_COLOR, GRID_COLOR

    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    vertical = orientation == "vertical"
    if not getattr(forest, "barcode", None):
        raise ValueError("No bars to plot: `forest.barcode` is empty.")

    color_map = _bar_color_map(forest, coloring)
    total = len([b for b in forest.barcode if b.lifespan() >= min_bar_length]) if min_bar_length > 0 \
        else len(list(forest.barcode))
    bars = _select_bars(forest, min_bar_length, max_bars)
    rows = _display_rows(forest, bars, sort, descending)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6) if vertical else (6, 4))
    else:
        fig = ax.figure

    # Rows fill the bar axis; few bars do not get thicker than 1/12 of it.
    n_rows = max(len(rows), 12)
    if bar_width is None:
        box = ax.get_position()
        length_pt = (box.width * fig.get_figwidth() if vertical else box.height * fig.get_figheight()) * 72
        bar_width = float(np.clip(0.6 * length_pt / n_rows, 0.6, 6.0))

    lower, limit, pad = _filtration_limits(bars)
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    segments, colors = [], []
    for index, bar in enumerate(rows):
        if bar is None:
            continue
        start, end = float(bar.birth), float(bar.death)
        if math.isfinite(end) and end < start:
            start, end = end, start
        if coloring == "grey":
            color = "black"
        elif coloring == "none":
            color = cycle_colors[len(colors) % len(cycle_colors)]
        else:
            color = color_map.get(bar, cycle_colors[0])
        stop = end if math.isfinite(end) else limit - 0.25 * pad
        segment = [(start, index), (stop, index)]
        segments.append([(y, x) for x, y in segment] if vertical else segment)
        colors.append(color)
        if not math.isfinite(end):
            tip, tail = (limit - 0.15 * pad, index), (start, index)
            ax.annotate("", xy=tip[::-1] if vertical else tip, xytext=tail[::-1] if vertical else tail,
                        arrowprops=dict(arrowstyle="->", lw=bar_width, color=color))
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=bar_width, capstyle="round"))

    bar_limits = (-0.5, n_rows - 0.5)
    if vertical:
        ax.set_ylim(lower, limit)
        ax.set_xlim(*bar_limits)
        ax.set_xticks([])
        ax.set_ylabel(xlabel if ylabel is None else ylabel)
        filtration_spine, grid_axis = "left", "y"
    else:
        ax.set_xlim(lower, limit)
        ax.set_ylim(bar_limits[::-1])
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        filtration_spine, grid_axis = "bottom", "x"
    for side, spine in ax.spines.items():
        spine.set_visible(side == filtration_spine)
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=AXIS_COLOR, width=0.8)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"$H_{{{forest.dim - 1}}}$ barcode" if title is None else title, loc="left")
    shown = len(bars)
    note = f"{shown} longest of {total} bars" if shown < total else f"{shown} bar{'s' if shown != 1 else ''}"
    ax.set_title(note, loc="right", fontsize="small", color="0.45")
    if tight_layout:
        fig.tight_layout()

    if show:
        plt.show()

    return ax
