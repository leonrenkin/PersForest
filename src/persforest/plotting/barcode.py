"""plotting.barcode: mechanically extracted reference implementation."""

from typing import Literal
from ._dependencies import require_plotting

def _bar_color_map(forest, coloring) -> dict:
    """Return the bar -> color map for a barcode coloring scheme."""
    # Same logic as plot_at_filtration; built lazily on first use.
    if coloring == "forest":
        if not hasattr(forest, "color_map_forest"):
            forest._build_color_map_forest()
        return forest.color_map_forest
    if coloring == "bars":
        if not hasattr(forest, "color_map_bars"):
            forest._build_color_map_bars()
        return forest.color_map_bars
    return {}


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


def _plot_barcode_generic(
        forest,
        *,
        ax=None,
        sort: str | None = "birth",   # "length" | "birth" | "death" | None
        title: str | None = None,
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
    title : str | None
        Plot title. Defaults to "$H_d$ Barcode", where d is the ambient
        dimension of the point cloud minus one.
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
    require_plotting()
    import math
    import matplotlib.pyplot as plt

    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    vertical = orientation == "vertical"
    if not getattr(forest, "barcode", None):
        raise ValueError("No bars to plot: `forest.barcode` is empty.")

    color_map = _bar_color_map(forest, coloring)
    bars = _select_bars(forest, min_bar_length, max_bars)

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

    lower, limit, pad = _filtration_limits(bars)
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

    ax.set_title(f"$H_{{{forest.dim - 1}}}$ Barcode" if title is None else title)
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
