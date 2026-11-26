"""Bar colors: one color per persistence tree ("forest") or per bar ("bars").

Both colorings rank by persistence, so the most persistent trees or bars get
the first palette colors; palettes repeat once they are used up. Ties keep
the barcode's iteration order, which is ``bar.index`` order for a forest.
"""

from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import Literal
from typing import Sequence

from ._dependencies import require_plotting

require_plotting()

from matplotlib import colormaps
from matplotlib import colors as mcolors

# Paul Tol's "vibrant" scheme without its grey: distinct for color-blind
# viewers and saturated enough to read on a pale complex or a gray image.
FOREST_PALETTE = ("#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988")
# tab20 with its ten dark colors first, so the longest bars do not get the
# dark and light shade of one hue.
BARS_PALETTE = tuple(mcolors.to_hex(c) for c in (*colormaps["tab20"].colors[0::2],
                                                 *colormaps["tab20"].colors[1::2]))


def _by_lifespan(bars: Iterable) -> list:
    return sorted(bars, key=lambda bar: bar.lifespan(), reverse=True)


def forest_color_map(bars: Iterable, palette: Sequence[Any] = FOREST_PALETTE) -> dict:
    """Give every bar the color of its tree.

    Parameters
    ----------
    bars : iterable of PFBar
        Bars exposing ``lifespan()`` and ``tree_id``. A bar without a
        ``tree_id`` forms a tree of its own.
    palette : sequence of colors
        Colors assigned to trees in order of their longest bar.

    Returns
    -------
    dict
        Mapping from bar to hex color.
    """
    palette = [mcolors.to_hex(color) for color in palette]
    tree_colors: dict = {}
    color_map = {}
    for bar in _by_lifespan(bars):
        tree = bar.tree_id if getattr(bar, "tree_id", None) is not None else ("bar", id(bar))
        if tree not in tree_colors:
            tree_colors[tree] = palette[len(tree_colors) % len(palette)]
        color_map[bar] = tree_colors[tree]
    return color_map


def bar_color_map(bars: Iterable, palette: Sequence[Any] = BARS_PALETTE) -> dict:
    """Color bars by lifespan rank alone, ignoring the trees.

    Parameters
    ----------
    bars : iterable of PFBar
        Bars exposing ``lifespan()``.
    palette : sequence of colors
        Colors assigned from the longest bar down.

    Returns
    -------
    dict
        Mapping from bar to hex color.
    """
    palette = [mcolors.to_hex(color) for color in palette]
    return {bar: palette[rank % len(palette)] for rank, bar in enumerate(_by_lifespan(bars))}


def _build_color_map_forest(self) -> None:
    """Store the tree coloring of ``self.barcode`` in ``self.color_map_forest``."""
    self.color_map_forest = forest_color_map(self.barcode)


def _build_color_map_bars(self) -> None:
    """Store the bar coloring of ``self.barcode`` in ``self.color_map_bars``."""
    self.color_map_bars = bar_color_map(self.barcode)


def _get_color_map(self, coloring: Literal["forest", "bars"] = "forest") -> dict:
    """Return the requested bar color map, building it on first use."""
    if coloring == "forest":
        if not hasattr(self, "color_map_forest"):
            self._build_color_map_forest()
        return self.color_map_forest
    if coloring == "bars":
        if not hasattr(self, "color_map_bars"):
            self._build_color_map_bars()
        return self.color_map_bars
    raise ValueError("coloring must be 'forest' or 'bars'")


def set_longest_bar_colors(
    self,
    colors: Sequence[Any],
    coloring: Literal["forest", "bars"] = "forest",
) -> None:
    """Overwrite colors for the longest bars in the selected color map."""
    color_map = self._get_color_map(coloring=coloring)
    for bar, color in zip(_by_lifespan(self.barcode), colors):
        color_map[bar] = mcolors.to_hex(color)
