"""Shared defaults of the static plots: figure sizes and style-dict merging.

Renderers take their default sizes from here, so plots of one kind match.
Figure resolution is left to Matplotlib's ``figure.dpi`` unless a caller
passes ``dpi``.
"""

from __future__ import annotations

from typing import Any
from typing import Optional

# Square panels for geometric scenes (complexes, cycles, interiors, loops).
SCENE_FIGSIZE = (5.0, 5.0)
# Charts with a filtration axis (profiles, landscapes, measurements).
CHART_FIGSIZE = (6.0, 4.0)
# Persistence forests need width for their trees.
FOREST_FIGSIZE = (10.0, 6.0)

# Chart frame: axis lines and ticks in dark grey, a faint grid behind the data.
AXIS_COLOR = "0.35"
GRID_COLOR = "0.92"
# Landscape levels run from dark (k = 1) to light along this colormap range.
LANDSCAPE_CMAP = ("plasma", 0.0, 0.8)


def style_chart_axes(ax) -> None:
    """Give a chart an open frame: left and bottom axes, a faint grid, a frameless legend."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS_COLOR, width=0.8)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)


def landscape_colors(n: int) -> list:
    """Return ``n`` colors from dark to light for landscape levels ``k = 1 .. n``."""
    from matplotlib import colormaps

    name, low, high = LANDSCAPE_CMAP
    positions = [0.5 * (low + high)] if n == 1 else [low + (high - low) * i / (n - 1) for i in range(n)]
    return [colormaps[name](position) for position in positions]


def resolve_style(style: Optional[dict[str, Any]], defaults: dict[str, Any], name: str) -> dict[str, Any]:
    """Merge style overrides into ``defaults``, rejecting unknown keys.

    Parameters
    ----------
    style : dict or None
        User overrides.
    defaults : dict
        Every supported key with its default value.
    name : str
        Parameter name used in the error message, e.g. ``"style_2d"``.

    Returns
    -------
    dict
        A new dictionary with the overrides applied.

    Raises
    ------
    ValueError
        If ``style`` contains a key that ``defaults`` does not.
    """
    resolved = dict(defaults)
    if style:
        unknown = sorted(set(style) - set(defaults))
        if unknown:
            raise ValueError(
                f"Unknown {name} keys {unknown}; supported keys are {sorted(defaults)}.")
        resolved.update(style)
    return resolved
