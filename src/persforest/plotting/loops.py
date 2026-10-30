"""Matplotlib rendering of closed planar loops.

``plot_planar_loops`` draws ``PlanarLoops`` from ``persforest.geometry.loops`` on new axes
or on top of a forest plot, e.g. one from ``forest.plot_at_filtration`` with
``show_cycles=False``. Pass the same ``image_orientation`` as that plot.
"""

from __future__ import annotations

from ._dependencies import require_plotting

require_plotting()

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from persforest.geometry.loops import PlanarLoops
from persforest.plotting._cubical_scene import display


def plot_planar_loops(
    loops: PlanarLoops,
    ax=None,
    show: bool = True,
    color=None,
    linewidth: float = 2.0,
    linestyle: str = "-",
    alpha: float = 1.0,
    label: Optional[str] = None,
    show_vertices: bool = False,
    show_orientation: bool = False,
    image_orientation: bool = False,
    figsize: tuple[float, float] = (5, 5),
    zorder: float = 6,
):
    """
    Draw loops as closed polylines in physical coordinates.

    Parameters
    ----------
    loops : PlanarLoops
        Loops from ``persforest.geometry.loops``, e.g. ``taut_loops``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; new equal-aspect axes when None. Limits of given
        axes are left to their owner.
    show : bool
        If True, call ``plt.show()`` after plotting.
    color : color, optional
        One color for all loops; the next color of the axes cycle when None.
    linewidth, linestyle, alpha : optional
        Line style of the loops.
    label : str, optional
        Legend label, attached to the first loop only.
    show_vertices : bool
        Mark the loop vertices, e.g. to see the samples of smoothed loops.
    show_orientation : bool
        Draw an arrowhead on each loop in its direction of travel.
    image_orientation : bool
        Draw array axis 0 downward and axis 1 to the right, matching cubical
        plots with ``image_orientation=True``.
    figsize : tuple of float
        Figure size for new axes.
    zorder : float
        Drawing order; the default lies above cubical cycles.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not isinstance(loops, PlanarLoops):
        raise TypeError("loops must be PlanarLoops; build them with persforest.geometry.loops, e.g. raw_loops(chain, forest.geometry)")
    new_axes = ax is None
    if new_axes:
        _, ax = plt.subplots(figsize=figsize)

    for index, loop in enumerate(loops):
        closed = display(np.vstack((loop, loop[:1])), image_orientation)
        [line] = ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth, linestyle=linestyle,
                         alpha=alpha, label=label if index == 0 else None, zorder=zorder,
                         marker="o" if show_vertices else None, markersize=2.5)
        color = line.get_color()
        if show_orientation and len(loop) > 1:
            ax.annotate("", xy=closed[1], xytext=closed[0], zorder=zorder + 1,
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": linewidth,
                                    "alpha": alpha, "mutation_scale": 6 * linewidth + 6})

    if new_axes:
        ax.set_aspect("equal", adjustable="datalim")
        if image_orientation:
            ax.invert_yaxis()
    if show:
        plt.show()
    return ax
