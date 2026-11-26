"""Reuse Matplotlib artists across frames of a 3D point-cloud animation."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..filtration_mpl import (
    _complex_collections_3d,
    _cycle_collection_3d,
    _filtration_title_3d,
    _resolve_style_3d,
)

if TYPE_CHECKING:
    from persforest.barcode import PFBar
    from persforest.forest import PersistenceForest


def _point_cloud_frame_drawer(
    forest: "PersistenceForest", ax: Axes3D, plot_kwargs: dict[str, Any],
) -> Callable[[float, tuple[float, float] | None], None]:
    """Draw successive frames while retaining only the current mesh and cycles."""
    points = np.asarray(forest.point_cloud, dtype=float)
    style = _resolve_style_3d(plot_kwargs.get("style_3d"))
    show_complex = plot_kwargs["show_complex"]
    signed = plot_kwargs.get("signed", False)
    min_bar_length = plot_kwargs.get("min_bar_length", 0)
    bars = []
    if plot_kwargs.get("show_cycles", True):
        bars = sorted(
            (bar for bar in forest.barcode if not bar.lifespan() < min_bar_length),
            key=lambda bar: bar.lifespan(), reverse=True,
        )
    color_map = None
    point_artist = None
    fixed_view = None
    complex_stop = None
    complex_collections = []
    cycles: dict["PFBar", tuple[float, float, Poly3DCollection | None]] = {}

    def draw(filt_val: float, camera_eye: tuple[float, float] | None) -> None:
        nonlocal color_map, point_artist, fixed_view, complex_stop, cycles
        if point_artist is None:
            # Use the public plotter once for option validation, scatter, and axes setup.
            initial_kwargs = dict(plot_kwargs, show_complex=False, show_cycles=False)
            if camera_eye is not None:
                initial_kwargs["style_3d"] = {**style, "camera_eye": camera_eye}
            forest.plot_at_filtration(filt_val=filt_val, ax=ax, **initial_kwargs)
            point_artist, = ax.collections
            fixed_view = ax.elev, ax.azim
            if bars:
                color_map = forest._get_color_map(coloring=plot_kwargs["coloring"])

        if show_complex:
            stop = np.searchsorted(forest._cells.event_values, filt_val, side="right")
            if stop != complex_stop:
                while complex_collections:
                    complex_collections.pop().remove()
                complex_collections.extend(_complex_collections_3d(forest, filt_val, points, style))
                complex_stop = stop

        active_cycles = {}
        for bar in bars:
            if not bar.birth <= filt_val < bar.death:
                continue
            entry = cycles.get(bar)
            if entry is None or not entry[0] <= filt_val < entry[1]:
                cycle = forest.cycle_for_bar_at(bar, filt_val)
                surface = _cycle_collection_3d(cycle, points, color_map[bar], style, signed=signed)
                entry = cycle.active_start, cycle.active_end, surface
            active_cycles[bar] = entry
        cycles = active_cycles

        # Match the static plot's order when projected depths tie.
        ordered = [*complex_collections, point_artist]
        ordered.extend(surface for _, _, surface in cycles.values() if surface is not None)
        current = list(ax.collections)
        if current != ordered:
            for collection in current:
                collection.remove()
            for collection in ordered:
                ax.add_collection(collection, autolim=False)
        ax.view_init(*(fixed_view if camera_eye is None else camera_eye))
        if plot_kwargs.get("title") is None:
            ax.set_title(_filtration_title_3d(forest, filt_val))

    return draw
