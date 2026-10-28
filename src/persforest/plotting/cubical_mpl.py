"""Matplotlib renderers for cycles and filtrations of 2D and 3D cubical grids.

Cells are drawn at their physical position in ``forest.geometry``. In 2D,
array axis 0 runs along x and axis 1 along y by default, the frame of the cell
orientations. ``image_orientation=True`` instead draws rows (axis 0) downward
and columns (axis 1) to the right, as an image viewer would; this reflects the
picture, so counterclockwise cycles appear clockwise. In 3D, axes 0, 1 and 2
run along x, y and z.
"""

from __future__ import annotations

from ._dependencies import require_plotting

require_plotting()

from typing import Any
from typing import Literal
from typing import Optional

from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from persforest.geometry.cubical import CubicalGeometry
from persforest.plotting import _cubical_scene as scene
from persforest.plotting._geometry import _camera_from_eye
from persforest.plotting._scene import _activity_rows_for_plotting
from persforest.plotting.interior_mpl import _activity_conflict
from persforest.plotting.interior_mpl import _activity_mask_field
from persforest.plotting.interior_mpl import _compose_activity_image
from persforest.plotting.interior_mpl import _gaussian_blur

_FILTRATION_STYLE_2D = {
    **scene.FLAT_FILTRATION_STYLE,
    "domain_edge_color": "0.3",
    "domain_edge_width": 0.6,
    "cycle_edge_width": 1.8,
    "show_orientation_arrows": False,
    "arrow_linewidth": 0.8,
    "arrow_scale": 4.0,
}

_FILTRATION_STYLE_3D = {
    "camera_eye": None,
    "remove_axes": False,
    "complex_color": "#add8e6",
    "complex_face_alpha": 0.2,
    "complex_edge_color": "0.35",
    "complex_edge_width": 0.3,
    "complex_edge_alpha": 0.15,
    "cycle_face_alpha": 0.55,
    "cycle_edge_color": None,
    "cycle_edge_width": 0.6,
    "cycle_edge_alpha": None,
    "antialiased": True,
    "zsort": "average",
}

_INTERIOR_STYLE_2D = {
    "complex_face_color": "0.5",
    "complex_face_alpha": 0.15,
    "domain_edge_color": "0.3",
    "domain_edge_width": 0.6,
    "background_color": "white",
    "remove_axes": True,
    "activity_alpha_range": (0.15, 0.95),
}

_INTERIOR_STYLE_3D = {
    "activity_edge_color": "white",
    "activity_edge_width": 0.25,
    "activity_edge_alpha": 0.8,
    "complex_edge_color": "0.25",
    "complex_edge_width": 0.35,
    "complex_edge_alpha": 0.25,
    "background_color": "white",
    "remove_axes": True,
    "activity_alpha_range": (0.15, 0.95),
    "cell_shrink": 1.0,
    "shade": True,
    "lightsource": None,
    "antialiased": True,
    "zsort": "average",
    "camera_eye": None,
}

_GRADIENT_STYLE = {
    "background_color": "white",
    "remove_axes": True,
    "domain_edge_color": "0.3",
    "domain_edge_width": 0.0,
    "resolution": 650,
    "blur_sigma": 7.0,
    "boundary_blur_sigma": 5.0,
    "intensity_gamma": 0.9,
    "activity_scale_fraction_of_max_bar": 0.75,
    "activity_scale_percentile": 95,
    "different_color_threshold": 0.18,
    "conflict_whitening": 0.85,
}


# ----- 2D helpers -------------------------------------------------------------

def _show_array(ax, data: NDArray, lower: NDArray, upper: NDArray, image_orientation: bool,
                zorder: float, interpolation: str = "nearest", **imshow_kwargs):
    """Draw an array indexed by array axes (values or RGBA) over the physical box ``[lower, upper]``."""
    (x0, y0), (x1, y1) = scene.display(lower, image_orientation), scene.display(upper, image_orientation)
    # imshow rows run along y: axis 0 in image orientation, axis 1 otherwise.
    data = data if image_orientation else np.swapaxes(data, 0, 1)
    return ax.imshow(data, origin="lower", extent=(x0, x1, y0, y1), interpolation=interpolation,
                     zorder=zorder, **imshow_kwargs)


def _attach_colorbar(ax, mappable, label: str):
    """Add a colorbar matching the height of ``ax``, reused when ``ax`` is redrawn.

    Animations clear and redraw the same axes every frame; reusing the colorbar
    keeps them from stacking a new one each time.
    """
    colorbar = getattr(ax, "_persforest_colorbar", None)
    if colorbar is not None and colorbar.ax in ax.figure.axes:
        colorbar.update_normal(mappable)
    else:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
        colorbar = ax.figure.colorbar(mappable, cax=cax)
        ax._persforest_colorbar = colorbar
    colorbar.set_label(label)
    return colorbar


def _draw_value_image(ax, forest, geometry, style: dict[str, Any], image_orientation: bool):
    """Draw ``grid_values`` with ``image_cmap`` over the full value range, with a colorbar."""
    values = np.asarray(forest.grid_values, dtype=float)
    lower, upper = scene.value_image_bounds(geometry)
    image = _show_array(ax, values, lower, upper, image_orientation, zorder=0.5,
                        cmap=style["image_cmap"], vmin=float(values.min()), vmax=float(values.max()),
                        alpha=float(style["image_alpha"]))
    if bool(style["image_colorbar"]):
        _attach_colorbar(ax, image, style["colorbar_label"])
    return image


def _mask_rgba(mask: NDArray, color, alpha: float) -> NDArray[np.float64]:
    canvas = np.zeros(mask.shape + (4,))
    canvas[mask] = mcolors.to_rgba(color, float(alpha))
    return canvas


def _draw_complex_2d(ax, forest, geometry, max_value: Optional[float], style: dict[str, Any],
                     image_orientation: bool, show_image: bool) -> None:
    """Draw the value image and/or the top cells present at ``max_value``.

    Without the image, present cells are tinted. Over the image, cells not yet
    present are veiled in white instead, so present cells keep their true
    intensities.
    """
    lower, upper = scene.domain_bounds(geometry)
    if show_image:
        _draw_value_image(ax, forest, geometry, style, image_orientation)
    if max_value is None and show_image:
        return
    present = scene.present_top_cells(forest, max_value)
    if show_image:
        rgba = _mask_rgba(~present, "white", style["absent_veil_alpha"])
    else:
        rgba = _mask_rgba(present, style["complex_face_color"], style["complex_face_alpha"])
    _show_array(ax, rgba, lower, upper, image_orientation, zorder=1)


def _draw_orientation_arrows(ax, segments: NDArray, color, style: dict[str, Any], zorder: float) -> None:
    for start, end in segments:
        midpoint, quarter = (start + end) / 2, (end - start) / 4
        arrow = ax.annotate(
            "",
            xy=midpoint + quarter,
            xytext=midpoint - quarter,
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=float(style["arrow_linewidth"]),
                color=color,
                mutation_scale=float(style["arrow_scale"]),
            ),
            zorder=zorder,
        )
        if arrow.arrow_patch is not None:
            arrow.arrow_patch.set_zorder(zorder)


def _finish_axes_2d(ax, geometry: CubicalGeometry, image_orientation: bool, style: dict[str, Any],
                    show_image: bool = False) -> None:
    """Outline the grid, fix limits and aspect, and label the array axes."""
    lower, upper = (scene.value_image_bounds if show_image else scene.domain_bounds)(geometry)
    (x0, y0), (x1, y1) = scene.display(lower, image_orientation), scene.display(upper, image_orientation)
    if float(style["domain_edge_width"]) > 0:
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False,
            edgecolor=style["domain_edge_color"], linewidth=float(style["domain_edge_width"]), zorder=0,
            clip_on=False))
    # The axes end at the grid; cycles drawn unclipped stay whole on its boundary.
    ax.set_xlim(x0, x1)
    ax.set_ylim((y1, y0) if image_orientation else (y0, y1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("axis 1" if image_orientation else "axis 0")
    ax.set_ylabel("axis 0" if image_orientation else "axis 1")


def _hide_axes_2d(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_cycle_2d(ax, cycle, color, geometry, image_orientation, linewidth, capstyle,
                   arrows: bool, style: dict[str, Any], zorder: float) -> None:
    segments = scene.display(scene.edge_segments(cycle, geometry), image_orientation)
    ax.add_collection(LineCollection(
        segments, linewidths=linewidth, colors=[color],
        capstyle=capstyle, joinstyle="round", zorder=zorder, clip_on=False))
    if arrows:
        _draw_orientation_arrows(ax, segments, color, style, zorder=zorder + 1)


# ----- 3D helpers -------------------------------------------------------------

def _new_3d_axes(ax, figsize, dpi):
    if ax is None:
        figure = plt.figure(figsize=figsize, dpi=dpi)
        return figure.add_subplot(111, projection="3d")
    if not hasattr(ax, "zaxis"):
        raise ValueError("3D plotting requires a Matplotlib 3D axis (projection='3d').")
    return ax


def _finish_axes_3d(ax, geometry: CubicalGeometry, camera_eye, remove_axes: bool) -> None:
    lower, upper = scene.domain_bounds(geometry)
    pad = 0.03 * float(np.max(upper - lower))
    limits = [(float(lower[a] - pad), float(upper[a] + pad)) for a in range(3)]
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect(tuple(high - low for low, high in limits))
    elevation, azimuth = _camera_from_eye(camera_eye)
    ax.view_init(elev=elevation, azim=azimuth)
    if remove_axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    else:
        ax.set_xlabel("axis 0")
        ax.set_ylabel("axis 1")
        ax.set_zlabel("axis 2")


# ----- filtration snapshots ---------------------------------------------------

def plot_cubical_at_filtration(
    forest,
    filt_val: float,
    ax=None,
    show: bool = True,
    show_complex: bool = True,
    figsize: tuple[float, float] = (5, 5),
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_cycles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0,
    cycle_zorder: float = 5,
    dpi: int = 100,
    style_2d: Optional[dict[str, Any]] = None,
    style_3d: Optional[dict[str, Any]] = None,
    image_orientation: bool = False,
    show_image: Optional[bool] = None,
):
    """
    Plot a cubical filtration and its active cycles at a filtration value.

    Target of ``PersistenceForest.plot_at_filtration`` for cubical forests.

    Parameters
    ----------
    forest : PersistenceForest
        Forest built with ``from_cubical_grid`` from a 2D or 3D array.
    filt_val : float
        Filtration threshold.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; if None, a new figure and axes are created. In 3D a
        supplied axes must use ``projection="3d"``.
    show : bool
        If True, call ``plt.show()`` after plotting.
    show_complex : bool
        2D: tint the pixels with value at most ``filt_val``, or with
        ``show_image``, veil the pixels not yet present. 3D: draw the boundary
        surface of the voxels present. Lower-dimensional cells already present
        are not drawn separately.
    figsize : tuple[float, float]
        Figure size used when ``ax`` is None.
    coloring : {"forest", "bars"}
        Bar color map for the cycles.
    title : str or None
        Axes title. Defaults to a filtration summary.
    show_cycles : bool
        Whether to draw the active cycle representatives.
    signed : bool
        If False, cancel cells that occur with both orientations.
    min_bar_length : float
        Only draw cycles of bars with lifespan at least this value.
    cycle_zorder : float
        Matplotlib z-order of cycle edges (2D only).
    dpi : int
        DPI used when a new figure is created.
    style_2d : dict or None
        2D style overrides: ``image_cmap``, ``image_alpha``,
        ``image_colorbar`` (with ``show_image``, default True),
        ``colorbar_label``, ``complex_face_color``, ``complex_face_alpha``,
        ``absent_veil_alpha``, ``domain_edge_color``, ``domain_edge_width``,
        ``cycle_edge_width``, ``show_orientation_arrows``,
        ``arrow_linewidth`` and ``arrow_scale``. Ignored in 3D.
    style_3d : dict or None
        3D style overrides: ``camera_eye``, ``remove_axes``,
        ``complex_color``, ``complex_face_alpha``, ``complex_edge_color``,
        ``complex_edge_width``, ``complex_edge_alpha``,
        ``cycle_face_alpha``, ``cycle_edge_color``, ``cycle_edge_width``,
        ``cycle_edge_alpha``, ``antialiased`` and ``zsort``. Ignored in 2D.
    image_orientation : bool
        2D only: draw axis 0 downward and axis 1 to the right.
    show_image : bool or None
        2D only: draw all grid values in grayscale beneath the cycles, with a
        colorbar of the values. None (default) shows them on 2D grids; with
        ``show_complex`` the pixels not yet present are veiled.

    Returns
    -------
    matplotlib.axes.Axes
    """
    geometry = scene.require_grid_dimension(forest, "plot_at_filtration")
    show_image = scene.resolve_flat_options(geometry, image_orientation, show_image)
    color_map = forest._get_color_map(coloring=coloring)
    active = forest._active_bars_with_cycles_at(filt_val, min_bar_length=min_bar_length) if show_cycles else []

    if geometry.dim == 2:
        style = scene.resolve_style(style_2d, _FILTRATION_STYLE_2D, "style_2d")
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if show_complex or show_image:
            _draw_complex_2d(ax, forest, geometry, filt_val if show_complex else None, style,
                             image_orientation, show_image)
        for bar, cycle in active:
            _draw_cycle_2d(ax, cycle if signed else cycle.unsigned(), color_map[bar], geometry,
                           image_orientation, float(style["cycle_edge_width"]), "butt",
                           bool(style["show_orientation_arrows"]), style, cycle_zorder)
        _finish_axes_2d(ax, geometry, image_orientation, style, show_image)
    else:
        style = scene.resolve_style(style_3d, _FILTRATION_STYLE_3D, "style_3d")
        ax = _new_3d_axes(ax, figsize, dpi)
        if show_complex:
            quads = scene.present_voxel_quads(forest, geometry, filt_val)
            if len(quads):
                ax.add_collection3d(Poly3DCollection(
                    quads,
                    facecolors=mcolors.to_rgba(style["complex_color"], float(style["complex_face_alpha"])),
                    edgecolors=mcolors.to_rgba(style["complex_edge_color"], float(style["complex_edge_alpha"])),
                    linewidths=float(style["complex_edge_width"]),
                    antialiased=bool(style["antialiased"]),
                    zsort=style["zsort"],
                ))
        cycle_face_alpha = float(style["cycle_face_alpha"])
        cycle_edge_alpha = style["cycle_edge_alpha"]
        if cycle_edge_alpha is None:
            cycle_edge_alpha = min(1.0, cycle_face_alpha + 0.25)
        for bar, cycle in sorted(active, key=lambda item: item[0].lifespan(), reverse=True):
            quads = scene.chain_face_quads(cycle if signed else cycle.unsigned(), geometry)
            if not len(quads):
                continue
            color = color_map[bar]
            edge_color = color if style["cycle_edge_color"] is None else style["cycle_edge_color"]
            ax.add_collection3d(Poly3DCollection(
                quads,
                facecolors=mcolors.to_rgba(color, cycle_face_alpha),
                edgecolors=mcolors.to_rgba(edge_color, float(cycle_edge_alpha)),
                linewidths=float(style["cycle_edge_width"]),
                antialiased=bool(style["antialiased"]),
                zsort=style["zsort"],
            ))
        _finish_axes_3d(ax, geometry, style["camera_eye"], bool(style["remove_axes"]))

    ax.set_title(fr"Filtration at value {filt_val:.4g}" if title is None else title)
    if show:
        plt.show()
    return ax


def plot_cubical_barcode_cycle_reps(
    forest,
    min_bar_length: float = 0,
    relative_position: float = 0.1,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (7, 7),
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_orientation_arrows: bool = False,
    remove_double_edges: bool = False,
    linewidth_cycle: float = 0.8,
    style_2d: Optional[dict[str, Any]] = None,
    image_orientation: bool = False,
    show_image: Optional[bool] = None,
):
    """
    Plot one representative of each sufficiently long bar of a 2D cubical forest.

    The representative of each bar is sampled at
    ``birth + lifespan * relative_position``; longer bars are drawn on top.
    Target of ``PersistenceForest.plot_barcode_cycle_reps`` for cubical
    forests. Parameters match ``plot_cubical_at_filtration`` where shared;
    ``style_2d["cycle_edge_width"]`` takes precedence over ``linewidth_cycle``.
    ``show_image`` draws all grid values in grayscale beneath the cycles;
    None (default) shows them.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if relative_position < 0 or relative_position > 1:
        raise ValueError("relative_position must be in [0,1]")
    geometry = scene.require_grid_dimension(forest, "plot_barcode_cycle_reps", supported=(2,))
    show_image = scene.resolve_flat_options(geometry, image_orientation, show_image)
    style = scene.resolve_style(style_2d, _FILTRATION_STYLE_2D, "style_2d")
    cycle_linewidth = (
        float(style["cycle_edge_width"])
        if style_2d is not None and "cycle_edge_width" in style_2d
        else linewidth_cycle
    )
    draw_orientation_arrows = show_orientation_arrows or bool(style["show_orientation_arrows"])
    color_map = forest._get_color_map(coloring=coloring)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if show_image:
        _draw_complex_2d(ax, forest, geometry, None, style, image_orientation, show_image=True)

    for bar in forest.barcode:
        if bar.lifespan() < min_bar_length:
            continue
        cycle = forest.cycle_for_bar_at(bar, bar.birth + bar.lifespan() * relative_position)
        # Longer bars on top, above the value image (z-order 0.5) and the outline.
        _draw_cycle_2d(ax, cycle.unsigned() if remove_double_edges else cycle, color_map[bar],
                       geometry, image_orientation, cycle_linewidth, "round",
                       draw_orientation_arrows, style, zorder=2 + bar.lifespan())

    _finish_axes_2d(ax, geometry, image_orientation, style, show_image)
    if title is None:
        title = fr"Barcode cycle representatives at relative position {relative_position:.2f}"
    ax.set_title(title)
    if show:
        plt.show()
    return ax


# ----- interior activity ------------------------------------------------------

def _activity_alpha(length, max_length: float, alpha_range):
    """Opacity growing linearly with activity length; accepts scalars or arrays."""
    alpha_min, alpha_max = alpha_range
    return alpha_min + (alpha_max - alpha_min) * np.clip(np.asarray(length) / max_length, 0.0, 1.0)


def plot_cubical_interior_activity(
    forest,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
    coloring: Literal["forest", "bars"] = "forest",
    show_complex: bool = False,
    complex_max_filtration: float | None = None,
    overlap: Literal["longest", "layer"] = "longest",
    min_activity_length: float = 0.0,
    style: dict[str, Any] | None = None,
    geometry: Literal["cells", "surface"] = "cells",
    image_orientation: bool = False,
):
    """
    Color the pixels or voxels of a cubical forest by interior activity.

    A cell takes the color of a bar while it lies in the interior of that
    bar's representative; opacity grows with the activity length. Target of
    ``PersistenceForest.plot_interior_simplex_activity`` for cubical forests.

    Parameters
    ----------
    show_complex : bool
        2D: shade the pixels with value at most ``complex_max_filtration`` (all
        pixels if None) beneath the activity. 3D: wireframe of the boundary of
        those voxels.
    overlap : {"longest", "layer"}
        Whether a multiply active cell shows its longest interval or all
        intervals, longer ones on top in 2D.
    style : dict or None
        2D keys: ``complex_face_color``, ``complex_face_alpha``,
        ``domain_edge_color``, ``domain_edge_width``, ``background_color``,
        ``remove_axes`` and ``activity_alpha_range``. 3D keys:
        ``activity_edge_color``, ``activity_edge_width``,
        ``activity_edge_alpha``, ``complex_edge_color``,
        ``complex_edge_width``, ``complex_edge_alpha``, ``background_color``,
        ``remove_axes``, ``activity_alpha_range``, ``cell_shrink``, ``shade``,
        ``lightsource``, ``antialiased``, ``zsort`` and ``camera_eye``.
    geometry : {"cells", "surface"}
        3D only: draw every active voxel, or the boundary of each bar's region.
    image_orientation : bool
        2D only; see ``plot_cubical_at_filtration``.

    Other parameters match ``plot_interior_simplex_activity``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    grid = scene.require_grid_dimension(forest, "plot_interior_simplex_activity")
    scene.resolve_flat_options(grid, image_orientation)
    if geometry not in ("cells", "surface"):
        raise ValueError("geometry must be 'cells' or 'surface'.")
    rows = _activity_rows_for_plotting(forest, overlap, min_activity_length)
    color_map = forest._get_color_map(coloring=coloring)
    max_activity_length = max([length for _, _, length in rows] + [np.finfo(float).eps])

    if grid.dim == 3:
        return _plot_interior_activity_3d(
            forest, grid, rows, color_map, max_activity_length, ax, show, figsize, dpi,
            show_complex, complex_max_filtration, style, geometry)

    plot_style = scene.resolve_style(style, _INTERIOR_STYLE_2D, "style")
    canvas = np.zeros(scene.top_cell_shape(grid) + (4,))
    # Rows are ordered so that "layer" composites longer intervals over shorter
    # ones; with "longest" every pixel occurs once and "over" reduces to assignment.
    for cell, bar, activity_length in rows:
        index = scene.top_cell_index(cell)
        alpha = float(_activity_alpha(activity_length, max_activity_length, plot_style["activity_alpha_range"]))
        color = np.asarray(mcolors.to_rgb(color_map[bar]))
        below_rgb, below_alpha = canvas[index][:3], canvas[index][3]
        out_alpha = alpha + below_alpha * (1 - alpha)
        canvas[index][:3] = (color * alpha + below_rgb * below_alpha * (1 - alpha)) / out_alpha
        canvas[index][3] = out_alpha

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(plot_style["background_color"])
    if show_complex:
        _draw_complex_2d(ax, forest, grid, complex_max_filtration, plot_style, image_orientation,
                         show_image=False)
    lower, upper = scene.domain_bounds(grid)
    _show_array(ax, canvas, lower, upper, image_orientation, zorder=2)

    _finish_axes_2d(ax, grid, image_orientation, plot_style)
    if bool(plot_style["remove_axes"]):
        _hide_axes_2d(ax)
    if show:
        plt.show()
    return ax


def _plot_interior_activity_3d(forest, grid, rows, color_map, max_activity_length, ax, show, figsize,
                               dpi, show_complex, complex_max_filtration, style, geometry):
    plot_style = scene.resolve_style(style, _INTERIOR_STYLE_3D, "style")
    cell_shrink = float(plot_style["cell_shrink"])
    if not 0.0 < cell_shrink <= 1.0:
        raise ValueError("cell_shrink must be greater than 0 and at most 1.")
    ax = _new_3d_axes(ax, figsize, dpi)
    ax.set_facecolor(plot_style["background_color"])
    ax.figure.patch.set_facecolor(plot_style["background_color"])

    polygons, face_colors = [], []
    for bar, quads, lengths, _ in scene.activity_voxel_quads(rows, grid, geometry == "surface", cell_shrink):
        rgba = np.tile(mcolors.to_rgba(color_map[bar]), (len(quads), 1))
        rgba[:, 3] = _activity_alpha(lengths, max_activity_length, plot_style["activity_alpha_range"])
        polygons.append(quads)
        face_colors.append(rgba)
    if polygons:
        ax.add_collection3d(Poly3DCollection(
            np.concatenate(polygons),
            facecolors=np.concatenate(face_colors),
            edgecolors=mcolors.to_rgba(plot_style["activity_edge_color"],
                                       float(plot_style["activity_edge_alpha"])),
            linewidths=float(plot_style["activity_edge_width"]),
            antialiased=bool(plot_style["antialiased"]),
            zsort=plot_style["zsort"],
            shade=bool(plot_style["shade"]),
            lightsource=plot_style["lightsource"],
        ))
    if show_complex:
        quads = scene.present_voxel_quads(forest, grid, complex_max_filtration)
        if len(quads):
            ax.add_collection3d(Line3DCollection(
                scene.quad_outlines(quads),
                colors=plot_style["complex_edge_color"],
                linewidths=float(plot_style["complex_edge_width"]),
                alpha=float(plot_style["complex_edge_alpha"]),
                antialiased=bool(plot_style["antialiased"]),
            ))
    _finish_axes_3d(ax, grid, plot_style["camera_eye"], bool(plot_style["remove_axes"]))
    if show:
        plt.show()
    return ax


def _stream_activity_image(rows, shape: tuple[int, int], color_map, activity_scale: float,
                           plot_style: dict[str, Any]) -> NDArray[np.float64]:
    """Blend per-bar activity on the pixel grid, one bar at a time.

    This computes the point-cloud blend of ``_blend_activity_fields`` without a
    raster per bar: each bar is blurred only within its bounding box widened by
    the blur radius, and the sums and two strongest fields are accumulated.
    Blur radii, given in raster pixels of a ``resolution``-pixel raster along
    the longer axis, are converted to grid pixels; the bilinear display then
    smooths the result.
    """
    pixels_per_cell = float(plot_style["resolution"]) / max(shape)
    sigma = float(plot_style["blur_sigma"]) / pixels_per_cell
    radius = max(1, int(np.ceil(3.0 * sigma))) if sigma > 0 else 0
    gamma = float(plot_style["intensity_gamma"])

    by_bar = scene.rows_by_bar(rows)

    weight_sum = np.zeros(shape)
    weighted_rgb = np.zeros(shape + (3,))
    first, second = np.zeros(shape), np.zeros(shape)
    first_rgb, second_rgb = np.zeros(shape + (3,)), np.zeros(shape + (3,))
    active_mask = np.zeros(shape)
    for bar, entries in by_bar.items():
        index = np.array([position for position, _ in entries])
        low = np.maximum(index.min(axis=0) - radius, 0)
        high = np.minimum(index.max(axis=0) + radius + 1, shape)
        window = tuple(slice(lo, hi) for lo, hi in zip(low, high))
        field = np.zeros(tuple(high - low))
        field[tuple((index - low).T)] = [length for _, length in entries]
        active_mask[tuple(index.T)] = 1.0
        # Zero padding at the window edge matches zero padding of the full grid,
        # since the window extends a blur radius past the bar's cells.
        weight = np.clip(_gaussian_blur(field, sigma) / activity_scale, 0.0, 1.0) ** gamma
        rgb = np.asarray(mcolors.to_rgb(color_map[bar]))

        weight_sum[window] += weight
        weighted_rgb[window] += weight[..., None] * rgb
        top, runner_up = first[window], second[window]
        top_rgb, runner_up_rgb = first_rgb[window], second_rgb[window]
        new_top = weight > top
        new_runner_up = ~new_top & (weight > runner_up)
        second[window] = np.where(new_top, top, np.maximum(runner_up, weight))
        second_rgb[window] = np.where(new_top[..., None], top_rgb,
                                      np.where(new_runner_up[..., None], rgb, runner_up_rgb))
        first[window] = np.where(new_top, weight, top)
        first_rgb[window] = np.where(new_top[..., None], rgb, top_rgb)

    mask_field = _activity_mask_field(active_mask, float(plot_style["boundary_blur_sigma"]) / pixels_per_cell)
    conflict = (_activity_conflict(first, second, first_rgb, second_rgb, plot_style)
                if len(by_bar) > 1 else np.zeros(shape))
    return _compose_activity_image(weight_sum, weighted_rgb, mask_field, conflict, plot_style)


def plot_cubical_interior_activity_gradient(
    forest,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
    coloring: Literal["forest", "bars"] = "forest",
    title: str | None = None,
    min_activity_length: float = 0.0,
    style: dict[str, Any] | None = None,
    image_orientation: bool = False,
):
    """
    Plot a smooth color field of interior activity on a 2D cubical grid.

    Each pixel carries the longest activity of its bar, as in the point-cloud
    version, and the per-bar fields are blurred and blended on the pixel grid,
    then displayed with bilinear interpolation. Blur radii keep their
    point-cloud meaning, raster pixels of a ``resolution``-pixel raster along
    the longer axis, and are converted to grid pixels. Target of
    ``PersistenceForest.plot_interior_simplex_activity_gradient`` for cubical
    forests.

    Parameters
    ----------
    style : dict or None
        ``background_color``, ``remove_axes``, ``domain_edge_color``,
        ``domain_edge_width`` and the raster and blending keys of the
        point-cloud gradient: ``resolution``, ``blur_sigma``,
        ``boundary_blur_sigma``, ``intensity_gamma``,
        ``activity_scale_fraction_of_max_bar``, ``activity_scale_percentile``,
        ``different_color_threshold`` and ``conflict_whitening``.
    image_orientation : bool
        See ``plot_cubical_at_filtration``.

    Other parameters match ``plot_interior_simplex_activity_gradient``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    grid = scene.require_grid_dimension(
        forest, "plot_interior_simplex_activity_gradient", supported=(2,))
    plot_style = scene.resolve_style(style, _GRADIENT_STYLE, "style")
    rows = _activity_rows_for_plotting(forest, "longest", min_activity_length)
    color_map = forest._get_color_map(coloring=coloring)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(plot_style["background_color"])

    if rows:
        lengths = np.array([length for _, _, length in rows])
        activity_scale = max(
            float(forest.max_bar().lifespan()) * float(plot_style["activity_scale_fraction_of_max_bar"]),
            float(np.percentile(lengths, float(plot_style["activity_scale_percentile"]))),
            np.finfo(float).eps,
        )
        image = _stream_activity_image(rows, scene.top_cell_shape(grid), color_map, activity_scale, plot_style)
        rgba = np.concatenate([image, np.ones(image.shape[:2] + (1,))], axis=2)
        lower, upper = scene.domain_bounds(grid)
        _show_array(ax, rgba, lower, upper, image_orientation, zorder=1, interpolation="bilinear")

    _finish_axes_2d(ax, grid, image_orientation, plot_style)
    if title is not None:
        ax.set_title(title)
    if bool(plot_style["remove_axes"]):
        _hide_axes_2d(ax)
    if show:
        plt.show()
    return ax
