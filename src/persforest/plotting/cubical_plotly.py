"""Plotly renderers for cycles and filtrations of 2D and 3D cubical grids.

Coordinates follow ``persforest.plotting.cubical_mpl``: array axes in order
along x, y (and z), or in 2D with ``image_orientation=True``, rows downward
and columns to the right.
"""

from __future__ import annotations

from ._dependencies import require_plotting

require_plotting()

from typing import Any
from typing import Literal
from typing import Optional

import matplotlib
import numpy as np
from numpy.typing import NDArray

from persforest.plotting import _cubical_scene as scene
from persforest.plotting._scene import _activity_rows_for_plotting
from persforest.plotting.interior_plotly import _bar_value
from persforest.plotting.interior_plotly import _darken_color
from persforest.plotting.interior_plotly import _mesh_trace_for_bar
from persforest.plotting.interior_plotly import _require_plotly
from persforest.plotting.interior_plotly import _rgba_css

_COMPLEX_COLOR_3D = "#add8e6"
_CYCLE_WIDTH_2D = 2.5
_OUTLINE_COLOR_2D = "rgba(60, 60, 60, 0.8)"

_INTERIOR_STYLE = {
    "activity_alpha_range": (0.35, 0.95),
    "show_activity_edges": False,
    "activity_edge_color": None,
    "activity_edge_width": 0.75,
    "activity_edge_alpha": 0.35,
    "complex_edge_color": "0.25",
    "complex_edge_width": 1.0,
    "complex_edge_alpha": 0.25,
    "background_color": "white",
    "remove_axes": True,
    "flatshading": True,
    "lighting": {"ambient": 0.45, "diffuse": 0.8, "roughness": 0.8, "specular": 0.2, "fresnel": 0.1},
    "lightposition": {"x": 100.0, "y": 200.0, "z": 100.0},
    "camera_eye": {"x": 1.45, "y": 1.45, "z": 1.15},
}


def _colorscale(cmap, samples: int = 11) -> list[list[Any]]:
    colormap = matplotlib.colormaps.get_cmap(cmap)
    return [[float(t), _rgba_css(colormap(float(t)), 1.0)] for t in np.linspace(0, 1, samples)]


def _heatmap(go, array: NDArray, lower: NDArray, upper: NDArray, image_orientation: bool, **kwargs):
    """Heatmap of an array indexed by array axes over the physical box ``[lower, upper]``."""
    centers = [lower[a] + (np.arange(n) + 0.5) * (upper[a] - lower[a]) / n for a, n in enumerate(array.shape)]
    if image_orientation:
        return go.Heatmap(z=array, x=centers[1], y=centers[0], **kwargs)
    return go.Heatmap(z=array.T, x=centers[0], y=centers[1], **kwargs)


def _mask_heatmap(go, mask: NDArray, geometry, image_orientation: bool, color, opacity: float, name: str):
    lower, upper = scene.domain_bounds(geometry)
    return _heatmap(
        go, np.where(mask, 1.0, np.nan), lower, upper, image_orientation,
        colorscale=[[0, _rgba_css(color, 1.0)], [1, _rgba_css(color, 1.0)]],
        zmin=0, zmax=1, opacity=float(opacity), showscale=False, hoverinfo="skip", name=name)


def _empty_mesh(go, opacity: float):
    return go.Mesh3d(x=[np.nan], y=[np.nan], z=[np.nan], i=[], j=[], k=[], opacity=opacity,
                     color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip")


def _quad_mesh(go, quads: NDArray, **kwargs):
    if not len(quads):
        return _empty_mesh(go, kwargs.get("opacity", 1.0))
    triangles = scene.quads_to_triangles(quads)
    x, y, z = triangles.reshape(-1, 3).T
    first = np.arange(0, 3 * len(triangles), 3)
    return go.Mesh3d(x=x, y=y, z=z, i=first, j=first + 1, k=first + 2, flatshading=True, **kwargs)


def _line_coordinates(segments: NDArray) -> NDArray:
    """Coordinates ``(k, 3n)`` tracing ``(n, 2, k)`` segments, separated by NaN gaps."""
    gaps = np.full((len(segments), 1, segments.shape[-1]), np.nan)
    return np.concatenate([segments, gaps], axis=1).reshape(-1, segments.shape[-1]).T


def _bar_hover(bar) -> str:
    return (f"birth={bar.birth:.4g}<br>death={bar.death:.4g}<br>"
            f"lifespan={bar.lifespan():.4g}<extra></extra>")


def _cycle_trace(go, geometry, bar, cycle, color_map, slot: int, image_orientation: bool,
                 cycle_opacity: float):
    common = dict(name=f"cycle {slot + 1}", legendgroup=f"bar-{id(bar)}", showlegend=slot < 12,
                  hovertemplate=_bar_hover(bar))
    if geometry.dim == 2:
        segments = scene.display(scene.edge_segments(cycle, geometry), image_orientation)
        x, y = _line_coordinates(segments)
        return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color_map[bar], width=_CYCLE_WIDTH_2D),
                          **common)
    return _quad_mesh(go, scene.chain_face_quads(cycle, geometry), opacity=cycle_opacity,
                      color=color_map[bar], **common)


def _empty_cycle_trace(go, geometry, slot: int, cycle_opacity: float):
    if geometry.dim == 2:
        return go.Scatter(x=[None], y=[None], mode="lines", name=f"cycle {slot + 1}",
                          showlegend=False, hoverinfo="skip")
    return _empty_mesh(go, cycle_opacity)


def _value_image_trace(go, forest, geometry, image_orientation: bool):
    """Grayscale heatmap of all grid values; static across filtration values."""
    image_lower, image_upper = scene.value_image_bounds(geometry)
    values = np.asarray(forest.grid_values, dtype=float)
    return _heatmap(
        go, values, image_lower, image_upper, image_orientation,
        colorscale=_colorscale(scene.IMAGE_STYLE["image_cmap"]), zmin=float(values.min()),
        zmax=float(values.max()), opacity=float(scene.IMAGE_STYLE["image_alpha"]),
        showscale=bool(scene.IMAGE_STYLE["image_colorbar"]),
        colorbar=dict(title=scene.IMAGE_STYLE["colorbar_label"], x=1.02, len=0.8, thickness=14),
        name="values", hovertemplate="value=%{z:.4g}<extra></extra>")


def _outline_trace_2d(go, forest, geometry, filt_val: float, image_orientation: bool):
    """Line trace around the pixels present at ``filt_val``."""
    present = scene.present_top_cells(forest, filt_val)
    segments = scene.display(scene.present_outline_segments(present, geometry), image_orientation)
    x, y = _line_coordinates(segments) if len(segments) else ([None], [None])
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=_OUTLINE_COLOR_2D, width=1),
                      name="complex", hoverinfo="skip")


def _snapshot_traces(go, forest, geometry, filt_val: float, color_map, active, *, signed: bool,
                     show_complex: bool, complex_opacity: float, cycle_opacity: float,
                     image_orientation: bool, show_image: bool,
                     max_cycle_slots: Optional[int] = None) -> list:
    """Traces of one filtration value with its ``active`` cycles, excluding the value image.

    With ``max_cycle_slots``, the trace count is fixed for animation frames, and
    in 2D the pixels present are outlined instead of shaded: frames then only
    move lines, which Plotly updates in place, whereas a changing heatmap makes
    the whole figure redraw and flash.
    """
    traces = []
    if geometry.dim == 2:
        if show_complex and max_cycle_slots is not None:
            traces.append(_outline_trace_2d(go, forest, geometry, filt_val, image_orientation))
        elif show_complex:
            present = scene.present_top_cells(forest, filt_val)
            if show_image:
                traces.append(_mask_heatmap(go, ~present, geometry, image_orientation, "white",
                                            scene.FLAT_FILTRATION_STYLE["absent_veil_alpha"],
                                            "not yet present"))
            else:
                traces.append(_mask_heatmap(go, present, geometry, image_orientation,
                                            scene.FLAT_FILTRATION_STYLE["complex_face_color"],
                                            complex_opacity, "complex"))
    elif show_complex:
        traces.append(_quad_mesh(go, scene.present_voxel_quads(forest, geometry, filt_val),
                                 opacity=complex_opacity, color=_COMPLEX_COLOR_3D, name="complex",
                                 hoverinfo="skip"))

    active = sorted(active, key=lambda item: item[0].lifespan(), reverse=True)
    slots = len(active) if max_cycle_slots is None else max_cycle_slots
    for slot in range(slots):
        if slot < len(active):
            bar, cycle = active[slot]
            traces.append(_cycle_trace(go, geometry, bar, cycle if signed else cycle.unsigned(),
                                       color_map, slot, image_orientation, cycle_opacity))
        else:
            traces.append(_empty_cycle_trace(go, geometry, slot, cycle_opacity))
    return traces


def _legend_x(show_image: bool) -> float:
    """Legend position right of the plot, past the value colorbar when there is one."""
    return 1.16 if show_image and scene.IMAGE_STYLE["image_colorbar"] else 1.02


def _layout(fig, geometry, image_orientation: bool, show_image: bool, **kwargs) -> None:
    if geometry.dim == 2:
        lower, upper = (scene.value_image_bounds if show_image else scene.domain_bounds)(geometry)
        (x0, y0), (x1, y1) = scene.display(lower, image_orientation), scene.display(upper, image_orientation)
        fig.update_layout(xaxis_title="axis 1" if image_orientation else "axis 0",
                          yaxis_title="axis 0" if image_orientation else "axis 1",
                          template="plotly_white", hovermode="closest", **kwargs)
        fig.update_xaxes(range=[x0, x1], constrain="domain", showgrid=False, zeroline=False)
        fig.update_yaxes(range=[y1, y0] if image_orientation else [y0, y1], scaleanchor="x",
                         scaleratio=1, constrain="domain", showgrid=False, zeroline=False)
    else:
        fig.update_layout(scene=dict(xaxis_title="axis 0", yaxis_title="axis 1", zaxis_title="axis 2",
                                     aspectmode="data"),
                          template="plotly_white", **kwargs)


def plot_cubical_at_filtration_plotly(
    forest,
    filt_val: float,
    coloring: Literal["forest", "bars"] = "forest",
    show_cycles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0.0,
    show_complex: bool = True,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    show: bool = True,
    renderer: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    image_orientation: bool = False,
    show_image: Optional[bool] = None,
):
    """
    Plot a 2D or 3D cubical filtration and its active cycles with Plotly.

    Target of ``PersistenceForest.plot_at_filtration_plotly`` for cubical
    forests. In 2D, ``show_complex`` tints the pixels present at ``filt_val``
    (opacity ``complex_opacity``), or with ``show_image`` veils those not yet
    present; ``show_image`` draws all grid values in grayscale with a
    colorbar, and None (default) shows them on 2D grids. In 3D, the
    boundary surface of the voxels present is drawn and cycles are meshes of
    opacity ``cycle_opacity``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    geometry = scene.require_grid_dimension(forest, "plot_at_filtration_plotly")
    show_image = scene.resolve_flat_options(geometry, image_orientation, show_image)
    color_map = forest._get_color_map(coloring=coloring)
    active = forest._active_bars_with_cycles_at(filt_val, min_bar_length=min_bar_length) if show_cycles else []
    image = [_value_image_trace(go, forest, geometry, image_orientation)] if show_image else []
    fig = go.Figure(data=image + _snapshot_traces(
        go, forest, geometry, filt_val, color_map, active, signed=signed, show_complex=show_complex,
        complex_opacity=complex_opacity, cycle_opacity=cycle_opacity,
        image_orientation=image_orientation, show_image=show_image))
    _layout(fig, geometry, image_orientation, show_image, title=f"Filtration value t = {filt_val:.4g}",
            width=width, height=height, legend=dict(x=_legend_x(show_image)))
    if show:
        fig.show(renderer=renderer)
    return fig


def plot_cubical_filtration_interactive(
    forest,
    coloring: Literal["forest", "bars"] = "forest",
    show_cycles: bool = True,
    signed: bool = False,
    filt_max: Optional[float] = None,
    min_bar_length: float = 0.0,
    show_complex: bool = True,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    resolution: int = 100,
    show: bool = True,
    renderer: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    image_orientation: bool = False,
    show_image: Optional[bool] = None,
):
    """
    Plotly slider through the filtration of a 2D or 3D cubical forest.

    The slider runs over ``resolution + 1`` equally spaced values from the
    smallest grid value to ``filt_max``, by default 3% of the value range
    past the last finite death. Options match
    ``plot_cubical_at_filtration_plotly``, except that in 2D ``show_complex``
    outlines the pixels present instead of shading them, so moving the slider
    updates lines only and the image does not flash.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    geometry = scene.require_grid_dimension(forest, "plot_filtration_interactive")
    show_image = scene.resolve_flat_options(geometry, image_orientation, show_image)
    start = float(forest.filtration[0][1])
    if filt_max is None:
        finite = [float(bar.death) for bar in forest.barcode if np.isfinite(bar.death)]
        end = max(finite) if finite else float(forest.filtration[-1][1])
        filt_max = end + 0.03 * max(end - start, np.finfo(float).eps)
    filtration_values = np.linspace(start, float(filt_max), int(resolution) + 1)
    color_map = forest._get_color_map(coloring=coloring)

    actives = [forest._active_bars_with_cycles_at(v, min_bar_length=min_bar_length) if show_cycles else []
               for v in filtration_values]
    options = dict(signed=signed, show_complex=show_complex, complex_opacity=complex_opacity,
                   cycle_opacity=cycle_opacity, image_orientation=image_orientation,
                   show_image=show_image, max_cycle_slots=max(map(len, actives)))
    # The value image does not change with the slider, so frames update only the traces after it.
    static = [_value_image_trace(go, forest, geometry, image_orientation)] if show_image else []
    frames = []
    for value, active in zip(filtration_values, actives):
        traces = _snapshot_traces(go, forest, geometry, value, color_map, active, **options)
        frames.append(go.Frame(name=f"{value:.12g}", data=traces,
                               traces=list(range(len(static), len(static) + len(traces)))))

    fig = go.Figure(data=static + list(frames[0].data), frames=frames)
    # 2D frames hold only line traces, updated without a full redraw; 3D meshes need one.
    redraw = geometry.dim != 2
    steps = [dict(method="animate", label=f"{value:.4g}",
                  args=[[f"{value:.12g}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": redraw},
                                             "transition": {"duration": 0}}])
             for value in filtration_values]
    _layout(fig, geometry, image_orientation, show_image, title="Interactive filtration plot",
            sliders=[dict(active=0, currentvalue={"prefix": "t = "}, pad={"t": 40}, steps=steps)],
            width=width, height=height)
    fig.update_layout(legend=dict(x=_legend_x(show_image), y=1, xanchor="left", yanchor="top"),
                      margin=dict(r=180))
    if show:
        fig.show(renderer=renderer)
    return fig


def _outline_trace(go, quads: NDArray, color: str, width: float, **kwargs):
    x, y, z = _line_coordinates(scene.quad_outlines(quads))
    return go.Scatter3d(x=x, y=y, z=z, mode="lines", line={"color": color, "width": width},
                        hoverinfo="skip", **kwargs)


def plot_cubical_interior_activity_plotly(
    forest,
    coloring: Literal["forest", "bars"] = "forest",
    show_complex: bool = False,
    complex_max_filtration: float | None = None,
    overlap: Literal["longest", "layer"] = "longest",
    min_activity_length: float = 0.0,
    geometry: Literal["cells", "surface"] = "surface",
    cell_shrink: float = 1.0,
    opacity: float | None = None,
    show: bool = True,
    renderer: str | None = None,
    width: int | None = None,
    height: int | None = None,
    style: dict[str, Any] | None = None,
    edge_mode: Literal["none", "surface", "all"] = "none",
):
    """
    Interactively plot the voxels of a 3D cubical forest by interior activity.

    Target of ``PersistenceForest.plot_interior_simplex_activity_plotly`` for
    cubical forests. ``geometry="surface"`` draws the boundary of each bar's
    voxel region, ``"cells"`` every active voxel (shrunk by ``cell_shrink``).
    Face opacity grows with activity length unless ``opacity`` is fixed;
    ``edge_mode`` outlines the region boundaries (``"surface"``) or every
    drawn face (``"all"``); ``show_complex`` adds a wireframe of the boundary
    of the voxels with value at most ``complex_max_filtration``. ``style``
    accepts the point-cloud keys except those for points.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    grid = scene.require_grid_dimension(forest, "plot_interior_simplex_activity_plotly", supported=(3,))
    if geometry not in ("cells", "surface"):
        raise ValueError("geometry must be 'cells' or 'surface'.")
    if edge_mode not in ("none", "surface", "all"):
        raise ValueError("edge_mode must be 'none', 'surface' or 'all'.")
    cell_shrink = float(cell_shrink)
    if not 0.0 < cell_shrink <= 1.0:
        raise ValueError("cell_shrink must be greater than 0 and at most 1.")
    plot_style = scene.resolve_style(style, _INTERIOR_STYLE, "style")
    alpha_range = ((float(opacity),) * 2 if opacity is not None
                   else tuple(float(value) for value in plot_style["activity_alpha_range"]))
    if len(alpha_range) != 2 or not all(0.0 <= value <= 1.0 for value in alpha_range):
        raise ValueError("opacity and activity_alpha_range values must be between 0 and 1.")
    if edge_mode == "none" and bool(plot_style["show_activity_edges"]):
        edge_mode = "surface" if geometry == "surface" else "all"

    go = _require_plotly()
    rows = _activity_rows_for_plotting(forest, overlap, min_activity_length)
    color_map = forest._get_color_map(coloring=coloring)
    groups = scene.activity_voxel_quads(rows, grid, geometry == "surface", cell_shrink)
    outline_groups = {}
    if edge_mode == "all" or (edge_mode == "surface" and geometry == "surface"):
        outline_groups = {bar: quads for bar, quads, _, _ in groups}
    elif edge_mode == "surface":
        outline_groups = {bar: quads for bar, quads, _, _ in scene.activity_voxel_quads(rows, grid, True)}
    max_activity_length = max([float(length) for _, _, length in rows] + [np.finfo(float).eps])

    traces = []
    for bar_index, (bar, quads, lengths, cells) in enumerate(
            sorted(groups, key=lambda group: _bar_value(group[0], "lifespan"), reverse=True)):
        # Two triangle records per quad, in the order quads_to_triangles emits them.
        triangles = scene.quads_to_triangles(quads)
        records = [(triangle, bar, length, cell) for triangle, length, cell
                   in zip(triangles, np.tile(lengths, 2), cells + cells)]
        traces.append(_mesh_trace_for_bar(
            go=go, face_records=records, bar=bar, bar_index=bar_index, color=color_map[bar],
            max_activity_length=max_activity_length, alpha_range=alpha_range,
            flatshading=bool(plot_style["flatshading"]), lighting=dict(plot_style["lighting"]),
            lightposition=dict(plot_style["lightposition"])))
        if bar in outline_groups:
            edge_color = plot_style["activity_edge_color"]
            if edge_color is None:
                edge_color = _darken_color(color_map[bar])
            traces.append(_outline_trace(
                go, outline_groups[bar], _rgba_css(edge_color, float(plot_style["activity_edge_alpha"])),
                float(plot_style["activity_edge_width"]), name=f"bar {bar_index + 1} edges",
                legendgroup=f"activity-bar-{bar_index}", showlegend=False))
    if show_complex:
        traces.append(_outline_trace(
            go, scene.present_voxel_quads(forest, grid, complex_max_filtration),
            _rgba_css(plot_style["complex_edge_color"], float(plot_style["complex_edge_alpha"])),
            float(plot_style["complex_edge_width"]), name="complex", showlegend=True))

    remove_axes = bool(plot_style["remove_axes"])
    axis_style = ({"visible": False, "showgrid": False, "showbackground": False,
                   "showticklabels": False, "zeroline": False} if remove_axes else {"showbackground": False})
    figure = go.Figure(data=traces)
    figure.update_layout(
        width=width,
        height=height,
        paper_bgcolor=plot_style["background_color"],
        plot_bgcolor=plot_style["background_color"],
        scene={
            "xaxis": {**axis_style, "title": "" if remove_axes else "axis 0"},
            "yaxis": {**axis_style, "title": "" if remove_axes else "axis 1"},
            "zaxis": {**axis_style, "title": "" if remove_axes else "axis 2"},
            "aspectmode": "data",
            "camera": {"eye": plot_style["camera_eye"]},
            "bgcolor": plot_style["background_color"],
        },
        legend={"groupclick": "togglegroup"},
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
    )
    if show:
        figure.show(renderer=renderer)
    return figure
