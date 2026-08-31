"""Interactive Plotly rendering for 3D interior-simplex activity."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import numpy as np
from matplotlib import colors as mcolors

from .interior_activity_plotting import (
    _activity_face_records,
    _activity_rows_for_plotting,
    _complex_edge_segments_3d,
    _validated_activity_tetrahedra,
)


def _require_plotly():
    """Import Plotly graph objects or raise an actionable installation hint."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError(
            "Interactive interior-activity plotting requires the optional "
            "dependency 'plotly'. Install it with `pip install \".[plotly]\"` "
            "(local repo) or `pip install persforest[plotly]`."
        ) from error
    return go


def _rgba_css(color: Any, alpha: float) -> str:
    """Convert a Matplotlib-compatible color to a Plotly rgba string."""
    red, green, blue, _ = mcolors.to_rgba(color)
    return (
        f"rgba({round(255 * red)}, {round(255 * green)}, "
        f"{round(255 * blue)}, {float(alpha):.6g})"
    )


def _darken_color(color: Any, factor: float = 0.65) -> tuple[float, float, float]:
    """Return a darkened RGB version of a Matplotlib-compatible color."""
    red, green, blue = mcolors.to_rgb(color)
    return red * factor, green * factor, blue * factor


def _bar_value(bar: Any, attribute: str) -> float:
    """Return a finite-or-infinite numeric bar attribute for hover labels."""
    value = getattr(bar, attribute, np.nan)
    if callable(value):
        value = value()
    return float(value)


def _mesh_trace_for_bar(
    go,
    face_records: list[tuple[np.ndarray, Any, float, tuple[int, ...]]],
    bar: Any,
    bar_index: int,
    color: Any,
    max_activity_length: float,
    alpha_range: tuple[float, float],
    flatshading: bool,
    lighting: dict[str, Any],
    lightposition: dict[str, float],
):
    """Build one toggleable Plotly mesh trace for an activity bar."""
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    i_values: list[int] = []
    j_values: list[int] = []
    k_values: list[int] = []
    face_colors: list[str] = []
    custom_data: list[list[Any]] = []

    alpha_min, alpha_max = alpha_range
    birth = _bar_value(bar, "birth")
    death = _bar_value(bar, "death")
    lifespan = _bar_value(bar, "lifespan")

    for face_points, _bar, activity_length, simplex_key in face_records:
        start_index = len(x_values)
        x_values.extend(float(value) for value in face_points[:, 0])
        y_values.extend(float(value) for value in face_points[:, 1])
        z_values.extend(float(value) for value in face_points[:, 2])
        i_values.append(start_index)
        j_values.append(start_index + 1)
        k_values.append(start_index + 2)

        alpha_scale = float(
            np.clip(activity_length / max_activity_length, 0.0, 1.0)
        )
        alpha = float(alpha_min + (alpha_max - alpha_min) * alpha_scale)
        face_colors.append(_rgba_css(color, alpha=alpha))
        hover_row = [
            str(tuple(simplex_key)),
            float(activity_length),
            birth,
            death,
            lifespan,
        ]
        custom_data.extend([hover_row, hover_row, hover_row])

    return go.Mesh3d(
        x=x_values,
        y=y_values,
        z=z_values,
        i=i_values,
        j=j_values,
        k=k_values,
        facecolor=face_colors,
        customdata=custom_data,
        flatshading=flatshading,
        lighting=lighting,
        lightposition=lightposition,
        opacity=1.0,
        name=f"bar {bar_index + 1}",
        legendgroup=f"activity-bar-{bar_index}",
        showlegend=True,
        hovertemplate=(
            "simplex=%{customdata[0]}<br>"
            "activity=%{customdata[1]:.4g}<br>"
            "birth=%{customdata[2]:.4g}<br>"
            "death=%{customdata[3]:.4g}<br>"
            "lifespan=%{customdata[4]:.4g}"
            "<extra></extra>"
        ),
    )


def _edge_trace_for_bar(
    go,
    face_records: list[tuple[np.ndarray, Any, float, tuple[int, ...]]],
    bar_index: int,
    color: Any,
    width: float,
    alpha: float,
):
    """Build a grouped line trace outlining triangular activity faces."""
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    seen_edges: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()

    for face_points, _bar, _activity_length, _simplex_key in face_records:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            first = tuple(float(value) for value in face_points[start])
            second = tuple(float(value) for value in face_points[end])
            edge_key = tuple(sorted((first, second)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            x_values.extend([first[0], second[0], None])
            y_values.extend([first[1], second[1], None])
            z_values.extend([first[2], second[2], None])

    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        line={"color": _rgba_css(color, alpha=alpha), "width": width},
        name=f"bar {bar_index + 1} edges",
        legendgroup=f"activity-bar-{bar_index}",
        showlegend=False,
        hoverinfo="skip",
    )


def _shrink_surface_edge_records(
    face_records: list[tuple[np.ndarray, Any, float, tuple[int, ...]]],
    point_cloud: np.ndarray,
    cell_shrink: float,
) -> list[tuple[np.ndarray, Any, float, tuple[int, ...]]]:
    """Align exterior edge faces with an exploded tetrahedral cell view."""
    if cell_shrink == 1.0:
        return face_records

    shrunken_records = []
    for face_points, bar, activity_length, simplex_key in face_records:
        tetrahedron_center = np.mean(point_cloud[list(simplex_key)], axis=0)
        shrunken_face = tetrahedron_center + cell_shrink * (
            face_points - tetrahedron_center
        )
        shrunken_records.append(
            (shrunken_face, bar, activity_length, simplex_key)
        )
    return shrunken_records


def plot_interior_simplex_activity_plotly(
    forest,
    coloring: Literal["forest", "bars"] = "forest",
    show_complex: bool = False,
    complex_max_filtration: float | None = None,
    overlap: Literal["longest", "layer"] = "longest",
    vertex_size: float = 2.0,
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
    Interactively plot 3D tetrahedra or surfaces by interior activity.

    The default is a clean per-bar boundary surface: shared faces within each
    bar are cancelled, activity edges are hidden, and face opacity scales with
    activity duration over the range ``(0.35, 0.95)``. Activity meshes are
    grouped by persistence bar, so clicking a legend item hides or reveals the
    corresponding geometry. Hover labels report the simplex, activity length
    and bar endpoints. Plotly is imported lazily.

    Parameters
    ----------
    forest : PersistenceForest
        A 3D forest whose interior cycle representatives have been computed.
    coloring : {"forest", "bars"}
        Bar color map to use.
    show_complex : bool
        If True, show a subdued wireframe of the filtered complex.
    complex_max_filtration : float or None
        Maximum filtration value included in the optional complex wireframe.
    overlap : {"longest", "layer"}
        Whether a multiply active tetrahedron uses its longest interval or is
        included once for every activity interval.
    vertex_size : float
        Plotly marker size for point-cloud vertices.
    min_activity_length : float
        Ignore activity intervals shorter than this value.
    geometry : {"cells", "surface"}
        ``"surface"`` draws the boundary of each per-bar cell union and is the
        recommended presentation view. It removes faces internal to a region,
        producing a clean exterior. ``"cells"`` draws every tetrahedron and is
        intended for inspecting the tetrahedral structure.
    cell_shrink : float
        Factor in ``(0, 1]`` used to shrink cells towards their barycenters.
        This applies only to ``geometry="cells"``. The default ``1`` preserves
        shared vertices and avoids artificial gaps. Values below ``1`` create
        an exploded-cell view and should be used deliberately.
    opacity : float or None
        Optional fixed opacity for every activity face. The default ``None``
        scales opacity linearly by activity duration, relative to the maximum
        displayed duration, using ``style["activity_alpha_range"]``. Supply a
        number in ``[0, 1]`` only when uniform opacity is desired.
    show : bool
        If True, display the figure.
    renderer : str or None
        Plotly renderer passed to ``Figure.show``.
    width, height : int or None
        Figure dimensions in pixels.
    style : dict or None
        Optional Plotly style overrides. Important keys are:

        - ``show_activity_edges=False``: compatibility alias for enabling
          edges. It maps to ``edge_mode="surface"`` in surface geometry and
          ``edge_mode="all"`` in cell geometry. Prefer ``edge_mode`` in new
          code.
        - ``activity_alpha_range=(0.35, 0.95)``: minimum and maximum opacity
          when ``opacity=None``.
        - ``flatshading=True``: retain faceted lighting without line edges.
        - ``lighting`` and ``lightposition``: Plotly ``Mesh3d`` lighting
          dictionaries.
        - ``camera_eye``: Plotly camera-eye dictionary with ``x``, ``y`` and
          ``z`` coordinates.
        - ``remove_axes=True``: hide axes, panes and tick labels.
        - ``point_color``, ``point_alpha`` and the ``activity_edge_*`` and
          ``complex_edge_*`` keys: point and line styling.
    edge_mode : {"none", "surface", "all"}
        Which activity edges to draw. ``"none"`` is the clean default.
        ``"surface"`` outlines only triangular faces exposed on the boundary
        of each per-bar activity region; this is the recommended opt-in mode.
        ``"all"`` outlines every tetrahedral cell and can reveal internal and
        rear-facing edges through transparent faces. When
        ``activity_edge_color`` is ``None``, edges use a darkened version of
        their bar color.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D figure.
    """
    if forest.dim != 3:
        raise ValueError(
            "plot_interior_simplex_activity_plotly only supports 3D "
            "PersistenceForest objects."
        )
    if geometry not in ("cells", "surface"):
        raise ValueError("geometry must be 'cells' or 'surface'.")
    if edge_mode not in ("none", "surface", "all"):
        raise ValueError("edge_mode must be 'none', 'surface' or 'all'.")
    cell_shrink = float(cell_shrink)
    if not 0.0 < cell_shrink <= 1.0:
        raise ValueError("cell_shrink must be greater than 0 and at most 1.")
    if opacity is not None and not 0.0 <= float(opacity) <= 1.0:
        raise ValueError("opacity must be between 0 and 1.")

    plot_style: dict[str, Any] = {
        "point_color": "black",
        "point_alpha": 0.9,
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
        "lighting": {
            "ambient": 0.45,
            "diffuse": 0.8,
            "roughness": 0.8,
            "specular": 0.2,
            "fresnel": 0.1,
        },
        "lightposition": {"x": 100.0, "y": 200.0, "z": 100.0},
        "camera_eye": {"x": 1.45, "y": 1.45, "z": 1.15},
    }
    if style is not None:
        plot_style.update(style)

    resolved_edge_mode = edge_mode
    if resolved_edge_mode == "none" and bool(plot_style["show_activity_edges"]):
        resolved_edge_mode = "surface" if geometry == "surface" else "all"

    if opacity is None:
        alpha_range = tuple(
            float(value) for value in plot_style["activity_alpha_range"]
        )
    else:
        alpha_range = (float(opacity), float(opacity))
    if (
        len(alpha_range) != 2
        or not 0.0 <= alpha_range[0] <= 1.0
        or not 0.0 <= alpha_range[1] <= 1.0
    ):
        raise ValueError("activity_alpha_range values must be between 0 and 1.")

    go = _require_plotly()
    point_cloud = np.asarray(forest.point_cloud, dtype=float)
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError("point_cloud must be an (n_points, 3) array-like.")
    if len(point_cloud) == 0:
        raise ValueError("point_cloud must contain at least one point.")

    rows = _activity_rows_for_plotting(
        forest=forest,
        overlap=overlap,
        min_activity_length=min_activity_length,
    )
    rows = _validated_activity_tetrahedra(point_cloud, rows)
    face_records = _activity_face_records(
        point_cloud=point_cloud,
        rows=rows,
        geometry=geometry,
        cell_shrink=cell_shrink,
    )
    color_map = forest._get_color_map(coloring=coloring)

    edge_face_records: list[
        tuple[np.ndarray, Any, float, tuple[int, ...]]
    ] = []
    if resolved_edge_mode == "surface":
        edge_face_records = _activity_face_records(
            point_cloud=point_cloud,
            rows=rows,
            geometry="surface",
            cell_shrink=1.0,
        )
        if geometry == "cells":
            edge_face_records = _shrink_surface_edge_records(
                face_records=edge_face_records,
                point_cloud=point_cloud,
                cell_shrink=cell_shrink,
            )
    elif resolved_edge_mode == "all":
        edge_face_records = _activity_face_records(
            point_cloud=point_cloud,
            rows=rows,
            geometry="cells",
            cell_shrink=cell_shrink if geometry == "cells" else 1.0,
        )

    records_by_bar: dict[
        Any,
        list[tuple[np.ndarray, Any, float, tuple[int, ...]]],
    ] = defaultdict(list)
    for record in face_records:
        records_by_bar[record[1]].append(record)

    edge_records_by_bar: dict[
        Any,
        list[tuple[np.ndarray, Any, float, tuple[int, ...]]],
    ] = defaultdict(list)
    for record in edge_face_records:
        edge_records_by_bar[record[1]].append(record)

    bars = sorted(
        records_by_bar,
        key=lambda bar: _bar_value(bar, "lifespan"),
        reverse=True,
    )
    max_activity_length = max(
        (record[2] for record in face_records),
        default=np.finfo(float).eps,
    )
    max_activity_length = max(max_activity_length, np.finfo(float).eps)

    traces: list[Any] = []
    for bar_index, bar in enumerate(bars):
        bar_records = records_by_bar[bar]
        traces.append(
            _mesh_trace_for_bar(
                go=go,
                face_records=bar_records,
                bar=bar,
                bar_index=bar_index,
                color=color_map[bar],
                max_activity_length=max_activity_length,
                alpha_range=alpha_range,
                flatshading=bool(plot_style["flatshading"]),
                lighting=dict(plot_style["lighting"]),
                lightposition=dict(plot_style["lightposition"]),
            )
        )
        if resolved_edge_mode != "none" and edge_records_by_bar[bar]:
            edge_color = plot_style["activity_edge_color"]
            if edge_color is None:
                edge_color = _darken_color(color_map[bar])
            traces.append(
                _edge_trace_for_bar(
                    go=go,
                    face_records=edge_records_by_bar[bar],
                    bar_index=bar_index,
                    color=edge_color,
                    width=float(plot_style["activity_edge_width"]),
                    alpha=float(plot_style["activity_edge_alpha"]),
                )
            )

    if show_complex:
        complex_segments = _complex_edge_segments_3d(
            forest=forest,
            point_cloud=point_cloud,
            complex_max_filtration=complex_max_filtration,
        )
        x_complex: list[float | None] = []
        y_complex: list[float | None] = []
        z_complex: list[float | None] = []
        for segment in complex_segments:
            x_complex.extend([float(segment[0, 0]), float(segment[1, 0]), None])
            y_complex.extend([float(segment[0, 1]), float(segment[1, 1]), None])
            z_complex.extend([float(segment[0, 2]), float(segment[1, 2]), None])
        traces.append(
            go.Scatter3d(
                x=x_complex,
                y=y_complex,
                z=z_complex,
                mode="lines",
                line={
                    "color": _rgba_css(
                        plot_style["complex_edge_color"],
                        alpha=float(plot_style["complex_edge_alpha"]),
                    ),
                    "width": float(plot_style["complex_edge_width"]),
                },
                name="complex",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    traces.append(
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode="markers",
            marker={
                "size": float(vertex_size),
                "color": plot_style["point_color"],
                "opacity": float(plot_style["point_alpha"]),
            },
            name="points",
            showlegend=True,
            hovertemplate="(%{x:.4g}, %{y:.4g}, %{z:.4g})<extra></extra>",
        )
    )

    remove_axes = bool(plot_style["remove_axes"])
    axis_style: dict[str, Any]
    if remove_axes:
        axis_style = {
            "visible": False,
            "showgrid": False,
            "showbackground": False,
            "showticklabels": False,
            "zeroline": False,
        }
    else:
        axis_style = {"showbackground": False}

    figure = go.Figure(data=traces)
    figure.update_layout(
        width=width,
        height=height,
        paper_bgcolor=plot_style["background_color"],
        plot_bgcolor=plot_style["background_color"],
        scene={
            "xaxis": {**axis_style, "title": "" if remove_axes else "x"},
            "yaxis": {**axis_style, "title": "" if remove_axes else "y"},
            "zaxis": {**axis_style, "title": "" if remove_axes else "z"},
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
