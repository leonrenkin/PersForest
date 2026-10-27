"""plotting.interior mpl: mechanically extracted reference implementation."""

from __future__ import annotations

from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from persforest.plotting._geometry import _activity_face_records
from persforest.plotting._geometry import _camera_angles
from persforest.plotting._geometry import _validated_activity_tetrahedra
from persforest.plotting._scene import _activity_rows_for_plotting
from persforest.plotting._scene import _complex_edge_segments_3d
from typing import Any
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np

def plot_interior_simplex_activity(
    forest,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
    coloring: Literal["forest", "bars"] = "forest",
    show_complex: bool = False,
    complex_max_filtration: float | None = None,
    overlap: Literal["longest", "layer"] = "longest",
    vertex_size: float = 2,
    min_activity_length: float = 0.0,
    style: dict[str, Any] | None = None,
    geometry: Literal["cells", "surface"] = "cells",
):
    """
    Plot full-dimensional simplices colored by interior activity.

    The input forest must already have interior cycle representatives available.
    Activity data is read from ``forest.interior_simplex_activity()``.
    If ``complex_max_filtration`` is set, the optional complex overlay only
    includes simplices with filtration value at most that threshold.

    For a 2D forest, the existing triangle renderer is used unchanged. For a
    3D forest, ``geometry="cells"`` draws tetrahedral cells, while
    ``geometry="surface"`` cancels faces shared by cells associated with the
    same bar and draws the resulting region boundaries.

    Parameters
    ----------
    forest : PersistenceForest
        Forest whose interior cycle representatives have been computed.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. A supplied 3D axes must use ``projection="3d"``.
    show : bool
        If True, call ``matplotlib.pyplot.show`` after plotting.
    figsize : tuple[float, float]
        Figure size used when creating an axes.
    dpi : int
        Figure resolution used when creating an axes.
    coloring : {"forest", "bars"}
        Bar color map to use.
    show_complex : bool
        If True, overlay complex edges.
    complex_max_filtration : float or None
        Maximum filtration value included in the optional complex overlay.
    overlap : {"longest", "layer"}
        Whether a multiply active simplex uses its longest interval or is
        drawn once for every activity interval.
    vertex_size : float
        Point-cloud marker size.
    min_activity_length : float
        Ignore activity intervals shorter than this value.
    style : dict or None
        Style overrides. In 3D these additionally include ``cell_shrink``,
        ``camera_eye``, ``shade``, ``lightsource``, ``zsort``,
        ``depthshade_points`` and ``activity_edge_alpha``.
    geometry : {"cells", "surface"}
        3D geometry representation. This option is ignored in 2D.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the activity plot.
    """
    if forest.dim == 2:
        return _plot_interior_simplex_activity_2d(
            forest=forest,
            ax=ax,
            show=show,
            figsize=figsize,
            dpi=dpi,
            coloring=coloring,
            show_complex=show_complex,
            complex_max_filtration=complex_max_filtration,
            overlap=overlap,
            vertex_size=vertex_size,
            min_activity_length=min_activity_length,
            style=style,
        )
    if forest.dim == 3:
        return _plot_interior_simplex_activity_3d(
            forest=forest,
            ax=ax,
            show=show,
            figsize=figsize,
            dpi=dpi,
            coloring=coloring,
            show_complex=show_complex,
            complex_max_filtration=complex_max_filtration,
            overlap=overlap,
            vertex_size=vertex_size,
            min_activity_length=min_activity_length,
            style=style,
            geometry=geometry,
        )
    raise ValueError(
        "plot_interior_simplex_activity only supports 2D and 3D "
        "PersistenceForest objects."
    )

def _plot_interior_simplex_activity_2d(
    forest,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
    coloring: Literal["forest", "bars"] = "forest",
    show_complex: bool = False,
    complex_max_filtration: float | None = None,
    overlap: Literal["longest", "layer"] = "longest",
    vertex_size: float = 2,
    min_activity_length: float = 0.0,
    style: dict[str, Any] | None = None,
):
    """Plot 2D filtration triangles colored by interior simplex activity."""
    if forest.dim != 2:
        raise ValueError(
            "_plot_interior_simplex_activity_2d requires a 2D PersistenceForest object."
        )
    if overlap not in ("longest", "layer"):
        raise ValueError("overlap must be 'longest' or 'layer'.")

    plot_style = {
        "point_color": "black",
        "point_alpha": 0.9,
        "activity_edge_color": "white",
        "activity_edge_width": 0.25,
        "complex_edge_color": "0",
        "complex_edge_width": 0.45,
        "complex_edge_alpha": 0.85,
        "background_color": "white",
        "remove_axes": True,
        "activity_alpha_range": (0.15, 0.95),
    }
    if style is not None:
        plot_style.update(style)

    pts = np.asarray(forest.point_cloud, dtype=float)
    color_map = forest._get_color_map(coloring=coloring)
    activity = forest.interior_simplex_activity()

    rows = []
    for simplex_key, simplex_activity in activity.items():
        for bar, active_start, active_end in simplex_activity:
            activity_length = float(active_end - active_start)
            if activity_length >= min_activity_length:
                rows.append((tuple(simplex_key), bar, activity_length))

    if overlap == "longest":
        longest_by_simplex = {}
        for simplex_key, bar, activity_length in rows:
            current = longest_by_simplex.get(simplex_key)
            if current is None or activity_length > current[1]:
                longest_by_simplex[simplex_key] = (bar, activity_length)
        rows = [
            (simplex_key, bar, activity_length)
            for simplex_key, (bar, activity_length) in longest_by_simplex.items()
        ]
    else:
        rows.sort(key=lambda row: row[2])

    max_activity_length = max((activity_length for _, _, activity_length in rows), default=0.0)
    alpha_min, alpha_max = plot_style["activity_alpha_range"]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.set_facecolor(plot_style["background_color"])

    if rows:
        triangle_polys = []
        triangle_colors = []

        for simplex_key, bar, activity_length in rows:
            alpha_scale = activity_length / max_activity_length
            alpha = float(alpha_min + (alpha_max - alpha_min) * alpha_scale)
            triangle_polys.append(pts[list(simplex_key)])
            triangle_colors.append(mcolors.to_rgba(color_map[bar], alpha=alpha))

        activity_collection = PolyCollection(
            triangle_polys,
            closed=True,
            facecolors=triangle_colors,
            edgecolors=plot_style["activity_edge_color"],
            linewidths=float(plot_style["activity_edge_width"]),
            zorder=1,
        )
        ax.add_collection(activity_collection)

    if show_complex:
        edge_segments = []
        for simplex, filtration in forest.filtration:
            if complex_max_filtration is not None and filtration > complex_max_filtration:
                continue
            if len(simplex) == 2:
                edge_segments.append(pts[list(simplex)])

        if edge_segments:
            edge_collection = LineCollection(
                edge_segments,
                colors=plot_style["complex_edge_color"],
                linewidths=float(plot_style["complex_edge_width"]),
                alpha=float(plot_style["complex_edge_alpha"]),
                zorder=2,
            )
            ax.add_collection(edge_collection)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=vertex_size,
        color=plot_style["point_color"],
        alpha=float(plot_style["point_alpha"]),
        edgecolors="none",
        zorder=3,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()

    if bool(plot_style["remove_axes"]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    if show:
        plt.show()

    return ax

def _plot_interior_simplex_activity_3d(
    forest,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
    coloring: Literal["forest", "bars"] = "forest",
    show_complex: bool = False,
    complex_max_filtration: float | None = None,
    overlap: Literal["longest", "layer"] = "longest",
    vertex_size: float = 2,
    min_activity_length: float = 0.0,
    style: dict[str, Any] | None = None,
    geometry: Literal["cells", "surface"] = "cells",
):
    """Plot 3D tetrahedra or region surfaces colored by interior activity."""
    if forest.dim != 3:
        raise ValueError(
            "_plot_interior_simplex_activity_3d requires a 3D PersistenceForest object."
        )
    if geometry not in ("cells", "surface"):
        raise ValueError("geometry must be 'cells' or 'surface'.")

    plot_style = {
        "point_color": "black",
        "point_alpha": 0.9,
        "depthshade_points": False,
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
    if style is not None:
        plot_style.update(style)

    cell_shrink = float(plot_style["cell_shrink"])
    if not 0.0 < cell_shrink <= 1.0:
        raise ValueError("cell_shrink must be greater than 0 and at most 1.")

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

    if ax is None:
        figure = plt.figure(figsize=figsize, dpi=dpi)
        ax = figure.add_subplot(111, projection="3d")
    elif not hasattr(ax, "zaxis"):
        raise ValueError("3D plotting requires a Matplotlib 3D axis (projection='3d').")

    ax.set_facecolor(plot_style["background_color"])
    ax.figure.patch.set_facecolor(plot_style["background_color"])

    if face_records:
        max_activity_length = max(record[2] for record in face_records)
        max_activity_length = max(max_activity_length, np.finfo(float).eps)
        alpha_min, alpha_max = plot_style["activity_alpha_range"]
        face_colors = []
        face_polygons = []
        for face_points, bar, activity_length, _simplex_key in face_records:
            alpha_scale = float(np.clip(activity_length / max_activity_length, 0.0, 1.0))
            alpha = float(alpha_min + (alpha_max - alpha_min) * alpha_scale)
            face_polygons.append(face_points)
            face_colors.append(mcolors.to_rgba(color_map[bar], alpha=alpha))

        activity_collection = Poly3DCollection(
            face_polygons,
            facecolors=face_colors,
            edgecolors=mcolors.to_rgba(
                plot_style["activity_edge_color"],
                alpha=float(plot_style["activity_edge_alpha"]),
            ),
            linewidths=float(plot_style["activity_edge_width"]),
            antialiased=bool(plot_style["antialiased"]),
            zsort=plot_style["zsort"],
            shade=bool(plot_style["shade"]),
            lightsource=plot_style["lightsource"],
        )
        ax.add_collection3d(activity_collection)

    if show_complex:
        edge_segments = _complex_edge_segments_3d(
            forest=forest,
            point_cloud=point_cloud,
            complex_max_filtration=complex_max_filtration,
        )
        if edge_segments:
            complex_collection = Line3DCollection(
                edge_segments,
                colors=plot_style["complex_edge_color"],
                linewidths=float(plot_style["complex_edge_width"]),
                alpha=float(plot_style["complex_edge_alpha"]),
                antialiased=bool(plot_style["antialiased"]),
            )
            ax.add_collection3d(complex_collection)

    ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        s=vertex_size,
        color=plot_style["point_color"],
        alpha=float(plot_style["point_alpha"]),
        edgecolors="none",
        depthshade=bool(plot_style["depthshade_points"]),
    )

    mins = np.min(point_cloud, axis=0)
    maxs = np.max(point_cloud, axis=0)
    spans = np.maximum(maxs - mins, np.finfo(float).eps)
    padding = 0.05 * float(np.max(spans))
    limits = [
        (float(mins[index] - padding), float(maxs[index] + padding))
        for index in range(3)
    ]
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect(tuple(upper - lower for lower, upper in limits))

    elevation, azimuth = _camera_angles(plot_style["camera_eye"])
    ax.view_init(elev=elevation, azim=azimuth)

    if bool(plot_style["remove_axes"]):
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
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if show:
        plt.show()

    return ax

def plot_interior_simplex_activity_gradient(
    forest,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
    coloring: Literal["forest", "bars"] = "forest",
    vertex_size: float = 2,
    title: str | None = None,
    min_activity_length: float = 0.0,
    style: dict[str, Any] | None = None,
):
    """
    Plot a smooth 2D color field from interior simplex activity.

    This representation remains 2D-only. ``style`` supports the keys
    documented by ``_plot_interior_simplex_activity_gradient_2d``, including
    raster resolution, blur radii, intensity scaling and conflict whitening.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the activity gradient.
    """
    if forest.dim == 2:
        return _plot_interior_simplex_activity_gradient_2d(
            forest=forest,
            ax=ax,
            show=show,
            figsize=figsize,
            dpi=dpi,
            coloring=coloring,
            vertex_size=vertex_size,
            title=title,
            min_activity_length=min_activity_length,
            style=style,
        )
    raise ValueError(
        "plot_interior_simplex_activity_gradient only supports 2D "
        "PersistenceForest objects."
    )

def _plot_interior_simplex_activity_gradient_2d(
    forest,
    ax=None,
    show: bool = True,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
    coloring: Literal["forest", "bars"] = "forest",
    vertex_size: float = 2,
    title: str | None = None,
    min_activity_length: float = 0.0,
    style: dict[str, Any] | None = None,
):
    """
    Plot a smooth 2D color field from interior simplex activity.

    The input forest must already have interior cycle representatives available.
    Activity data is read from ``forest.interior_simplex_activity()``.

    Style keys
    ----------
    point_color="black", point_alpha=0.9
        Point cloud marker color and opacity.
    background_color="white"
        Color used outside the activity field.
    remove_axes=True
        If True, hide ticks, labels, and spines.
    resolution=650
        Maximum pixel dimension of the rasterized gradient field.
    blur_sigma=7.0
        Gaussian smoothing radius for per-bar activity fields.
    boundary_blur_sigma=5.0
        Smoothing radius for the active-region mask; larger values create a
        wider fade to white near boundaries.
    intensity_gamma=0.9
        Exponent applied to normalized activity; smaller values make weak
        activity more visible.
    activity_scale_fraction_of_max_bar=0.75
        Fraction of ``forest.max_bar().lifespan()`` used in activity scaling.
    activity_scale_percentile=95
        Percentile of activity lengths used in activity scaling.
    different_color_threshold=0.18
        Normalized RGB distance in ``[0, 1]`` above which neighboring bar
        colors start fading to white; values at or above ``1`` disable
        color-conflict whitening.
    conflict_whitening=0.85
        Strength of whitening where different-colored bar fields compete.
    """
    if forest.dim != 2:
        raise ValueError(
            "_plot_interior_simplex_activity_gradient_2d requires a 2D "
            "PersistenceForest object."
        )

    from collections import defaultdict

    plot_style = {
        "point_color": "black",
        "point_alpha": 0.9,
        "background_color": "white",
        "remove_axes": True,
        "resolution": 650,
        "blur_sigma": 7.0,
        "boundary_blur_sigma": 5.0,
        "intensity_gamma": 0.9,
        "activity_scale_fraction_of_max_bar": 0.75,
        "activity_scale_percentile": 95,
        "different_color_threshold": 0.18,
        "conflict_whitening": 0.85,
    }
    if style is not None:
        plot_style.update(style)

    def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
        sigma = float(sigma)
        if sigma <= 0:
            return arr

        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()

        blurred = np.apply_along_axis(
            lambda values: np.convolve(
                np.pad(values, radius, mode="constant"),
                kernel,
                mode="valid",
            ),
            -1,
            arr,
        )
        blurred = np.apply_along_axis(
            lambda values: np.convolve(
                np.pad(values, radius, mode="constant"),
                kernel,
                mode="valid",
            ),
            -2,
            blurred,
        )
        return blurred

    pts = np.asarray(forest.point_cloud, dtype=float)
    color_map = forest._get_color_map(coloring=coloring)
    activity = forest.interior_simplex_activity()

    longest_by_simplex = {}
    for simplex_key, simplex_activity in activity.items():
        for bar, active_start, active_end in simplex_activity:
            activity_length = float(active_end - active_start)
            if activity_length < min_activity_length:
                continue
            simplex_key = tuple(simplex_key)
            current = longest_by_simplex.get(simplex_key)
            if current is None or activity_length > current[1]:
                longest_by_simplex[simplex_key] = (bar, activity_length)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.set_facecolor(plot_style["background_color"])

    if longest_by_simplex:
        activity_lengths = np.array(
            [length for _bar, length in longest_by_simplex.values()],
            dtype=float,
        )
        activity_scale = max(
            float(forest.max_bar().lifespan()) * float(plot_style["activity_scale_fraction_of_max_bar"]),
            float(np.percentile(activity_lengths, float(plot_style["activity_scale_percentile"]))),
        )
        activity_scale = max(activity_scale, np.finfo(float).eps)

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        spans = np.maximum(maxs - mins, np.finfo(float).eps)
        pad = 0.04 * float(np.max(spans))
        xmin, ymin = mins - pad
        xmax, ymax = maxs + pad
        width = xmax - xmin
        height = ymax - ymin
        max_resolution = int(plot_style["resolution"])
        if width >= height:
            nx = max_resolution
            ny = max(2, int(np.ceil(max_resolution * height / width)))
        else:
            ny = max_resolution
            nx = max(2, int(np.ceil(max_resolution * width / height)))

        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        bar_order = []
        bar_to_idx = {}
        for _simplex_key, (bar, _activity_length) in longest_by_simplex.items():
            if bar not in bar_to_idx:
                bar_to_idx[bar] = len(bar_order)
                bar_order.append(bar)

        fields = np.zeros((len(bar_order), ny, nx), dtype=float)
        active_mask = np.zeros((ny, nx), dtype=float)

        for simplex_key, (bar, activity_length) in longest_by_simplex.items():
            tri = pts[list(simplex_key)]
            tri_min = tri.min(axis=0)
            tri_max = tri.max(axis=0)
            x0 = max(0, int(np.searchsorted(xs, tri_min[0], side="left") - 1))
            x1 = min(nx, int(np.searchsorted(xs, tri_max[0], side="right") + 1))
            y0 = max(0, int(np.searchsorted(ys, tri_min[1], side="left") - 1))
            y1 = min(ny, int(np.searchsorted(ys, tri_max[1], side="right") + 1))
            if x1 <= x0 or y1 <= y0:
                continue

            xx, yy = np.meshgrid(xs[x0:x1], ys[y0:y1])
            p0, p1, p2 = tri
            denom = (
                (p1[1] - p2[1]) * (p0[0] - p2[0])
                + (p2[0] - p1[0]) * (p0[1] - p2[1])
            )
            if abs(denom) <= np.finfo(float).eps:
                continue

            a = ((p1[1] - p2[1]) * (xx - p2[0]) + (p2[0] - p1[0]) * (yy - p2[1])) / denom
            b = ((p2[1] - p0[1]) * (xx - p2[0]) + (p0[0] - p2[0]) * (yy - p2[1])) / denom
            c = 1.0 - a - b
            inside = (a >= -1e-12) & (b >= -1e-12) & (c >= -1e-12)
            if not np.any(inside):
                continue

            field = fields[bar_to_idx[bar], y0:y1, x0:x1]
            field[inside] = np.maximum(field[inside], activity_length)
            mask = active_mask[y0:y1, x0:x1]
            mask[inside] = 1.0

        blurred_fields = _gaussian_blur(fields, float(plot_style["blur_sigma"]))
        weights = np.clip(blurred_fields / activity_scale, 0.0, 1.0)
        weights = weights ** float(plot_style["intensity_gamma"])
        mask_field = _gaussian_blur(active_mask, float(plot_style["boundary_blur_sigma"]))
        if np.max(mask_field) > 0:
            mask_field = mask_field / np.max(mask_field)
        mask_field = np.clip(mask_field, 0.0, 1.0)

        weight_sum = np.sum(weights, axis=0)
        rgb_colors = np.array([mcolors.to_rgb(color_map[bar]) for bar in bar_order], dtype=float)
        weighted_rgb = np.einsum("bhw,bc->hwc", weights, rgb_colors)
        blended_rgb = np.ones((ny, nx, 3), dtype=float)
        nonzero = weight_sum > np.finfo(float).eps
        blended_rgb[nonzero] = weighted_rgb[nonzero] / weight_sum[nonzero, None]

        if len(bar_order) > 1:
            top_indices = np.argpartition(weights, -2, axis=0)[-2:]
            top_values = np.take_along_axis(weights, top_indices, axis=0)
            order = np.argsort(top_values, axis=0)
            second_idx = np.take_along_axis(top_indices, order[:1], axis=0)[0]
            first_idx = np.take_along_axis(top_indices, order[1:], axis=0)[0]
            first_values = np.take_along_axis(weights, first_idx[None, :, :], axis=0)[0]
            second_values = np.take_along_axis(weights, second_idx[None, :, :], axis=0)[0]
            first_rgb = rgb_colors[first_idx]
            second_rgb = rgb_colors[second_idx]
            color_distance = np.linalg.norm(first_rgb - second_rgb, axis=2) / np.sqrt(3.0)
            threshold = float(plot_style["different_color_threshold"])
            max_color_distance = 1.0
            if threshold >= max_color_distance:
                color_conflict = np.zeros_like(color_distance)
            else:
                threshold = max(0.0, threshold)
                color_conflict = np.clip(
                    (color_distance - threshold) / (max_color_distance - threshold),
                    0.0,
                    1.0,
                )
            competition = second_values / np.maximum(first_values, np.finfo(float).eps)
            conflict = (
                color_conflict
                * np.clip(competition, 0.0, 1.0)
                * float(plot_style["conflict_whitening"])
            )
        else:
            conflict = np.zeros((ny, nx), dtype=float)

        strength_base = np.clip(weight_sum, 0.0, 1.0)
        strength = np.clip(strength_base * mask_field * (1.0 - conflict), 0.0, 1.0)
        background_rgb = np.array(mcolors.to_rgb(plot_style["background_color"]), dtype=float)
        image = background_rgb + strength[:, :, None] * (blended_rgb - background_rgb)
        image = np.clip(image, 0.0, 1.0)

        ax.imshow(
            image,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            interpolation="bilinear",
            zorder=1,
        )

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=vertex_size,
        color=plot_style["point_color"],
        alpha=float(plot_style["point_alpha"]),
        edgecolors="none",
        zorder=3,
    )

    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title)
    ax.autoscale()

    if bool(plot_style["remove_axes"]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    if show:
        plt.show()

    return ax
