"""Matplotlib rendering of simplicial filtrations and cycle representatives."""

from __future__ import annotations

from ._dependencies import require_plotting

require_plotting()

from collections import defaultdict
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from persforest.barcode import _value_at_relative_position
from persforest.chains import SignedChain
from persforest.plotting._geometry import _camera_from_eye
from persforest.plotting._geometry import boundary_triangles
from persforest.plotting._geometry import shaded_face_colors
from persforest.plotting._lines import add_cycle_lines
from persforest.plotting._style import filtration_title
from persforest.plotting._style import representatives_title
from persforest.plotting._style import resolve_style
from persforest.plotting._style import style_scene_axes
from typing import Any
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
import itertools
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from persforest.forest import PersistenceForest

def _desaturate_color(color: Any, amount: float) -> tuple[float, float, float]:
    """
    Return an RGB color with saturation reduced by `amount` in [0, 1].
    """
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[1] *= 1.0 - amount
    out = mcolors.hsv_to_rgb(hsv)
    return (float(out[0]), float(out[1]), float(out[2]))

def _resolve_style_2d(style_2d: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge 2D filtration style overrides with default values.

    Supported keys are ``point_color``, ``point_alpha``,
    ``complex_face_color``, ``complex_face_alpha``, ``complex_edge_color``,
    ``complex_edge_width``, ``complex_edge_alpha``, ``cycle_edge_width``,
    ``cycle_halo``, ``cycle_halo_width``, ``cycle_halo_color``,
    ``show_orientation_arrows``, ``arrow_linewidth`` and ``arrow_scale``.
    """
    # A neutral complex keeps every palette color readable on top of it.
    style = {
        "point_color": "0.2",
        "point_alpha": 1.0,
        "complex_face_color": "#e2e5e9",
        "complex_face_alpha": 1.0,
        "complex_edge_color": "0.62",
        "complex_edge_width": 0.45,
        "complex_edge_alpha": None,
        "cycle_edge_width": 2.2,
        "cycle_halo": False,
        "cycle_halo_width": 2.4,
        "cycle_halo_color": "white",
        "show_orientation_arrows": False,
        "arrow_linewidth": 0.8,
        "arrow_scale": 4.0,
    }
    return resolve_style(style_2d, style, "style_2d")

def _resolve_style_3d(style_3d: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge 3D filtration style overrides with default values.

    Supported keys are ``camera_eye``, ``remove_axes``, ``point_color``,
    ``point_alpha``, ``depthshade_points``, ``complex_color``,
    ``complex_face_alpha``, ``cycle_face_alpha``, ``complex_edge_color``,
    ``cycle_edge_color``, ``complex_edge_width``, ``cycle_edge_width``,
    ``complex_edge_alpha``, ``cycle_edge_alpha``, ``antialiased``, ``zsort``,
    ``desaturate_complex`` and ``shade``.

    The complex is drawn as a faint shell of its boundary surface and the
    cycles as opaque, shaded surfaces with edges in a darker shade of their
    color (``cycle_edge_color=None``); ``complex_edge_width=0`` omits complex
    edges.
    """
    style = {
        "camera_eye": None,
        "remove_axes": True,
        "point_color": "0.35",
        "point_alpha": 0.35,
        "depthshade_points": False,
        "complex_color": "#8c99a8",
        "complex_face_alpha": 0.07,
        "cycle_face_alpha": 1.0,
        "complex_edge_color": "0.35",
        "cycle_edge_color": None,
        "complex_edge_width": 0.0,
        "cycle_edge_width": 0.25,
        "complex_edge_alpha": None,
        "cycle_edge_alpha": None,
        "antialiased": True,
        "zsort": "average",
        "desaturate_complex": 0.0,
        "shade": True,
    }
    return resolve_style(style_3d, style, "style_3d")

def _plot_at_filtration_generic(
    forest,
    filt_val: float,
    ax=None,
    show: bool = False,
    show_complex: bool = True,
    figsize: tuple[float, float] = (5, 5),
    vertex_size: float | None = None,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_cycles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0,
    point_zorder: float = 4,
    cycle_zorder: float = 5,
    dpi: int | None = None,
    style_2d: Optional[dict[str, Any]] = None,
    style_3d: Optional[dict[str, Any]] = None,
):
    """
    Plot simplicial filtration at a fixed filtration value.

    Dispatches to the 2D or 3D renderer based on ``forest.dim``. This is the
    generic target for ``PersistenceForest.plot_at_filtration``.

    Parameters
    ----------
    forest : object
        Forest-like object exposing ``dim``, ``point_cloud``, ``filtration``,
        ``barcode``, ``_get_color_map``, ``_active_bars_with_cycles_at`` and
        ``_complex_snapshot_at_filtration`` as needed by the selected
        dimension.
    filt_val : float
        Filtration threshold to render.
    ax : matplotlib.axes.Axes | None
        Axes to draw on. A new axes is created when omitted.
    show : bool
        If True, call ``plt.show()`` after plotting.
    show_complex : bool
        Whether to draw complex edges/faces.
    figsize : tuple[float, float]
        Figure size when creating a new figure.
    vertex_size : float or None
        Marker size of the points; None uses 5 in 2D and 1.5 in 3D.
    coloring : {"forest", "bars"}
        Bar color map to use for active cycle representatives.
    title : str | None
        Optional axes title.
    show_cycles : bool
        Whether to draw active cycle representatives.
    signed : bool
        If False, cancel opposite-oriented duplicate simplices before drawing
        cycle representatives.
    min_bar_length : float
        Only draw cycles for bars with lifespan at least this value.
    point_zorder, cycle_zorder : float
        Matplotlib z-order controls used by the 2D renderer.
    dpi : int or None
        Resolution of a newly created figure; None uses Matplotlib's ``figure.dpi``.
    style_2d, style_3d : dict | None
        Renderer-specific style overrides.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the rendered filtration snapshot.
    """
    if vertex_size is None:
        # Points are context in 3D, where hundreds of them crowd the surfaces.
        vertex_size = 5.0 if forest.dim == 2 else 1.5
    if forest.dim == 2:
        return _plot_at_filtration_2d(
            forest=forest,
            filt_val=filt_val,
            ax=ax,
            show=show,
            show_complex=show_complex,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            title=title,
            show_cycles=show_cycles,
            signed=signed,
            min_bar_length = min_bar_length,
            point_zorder=point_zorder,
            cycle_zorder=cycle_zorder,
            dpi = dpi,
            style_2d=style_2d,
        )
    if forest.dim == 3:
        return _plot_at_filtration_3d(
            forest=forest,
            filt_val=filt_val,
            ax=ax,
            show=show,
            show_complex=show_complex,
            show_cycles=show_cycles,
            signed=signed,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            title=title,
            min_bar_length=min_bar_length,
            dpi = dpi,
            style_3d=style_3d,
        )

    raise ValueError("plot_at_filtration is only implemented for ambient dimensions 2 and 3.")

def _plot_at_filtration_2d(
    forest,
    filt_val: float,
    ax=None,
    show: bool = False,
    show_complex: bool = True,
    figsize: tuple[float, float] = (5, 5),
    vertex_size: float = 3,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_cycles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0,
    point_zorder: float = 4,
    cycle_zorder: float = 5,
    dpi: int | None = None,
    style_2d: Optional[dict[str, Any]] = None,
):
    """
    Plot a 2D filtration snapshot with optional active cycle overlays.

    Parameters are the 2D subset forwarded by
    ``_plot_at_filtration_generic``. ``style_2d`` accepts the keys documented
    in ``_resolve_style_2d``.
    """
    color_map = forest._get_color_map(coloring=coloring)
    style = _resolve_style_2d(style_2d)

    pts = np.asarray(forest.point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("point_cloud must be an (n_points, 2) array-like.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    edges_xy = []
    tris_xy = []
    for simplex, f in forest.filtration:
        if f > filt_val:
            break
        if len(simplex) == 2:
            i, j = simplex
            edges_xy.append([pts[i], pts[j]])
        elif len(simplex) == 3:
            i, j, k = simplex
            tris_xy.append([pts[i], pts[j], pts[k]])

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=vertex_size,
        color=style["point_color"],
        alpha=float(style["point_alpha"]),
        label="points",
        marker="o",
        edgecolors="none",
        zorder=point_zorder,
    )

    if show_complex and tris_xy:
        tri_coll = PolyCollection(
            tris_xy,
            closed=True,
            edgecolors="none",
            facecolors=style["complex_face_color"],
            alpha=float(style["complex_face_alpha"]),
            zorder=1,
        )
        ax.add_collection(tri_coll)

    if show_complex and edges_xy:
        edge_coll = LineCollection(
            edges_xy,
            linewidths=float(style["complex_edge_width"]),
            colors=style["complex_edge_color"],
            alpha=(None if style["complex_edge_alpha"] is None
                   else float(style["complex_edge_alpha"])),
            zorder=2,
            label="edges",
        )
        ax.add_collection(edge_coll)

    if show_cycles:
        for bar in forest.barcode:
            if filt_val >= bar.birth and filt_val < bar.death and bar.lifespan()>=min_bar_length:
                cycle = forest.cycle_for_bar_at(bar, filt_val)
                if cycle.dim() != 1:
                    raise ValueError(
                        f"plot_at_filtration expected a 1-chain, got dim={cycle.dim()}"
                    )
                segments = (
                    cycle.segments(point_cloud=forest.point_cloud)
                    if signed
                    else cycle.unsigned().segments(point_cloud=forest.point_cloud)
                )

                add_cycle_lines(ax, segments, color_map[bar], float(style["cycle_edge_width"]), cycle_zorder,
                                halo=bool(style["cycle_halo"]), halo_width=float(style["cycle_halo_width"]),
                                halo_color=style["cycle_halo_color"])

                if bool(style["show_orientation_arrows"]):
                    for seg in segments:
                        (x0, y0), (x1, y1) = np.asarray(seg, dtype=float)
                        dx = x1 - x0
                        dy = y1 - y0
                        length = float(np.hypot(dx, dy))
                        if length == 0.0:
                            continue

                        frac = 0.5
                        mx = 0.5 * (x0 + x1)
                        my = 0.5 * (y0 + y1)
                        ux = dx / length
                        uy = dy / length

                        half = 0.5 * frac * length
                        x_start = mx - ux * half
                        y_start = my - uy * half
                        x_end = mx + ux * half
                        y_end = my + uy * half

                        ax.annotate(
                            "",
                            xy=(x_end, y_end),
                            xytext=(x_start, y_start),
                            arrowprops=dict(
                                arrowstyle="-|>",
                                linewidth=float(style["arrow_linewidth"]),
                                color=color_map[bar],
                                mutation_scale=float(style["arrow_scale"]),
                            ),
                            zorder=6,
                        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(filtration_title(forest, filt_val) if title is None else title)
    style_scene_axes(ax)

    ax.autoscale()
    if show:
        plt.show()
    return ax

def _plot_at_filtration_with_dual_2d(
    forest,
    filt_val: float,
    ax=None,
    fill_triangles: bool = True,
    figsize: tuple[float, float] = (5, 5),
    vertex_size: float = 12,
    coloring: Literal["forest", "bars"] = "forest",
    dual_vertex_size: float = 12,
    show_cycles: bool = True,
    linewidth_filt: float = 0.6,
    linewidth_cycle: float = 1.8,
    linewidth_dual_edge: float = 0.4,
    show_dual: bool = True,
    show: bool = False,
):
    """
    Plot primal and dual edges of the alpha complex at a filtration value.

    Parameters
    ----------
    forest : object
        Forest-like object exposing ``point_cloud``, ``filtration``,
        ``barcode``, ``_get_color_map`` and cycle representatives.
    filt_val : float
        Filtration threshold for displaying simplices.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; a new figure is created if None.
    fill_triangles : bool
        If True, fill triangles that are present at or before filt_val.
    figsize : tuple[float, float]
        Size of the figure if ax is None.
    vertex_size : float
        Marker size for input points.
    coloring : {"forest","bars"}
        Color scheme to use for active cycles.
    dual_vertex_size : float
        Marker size for dual vertices.
    show_cycles : bool
        If True, overlay the active cycles at the filtration value.
    linewidth_filt : float
        Line width for primal edges.
    linewidth_cycle : float
        Line width for cycle edges.
    linewidth_dual_edge : float
        Line width for dual edges.
    show_dual : bool
        If True, overlay the dual edges and dual vertices.
    show : bool
        If True, calls plt.show() when done.

    Returns
    -------
    matplotlib.axes.Axes
    """
    color_map = forest._get_color_map(coloring=coloring)

    pts = np.asarray(forest.point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("point_cloud must be an (n_points, 2) array-like.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    edge_info: dict[Tuple[int, int], tuple[np.ndarray, float]] = {}
    triangle_info: dict[Tuple[int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}
    edge_to_triangles: dict[Tuple[int, int], list[Tuple[int, int, int]]] = defaultdict(list)

    for simplex, filtration in forest.filtration:
        if len(simplex) == 2:
            i, j = simplex
            edge_key = tuple(sorted((i, j)))
            edge_info[edge_key] = (np.array([pts[i], pts[j]]), filtration)
        elif len(simplex) == 3:
            i, j, k = simplex
            tri_key = tuple(sorted((i, j, k)))
            coords = np.array([pts[i], pts[j], pts[k]])
            triangle_info[tri_key] = (coords, coords.mean(axis=0), filtration)
            for edge in itertools.combinations(tri_key, 2):
                edge_to_triangles[tuple(sorted(edge))].append(tri_key)

    edges_present: list[np.ndarray] = []
    edges_future: list[np.ndarray] = []
    for segment, filtration in edge_info.values():
        if filt_val >= filtration:
            edges_present.append(segment)
        else:
            edges_future.append(segment)

    tris_present: list[np.ndarray] = []
    dual_vertices_present: list[np.ndarray] = []
    for coords, barycenter, filtration in triangle_info.values():
        if filtration <= filt_val:
            tris_present.append(coords)
        else:
            dual_vertices_present.append(barycenter)

    dual_edges_present: list[np.ndarray] = []
    dual_edges_future: list[np.ndarray] = []
    for edge_key, tri_keys in edge_to_triangles.items():
        if len(tri_keys) != 2:
            continue
        edge_data = edge_info.get(edge_key)
        tri1 = triangle_info.get(tri_keys[0])
        tri2 = triangle_info.get(tri_keys[1])
        if edge_data is None or tri1 is None or tri2 is None:
            continue

        segment = np.array([tri1[1], tri2[1]])
        if filt_val < edge_data[1]:
            dual_edges_present.append(segment)
        else:
            dual_edges_future.append(segment)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=vertex_size,
        color="k",
        label="points",
        marker="o",
        edgecolors="none",
        zorder=2.8,
    )

    if fill_triangles and tris_present:
        ax.add_collection(
            PolyCollection(
                tris_present,
                closed=True,
                edgecolors="none",
                facecolors="C0",
                alpha=0.2,
                zorder=1,
            )
        )

    if dual_vertices_present and show_dual:
        present_arr = np.array(dual_vertices_present)
        ax.scatter(
            present_arr[:, 0],
            present_arr[:, 1],
            s=dual_vertex_size,
            c="C3",
            marker="o",
            edgecolors="none",
            zorder=2.8,
        )

    if edges_future:
        ax.add_collection(
            LineCollection(
                edges_future,
                linewidths=linewidth_filt,
                colors="0.45",
                alpha=0.5,
                zorder=1.6,
            )
        )

    if edges_present:
        ax.add_collection(
            LineCollection(
                edges_present,
                linewidths=linewidth_filt,
                colors="0.3",
                zorder=2,
                label="edges",
            )
        )

    if show_dual:
        if dual_edges_future:
            ax.add_collection(
                LineCollection(
                    dual_edges_future,
                    colors="C3",
                    linewidths=linewidth_dual_edge,
                    alpha=0.5,
                    linestyle="dotted",
                    zorder=3.6,
                )
            )

        if dual_edges_present:
            ax.add_collection(
                LineCollection(
                    dual_edges_present,
                    colors="C3",
                    linewidths=linewidth_dual_edge,
                    alpha=1,
                    zorder=4,
                    label="dual edges",
                )
            )

    if show_cycles:
        for bar in forest.barcode:
            if filt_val >= bar.birth and filt_val < bar.death:
                cycle = forest.cycle_for_bar_at(bar, filt_val)
                segments = [
                    np.array(pts[list(signed_simplex[0])])
                    for signed_simplex in cycle.signed_simplices
                ]
                add_cycle_lines(ax, segments, color_map[bar], linewidth_cycle, 5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(filtration_title(forest, filt_val))
    style_scene_axes(ax)

    if show:
        plt.show()

    return ax

def _complex_collections_3d(
    forest: PersistenceForest, filt_val: float, points: np.ndarray, style: dict[str, Any],
) -> list[Line3DCollection | Poly3DCollection]:
    """Build the complex's boundary surface (and optional edges) in drawing order.

    Triangles shared by two tetrahedra lie inside the complex and are never
    seen through the shell, so only the boundary is drawn.
    """
    snapshot = forest._complex_snapshot_at_filtration(filt_val=filt_val)
    collections = []
    face_alpha = float(style["complex_face_alpha"])
    edges = snapshot.get("edges", [])
    if edges and float(style["complex_edge_width"]) > 0:
        edge_alpha = style["complex_edge_alpha"]
        collections.append(Line3DCollection(
            [points[list(edge)] for edge in edges],
            colors=style["complex_edge_color"],
            linewidths=float(style["complex_edge_width"]),
            alpha=max(0.2, face_alpha) if edge_alpha is None else float(edge_alpha),
            antialiased=bool(style["antialiased"]),
        ))

    triangles = boundary_triangles(snapshot.get("triangles", []), snapshot.get("tetrahedra", []))
    if triangles:
        complex_rgb = _desaturate_color(style["complex_color"], float(style["desaturate_complex"]))
        collections.append(Poly3DCollection(
            [points[list(triangle)] for triangle in triangles],
            facecolors=(*complex_rgb, face_alpha),
            edgecolors="none",
            antialiased=bool(style["antialiased"]),
            zsort=style["zsort"],
        ))
    return collections


def _cycle_faces_3d(cycle: SignedChain, points: np.ndarray, color: Any, style: dict[str, Any], *,
                    signed: bool) -> tuple[list, np.ndarray, np.ndarray]:
    """Return the triangles of a 2-cycle with their face and edge colors."""
    if cycle.dim() != 2:
        raise ValueError(
            f"plot_at_filtration expected a 2-chain, got dim={cycle.dim()}"
        )
    triangles = [points[list(face)] for face in cycle.simplices(signed=signed)]
    face_alpha = float(style["cycle_face_alpha"])
    if bool(style["shade"]):
        faces = shaded_face_colors(triangles, color, alpha=face_alpha)
    else:
        faces = np.tile(mcolors.to_rgba(color, face_alpha), (len(triangles), 1))
    edge_alpha = 1.0 if style["cycle_edge_alpha"] is None else float(style["cycle_edge_alpha"])
    if style["cycle_edge_color"] is None:
        edges = faces.copy()
        edges[:, :3] *= 0.75
        edges[:, 3] = edge_alpha
    else:
        edges = np.tile(mcolors.to_rgba(style["cycle_edge_color"], edge_alpha), (len(triangles), 1))
    return triangles, faces, edges


def _cycle_collection_3d(
    cycle: SignedChain, points: np.ndarray, color: Any, style: dict[str, Any], *, signed: bool,
) -> Poly3DCollection | None:
    """Build a shaded surface for a single representative."""
    triangles, faces, edges = _cycle_faces_3d(cycle, points, color, style, signed=signed)
    if not triangles:
        return None
    return Poly3DCollection(triangles, facecolors=faces, edgecolors=edges,
                            linewidths=float(style["cycle_edge_width"]),
                            antialiased=bool(style["antialiased"]), zsort=style["zsort"])


def _filtration_title_3d(forest, filt_val: float) -> str:
    return filtration_title(forest, filt_val)


def _plot_at_filtration_3d(
    forest,
    filt_val: float,
    ax=None,
    show: bool = False,
    show_complex: bool = True,
    show_cycles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0.0,
    figsize: tuple[float, float] = (5, 5),
    vertex_size: float = 3.0,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    dpi: int | None = None,
    style_3d: Optional[dict[str, Any]] = None,
):
    """
    Plot a 3D filtration snapshot with optional complex and cycle surfaces.

    Parameters are the 3D subset forwarded by
    ``_plot_at_filtration_generic``. ``style_3d`` accepts the keys documented
    in ``_resolve_style_3d``.
    """
    color_map = forest._get_color_map(coloring=coloring)
    style = _resolve_style_3d(style_3d)
    pts = np.asarray(forest.point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("point_cloud must be an (n_points, 3) array-like.")

    if ax is None:
        fig = plt.figure(figsize=figsize,dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
    elif not hasattr(ax, "zaxis"):
        raise ValueError("3D plotting requires a matplotlib 3D axis (projection='3d').")

    if show_complex:
        for collection in _complex_collections_3d(forest, float(filt_val), pts, style):
            ax.add_collection3d(collection)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=vertex_size,
        c=style["point_color"],
        alpha=float(style["point_alpha"]),
        depthshade=bool(style["depthshade_points"]),
    )

    if show_cycles:
        active = forest._active_bars_with_cycles_at(
            filt_val=float(filt_val),
            min_bar_length=min_bar_length,
        )
        active = sorted(active, key=lambda bc: bc[0].lifespan(), reverse=True)

        # One collection, so faces of different cycles are depth-sorted together.
        triangles, faces, edges = [], [], []
        for bar, cycle in active:
            bar_triangles, bar_faces, bar_edges = _cycle_faces_3d(cycle, pts, color_map[bar], style, signed=signed)
            triangles.extend(bar_triangles)
            faces.append(bar_faces)
            edges.append(bar_edges)
        if triangles:
            ax.add_collection3d(Poly3DCollection(
                triangles, facecolors=np.concatenate(faces), edgecolors=np.concatenate(edges),
                linewidths=float(style["cycle_edge_width"]), antialiased=bool(style["antialiased"]),
                zsort=style["zsort"]))

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    pad = 0.05 * np.max(spans)
    xlim = (mins[0] - pad, maxs[0] + pad)
    ylim = (mins[1] - pad, maxs[1] + pad)
    zlim = (mins[2] - pad, maxs[2] + pad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))

    elev, azim = _camera_from_eye(camera_eye=style["camera_eye"])
    ax.view_init(elev=elev, azim=azim)
    if bool(style["remove_axes"]):
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
    ax.set_title(filtration_title(forest, filt_val) if title is None else title)

    if show:
        plt.show()

    return ax

def plot_barcode_cycle_reps(
    forest,
    min_bar_length: float = 0,
    relative_position: float = 0.1,
    ax=None,
    show: bool = False,
    figsize: tuple[float, float] = (5, 5),
    vertex_size: float = 3,
    point_zorder: float = 3,
    coloring: Literal["forest", "bars"] = "forest",
    title: Optional[str] = None,
    show_orientation_arrows: bool = False,
    remove_double_edges: bool = False,
    linewidth_cycle: float = 1.5,
    style_2d: Optional[dict[str, Any]] = None,
):
    """
    Plot one representative cycle for each sufficiently long barcode bar.

    The representative for each selected bar is sampled at
    ``birth + lifespan * relative_position``. This method is implemented
    only for 2D point clouds.

    Parameters
    ----------
    forest : object
        Forest-like object exposing ``dim``, ``point_cloud``, ``barcode`` and
        ``_get_color_map``.
    min_bar_length : float
        Minimum lifespan of bars to consider.
    relative_position : float
        Relative position in each bar's lifespan, in ``[0, 1)``; bars with
        ``birth == death`` are skipped.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; if None, a new figure+axes are created.
    show : bool
        If True, calls plt.show() when done.
    figsize : tuple[float, float]
        Figure size used when ``ax`` is None.
    vertex_size : float
        Marker size for point cloud.
    point_zorder : float
        Matplotlib z-order for the point cloud markers.
    coloring : {"forest","bars"}
        Color scheme; builds the map on first use.
    title : str | None
        Title for the axes. Defaults to a relative-position summary.
    show_orientation_arrows : bool
        If True, draw small arrows along each cycle edge to indicate
        the orientation of the cycle representatives.
    remove_double_edges : bool
        If True, cancel edges appearing with opposite orientations before
        plotting.
    linewidth_cycle : float
        Line width for cycle edges.
    style_2d : dict | None
        2D style overrides. Supports the same keys as ``plot_at_filtration``.
        For vertex styling, use ``point_color`` and ``point_alpha``.
        ``cycle_edge_width`` is used when supplied; otherwise
        ``linewidth_cycle`` is kept for backward compatibility.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if relative_position < 0 or relative_position >= 1:
        raise ValueError("relative_position must be in [0,1)")

    if forest.dim != 2:
        raise ValueError("plot_barcode_cycle_reps only implemented for dimension 2")

    color_map = forest._get_color_map(coloring=coloring)
    style = _resolve_style_2d(style_2d)
    cycle_linewidth = (
        float(style["cycle_edge_width"])
        if style_2d is not None and "cycle_edge_width" in style_2d
        else linewidth_cycle
    )
    draw_orientation_arrows = show_orientation_arrows or bool(style["show_orientation_arrows"])

    pts = np.asarray(forest.point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("point_cloud must be an (n_points, 2) array-like.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=vertex_size,
        color=style["point_color"],
        alpha=float(style["point_alpha"]),
        zorder=point_zorder,
        label="points",
        marker="o",
        edgecolors="none",
    )

    longest = max((bar.lifespan() for bar in forest.barcode), default=0.0) or 1.0
    for bar in forest.barcode:
        value = _value_at_relative_position(bar, relative_position)
        if bar.lifespan() < min_bar_length or value is None:
            continue
        # Above the points, longer bars on top.
        cycle_zorder = point_zorder + 1 + bar.lifespan() / longest

        cycle = forest.cycle_for_bar_at(bar, value)
        if cycle.dim() != 1:
            raise ValueError(
                f"plot_barcode_cycle_reps expected a 1-chain, got dim={cycle.dim()}"
            )
        segments = (
            cycle.segments(point_cloud=forest.point_cloud)
            if not remove_double_edges
            else cycle.unsigned().segments(point_cloud=forest.point_cloud)
        )
        add_cycle_lines(ax, segments, color_map[bar], cycle_linewidth, cycle_zorder,
                        halo=bool(style["cycle_halo"]), halo_width=float(style["cycle_halo_width"]),
                        halo_color=style["cycle_halo_color"])

        if draw_orientation_arrows:
            for seg in segments:
                (x0, y0), (x1, y1) = np.asarray(seg, dtype=float)
                dx = x1 - x0
                dy = y1 - y0
                length = float(np.hypot(dx, dy))
                if length == 0.0:
                    continue

                frac = 0.5
                mx = 0.5 * (x0 + x1)
                my = 0.5 * (y0 + y1)
                ux = dx / length
                uy = dy / length

                half = 0.5 * frac * length
                x_start = mx - ux * half
                y_start = my - uy * half
                x_end = mx + ux * half
                y_end = my + uy * half

                arrow = ax.annotate(
                    "",
                    xy=(x_end, y_end),
                    xytext=(x_start, y_start),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        linewidth=float(style["arrow_linewidth"]),
                        color=color_map[bar],
                        mutation_scale=float(style["arrow_scale"]),
                    ),
                    zorder=cycle_zorder,
                )
                if arrow.arrow_patch is not None:
                    arrow.arrow_patch.set_zorder(cycle_zorder)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(representatives_title(relative_position) if title is None else title)
    style_scene_axes(ax)
    ax.autoscale()
    if show:
        plt.show()
    return ax
