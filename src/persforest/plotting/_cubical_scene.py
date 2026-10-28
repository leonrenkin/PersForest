"""Backend-independent geometry of cubical cells for the grid renderers.

Cells are tuples of doubled coordinates (see ``filtrations.cubical``); this
module turns them into physical points, segments and quadrilaterals. In 2D,
plot coordinates put array axis 0 along x and axis 1 along y unless
``image_orientation`` is set, which draws axis 0 downward and axis 1 to the
right as an image viewer would.
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from persforest.filtrations.cubical import _neighbor_extreme
from persforest.geometry.cubical import CubicalGeometry

# The value image sits beneath colored cycles, so it is a neutral grayscale
# softened toward white: intensities stay readable without competing with the
# bar colors.
IMAGE_STYLE = {
    "image_cmap": "gray",
    "image_alpha": 0.75,
    "image_colorbar": True,
    "colorbar_label": "value",
}

# Backend-neutral 2D filtration defaults; the Matplotlib keys for outlines and
# arrows are added by ``cubical_mpl``.
FLAT_FILTRATION_STYLE = {
    **IMAGE_STYLE,
    "complex_face_color": "C0",
    "complex_face_alpha": 0.2,
    "absent_veil_alpha": 0.65,
}


def resolve_style(style: Optional[dict[str, Any]], defaults: dict[str, Any], name: str) -> dict[str, Any]:
    """Merge overrides, rejecting point-cloud keys that have no grid meaning."""
    resolved = dict(defaults)
    if style:
        unknown = sorted(set(style) - set(defaults))
        if unknown:
            raise ValueError(
                f"{name} keys {unknown} are not supported for cubical grids; "
                f"supported keys are {sorted(defaults)}.")
        resolved.update(style)
    return resolved


def require_grid_dimension(forest, method: str, supported: tuple[int, ...] = (2, 3)) -> CubicalGeometry:
    """Return the forest's grid geometry, rejecting unsupported numbers of axes."""
    geometry = forest.geometry
    if geometry.dim == 1:
        raise ValueError(
            f"{method} is not defined for 1D cubical grids, whose bars track H0 of a "
            f"signal; use plot_barcode or plot_persistence_forest.")
    if geometry.dim not in supported:
        dimensions = " and ".join(f"{d}D" for d in supported)
        raise ValueError(
            f"{method} is implemented for {dimensions} cubical grids; this grid has "
            f"{geometry.dim} axes.")
    return geometry


def resolve_flat_options(geometry: CubicalGeometry, image_orientation: bool,
                         show_image: bool | None = None) -> bool:
    """Reject the 2D-only display options on other grids and resolve ``show_image``.

    ``show_image=None`` shows the value image on 2D grids, the default view.
    """
    for name, value in (("image_orientation", image_orientation), ("show_image", show_image)):
        if value and geometry.dim != 2:
            raise ValueError(f"{name} only applies to 2D grids.")
    return geometry.dim == 2 if show_image is None else bool(show_image)


def top_cell_shape(geometry: CubicalGeometry) -> tuple[int, ...]:
    if geometry.filtration_type == "top_cells":
        return geometry.shape
    return tuple(n - 1 for n in geometry.shape)


def top_cell_values(forest) -> NDArray[np.float64]:
    """Filtration value of every top cell, indexed like the top cells."""
    values = np.asarray(forest.grid_values, dtype=float)
    if forest.geometry.filtration_type == "vertices":
        # Lower-star: a top cell takes the maximum of its vertices.
        for axis in range(values.ndim):
            values = _neighbor_extreme(values, axis, np.maximum, None)
    return values


def present_top_cells(forest, max_value: float | None) -> NDArray[np.bool_]:
    """Mask of the top cells with value at most ``max_value`` (all if None)."""
    values = top_cell_values(forest)
    return np.ones(values.shape, dtype=bool) if max_value is None else values <= max_value


def physical(geometry: CubicalGeometry, cells: NDArray) -> NDArray[np.float64]:
    """Vectorized ``geometry.coordinates`` for an array of doubled coordinates."""
    return np.asarray(geometry.origin) + cells / 2 * np.asarray(geometry.spacing)


def domain_bounds(geometry: CubicalGeometry) -> tuple[NDArray, NDArray]:
    """Lower and upper physical corner of the union of top cells."""
    lower = np.asarray(geometry.origin, dtype=float)
    return lower, lower + np.asarray(top_cell_shape(geometry)) * np.asarray(geometry.spacing)


def value_image_bounds(geometry: CubicalGeometry) -> tuple[NDArray, NDArray]:
    """Physical extent of ``grid_values`` drawn as one pixel per entry.

    Top-cell entries are the top cells themselves; vertex entries are drawn as
    pixels centered on their vertices and so overhang the domain by half a cell.
    """
    lower, upper = domain_bounds(geometry)
    if geometry.filtration_type == "vertices":
        half = np.asarray(geometry.spacing) / 2
        return lower - half, upper + half
    return lower, upper


def display(points: NDArray, image_orientation: bool) -> NDArray:
    """Map physical 2D ``(axis 0, axis 1)`` coordinates to plot ``(x, y)``."""
    return points[..., ::-1] if image_orientation else points


def _signed_cells(chain, dimension: int, ambient: int) -> tuple[NDArray, NDArray]:
    if not chain.signed_simplices:
        return np.empty((0, ambient), dtype=int), np.empty(0, dtype=int)
    cells, signs = zip(*sorted(chain.signed_simplices))
    cells = np.asarray(cells, dtype=int)
    if cells.shape[1] != ambient or not ((cells & 1).sum(axis=1) == dimension).all():
        raise ValueError(f"Expected a chain of {dimension}-cells of a {ambient}D cubical grid.")
    return cells, np.asarray(signs, dtype=int)


def edge_segments(chain, geometry: CubicalGeometry) -> NDArray[np.float64]:
    """Return oriented physical ``(n, 2, 2)`` segments of a 1-chain of grid edges.

    The boundary of an edge is its upper minus its lower vertex, so an edge
    with orientation +1 runs toward increasing coordinate along its axis.
    """
    cells, signs = _signed_cells(chain, 1, 2)
    extent = cells & 1
    forward = (signs > 0)[:, None]
    start = np.where(forward, cells - extent, cells + extent)
    end = np.where(forward, cells + extent, cells - extent)
    return np.stack([physical(geometry, start), physical(geometry, end)], axis=1).reshape(-1, 2, 2)


def face_quads(cells: NDArray, geometry: CubicalGeometry, signs: NDArray | None = None) -> NDArray[np.float64]:
    """Return physical ``(n, 4, 3)`` corners of 2-cells of a 3D grid.

    Corners circle counterclockwise in the frame of the two extended axes,
    which is the reference orientation; a sign of -1 reverses the order.
    """
    cells = np.asarray(cells, dtype=int).reshape(-1, 3)
    extended = np.argsort(1 - (cells & 1), axis=1, kind="stable")[:, :2]
    first, second = np.zeros_like(cells), np.zeros_like(cells)
    rows = np.arange(len(cells))
    first[rows, extended[:, 0]] = 1
    second[rows, extended[:, 1]] = 1
    corners = np.stack([cells - first - second, cells + first - second,
                        cells + first + second, cells - first + second], axis=1)
    if signs is not None:
        corners = np.where((np.asarray(signs) < 0)[:, None, None], corners[:, ::-1], corners)
    return physical(geometry, corners)


def chain_face_quads(chain, geometry: CubicalGeometry) -> NDArray[np.float64]:
    """Return oriented corners of the 2-cells of a 2-chain of a 3D grid."""
    cells, signs = _signed_cells(chain, 2, 3)
    return face_quads(cells, geometry, signs)


def voxel_faces(voxels: NDArray, exposed_only: bool) -> tuple[NDArray, NDArray]:
    """Return the facets of voxels and, for each, the row of its voxel.

    ``voxels`` holds top-cell indices. With ``exposed_only``, facets shared by
    two of the voxels cancel, leaving the boundary of their union.
    """
    voxels = np.asarray(voxels, dtype=int).reshape(-1, 3)
    centers = 2 * voxels + 1
    offsets = np.concatenate([np.eye(3, dtype=int), -np.eye(3, dtype=int)])
    faces = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    owners = np.repeat(np.arange(len(voxels)), len(offsets))
    if exposed_only and len(faces):
        _, first, counts = np.unique(faces, axis=0, return_index=True, return_counts=True)
        keep = np.sort(first[counts == 1])
        faces, owners = faces[keep], owners[keep]
    return faces, owners


def quads_to_triangles(quads: NDArray) -> NDArray:
    """Split ``(n, 4, k)`` quadrilaterals into ``(2n, 3, k)`` triangles."""
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def boundary_faces(present: NDArray[np.bool_]) -> NDArray:
    """Doubled coordinates of the facets of the union of the ``present`` voxels.

    A facet is on the boundary where a present voxel has an absent neighbor
    across it; padding makes the domain boundary count as absent.
    """
    padded = np.pad(present, 1)
    inner = (slice(1, -1),) * present.ndim
    faces = []
    for axis in range(present.ndim):
        offset = np.eye(present.ndim, dtype=int)[axis]
        for step in (1, -1):
            neighbor = np.roll(padded, -step, axis=axis)[inner]
            faces.append(2 * np.argwhere(present & ~neighbor) + 1 + step * offset)
    return np.concatenate(faces)


def present_voxel_quads(forest, geometry: CubicalGeometry, max_value: float | None) -> NDArray[np.float64]:
    """Corners of the boundary faces of the voxels with value at most ``max_value``."""
    return face_quads(boundary_faces(present_top_cells(forest, max_value)), geometry)


def present_outline_segments(present: NDArray[np.bool_], geometry: CubicalGeometry) -> NDArray[np.float64]:
    """Physical ``(n, 2, 2)`` grid edges separating present from absent pixels.

    The domain boundary counts as absent, so the segments outline the union of
    the ``present`` top cells of a 2D grid.
    """
    padded = np.pad(present, 1)
    segments = []
    for axis in range(2):
        # Edges normal to ``axis`` lie between consecutive top cells along it.
        lower = padded[(slice(None, -1), slice(1, -1)) if axis == 0 else (slice(1, -1), slice(None, -1))]
        upper = padded[(slice(1, None), slice(1, -1)) if axis == 0 else (slice(1, -1), slice(1, None))]
        index = np.argwhere(lower != upper)
        extent = np.array([0, 1] if axis == 0 else [1, 0])
        vertex = 2 * index
        segments.append(np.stack([vertex, vertex + 2 * extent], axis=1))
    return physical(geometry, np.concatenate(segments))


def quad_outlines(quads: NDArray) -> NDArray:
    """Segments ``(4n, 2, 3)`` along the edges of quadrilaterals."""
    return np.stack([quads, np.roll(quads, -1, axis=1)], axis=2).reshape(-1, 2, 3)


def top_cell_index(cell: tuple[int, ...]) -> tuple[int, ...]:
    if not all(coordinate & 1 for coordinate in cell):
        raise ValueError(f"Expected a top cell of a cubical grid, got {cell}.")
    return tuple(coordinate // 2 for coordinate in cell)


def rows_by_bar(rows) -> dict[Any, list[tuple[tuple[int, ...], float]]]:
    """Group activity rows ``(cell, bar, length)`` into ``{bar: [(top-cell index, length)]}``."""
    by_bar: dict = {}
    for cell, bar, length in rows:
        by_bar.setdefault(bar, []).append((top_cell_index(cell), float(length)))
    return by_bar


def activity_voxel_quads(rows, geometry, surface: bool, cell_shrink: float = 1.0):
    """Faces of active voxels per bar: ``[(bar, quads, activity_lengths, cells)]``.

    With ``surface``, faces shared by two voxels of the same bar cancel, so each
    bar shows the boundary of its region; otherwise every voxel shows all six
    faces, shrunk toward its center by ``cell_shrink``.
    """
    groups = []
    for bar, entries in rows_by_bar(rows).items():
        voxels = np.array([index for index, _ in entries], dtype=int).reshape(-1, 3)
        lengths = np.array([length for _, length in entries], dtype=float)
        faces, owners = voxel_faces(voxels, exposed_only=surface)
        quads = face_quads(faces, geometry)
        if not surface and cell_shrink != 1.0:
            centers = physical(geometry, 2 * voxels[owners] + 1)[:, None, :]
            quads = centers + cell_shrink * (quads - centers)
        cells = [tuple(int(c) for c in 2 * voxels[owner] + 1) for owner in owners]
        groups.append((bar, quads, lengths[owners], cells))
    return groups
