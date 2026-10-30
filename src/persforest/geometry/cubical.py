"""Physical geometry of cubical grids."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from persforest.chains import SignedChain


@dataclass(frozen=True, slots=True)
class CubicalGeometry:
    """Physical placement of a cubical grid.

    Along axis ``a``, doubled coordinate ``c`` lies at
    ``origin[a] + (c / 2) * spacing[a]``. With ``filtration_type="top_cells"``
    the origin is the lower corner of the first pixel or voxel; with
    ``"vertices"`` it is the first vertex.

    Attributes
    ----------
    shape : tuple of int
        Shape of the value array.
    spacing : tuple of float
        Positive cell size per array axis.
    origin : tuple of float
        Position of doubled coordinate 0 per array axis.
    filtration_type : {"top_cells", "vertices"}
        Convention of the value array.
    """

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    filtration_type: str

    @property
    def dim(self) -> int:
        return len(self.shape)

    def coordinates(self, cell: tuple[int, ...]) -> NDArray[np.float64]:
        """Return the physical center of ``cell``."""
        return np.asarray(self.origin) + np.asarray(cell, dtype=float) / 2 * np.asarray(self.spacing)

    def cell_measure(self, cell: tuple[int, ...]) -> float:
        """Return the length, area or volume of ``cell`` (1 for a vertex)."""
        return math.prod(s for c, s in zip(cell, self.spacing) if c & 1)

    # Simplicial measurements index or convert their second argument as a point
    # array; fail with an actionable message instead of an unrelated error.
    def _reject_point_cloud_use(self, *args, **kwargs):
        raise TypeError(
            "This measurement expects a point cloud, but the forest was built from a "
            "cubical grid. Use persforest.measurements.cubical (e.g. "
            "cubical_chain_volume, cubical_chain_interior_volume), a loop functional "
            "wrapped by persforest.measurements.loops.on_raw_loops (or on_taut_loops, "
            "on_smoothed_loops, ...), or a callable "
            "accepting (chain, CubicalGeometry).")

    __array__ = _reject_point_cloud_use
    __getitem__ = _reject_point_cloud_use


def cubical_chain_to_vertex_chain(signed_chain: SignedChain, geometry: CubicalGeometry) -> tuple[SignedChain, NDArray[np.float64]]:
    """Rewrite a chain of grid edges as a chain of vertex-index edges.

    Grid vertices are numbered in sorted cell order and placed at their
    physical position. An edge cell runs along its odd axis, in the positive
    direction for orientation +1. Its vertex edge is stored as an increasing
    index pair whose orientation is +1 exactly when travel goes from the lower
    to the higher index, the convention of simplicial 1-chains. Opposite copies
    of an edge are kept, so the result can be passed, with the returned points,
    to the simplicial functionals and path extraction.

    Parameters
    ----------
    signed_chain : SignedChain
        Chain of edges of a 2D grid (doubled coordinates).
    geometry : CubicalGeometry
        Grid geometry of the forest (``forest.geometry``).

    Returns
    -------
    vertex_chain : SignedChain
        Chain of oriented vertex-index edges, without interior data.
    points : ndarray, shape (n_vertices, 2)
        Physical vertex positions.

    Raises
    ------
    TypeError
        If ``geometry`` is not a ``CubicalGeometry``.
    ValueError
        If the grid is not 2D or the chain contains cells other than edges.
    """
    if not isinstance(geometry, CubicalGeometry):
        raise TypeError("cubical_chain_to_vertex_chain needs the CubicalGeometry of a cubical forest")
    if geometry.dim != 2:
        raise ValueError(f"Cubical 1-cycles live on 2D grids; this grid has {geometry.dim} axes")
    edges = []
    for cell, orientation in sorted(signed_chain.signed_simplices):
        odd_axes = [axis for axis, coordinate in enumerate(cell) if coordinate & 1]
        if len(cell) != 2 or len(odd_axes) != 1:
            raise ValueError(f"Expected edges of a 2D grid, got cell {cell}")
        lower, upper = list(cell), list(cell)
        lower[odd_axes[0]] -= 1
        upper[odd_axes[0]] += 1
        edges.append((tuple(lower), tuple(upper), orientation))

    index = {vertex: i for i, vertex in enumerate(sorted({v for edge in edges for v in edge[:2]}))}
    points = np.array([geometry.coordinates(vertex) for vertex in index], dtype=float).reshape(-1, 2)
    # Grid vertices are numbered in increasing order, so lower < upper.
    vertex_chain = SignedChain({((index[lower], index[upper]), orientation) for lower, upper, orientation in edges})
    return vertex_chain, points


def make_cubical_geometry(shape: tuple[int, ...], spacing, origin, filtration_type: str) -> CubicalGeometry:
    """Validate ``spacing`` and ``origin`` (scalars or one value per axis)."""
    d = len(shape)

    def per_axis(value, default, name):
        if value is None:
            return (default,) * d
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
        if array.ndim == 0:
            array = np.full(d, float(array))
        if array.shape != (d,):
            raise ValueError(f"{name} must be a scalar or have one value per axis ({d})")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return tuple(float(v) for v in array)

    spacing = per_axis(spacing, 1.0, "spacing")
    if min(spacing) <= 0:
        raise ValueError("spacing must be positive")
    return CubicalGeometry(tuple(shape), spacing, per_axis(origin, 0.0, "origin"), filtration_type)
