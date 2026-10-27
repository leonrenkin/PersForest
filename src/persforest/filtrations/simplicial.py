"""Simplicial filtrations embedded in a point cloud."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from numpy.typing import NDArray

from persforest.filtrations._validation import _validate_filtration
from persforest.geometry.simplicial import simplex_orientation

Simplex = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SimplicialFiltration:
    """Simplicial adapter for the construction engine.

    Simplices are increasing vertex tuples into ``point_cloud``. The reference
    orientation of a simplex is its vertex order; the ambient orientation of a
    top simplex is the sign of its edge-vector determinant.

    Attributes
    ----------
    point_cloud : ndarray, shape (n_points, top_dimension)
        Vertex coordinates.
    events : list of (tuple of int, float)
        Top simplices and their facets in forward filtration order.
    """

    point_cloud: NDArray
    events: list[tuple[Simplex, float]]

    @property
    def top_dimension(self) -> int:
        return self.point_cloud.shape[1]

    def dimension(self, cell: Simplex) -> int:
        return len(cell) - 1

    def boundary(self, cell: Simplex) -> tuple[tuple[Simplex, int], ...]:
        # Deleting vertex i has incidence (-1)**i; deletion keeps faces sorted.
        if len(cell) == 1:
            return ()
        return tuple((cell[:i] + cell[i + 1:], -1 if i % 2 else 1) for i in range(len(cell)))

    def orientation(self, cell: Simplex) -> int:
        return simplex_orientation(simplex=cell, point_cloud=self.point_cloud)


def simplicial_events(
    filtration: Iterable[tuple[Sequence[int], float]], top_dimension: int,
) -> list[tuple[Simplex, float]]:
    """Keep top simplices and their facets in order, as vertex tuples.

    Parameters
    ----------
    filtration : iterable of (sequence of int, float)
        Ordered simplices with increasing vertex order.
    top_dimension : int
        Dimension of the top simplices.

    Returns
    -------
    list of (tuple of int, float)
        Entries with at least ``top_dimension`` vertices, in their original order.
    """
    return [(tuple(simplex), filtration_value) for simplex, filtration_value in filtration
            if len(simplex) >= top_dimension]


def prepare_simplicial_filtration(
    filtration: Iterable[tuple[Sequence[int], float]], point_cloud: NDArray,
) -> SimplicialFiltration:
    """Validate custom input without changing its values or caller tie order.

    Parameters
    ----------
    filtration : iterable of (sequence of int, float)
        Ordered simplicial input.
    point_cloud : ndarray, shape (n_points, dim)
        Validated vertex coordinates.

    Returns
    -------
    SimplicialFiltration
        Adapter over the validated, canonically ordered simplices.
    """
    n_points, ambient_dim = point_cloud.shape
    entries = _validate_filtration(filtration, n_points, ambient_dim)
    return SimplicialFiltration(point_cloud, simplicial_events(entries, ambient_dim))
