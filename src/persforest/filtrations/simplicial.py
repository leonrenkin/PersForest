"""Simplicial filtrations embedded in a point cloud."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from persforest.filtrations._validation import _validate_filtration
from persforest.geometry.simplicial import simplex_orientation

Simplex = tuple[int, ...]


def simplicial_boundary(cell: Simplex) -> tuple[tuple[Simplex, int], ...]:
    """Facets of the vertex-ordered simplex with their incidences.

    Deleting vertex i has incidence (-1)**i; deletion keeps faces sorted.
    """
    if len(cell) == 1:
        return ()
    return tuple((cell[:i] + cell[i + 1:], -1 if i % 2 else 1) for i in range(len(cell)))


@dataclass(frozen=True, slots=True, eq=False)
class SimplicialFiltration:
    """Simplicial adapter for the construction engine.

    Cell keys are increasing vertex tuples into ``point_cloud``. The reference
    orientation of a simplex is its vertex order; the ambient orientation of a
    top simplex is the sign of its edge-vector determinant. Top simplices get
    IDs ``0 .. n_top - 1`` and facets the following IDs, each in order of
    first appearance in the events. Facet positions in ``top_facets`` follow
    ``simplicial_boundary``.

    Attributes
    ----------
    point_cloud : ndarray, shape (n_points, top_dimension)
        Vertex coordinates.
    simplices : list of tuple of int
        Cell key of every ID.
    top_dimension, n_top, n_cells, event_cells, event_values, top_facets, facet_incidence
        See ``filtrations._interface.CellFiltration``.
    """

    point_cloud: NDArray
    simplices: list[Simplex]
    n_top: int
    event_cells: NDArray[np.int64]
    event_values: NDArray[np.float64]
    top_facets: NDArray[np.int64]

    @classmethod
    def from_events(cls, point_cloud: NDArray, events: Sequence[tuple[Simplex, float]]) -> "SimplicialFiltration":
        """Number the top simplices and facets of ``events`` (forward order).

        Raises
        ------
        ValueError
            If a facet of a top simplex is not among the events.
        """
        top_dimension = point_cloud.shape[1]
        tops = [simplex for simplex, _ in events if len(simplex) == top_dimension + 1]
        facets = [simplex for simplex, _ in events if len(simplex) == top_dimension]
        simplices = tops + facets
        ids = {simplex: index for index, simplex in enumerate(simplices)}
        try:
            top_facets = [[ids[facet] for facet, _ in simplicial_boundary(top)] for top in tops]
        except KeyError as error:
            raise ValueError(f"Facet {error.args[0]} of a top simplex is missing from the filtration") from None
        return cls(
            point_cloud=point_cloud,
            simplices=simplices,
            n_top=len(tops),
            event_cells=np.fromiter((ids[simplex] for simplex, _ in events), dtype=np.int64, count=len(events)),
            event_values=np.fromiter((value for _, value in events), dtype=np.float64, count=len(events)),
            top_facets=np.asarray(top_facets, dtype=np.int64).reshape(len(tops), top_dimension + 1),
        )

    @property
    def top_dimension(self) -> int:
        return self.point_cloud.shape[1]

    @property
    def n_cells(self) -> int:
        return len(self.simplices)

    @property
    def facet_incidence(self) -> tuple[int, ...]:
        return tuple(-1 if i % 2 else 1 for i in range(self.top_dimension + 1))

    def dimension(self, cell: Simplex) -> int:
        return len(cell) - 1

    def boundary(self, cell: Simplex) -> tuple[tuple[Simplex, int], ...]:
        return simplicial_boundary(cell)

    def orientation(self, top_id: int) -> int:
        return simplex_orientation(simplex=self.simplices[top_id], point_cloud=self.point_cloud)

    def cell_keys(self, cell_ids) -> list[Simplex]:
        simplices = self.simplices
        return [simplices[cell_id] for cell_id in cell_ids]


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
    return SimplicialFiltration.from_events(point_cloud, simplicial_events(entries, ambient_dim))
