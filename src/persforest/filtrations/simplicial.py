"""Simplicial filtrations embedded in a point cloud."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain, compress, islice
from operator import itemgetter

import numpy as np
from numpy.typing import NDArray

from persforest import _engine
from persforest.filtrations._validation import _validate_filtration
from persforest.geometry.simplicial import simplex_orientation, simplex_orientations

Simplex = tuple[int, ...]

# Events read per chunk in SimplicialFiltration.from_events: bounds the Python
# objects alive at once while keeping the per-chunk NumPy overhead negligible.
_EVENT_CHUNK = 1 << 20


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
    ``simplicial_boundary``. Vertices are stored as arrays; the key tuples
    are built on the first ``cell_keys`` call and kept.

    Attributes
    ----------
    point_cloud : ndarray, shape (n_points, top_dimension)
        Vertex coordinates.
    top_vertices : ndarray of int64, shape (n_top, top_dimension + 1)
        Increasing vertex ids of top cell ``i`` in row ``i``.
    facet_vertices : ndarray of int64, shape (n_cells - n_top, top_dimension)
        Increasing vertex ids of facet ``n_top + i`` in row ``i``.
    top_dimension, n_top, n_cells, event_cells, event_values, top_facets, facet_incidence
        See ``filtrations._interface.CellFiltration``.
    """

    point_cloud: NDArray
    n_top: int
    n_cells: int
    top_vertices: NDArray[np.int64]
    facet_vertices: NDArray[np.int64]
    event_cells: NDArray[np.int64]
    event_values: NDArray[np.float64]
    top_facets: NDArray[np.int64]
    _keys: list[Simplex] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_events(cls, point_cloud: NDArray,
                    events: Iterable[tuple[Sequence[int], float]]) -> "SimplicialFiltration":
        """Number the top simplices and facets of ordered events.

        Parameters
        ----------
        point_cloud : ndarray, shape (n_points, dim)
            Vertex coordinates; top simplices have ``dim + 1`` vertices.
        events : iterable of (sequence of int, float)
            Simplices with increasing vertex ids and their filtration values,
            in forward filtration order. Simplices with fewer than ``dim``
            vertices are skipped. The events are read once, in chunks, so an
            iterator over a large filtration is never held in memory whole.

        Raises
        ------
        ValueError
            If a simplex has more than ``dim + 1`` vertices or a vertex id out
            of range or out of order, if a facet repeats, or if a facet of a
            top simplex is not among the events.
        """
        top_dimension = point_cloud.shape[1]
        events = iter(events)
        parts = [_top_and_facet_arrays(chunk, top_dimension)
                 for chunk in iter(lambda: list(islice(events, _EVENT_CHUNK)), [])]
        lengths, values, vertices = (np.concatenate(part) for part in zip(*parts)) if parts else (
            np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0, dtype=np.int64))
        starts = np.cumsum(lengths) - lengths
        is_top = lengths == top_dimension + 1
        top_vertices = vertices[starts[is_top][:, None] + np.arange(top_dimension + 1)]
        facet_vertices = vertices[starts[~is_top][:, None] + np.arange(top_dimension)]
        n_top = len(top_vertices)
        top_facets = _engine.simplicial_top_facets(top_vertices, facet_vertices, len(point_cloud))
        return cls(
            point_cloud=point_cloud,
            n_top=n_top,
            n_cells=n_top + len(facet_vertices),
            top_vertices=top_vertices,
            facet_vertices=facet_vertices,
            # IDs in event order: top cells 0 .. n_top - 1, then facets n_top .. n_cells - 1.
            event_cells=np.where(is_top, np.cumsum(is_top) - 1, n_top + np.cumsum(~is_top) - 1),
            event_values=values,
            top_facets=top_facets.reshape(n_top, top_dimension + 1),
        )

    @property
    def top_dimension(self) -> int:
        return self.point_cloud.shape[1]

    @property
    def facet_incidence(self) -> tuple[int, ...]:
        return tuple(-1 if i % 2 else 1 for i in range(self.top_dimension + 1))

    def dimension(self, cell: Simplex) -> int:
        return len(cell) - 1

    def boundary(self, cell: Simplex) -> tuple[tuple[Simplex, int], ...]:
        return simplicial_boundary(cell)

    def orientation(self, top_id: int) -> int:
        return simplex_orientation(simplex=self.top_vertices[top_id], point_cloud=self.point_cloud)

    def top_orientations(self) -> NDArray[np.int8]:
        return simplex_orientations(self.top_vertices, self.point_cloud)

    def cell_keys(self, cell_ids) -> list[Simplex]:
        keys = self._keys
        if keys is None:
            keys = self._build_keys()
        if isinstance(cell_ids, np.ndarray):
            cell_ids = cell_ids.tolist()
        elif not isinstance(cell_ids, (list, tuple, range)):
            cell_ids = list(cell_ids)
        try:
            result = [keys[cell_id] for cell_id in cell_ids]
        except IndexError:
            raise ValueError("cell ID out of range") from None
        if result and min(cell_ids) < 0:
            raise ValueError("cell ID out of range")
        return result

    def _build_keys(self) -> list[Simplex]:
        # Built on first use and kept: representatives look up a few keys at a
        # time, for which indexing a list of tuples is fastest.
        keys = list(map(tuple, self.top_vertices.tolist()))
        keys += map(tuple, self.facet_vertices.tolist())
        object.__setattr__(self, "_keys", keys)
        return keys


def _top_and_facet_arrays(
    events: list[tuple[Sequence[int], float]], top_dimension: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    """Vertex counts, values and concatenated vertices of the events with at least ``top_dimension`` vertices."""
    simplices = list(map(itemgetter(0), events))
    lengths = np.fromiter(map(len, simplices), dtype=np.int64, count=len(simplices))
    if lengths.max() > top_dimension + 1:
        raise ValueError(f"Simplices in R^{top_dimension} have at most {top_dimension + 1} vertices")
    kept = lengths >= top_dimension
    values = np.fromiter(map(itemgetter(1), events), dtype=np.float64, count=len(events))[kept]
    vertices = np.fromiter(chain.from_iterable(compress(simplices, kept.tolist())), dtype=np.int64,
                           count=int(lengths[kept].sum()))
    return lengths[kept], values, vertices


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
    return SimplicialFiltration.from_events(point_cloud, entries)
