"""Cubical filtrations of full rectangular grids."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from persforest import _engine

CUBICAL_FILTRATION_TYPES = ("top_cells", "vertices")

Cell = tuple[int, ...]


def cubical_dimension(cell: Cell) -> int:
    """Number of odd doubled coordinates, i.e. axes along which ``cell`` extends."""
    return sum(coordinate & 1 for coordinate in cell)


def cubical_boundary(cell: Cell) -> tuple[tuple[Cell, int], ...]:
    """Facets of the reference-oriented cell with their incidences.

    Along the i-th extended axis, the upper facet has incidence (-1)**i and the
    lower facet the opposite sign.
    """
    facets = []
    position = 0
    for axis, coordinate in enumerate(cell):
        if coordinate & 1:
            sign = -1 if position % 2 else 1
            upper, lower = list(cell), list(cell)
            upper[axis] += 1
            lower[axis] -= 1
            facets.append((tuple(upper), sign))
            facets.append((tuple(lower), -sign))
            position += 1
    return tuple(facets)


@dataclass(frozen=True, slots=True, eq=False)
class CubicalFiltration:
    """Cubical adapter for the construction engine.

    A cell key is the tuple of its doubled coordinates in array-axis order: an
    odd entry marks an axis along which the cell extends, so the dimension is
    the number of odd entries. Every top cell has the reference orientation of
    the array-axis frame (see ``cubical_boundary``).

    Cell IDs follow the event order, top cells first: top cells are numbered
    ``0 .. n_top - 1`` and facets ``n_top .. n_cells - 1``, each in the order in
    which they enter the filtration, so the engine's sweep reads its per-cell
    arrays sequentially. Facet positions in ``top_facets`` follow
    ``cubical_boundary``: (upper, lower) along axis 0, then axis 1, ...

    Attributes
    ----------
    top_dimension, n_top, n_cells, event_cells, event_values, top_facets, facet_incidence
        See ``filtrations._interface.CellFiltration``. Events are ordered by
        (value, dimension, doubled coordinates in row-major order).
    doubled_shape : tuple of int
        Extent ``2 n + 1`` of the doubled coordinates along each axis.
    cell_positions : ndarray of int32, shape (n_cells,)
        Row-major index of each cell's doubled coordinates in ``doubled_shape``.
    """

    top_dimension: int
    n_top: int
    n_cells: int
    event_cells: NDArray[np.int64]
    event_values: NDArray[np.float64]
    top_facets: NDArray[np.int64]
    facet_incidence: tuple[int, ...]
    doubled_shape: tuple[int, ...]
    cell_positions: NDArray[np.int32]

    def dimension(self, cell: Cell) -> int:
        return cubical_dimension(cell)

    def boundary(self, cell: Cell) -> tuple[tuple[Cell, int], ...]:
        return cubical_boundary(cell)

    def orientation(self, top_id: int) -> int:
        return 1

    def top_orientations(self) -> NDArray[np.int8]:
        return np.ones(self.n_top, dtype=np.int8)

    def cell_keys(self, cell_ids) -> list[Cell]:
        if isinstance(cell_ids, (list, tuple)) and len(cell_ids) <= 64:
            # Short lookups (replay steps) are faster without NumPy overhead.
            return [self._cell_key(cell_id) for cell_id in cell_ids]
        ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return []
        if ids.min() < 0 or ids.max() >= self.n_cells:
            raise ValueError("cell ID out of range")
        coordinates = np.unravel_index(self.cell_positions[ids], self.doubled_shape)
        return list(map(tuple, np.stack(coordinates, axis=1).tolist()))

    def _cell_key(self, cell_id: int) -> Cell:
        if not 0 <= cell_id < self.n_cells:
            raise ValueError("cell ID out of range")
        position = int(self.cell_positions[cell_id])
        coordinates = []
        for extent in reversed(self.doubled_shape):
            position, coordinate = divmod(position, extent)
            coordinates.append(coordinate)
        return tuple(reversed(coordinates))


def _validated_grid(values, filtration_type: str) -> NDArray[np.float64]:
    if filtration_type not in CUBICAL_FILTRATION_TYPES:
        raise ValueError(
            f"filtration_type must be one of {CUBICAL_FILTRATION_TYPES}, not {filtration_type!r}")
    try:
        grid = np.array(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("values must be a finite numeric array") from error
    if grid.ndim < 1:
        raise ValueError("values must have at least one axis")
    minimum = 1 if filtration_type == "top_cells" else 2
    if min(grid.shape) < minimum:
        raise ValueError(
            f"values must have at least {minimum} entries along every axis "
            f"for filtration_type={filtration_type!r}")
    if not np.isfinite(grid).all():
        raise ValueError("values must be finite; masks and missing cells are not supported")
    return grid


def _neighbor_extreme(array: NDArray, axis: int, reduce, pad_value: float | None) -> NDArray:
    """Combine each pair of neighbors along ``axis``, optionally padding both ends."""
    if pad_value is not None:
        width = [(0, 0)] * array.ndim
        width[axis] = (1, 1)
        array = np.pad(array, width, constant_values=pad_value)
    before = (slice(None),) * axis
    return reduce(array[before + (slice(None, -1),)], array[before + (slice(1, None),)])


def _cell_values(grid: NDArray, filtration_type: str, above_all=np.inf) -> list[tuple[NDArray, tuple[int, ...]]]:
    """Return ``(values, parities)`` blocks for the top cells and each facet family.

    ``parities[a]`` is 1 if cells in the block extend along axis ``a``; entry
    ``index`` of ``values`` belongs to the cell with doubled coordinates
    ``2 * index + parities``. ``above_all`` exceeds every entry of ``grid``.
    """
    d = grid.ndim
    blocks = []
    if filtration_type == "top_cells":
        blocks.append((grid, (1,) * d))
        for axis in range(d):
            # Facets normal to ``axis`` take the minimum of their incident top
            # cells; facets on the domain boundary have a single neighbor.
            facet_values = _neighbor_extreme(grid, axis, np.minimum, above_all)
            blocks.append((facet_values, tuple(0 if a == axis else 1 for a in range(d))))
    else:
        # Lower-star: a cell takes the maximum of its vertices.
        top = grid
        for axis in range(d):
            top = _neighbor_extreme(top, axis, np.maximum, None)
        blocks.append((top, (1,) * d))
        for axis in range(d):
            facet_values = grid
            for other in range(d):
                if other != axis:
                    facet_values = _neighbor_extreme(facet_values, other, np.maximum, None)
            blocks.append((facet_values, tuple(0 if a == axis else 1 for a in range(d))))
    return blocks


def prepare_cubical_filtration(values, filtration_type: str = "top_cells") -> tuple[NDArray, CubicalFiltration]:
    """Validate a grid and build its ordered top cells and facets.

    Parameters
    ----------
    values : array-like
        Finite values on a full rectangular grid with at least one axis.
    filtration_type : {"top_cells", "vertices"}
        ``"top_cells"``: entries are top cells (pixels or voxels) and a face
        enters at the minimum of its incident top cells. ``"vertices"``: entries
        are vertices and a cell enters at the maximum of its vertices.

    Returns
    -------
    tuple
        The validated value array and the cubical adapter.

    Notes
    -----
    Ties are ordered by dimension (faces first) and then by doubled coordinates
    in row-major order, so the event order is deterministic.
    """
    grid = _validated_grid(values, filtration_type)
    d = grid.ndim
    top_shape = grid.shape if filtration_type == "top_cells" else tuple(n - 1 for n in grid.shape)
    # Every cell takes the minimum or maximum of some grid entries, so ranks
    # among the distinct grid values order the cells as their values do. The
    # engine then sorts by (rank, dimension, doubled coordinates in row-major
    # order) in linear time instead of comparison-sorting all cells, and
    # numbers the cells in that order.
    levels, rank = np.unique(grid, return_inverse=True)
    n_levels = len(levels)
    del levels
    rank_blocks = _cell_values(rank.reshape(grid.shape).astype(np.int32), filtration_type, above_all=n_levels)
    del rank
    cell_rank = np.concatenate([block_rank.reshape(-1) for block_rank, _ in rank_blocks])
    del rank_blocks
    cell_value = np.concatenate([block_values.reshape(-1) for block_values, _ in _cell_values(grid, filtration_type)])
    events = _engine.cubical_events(list(top_shape), cell_rank, cell_value, n_levels)
    del cell_rank, cell_value
    n_top = int(np.prod(top_shape))
    cells = CubicalFiltration(
        top_dimension=d,
        n_top=n_top,
        n_cells=len(events["event_cells"]),
        event_cells=events["event_cells"],
        event_values=events["event_values"],
        top_facets=events["top_facets"].reshape(n_top, 2 * d),
        facet_incidence=tuple(sign for axis in range(d) for sign in ((1, -1) if axis % 2 == 0 else (-1, 1))),
        doubled_shape=tuple(2 * n + 1 for n in top_shape),
        cell_positions=events["cell_positions"],
    )
    return grid, cells
