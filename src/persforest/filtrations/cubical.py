"""Cubical filtrations of full rectangular grids."""

import bisect
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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

    Cell IDs number the blocks of ``_cell_values`` consecutively: first the top
    cells in row-major order, then the facets normal to axis 0, axis 1, ...,
    each block in row-major order. Facet positions in ``top_facets`` follow
    ``cubical_boundary``: (upper, lower) along axis 0, then axis 1, ...

    Attributes
    ----------
    top_dimension, n_top, n_cells, event_cells, event_values, top_facets, facet_incidence
        See ``filtrations._interface.CellFiltration``. Events are ordered by
        (value, dimension, doubled coordinates in row-major order).
    block_offsets : tuple of int
        First cell ID of each block, followed by ``n_cells``.
    block_shapes : tuple of tuple of int
        Index shape of each block.
    block_parities : tuple of tuple of int
        Doubled-coordinate parity of each block's cells.
    """

    top_dimension: int
    n_top: int
    n_cells: int
    event_cells: NDArray[np.int64]
    event_values: NDArray[np.float64]
    top_facets: NDArray[np.int64]
    facet_incidence: tuple[int, ...]
    block_offsets: tuple[int, ...]
    block_shapes: tuple[tuple[int, ...], ...]
    block_parities: tuple[tuple[int, ...], ...]

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
        coordinates = np.empty((ids.size, self.top_dimension), dtype=np.int64)
        block_of = np.searchsorted(np.asarray(self.block_offsets), ids, side="right") - 1
        for block in np.unique(block_of).tolist():
            mask = block_of == block
            index = np.unravel_index(ids[mask] - self.block_offsets[block], self.block_shapes[block])
            coordinates[mask] = 2 * np.stack(index, axis=1) + np.asarray(self.block_parities[block])
        return list(map(tuple, coordinates.tolist()))

    def _cell_key(self, cell_id: int) -> Cell:
        if not 0 <= cell_id < self.n_cells:
            raise ValueError("cell ID out of range")
        block = bisect.bisect_right(self.block_offsets, cell_id) - 1
        local = cell_id - self.block_offsets[block]
        coordinates = []
        for extent, parity in zip(reversed(self.block_shapes[block]), reversed(self.block_parities[block])):
            local, index = divmod(local, extent)
            coordinates.append(2 * index + parity)
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
    lower = np.take(array, np.arange(array.shape[axis] - 1), axis=axis)
    upper = np.take(array, np.arange(1, array.shape[axis]), axis=axis)
    return reduce(lower, upper)


def _cell_values(grid: NDArray, filtration_type: str) -> list[tuple[NDArray, tuple[int, ...]]]:
    """Return ``(values, parities)`` blocks for the top cells and each facet family.

    ``parities[a]`` is 1 if cells in the block extend along axis ``a``; entry
    ``index`` of ``values`` belongs to the cell with doubled coordinates
    ``2 * index + parities``.
    """
    d = grid.ndim
    blocks = []
    if filtration_type == "top_cells":
        blocks.append((grid, (1,) * d))
        for axis in range(d):
            # Facets normal to ``axis`` take the minimum of their incident top
            # cells; facets on the domain boundary have a single neighbor.
            facet_values = _neighbor_extreme(grid, axis, np.minimum, np.inf)
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


def _top_facets(top_shape: tuple[int, ...], block_offsets: list[int]) -> NDArray[np.int64]:
    """Facet IDs of every top cell, in ``cubical_boundary`` order.

    Top cell ``i`` (doubled coordinates ``2 i + 1``) has, along axis ``a``, the
    facets ``i + e_a`` (upper) and ``i`` (lower) of block ``1 + a``, whose
    shape is ``top_shape + e_a``; this holds for both filtration types.
    """
    d = len(top_shape)
    index = np.indices(top_shape).reshape(d, -1)
    columns = []
    for axis in range(d):
        shape = tuple(n + (a == axis) for a, n in enumerate(top_shape))
        upper = index.copy()
        upper[axis] += 1
        columns.append(block_offsets[1 + axis] + np.ravel_multi_index(upper, shape))
        columns.append(block_offsets[1 + axis] + np.ravel_multi_index(index, shape))
    return np.stack(columns, axis=1).astype(np.int64, copy=False)


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
    blocks = _cell_values(grid, filtration_type)
    shapes = [block_values.shape for block_values, _ in blocks]
    parities = [parities for _, parities in blocks]
    offsets = np.concatenate(([0], np.cumsum([block_values.size for block_values, _ in blocks]))).tolist()
    # Doubled coordinates span 2 * (top shape) + 1 along every axis.
    doubled_shape = tuple(2 * n + 1 for n in shapes[0])
    row_major, cell_values, dimensions = [], [], []
    for block_values, block_parities in blocks:
        index = np.indices(block_values.shape).reshape(d, -1)
        coordinates = 2 * index + np.array(block_parities)[:, None]
        row_major.append(np.ravel_multi_index(coordinates, doubled_shape))
        cell_values.append(block_values.reshape(-1))
        dimensions.append(np.full(block_values.size, sum(block_parities), dtype=np.int8))
    row_major = np.concatenate(row_major)
    cell_values = np.concatenate(cell_values)
    order = np.lexsort((row_major, np.concatenate(dimensions), cell_values))
    del row_major
    cells = CubicalFiltration(
        top_dimension=d,
        n_top=int(offsets[1]),
        n_cells=int(offsets[-1]),
        event_cells=order.astype(np.int64, copy=False),
        event_values=cell_values[order],
        top_facets=_top_facets(shapes[0], offsets),
        facet_incidence=tuple(sign for axis in range(d) for sign in ((1, -1) if axis % 2 == 0 else (-1, 1))),
        block_offsets=tuple(offsets),
        block_shapes=tuple(tuple(shape) for shape in shapes),
        block_parities=tuple(tuple(p) for p in parities),
    )
    return grid, cells
