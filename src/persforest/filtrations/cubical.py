"""Cubical filtrations of full rectangular grids."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CUBICAL_FILTRATION_TYPES = ("top_cells", "vertices")

Cell = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CubicalFiltration:
    """Cubical adapter for the construction engine.

    A cell is the tuple of its doubled coordinates in array-axis order: an odd
    entry marks an axis along which the cell extends, so the dimension is the
    number of odd entries. Every top cell has the reference orientation of the
    array-axis frame; along its i-th extended axis, the upper facet has
    incidence (-1)**i and the lower facet the opposite sign.

    Attributes
    ----------
    top_dimension : int
        Number of array axes.
    events : list of (tuple of int, float)
        Top cells and their facets, ordered by (value, dimension, cell).
    """

    top_dimension: int
    events: list[tuple[Cell, float]]

    def dimension(self, cell: Cell) -> int:
        return sum(coordinate & 1 for coordinate in cell)

    def boundary(self, cell: Cell) -> tuple[tuple[Cell, int], ...]:
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

    def orientation(self, cell: Cell) -> int:
        return 1


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
    coordinates, cell_values, dimensions = [], [], []
    for block_values, parities in _cell_values(grid, filtration_type):
        index = np.indices(block_values.shape).reshape(d, -1).T
        coordinates.append(2 * index + np.array(parities))
        cell_values.append(block_values.reshape(-1))
        dimensions.append(np.full(block_values.size, sum(parities)))
    coordinates = np.concatenate(coordinates)
    cell_values = np.concatenate(cell_values)
    dimensions = np.concatenate(dimensions)
    row_major = np.ravel_multi_index(coordinates.T, tuple(coordinates.max(axis=0) + 1))
    order = np.lexsort((row_major, dimensions, cell_values))
    events = list(zip(map(tuple, coordinates[order].tolist()), cell_values[order].tolist()))
    return grid, CubicalFiltration(d, events)
