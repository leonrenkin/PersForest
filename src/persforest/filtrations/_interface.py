"""Internal contract between filtration adapters and the construction engine."""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Protocol


class CellFiltration(Protocol):
    """Filtered cell complex as consumed by the reverse sweep.

    The engine works on integer cell IDs. IDs ``0 .. n_top - 1`` are the top
    cells and IDs ``n_top .. n_cells - 1`` their facets; lower-dimensional
    cells have no ID. Public chains use the adapter's hashable cell keys
    (doubled coordinates or vertex tuples), obtained with ``cell_keys``.

    Attributes
    ----------
    top_dimension : int
        Dimension of the top cells. The tracked homology has dimension
        ``top_dimension - 1``.
    n_top, n_cells : int
        Number of top cells, and of top cells and facets together.
    event_cells : ndarray of int64, shape (n_cells,)
        Cell IDs in forward filtration order.
    event_values : ndarray of float64, shape (n_cells,)
        Filtration values of ``event_cells``.
    top_facets : ndarray of int64, shape (n_top, k)
        Facet IDs of each top cell, in a fixed deterministic order.
    facet_incidence : tuple of int, length k
        Incidence of the facet at each position of ``top_facets`` in the
        boundary of the reference-oriented top cell. The linear extension
        satisfies boundary-of-boundary zero.
    """

    top_dimension: int
    n_top: int
    n_cells: int
    event_cells: Any
    event_values: Any
    top_facets: Any
    facet_incidence: tuple[int, ...]

    def orientation(self, top_id: int) -> int:
        """Return +1 or -1, aligning a top cell's reference orientation with the ambient one.

        Called once per top cell, in sweep order. May raise ``ValueError`` for
        a degenerate cell.
        """
        ...

    def cell_keys(self, cell_ids: Iterable[int]) -> list[Hashable]:
        """Return the public keys of ``cell_ids``, in the given order."""
        ...

    def boundary(self, cell_key: Hashable) -> tuple[tuple[Hashable, int], ...]:
        """Return the facet keys of the reference-oriented cell with their incidences.

        Agrees with ``top_facets`` and ``facet_incidence`` on top cells. Used
        to assemble public chains from keyed diffs.
        """
        ...


def signed_incidences(cells: CellFiltration, orientation: int) -> tuple[int, ...]:
    """Return ``facet_incidence`` multiplied by ``orientation`` (+1 or -1)."""
    if orientation > 0:
        return tuple(cells.facet_incidence)
    return tuple(-incidence for incidence in cells.facet_incidence)


def facet_ids(cells: CellFiltration, top_id: int) -> Sequence[int]:
    """Return the facet IDs of one top cell as a list of ints."""
    return cells.top_facets[top_id].tolist()
