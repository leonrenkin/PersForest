"""Internal contract between filtration adapters and the construction engine."""

from collections.abc import Hashable, Sequence
from typing import Protocol, TypeVar

CellId = TypeVar("CellId", bound=Hashable)


class CellFiltration(Protocol[CellId]):
    """Filtered cell complex as consumed by the reverse sweep.

    Attributes
    ----------
    top_dimension : int
        Dimension of the top cells. The tracked homology has dimension
        ``top_dimension - 1``.
    events : sequence of (cell, float)
        Cells in forward filtration order with their filtration values. It
        contains every cell of dimension ``top_dimension`` and
        ``top_dimension - 1``; the sweep ignores cells of lower dimension.

    Notes
    -----
    Cell IDs are tuples, canonical within the adapter, and identical in
    ``events`` and ``boundary``. Signed chains store ``(cell, orientation)``
    pairs, so an ID must not depend on orientation.
    """

    top_dimension: int
    events: Sequence[tuple[CellId, float]]

    def dimension(self, cell: CellId) -> int:
        """Return the dimension of ``cell``."""
        ...

    def boundary(self, cell: CellId) -> tuple[tuple[CellId, int], ...]:
        """Return facets of the reference-oriented cell with their incidences.

        The order is deterministic, every facet has dimension one less than
        ``cell``, vertices have empty boundary, and the linear extension
        satisfies boundary-of-boundary zero.
        """
        ...

    def orientation(self, cell: CellId) -> int:
        """Return +1 or -1, aligning a top cell's reference orientation with the ambient one.

        Called once per top cell, in sweep order. May raise ``ValueError`` for
        a degenerate cell.
        """
        ...


def oriented_boundary(
    cells: CellFiltration[CellId], cell: CellId, orientation: int
) -> tuple[tuple[CellId, int], ...]:
    """Return the boundary of ``cell`` multiplied by ``orientation`` (+1 or -1)."""
    return tuple((facet, orientation * coefficient) for facet, coefficient in cells.boundary(cell))
