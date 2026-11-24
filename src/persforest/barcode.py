"""barcode: the bars of a persistence forest."""

from collections.abc import Sequence
from typing import Any


class PFBar:
    """
    A bar of a persistence forest: a codimension-one class from birth to death.

    The representative of the class changes at ``critical_values``; between
    consecutive critical values it is constant. Representatives are obtained
    from the forest (``cycle_for_bar_at``, ``iter_bar_cycle_reps``), and so is
    the forest structure (``parent_bar``, ``child_bars``). Bars compare and
    hash by identity.

    Attributes
    ----------
    birth, death : float
        Filtration values at which the class is born and dies.
    tree_id : int or None
        Index of the tree of the forest that contains the bar; bars of one tree
        are connected through ``parent_bar``.
    index : int or None
        Position in the forest's deterministic bar order.
    """

    __slots__ = ("birth", "death", "tree_id", "_critical_values", "_arrays", "_index")

    def __init__(self, birth: float, death: float, critical_values: Sequence[float] = (),
                 tree_id: int | None = None):
        """
        Initialize a bar.

        Parameters
        ----------
        birth, death : float
            Filtration values at which the class is born and dies.
        critical_values : sequence of float
            Values at which the representative changes, ascending, ending at
            ``death``.
        tree_id : int, optional
            Index of the tree that contains the bar.
        """
        self.birth = birth
        self.death = death
        self.tree_id = tree_id
        self._critical_values: tuple[float, ...] | None = tuple(critical_values)
        self._arrays: Any = None
        self._index: int | None = None

    @classmethod
    def _from_forest(cls, arrays: Any, index: int, birth: float, death: float, tree_id: int) -> "PFBar":
        """Bar ``index`` of a forest's engine arrays; critical values are read on first use."""
        bar = cls(birth, death, (), tree_id)
        bar._critical_values = None
        bar._arrays = arrays
        bar._index = index
        return bar

    @property
    def critical_values(self) -> tuple[float, ...]:
        """Values at which the representative changes, ascending, ending at ``death``.

        The representative active on ``[birth, critical_values[0])`` changes at
        each critical value; the last one is the death.
        """
        if self._critical_values is None:
            self._critical_values = tuple(self._arrays.critical_values(self._index).tolist())
        return self._critical_values

    @property
    def index(self) -> int | None:
        """Position of the bar in its forest's deterministic bar order.

        The order is the same for the same input and options, so it breaks
        ties between otherwise equal bars reproducibly. None for bars not
        created by a forest.
        """
        return self._index

    def lifespan(self) -> float:
        """Return ``death - birth``."""
        return self.death - self.birth

    def __repr__(self) -> str:
        return f"PFBar(birth={self.birth!r}, death={self.death!r})"
