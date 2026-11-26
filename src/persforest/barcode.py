"""barcode: the bars of a persistence forest."""

from collections.abc import Sequence
from typing import Any


class PFBar:
    """
    A bar of a persistence forest: a codimension-one class from birth to death.

    The representative of the class changes at ``critical_values``; between
    consecutive critical values it is constant. Representatives are obtained
    from the forest (``cycle_for_bar_at``, ``iter_bar_cycle_reps``), and so is
    the forest structure (``parent_bar``, ``child_bars``). Bars compare by
    identity. A forest's bars hash by ``index``, so iterating over
    ``forest.barcode`` follows the deterministic bar order in every process;
    other bars hash by identity.

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
    def _all_from_forest(cls, arrays: Any) -> list["PFBar"]:
        """All bars of a forest's engine arrays, in bar order; critical values are read on first use."""
        # One bar per class of the forest: setting the slots directly instead
        # of calling __init__ halves the creation time.
        new = cls.__new__
        bars: list[PFBar] = []
        append = bars.append
        for index, (birth, death, tree_id) in enumerate(zip(
                arrays.bar_birth.tolist(), arrays.bar_death.tolist(), arrays.bar_tree.tolist())):
            bar = new(cls)
            bar.birth = birth
            bar.death = death
            bar.tree_id = tree_id
            bar._critical_values = None
            bar._arrays = arrays
            bar._index = index
            append(bar)
        return bars

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

    def __hash__(self) -> int:
        # Identity hashes depend on memory addresses, which would make the
        # iteration order of a barcode set, and every tie broken by it, vary
        # between runs. Equality stays identity: equal bars have equal indices.
        index = self._index
        return object.__hash__(self) if index is None else hash(index)

    def __repr__(self) -> str:
        return f"PFBar(birth={self.birth!r}, death={self.death!r})"
