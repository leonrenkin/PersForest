"""barcode: mechanically extracted reference implementation."""

from persforest.chains import SignedChain
from typing import Optional

class PFBar:
    """ 
    Persistence bar together with a progression of cycle representatives.

    For an ambient ``dim``-dimensional point cloud, these bars represent
    codimension-one homology classes. Each cycle representative has
    ``active_start`` and ``active_end`` attributes giving the interval on which
    this representative is active.
    The cycle reps are a strictly decreasing chain w.r.t. inclusion.
    """

    def __init__(self, 
                 birth: float, 
                 death: float, 
                 _node_progression: tuple[int,...], 
                 cycle_reps: list[SignedChain], 
                 is_max_tree_bar: Optional[bool]=None, 
                 root_id: Optional[int]=None):
        """
        Initialize a persistence bar together with its representative cycles.

        Parameters
        ----------
        birth : float
            Filtration value where the class is born.
        death : float
            Filtration value where the class dies.
        _node_progression : tuple[int, ...]
            Node ids (from leaves to root) describing where the representative changes.
        cycle_reps : list[SignedChain]
            Cycle representatives active on consecutive subintervals.
        is_max_tree_bar : bool | None
            True if this bar is the longest in its tree, False if truncated by
            another bar, None if unknown.
        root_id : int | None
            Id of the tree root that contains this bar.
        """
        self.birth = birth
        self.death = death
        self._node_progression = _node_progression #nodes saved as node_ids
        self.cycle_reps = cycle_reps
        self.is_max_tree_bar = is_max_tree_bar
        self.root_id = root_id


    def cycle_at_filtration_value(self, filt_val)->SignedChain:
        """
        Return the active representative cycle at a given filtration value.

        Parameters
        ----------
        filt_val : float
            Filtration value inside ``[birth, death)`` of the bar.

        Returns
        -------
        SignedChain
            Cycle representative active at ``filt_val``.

        Raises
        ------
        ValueError
            If ``filt_val`` lies outside the lifespan, no representative is
            active, or representatives were not stored (as in diff-only mode).
        """

        if filt_val < self.birth:
            raise ValueError(f"Filtration value {filt_val} is too small and not in lifespan of the bar")
        if filt_val >= self.death:
            raise ValueError(f"Filtration value {filt_val} is too large and not in lifespan of the bar")
        if not self.cycle_reps or any(cycle is None for cycle in self.cycle_reps):
            raise ValueError(
                "This bar has no stored cycle representative (possibly because "
                "PersistenceForest was built with diff_only_mode=True). "
                "Use forest.cycle_for_bar_at(bar, filt_val) instead."
            )

        if len(self._node_progression)==1:
            return self.cycle_reps[0]

        first = 0
        last = len(self.cycle_reps)-1
        best = None

        while first<=last:
            midpoint = (first+last) // 2

            if self.cycle_reps[midpoint].active_start <= filt_val:
                best = self.cycle_reps[midpoint]
                first = midpoint+1
            else:
                last = midpoint-1

        if best == None:
            raise ValueError(f"Binary search in barcode returned {None}, check if bar is empty list")
        elif best.active_start>filt_val or best.active_end<= filt_val:
            raise ValueError("Output of binary search incorrect, loop not active at filtration value. Check correctness of loop_at_filtration method.")
        
        return best
    
    def lifespan(self):
        """Returns lifespan of bar"""
        return self.death - self.birth
