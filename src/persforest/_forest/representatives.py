"""_forest.representatives: mechanically extracted reference implementation."""

from collections import defaultdict
from persforest.barcode import PFBar
from persforest.chains import SignedChain
from persforest.filtrations.simplicial import key
from persforest.filtrations.simplicial import signed_boundary
from typing import Iterator
from typing import List
from typing import Set
from typing import Tuple
import time

def _require_barcode_diffs(self) -> None:
    """Reject queries without a completed barcode and accumulated simplex diffs."""
    if not self.keep_simplex_diff:
        raise ValueError("Construct with keep_simplex_diff=True to query interior activity")
    if not self._barcode_computed:
        raise RuntimeError("Barcode has not been computed; call compute_barcode() first")
    if not self._barcode_diffs_ready:
        raise RuntimeError(
            "Barcode must be computed from simplex diffs for this operation. "
            "Rebuild with keep_simplex_diff=True and use compute_barcode() "
            "instead of _compute_barcode_cycles()."
        )

def update_chain_with_diff(signed_simplices: Set[tuple], interior_diff: Set[tuple] | None, codim1_simplex_diff: Set[tuple] | None):
    """
    Apply stored simplex differences to a mutable set of signed simplices.

    Parameters
    ----------
    signed_simplices : set[tuple]
        Current oriented codimension-one simplices, stored as
        ``(simplex_tuple, orientation)``.
    interior_diff : set[tuple] | None
        Oriented full-dimensional simplices whose signed boundaries should be
        added to ``signed_simplices``.
    codim1_simplex_diff : set[tuple] | None
        Oriented codimension-one simplices to remove from ``signed_simplices``.

    Returns
    -------
    set[tuple]
        The same set object after applying the additions and removals.
    """

    if interior_diff is not None:
        for oriented_simplex in interior_diff:
            signed_simplices.update(signed_boundary(simplex =oriented_simplex[0], orientation=oriented_simplex[1]))

    if codim1_simplex_diff is not None:
        for oriented_simplex in codim1_simplex_diff:
            signed_simplices.remove(oriented_simplex)
    
    return signed_simplices

def _compute_loop_activity(self):
    """Annotate each non-root node cycle with its active interval."""

    for node in self.nodes.values():

        if node.parent == None:
            continue
        
        parent = self.nodes[node.parent]
        node.cycle.active_end = node.filt_val
        node.cycle.active_start = parent.filt_val

    return

def iter_bar_cycle_reps(self, bar: PFBar) -> Iterator[SignedChain]:
    """Yield iterator of representatives of a bar in descending filtration order.

        Normal mode yields stored representatives; 
        diff-only mode reconstructs boundaries and interiors with independent simplex sets.

        Raises
        ------
        ValueError
            On iteration, if the bar is foreign or required diffs are missing.
        """
    if bar not in self.barcode:
        raise ValueError("Bar is not in barcode of forest")

    if not self.diff_only_mode:
        yield from reversed(bar.cycle_reps)
        return

    _require_barcode_diffs(self)
    interior = set()
    simplices = set()
    for nid in reversed(bar._node_progression):
        node = self.nodes[nid]
        if not node._simplex_diff_available:
            raise ValueError("node._simplex_diff_available is not available; this should not happen")

        active_start, active_end = self._node_activity(node)
        simplices = update_chain_with_diff(
            signed_simplices=simplices,
            interior_diff=node._barcode_interior_diff,
            codim1_simplex_diff=node._barcode_codim1_simplex_diff,
        )
        simplices = update_chain_with_diff(
            signed_simplices=simplices,
            interior_diff=node._interior_diff,
            codim1_simplex_diff=node._codim1_simplex_diff,
        )
        if node._interior_diff is not None:
            interior.update(node._interior_diff)
        if node._barcode_interior_diff is not None:
            interior.update(node._barcode_interior_diff)

        yield SignedChain(
            signed_simplices=simplices.copy(),
            active_start=active_start,
            active_end=active_end,
            interior_available=True,
            interior=interior.copy(),
        )

def cycle_for_bar_at(self, bar: PFBar, filt_val: float) -> SignedChain:
    """Return a bar's representative active at a filtration value.

        Stored representatives are used in normal mode. In diff-only mode,
        replay the bar's simplex diffs from death toward birth and return the
        first representative active on ``[active_start, active_end)``.

        Parameters
        ----------
        bar : PFBar
            Bar belonging to this forest.
        filt_val : float
            Filtration value in ``[bar.birth, bar.death)``.

        Returns
        -------
        SignedChain
            Representative active at ``filt_val``.
        """
    if bar not in self.barcode:
        raise ValueError("Bar is not in barcode of forest")
    if filt_val < bar.birth or filt_val >= bar.death:
        raise ValueError(f"Filtration value {filt_val} is not in the bar's lifespan")
    if not self.diff_only_mode:
        return bar.cycle_at_filtration_value(filt_val)

    for cycle_rep in self.iter_bar_cycle_reps(bar):
        if cycle_rep.active_start <= filt_val < cycle_rep.active_end:
            return cycle_rep
    raise ValueError(f"Filtration value is in lifespan but no cycle representative was found; this should not happen")

def _add_interior_to_bar(self, bar: PFBar):
    """
        Adds an ``interior`` set containing accumulated full-dimensional simplices to each SignedChain in bar, 
        and sets ``interior_available=True``.

        Raises
        ------
        ValueError
            If any node in the progression does not have simplex diffs.
        """

    if self.diff_only_mode:
        raise ValueError("Saving interiors is not possible with diff_only_mode=True")

    _require_barcode_diffs(self)
    interior = set()
    simplices = set()

    for nid in reversed(bar._node_progression):
        node = self.nodes[nid]

        if not node._simplex_diff_available:
            raise ValueError("set PersistenceForest(..., keep_simplex_diff=True)")

        if node._interior_diff is not None:
            interior.update(node._interior_diff)
        if node._barcode_interior_diff is not None:
            interior.update(node._barcode_interior_diff)

        if node.cycle is None:
            raise ValueError(f"Node with id {node.id} has node.cycle = None. This should not happen.")
        
        node.cycle.interior = interior.copy()
        node.cycle.interior_available = True

    return

def _add_interior_to_barcode(self,print_info: bool =False):
    """
        Replace barcode cycle representatives by diff-reconstructed chains.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.

        Notes
        -----
        Requires the forest to have been built with ``keep_simplex_diff=True``.
        The reconstructed representatives carry accumulated interior simplex
        data used by ``interior_simplex_activity``.
        """
    _require_barcode_diffs(self)
    if print_info:
        print("Computing interior of cycle reps")
    interior_start = time.perf_counter()

    for bar in self.barcode:
        self._add_interior_to_bar(bar)
    
    interior_end = time.perf_counter()-interior_start
    if print_info:
        print(f"interior of cycle reps computed in {interior_end}")

    return

def interior_simplex_activity(self) -> dict[tuple[int, ...], list[tuple[PFBar, float, float]]]:
    """
        Return active interior intervals for full-dimensional simplices.

        The return value maps each unsigned full-dimensional simplex to a list
        of ``(bar, active_start, active_end)`` tuples. The interval is the
        half-open filtration range where the simplex is in the interior of the
        active cycle representative for that bar.

        Returns
        -------
        dict[tuple[int, ...], list[tuple[PFBar, float, float]]]
            Mapping from simplex keys to bars and active intervals.

        Notes
        -----
        Requires ``keep_simplex_diff=True``.
        """
    _require_barcode_diffs(self)

    activity = defaultdict(list)

    for bar in self.barcode:
        active_end_by_simplex: dict[tuple[int, ...], float] = {}
        for node_id in reversed(bar._node_progression):
            node = self.nodes[node_id]
            if not node._simplex_diff_available:
                raise ValueError(f"Interior simplex diffs are unavailable for node {node_id}")

            # The first occurrence in descending order is the last active
            # representative in forward time, including merged branches.
            for interior_diff in (node._interior_diff, node._barcode_interior_diff):
                if interior_diff is not None:
                    for simplex, _orientation in interior_diff:
                        active_end_by_simplex.setdefault(key(simplex), node.filt_val)

        for simplex_key, active_end in active_end_by_simplex.items():
            activity[simplex_key].append(
                (bar, bar.birth, active_end)
            )

    return dict(activity)

def _active_bars_with_cycles_at(
    self,
    filt_val: float,
    min_bar_length: float = 0.0,
) -> List[Tuple["PFBar", SignedChain]]:
    """Return active bars paired with their cycles at ``filt_val``."""
    active = []
    for bar in self.barcode:
        if bar.lifespan() < min_bar_length:
            continue
        if bar.birth <= filt_val < bar.death:
            active.append((bar, self.cycle_for_bar_at(bar, filt_val)))
    return active

def barcode_cycle_reps(self, relative_position=0.1, min_bar_length: float = 0) -> List[SignedChain]:
    """
        Return the list of cycle representatives for all bars in the barcode.

        Parameters
        ----------
        relative_position : float
            Relative position in the barcode (between 0 and 1).
        min_bar_length : float
            Minimum lifespan of bars to consider.

        Returns
        -------
        list[SignedChain]
            Cycle representatives for all bars in the barcode.
        """
    if relative_position < 0 or relative_position >= 1:
        raise ValueError("relative_position must be in [0,1)")

    # Get all bars in the barcode
    all_bars = sorted(list(self.barcode), key=lambda bar: bar.lifespan(), reverse=True)
    
    # Select bars based on relative position and minimum bar length
    selected_bars = [bar for bar in all_bars if bar.lifespan() >= min_bar_length]
    
    # Compute cycle representatives for each selected bar
    cycles = [self.cycle_for_bar_at(bar, bar.birth + (bar.lifespan() * relative_position)) for bar in selected_bars]
    
    return cycles
