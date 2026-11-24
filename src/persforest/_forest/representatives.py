"""_forest.representatives: mechanically extracted reference implementation."""

from collections import defaultdict
from persforest.barcode import PFBar
from persforest.chains import SignedChain
from persforest.filtrations._interface import CellFiltration
from persforest.filtrations._interface import facet_ids
from persforest.filtrations._interface import signed_incidences
from typing import Iterator
from typing import List
from typing import Set
from typing import Tuple
import time

def _require_barcode_diffs(self) -> None:
    """Reject queries without a completed barcode and accumulated simplex diffs."""
    if not self._barcode_computed:
        raise RuntimeError("Barcode has not been computed; call compute_barcode() first")
    if not self._barcode_diffs_ready:
        raise RuntimeError(
            "Accumulated barcode diffs are unavailable; the barcode build did not complete."
        )

def update_chain_with_diff(cells: CellFiltration, signed_simplices: Set[tuple], interior_diff: Set[tuple] | None, codim1_simplex_diff: Set[tuple] | None):
    """
    Apply stored simplex differences to a mutable set of signed simplices.

    Parameters
    ----------
    cells : CellFiltration
        Cell filtration providing the boundary map on cell keys.
    signed_simplices : set[tuple]
        Current oriented codimension-one cells, stored as ``(cell_key, orientation)``.
    interior_diff : set[tuple] | None
        Oriented top cells ``(cell_key, orientation)`` whose signed boundaries
        should be added to ``signed_simplices``.
    codim1_simplex_diff : set[tuple] | None
        Oriented codimension-one cells to remove from ``signed_simplices``.

    Returns
    -------
    set[tuple]
        The same set object after applying the additions and removals.
    """

    if interior_diff is not None:
        for top, orientation in interior_diff:
            signed_simplices.update((facet, orientation * incidence) for facet, incidence in cells.boundary(top))

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

def _iter_bar_diff_steps(self, bar: PFBar) -> Iterator[tuple]:
    """Yield the diff replay of a bar in descending filtration order.

    Each item is ``(node, active_start, active_end, diffs)`` where ``diffs``
    is ``((barcode_interior, barcode_codim1), (interior, codim1))``: the
    accumulated diffs of merged branches, then the node's own diffs, in the
    order in which they must be applied. Applying the diffs of all items up to
    and including one gives the representative of that node.
    """
    _require_barcode_diffs(self)
    for nid in reversed(bar._node_progression):
        node = self.nodes[nid]
        if not node._simplex_diff_available:
            raise ValueError("node._simplex_diff_available is not available; this should not happen")
        active_start, active_end = self._node_activity(node)
        yield node, active_start, active_end, (
            (node._barcode_interior_diff, node._barcode_codim1_simplex_diff),
            (node._interior_diff, node._codim1_simplex_diff),
        )


def _keyed_diffs(cells: CellFiltration, diffs: tuple) -> tuple:
    """Convert the cell IDs of one replay step's diffs to cell keys with a single lookup.

    ``diffs`` is a tuple of ``(interior_diff, codim1_simplex_diff)`` pairs of
    ``(cell_id, orientation)`` sets or None. Converting per step keeps the
    cost proportional to the diffs, not to the representative.
    """
    ids = [cell for pair in diffs for diff in pair if diff is not None for cell, _ in diff]
    if not ids:
        return diffs
    keys = iter(cells.cell_keys(ids))
    # Iterating each unchanged set again visits its entries in the same order.
    return tuple(tuple(None if diff is None else {(next(keys), sign) for _, sign in diff} for diff in pair)
                 for pair in diffs)


def _apply_diff_step(cells: CellFiltration, signed_simplices: set, interior: set, diffs: tuple) -> None:
    """Apply one item of ``_iter_bar_diff_steps`` to live keyed boundary and interior sets."""
    for interior_diff, codim1_simplex_diff in _keyed_diffs(cells, diffs):
        update_chain_with_diff(cells, signed_simplices, interior_diff, codim1_simplex_diff)
        if interior_diff is not None:
            interior.update(interior_diff)


def _apply_id_diff_step(cells: CellFiltration, signed_ids: set, interior_ids: set, diffs: tuple) -> None:
    """Apply one item of ``_iter_bar_diff_steps`` to live sets of ``(cell_id, sign)`` pairs."""
    for interior_diff, codim1_simplex_diff in diffs:
        if interior_diff is not None:
            for top_id, orientation in interior_diff:
                signed_ids.update(zip(facet_ids(cells, top_id), signed_incidences(cells, orientation)))
            interior_ids.update(interior_diff)
        if codim1_simplex_diff is not None:
            for entry in codim1_simplex_diff:
                signed_ids.remove(entry)


def _keyed_set(cells: CellFiltration, signed_ids: set) -> set:
    """Convert a set of ``(cell_id, sign)`` pairs to public ``(cell_key, sign)`` pairs in one lookup."""
    if not signed_ids:
        return set()
    ids, signs = zip(*signed_ids)
    return set(zip(cells.cell_keys(ids), signs))


def iter_bar_cycle_reps(self, bar: PFBar) -> Iterator[SignedChain]:
    """Yield iterator of representatives of a bar in descending filtration order.

        Normal mode yields stored representatives; 
        ``storage="diffs"`` reconstructs boundaries and interiors with independent cell sets.

        Raises
        ------
        ValueError
            On iteration, if the bar is foreign or required diffs are missing.
        """
    if bar not in self.barcode:
        raise ValueError("Bar is not in barcode of forest")

    if self.storage == "cycles":
        yield from reversed(bar.cycle_reps)
        return

    interior = set()
    simplices = set()
    for _node, active_start, active_end, diffs in _iter_bar_diff_steps(self, bar):
        _apply_diff_step(self._cells, simplices, interior, diffs)
        yield SignedChain(
            signed_simplices=simplices.copy(),
            active_start=active_start,
            active_end=active_end,
            interior_available=True,
            interior=interior.copy(),
        )

def cycle_for_bar_at(self, bar: PFBar, filt_val: float) -> SignedChain:
    """Return a bar's representative active at a filtration value.

        Stored representatives are used with ``storage="cycles"``. With ``storage="diffs"``,
        replay the bar's simplex diffs from death toward birth and return the
        first representative active on ``[active_start, active_end)``. The
        replay works on live sets and copies only the returned representative.

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
    if self.storage == "cycles":
        return bar.cycle_at_filtration_value(filt_val)

    # Only the returned representative needs public keys: replay on IDs and
    # convert once, instead of converting every step as the iterator must.
    interior = set()
    simplices = set()
    for _node, active_start, active_end, diffs in _iter_bar_diff_steps(self, bar):
        _apply_id_diff_step(self._cells, simplices, interior, diffs)
        if active_start <= filt_val < active_end:
            return SignedChain(
                signed_simplices=_keyed_set(self._cells, simplices),
                active_start=active_start,
                active_end=active_end,
                interior_available=True,
                interior=_keyed_set(self._cells, interior),
            )
    raise ValueError(f"Filtration value is in lifespan but no cycle representative was found; this should not happen")

def _support_cells(self, support: str) -> list[int]:
    """Return the IDs of all cells an additive measurement with ``support`` can weigh.

    These are the codimension-one cells (``"boundary"``) or the top cells
    (``"interior"``) of the filtration, in event order.
    """
    cells = self._cells
    n_top = cells.n_top
    if support == "boundary":
        return [cell for cell in cells.event_cells.tolist() if cell >= n_top]
    return [cell for cell in cells.event_cells.tolist() if cell < n_top]

# Contribution, in multiples of the cell weight, of a boundary cell whose
# present orientations are encoded as a mask (bit 1: +1, bit 2: -1).
# Signed chains count each stored orientation. Unsigned chains keep a cell iff
# its net coefficient is nonzero, i.e. exactly one orientation is present,
# which is what SignedChain.unsigned() keeps.
_SIGNED_MULTIPLICITY = (0, 1, 1, 2)
_UNSIGNED_MULTIPLICITY = (0, 1, 1, 0)

def _iter_bar_additive_values(self, bar: PFBar, weights: dict, support: str, signed: bool) -> Iterator[tuple[float, float, float]]:
    """Yield ``(active_start, active_end, value)`` of an additive measurement.

    Evaluates the sum of ``weights`` over the support of each representative
    of ``bar``, in the order of ``iter_bar_cycle_reps``, by replaying the
    simplex diffs once. ``weights`` maps the ID of every cell that can occur
    to its weight. The cost is proportional to the size of the diffs, not of the
    representatives.

    The value is a Neumaier-compensated running sum. It is exactly zero when
    no cell contributes, so empty representatives never give rounding residue.
    """
    cells = self._cells
    total = 0.0
    compensation = 0.0
    count = 0

    if support == "interior":
        # Interiors only grow along the replay; entries are (cell, orientation).
        present = set()
        for _node, active_start, active_end, diffs in _iter_bar_diff_steps(self, bar):
            for interior_diff, _codim1_simplex_diff in diffs:
                if interior_diff is None:
                    continue
                for entry in interior_diff:
                    if entry in present:
                        continue
                    present.add(entry)
                    count += 1
                    weight = weights[entry[0]]
                    updated = total + weight
                    if abs(total) >= abs(weight):
                        compensation += (total - updated) + weight
                    else:
                        compensation += (weight - updated) + total
                    total = updated
            yield active_start, active_end, (total + compensation) if count else 0.0
        return

    multiplicity = _SIGNED_MULTIPLICITY if signed else _UNSIGNED_MULTIPLICITY
    masks: dict = {}
    for _node, active_start, active_end, diffs in _iter_bar_diff_steps(self, bar):
        for interior_diff, codim1_simplex_diff in diffs:
            changes = []
            if interior_diff is not None:
                # Adding the boundary of an interior cell; adding a present
                # oriented cell is a no-op, as for the reconstructed sets.
                for interior_cell, orientation in interior_diff:
                    for facet, facet_orientation in zip(facet_ids(cells, interior_cell),
                                                        signed_incidences(cells, orientation)):
                        bit = 1 if facet_orientation > 0 else 2
                        old = masks.get(facet, 0)
                        if old & bit:
                            continue
                        masks[facet] = old | bit
                        changes.append((facet, old, old | bit))
            if codim1_simplex_diff is not None:
                # Removing oriented cells that must be present.
                for facet, facet_orientation in codim1_simplex_diff:
                    bit = 1 if facet_orientation > 0 else 2
                    old = masks.get(facet, 0)
                    if not old & bit:
                        raise KeyError((facet, facet_orientation))
                    masks[facet] = old & ~bit
                    changes.append((facet, old, old & ~bit))
            for facet, old, new in changes:
                delta_units = multiplicity[new] - multiplicity[old]
                if delta_units == 0:
                    continue
                count += delta_units
                value = delta_units * weights[facet]
                updated = total + value
                if abs(total) >= abs(value):
                    compensation += (total - updated) + value
                else:
                    compensation += (value - updated) + total
                total = updated
        yield active_start, active_end, (total + compensation) if count else 0.0

def _add_interior_to_bar(self, bar: PFBar):
    """
        Adds an ``interior`` set containing accumulated full-dimensional simplices to each SignedChain in bar, 
        and sets ``interior_available=True``.

        Raises
        ------
        ValueError
            If any node in the progression does not have simplex diffs.
        """

    if self.storage == "diffs":
        raise ValueError('Saving interiors requires storage="cycles"')

    _require_barcode_diffs(self)
    interior = set()

    for nid in reversed(bar._node_progression):
        node = self.nodes[nid]

        if not node._simplex_diff_available:
            raise ValueError(f"Simplex diffs are unavailable for node {nid}")

        for interior_diff, _ in _keyed_diffs(self._cells, ((node._interior_diff, None),
                                                          (node._barcode_interior_diff, None))):
            if interior_diff is not None:
                interior.update(interior_diff)

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
        Requires ``storage="cycles"``. The reconstructed representatives carry accumulated interior simplex
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
        Requires a computed barcode.
        """
    _require_barcode_diffs(self)

    activity = defaultdict(list)

    for bar in self.barcode:
        active_end_by_simplex: dict[int, float] = {}
        for node_id in reversed(bar._node_progression):
            node = self.nodes[node_id]
            if not node._simplex_diff_available:
                raise ValueError(f"Interior simplex diffs are unavailable for node {node_id}")

            # The first occurrence in descending order is the last active
            # representative in forward time, including merged branches.
            for interior_diff in (node._interior_diff, node._barcode_interior_diff):
                if interior_diff is not None:
                    for simplex, _orientation in interior_diff:
                        active_end_by_simplex.setdefault(simplex, node.filt_val)

        for simplex, active_end in active_end_by_simplex.items():
            activity[simplex].append(
                (bar, bar.birth, active_end)
            )

    keys = self._cells.cell_keys(list(activity))
    return dict(zip(keys, activity.values()))

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
