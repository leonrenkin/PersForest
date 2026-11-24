"""_forest.representatives: cycle representatives from the engine's forest arrays.

The representative of a node consists of all diffs in its subtree, which is
one contiguous slice of the arrays (see ``_forest.arrays``). Moving along a bar
from its death toward its birth adds at most two slices per step; the replay
converts only those entries to public cell keys.
"""

from collections import defaultdict
from typing import Iterator, List, Tuple

import numpy as np

from persforest._forest.arrays import chain_ids
from persforest.barcode import PFBar
from persforest.chains import SignedChain


def _keyed(cells, ids, signs) -> set:
    """``(cell_key, sign)`` pairs of ID and sign lists or arrays."""
    if len(ids) == 0:
        return set()
    if isinstance(ids, np.ndarray):
        ids, signs = ids.tolist(), signs.tolist()
    return set(zip(cells.cell_keys(ids), signs))


def _replay_steps(self, index: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(node, cells, signs)`` along bar ``index`` from its death toward its birth.

    The first item holds the diff entries of the leaf's subtree, every later
    one the entries the next node's subtree adds.
    """
    arrays = self._forest
    previous = None
    for node in reversed(arrays.progression(index).tolist()):
        entries = arrays.subtree_entries(node) if previous is None else arrays.step_entries(previous, node)
        yield node, entries[0], entries[1]
        previous = node


def iter_bar_cycle_reps(self, bar: PFBar) -> Iterator[SignedChain]:
    """Yield the representatives of a bar in descending filtration order.

    Raises
    ------
    ValueError
        On iteration, if the bar does not belong to the forest.
    """
    index = self._require_own_bar(bar)
    arrays, cells = self._forest, self._cells
    top_facets = np.asarray(cells.top_facets)
    incidence = np.asarray(cells.facet_incidence, dtype=np.int64)
    simplices: set = set()
    interior: set = set()
    for node, entries, signs in _replay_steps(self, index):
        is_top = entries < cells.n_top
        tops, orientations = entries[is_top], signs[is_top].astype(np.int64)
        if tops.size:
            facet_signs = (orientations[:, None] * incidence[None, :]).reshape(-1)
            simplices.update(_keyed(cells, top_facets[tops].reshape(-1), facet_signs))
            interior.update(_keyed(cells, tops, orientations))
        removed = entries[~is_top]
        if removed.size:
            # Both top cells of a removed facet lie in the subtree: drop both orientations.
            for key in cells.cell_keys(removed.tolist()):
                simplices.discard((key, 1))
                simplices.discard((key, -1))
        active_start, active_end = arrays.active_interval(node)
        yield SignedChain(signed_simplices=simplices.copy(), active_start=active_start,
                          active_end=active_end, interior_available=True, interior=interior.copy())


def cycle_for_bar_at(self, bar: PFBar, filt_val: float) -> SignedChain:
    """Return a bar's representative active at a filtration value.

    Parameters
    ----------
    bar : PFBar
        Bar belonging to this forest.
    filt_val : float
        Filtration value in ``[bar.birth, bar.death)``.

    Returns
    -------
    SignedChain
        Representative active at ``filt_val``, with its interior.
    """
    index = self._require_own_bar(bar)
    if filt_val < bar.birth or filt_val >= bar.death:
        raise ValueError(f"Filtration value {filt_val} is not in the bar's lifespan")
    arrays, cells = self._forest, self._cells
    nodes = arrays.progression(index)
    # Node j is active on [value(node j - 1), value(node j)); its predecessor's value is the birth.
    node = int(nodes[np.searchsorted(arrays.node_value[nodes], filt_val, side="right")])
    facets, facet_signs, tops, orientations = chain_ids(cells, *arrays.subtree_entries(node))
    active_start, active_end = arrays.active_interval(node)
    return SignedChain(signed_simplices=_keyed(cells, facets, facet_signs), active_start=active_start,
                       active_end=active_end, interior_available=True,
                       interior=_keyed(cells, tops, orientations))


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


def _iter_bar_additive_values(self, bar: PFBar, weights: dict, support: str, signed: bool,
                              ) -> Iterator[tuple[float, float, float]]:
    """Yield ``(active_start, active_end, value)`` of an additive measurement.

    Evaluates the sum of ``weights`` over the support of each representative
    of ``bar``, in the order of ``iter_bar_cycle_reps``, by replaying the
    diffs once. ``weights`` maps the ID of every cell that can occur to its
    weight. The cost is proportional to the size of the diffs, not of the
    representatives.

    The value is a Neumaier-compensated running sum. It is exactly zero when
    no cell contributes, so empty representatives never give rounding residue.
    """
    index = self._require_own_bar(bar)
    arrays, cells = self._forest, self._cells
    n_top = cells.n_top
    total = 0.0
    compensation = 0.0
    count = 0

    def add(value):
        nonlocal total, compensation
        updated = total + value
        if abs(total) >= abs(value):
            compensation += (total - updated) + value
        else:
            compensation += (value - updated) + total
        total = updated

    if support == "interior":
        for node, entries, _signs in _replay_steps(self, index):
            for top in entries[entries < n_top].tolist():
                count += 1
                add(weights[top])
            active_start, active_end = arrays.active_interval(node)
            yield active_start, active_end, (total + compensation) if count else 0.0
        return

    top_facets = np.asarray(cells.top_facets)
    incidence = tuple(cells.facet_incidence)
    multiplicity = _SIGNED_MULTIPLICITY if signed else _UNSIGNED_MULTIPLICITY
    masks: dict = {}
    for node, entries, signs in _replay_steps(self, index):
        is_top = entries < n_top
        changes = []
        tops = entries[is_top]
        # Adding the boundary of an interior cell; adding a present oriented
        # cell is a no-op, as for the reconstructed sets.
        for facets, orientation in zip(top_facets[tops].tolist(), signs[is_top].tolist()):
            for facet, facet_incidence in zip(facets, incidence):
                bit = 1 if orientation * facet_incidence > 0 else 2
                old = masks.get(facet, 0)
                if old & bit:
                    continue
                masks[facet] = old | bit
                changes.append((facet, old, old | bit))
        # Removing a facet in both orientations, which must be present.
        for facet in entries[~is_top].tolist():
            old = masks.get(facet, 0)
            if old != 3:
                raise KeyError(facet)
            masks[facet] = 0
            changes.append((facet, old, 0))
        for facet, old, new in changes:
            delta_units = multiplicity[new] - multiplicity[old]
            if delta_units == 0:
                continue
            count += delta_units
            add(delta_units * weights[facet])
        active_start, active_end = arrays.active_interval(node)
        yield active_start, active_end, (total + compensation) if count else 0.0


def interior_simplex_activity(self) -> dict[tuple[int, ...], list[tuple[PFBar, float, float]]]:
    """
    Return active interior intervals for full-dimensional simplices.

    The return value maps each unsigned full-dimensional simplex to a list of
    ``(bar, active_start, active_end)`` tuples: the simplex lies in the interior
    of the bar's representatives on that half-open interval.
    """
    if not self._barcode_computed:
        raise RuntimeError("Barcode has not been computed; call compute_barcode() first")
    arrays, cells = self._forest, self._cells
    activity = defaultdict(list)
    for bar in self._bars:
        # Walking from the death toward the birth, a top cell's activity ends
        # at the first node whose subtree contains it.
        active_end_by_top: dict[int, float] = {}
        for node, entries, _signs in _replay_steps(self, bar._index):
            value = float(arrays.node_value[node])
            for top in entries[entries < cells.n_top].tolist():
                active_end_by_top.setdefault(top, value)
        for top, active_end in active_end_by_top.items():
            activity[top].append((bar, bar.birth, active_end))
    keys = cells.cell_keys(list(activity))
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
