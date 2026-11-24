"""_forest.arrays: the persistence forest as flat arrays from the compiled engine.

Nodes are numbered in depth-first post-order, so the subtree of node ``v`` is
the contiguous node range ``[node_first[v], v]`` and its diff entries are one
slice of ``diff_cells``/``diff_signs``. The representative of a node consists of
all diffs in its subtree: every top cell in it enters with its orientation
(adding its signed boundary), and every facet in it is removed in both
orientations. Moving along a bar from one node to its parent therefore adds at
most two contiguous slices.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from persforest import _engine
from persforest.filtrations._interface import CellFiltration


@dataclass(frozen=True, slots=True, eq=False)
class ForestArrays:
    """Output of ``persforest._engine.build_forest``; see ``cpp/engine.hpp``."""

    node_value: NDArray[np.float64]
    node_parent: NDArray[np.int64]
    node_first: NDArray[np.int64]
    diff_offsets: NDArray[np.int64]
    diff_cells: NDArray[np.int64]
    diff_signs: NDArray[np.int8]
    bar_birth: NDArray[np.float64]
    bar_death: NDArray[np.float64]
    bar_parent: NDArray[np.int64]
    bar_tree: NDArray[np.int64]
    progression_offsets: NDArray[np.int64]
    progression_nodes: NDArray[np.int64]

    @property
    def n_bars(self) -> int:
        return len(self.bar_birth)

    def progression(self, bar: int) -> NDArray[np.int64]:
        """Nodes of ``bar`` by ascending value, ending with the leaf at its death."""
        return self.progression_nodes[self.progression_offsets[bar]:self.progression_offsets[bar + 1]]

    def critical_values(self, bar: int) -> NDArray[np.float64]:
        """Values at which the representative of ``bar`` changes, ascending, ending at its death."""
        return self.node_value[self.progression(bar)]

    def active_interval(self, node: int) -> tuple[float, float]:
        """Half-open interval ``[start, end)`` on which the representative of ``node`` is active."""
        parent = self.node_parent[node]
        end = float(self.node_value[node])
        return (end if parent < 0 else float(self.node_value[parent])), end

    def diff_entries(self, first_node: int, last_node: int) -> tuple[NDArray[np.int64], NDArray[np.int8]]:
        """Diff entries of the nodes ``first_node .. last_node`` (inclusive; empty if reversed)."""
        if last_node < first_node:
            return self.diff_cells[:0], self.diff_signs[:0]
        start, stop = self.diff_offsets[first_node], self.diff_offsets[last_node + 1]
        return self.diff_cells[start:stop], self.diff_signs[start:stop]

    def subtree_entries(self, node: int) -> tuple[NDArray[np.int64], NDArray[np.int8]]:
        """Diff entries of the subtree of ``node``, whose representative they form."""
        return self.diff_entries(int(self.node_first[node]), node)

    def step_entries(self, child: int, node: int) -> tuple[NDArray[np.int64], NDArray[np.int8]]:
        """Diff entries in the subtree of ``node`` but not in that of its descendant ``child``."""
        before = self.diff_entries(int(self.node_first[node]), int(self.node_first[child]) - 1)
        after = self.diff_entries(child + 1, node)
        return np.concatenate((before[0], after[0])), np.concatenate((before[1], after[1]))


@dataclass(slots=True)
class TreeNode:
    """Node of the forest's tree view used for plotting: ID, value, parent ID (or None), child IDs."""

    id: int
    filt_val: float
    parent: int | None
    children: set[int]


def build_forest_arrays(cells: CellFiltration, *, reduce: bool, filtration_tol: float) -> ForestArrays:
    """Run the compiled engine on a cell filtration."""
    top_facets = np.ascontiguousarray(cells.top_facets, dtype=np.int64).reshape(
        cells.n_top, len(cells.facet_incidence))
    result = _engine.build_forest(
        event_cells=np.ascontiguousarray(cells.event_cells, dtype=np.int64),
        event_values=np.ascontiguousarray(cells.event_values, dtype=np.float64),
        n_top=int(cells.n_top),
        n_cells=int(cells.n_cells),
        top_facets=top_facets,
        top_orientation=np.ascontiguousarray(cells.top_orientations(), dtype=np.int8),
        reduce=bool(reduce),
        filtration_tol=float(filtration_tol),
    )
    return ForestArrays(**result)


def chain_ids(cells: CellFiltration, diff_cells: NDArray[np.int64], diff_signs: NDArray[np.int8],
              ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Signed boundary and interior of a set of diff entries, as ID arrays.

    Returns ``(facets, facet_signs, tops, top_orientations)``: the signed facets
    of the entered top cells that no entry removes (each pair once), and the
    entered top cells with their orientations.
    """
    is_top = diff_cells < cells.n_top
    tops = diff_cells[is_top]
    orientations = diff_signs[is_top].astype(np.int64)
    removed = diff_cells[~is_top]
    facets = np.asarray(cells.top_facets)[tops].reshape(-1)
    signs = (orientations[:, None] * np.asarray(cells.facet_incidence, dtype=np.int64)[None, :]).reshape(-1)
    keep = ~np.isin(facets, removed)
    facets, signs = facets[keep], signs[keep]
    _, first = np.unique(2 * facets + (signs > 0), return_index=True)
    return facets[first], signs[first], tops, orientations
