"""Frozen copy of the former Python construction engine, used as a test reference.

This is the pure-Python engine that persforest used before the compiled
engine (sweep in ``_forest/build.py``, reduction, root normalization and
elder-rule barcode with accumulated diffs, representative replay), restricted
to diff storage and gathered in one module. Do not change its logic; it exists
to check the compiled engine on inputs the golden snapshots do not cover.

``ReferenceForest(cells, reduce=..., filtration_tol=...)`` runs it on a
``CellFiltration`` and offers the queries the snapshots use: ``barcode``,
``bar.critical_values``, ``bar.tree_id``, ``parent_bar``,
``iter_bar_cycle_reps`` and ``interior_simplex_activity``.
"""
from collections import defaultdict
from dataclasses import dataclass
import itertools
from typing import Optional

from persforest.chains import SignedChain
from persforest.filtrations._interface import signed_incidences


@dataclass(slots=True)
class PFNode:
    id: int
    filt_val: float
    children: set[int]
    parent: Optional[int] = None
    _is_inactive_leaf: bool = False
    _barcode_covered: int = 0
    _interior_diff: set | None = None
    _codim1_simplex_diff: set | None = None
    _barcode_interior_diff: set | None = None
    _barcode_codim1_simplex_diff: set | None = None


class ReferenceBar:
    def __init__(self, birth, death, node_progression, is_max_tree_bar, root_id):
        self.birth = birth
        self.death = death
        self._node_progression = node_progression
        self.is_max_tree_bar = is_max_tree_bar
        self.root_id = root_id
        self.critical_values: tuple = ()
        self.tree_id = root_id

    def lifespan(self):
        return self.death - self.birth


def _cached_root_id(nodes, ancestors, node_id):
    path = []
    current = node_id
    while True:
        cached = ancestors.get(current, current)
        if cached == current:
            parent = nodes[current].parent
            if parent is None:
                break
            cached = parent
        path.append(current)
        current = cached
    for visited in path:
        ancestors[visited] = current
    return current


def merge_owned_sets(kept, absorbed):
    if kept is None:
        return absorbed
    if absorbed is None:
        return kept
    if len(kept) < len(absorbed):
        kept, absorbed = absorbed, kept
    kept |= absorbed
    return kept


class ReferenceForest:
    def __init__(self, cells, *, reduce=True, filtration_tol=1e-12):
        self._cells = cells
        self.filtration_tol = filtration_tol
        self._node_id = itertools.count(1)
        self.nodes: dict[int, PFNode] = {}
        self._active_node_ids: set[int] = set()
        self.roots: set[int] = set()
        self.barcode: set = set()
        self._compute_forest()
        if reduce:
            self._reduce_forest()
        self._compute_barcode_diff()
        owner = {node_id: bar for bar in self.barcode for node_id in bar._node_progression}
        self._parent = {}
        for bar in self.barcode:
            bar.critical_values = tuple(self.nodes[n].filt_val for n in bar._node_progression)
            self._parent[bar] = owner.get(self.nodes[bar._node_progression[0]].parent)

    # ----- sweep (former _forest/build.py) -----

    def _add_leaf(self, cell, filt_val, orientation):
        nid = next(self._node_id)
        node = PFNode(id=nid, filt_val=filt_val, children=set(), _interior_diff={(cell, orientation)})
        self.nodes[nid] = node
        self._active_node_ids.add(nid)
        return node

    def _leaves_below_node(self, node):
        leaf_ids, stack = set(), [node.id]
        while stack:
            vid = stack.pop()
            children_ids = self.nodes[vid].children
            if not children_ids:
                leaf_ids.add(vid)
            else:
                stack.extend(children_ids)
        return leaf_ids

    def _make_root(self, node, filt_val):
        nid = next(self._node_id)
        self._active_node_ids.remove(node.id)
        node.parent = nid
        root = PFNode(id=nid, filt_val=filt_val, children={node.id})
        self.nodes[nid] = root
        self.roots.add(nid)
        for leaf_id in self._leaves_below_node(root):
            self.nodes[leaf_id]._is_inactive_leaf = True

    def _merge_nodes(self, node1, node2, cell, filt_val):
        nid = next(self._node_id)
        node1.parent = nid
        node2.parent = nid
        self.nodes[nid] = PFNode(id=nid, filt_val=filt_val, children={node1.id, node2.id},
                                 _codim1_simplex_diff={(cell, 1), (cell, -1)})
        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node1.id)
        self._active_node_ids.remove(node2.id)

    def _update_node(self, node, cell, filt_val):
        nid = next(self._node_id)
        node.parent = nid
        self.nodes[nid] = PFNode(id=nid, filt_val=filt_val, children={node.id},
                                 _codim1_simplex_diff={(cell, 1), (cell, -1)})
        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node.id)

    def _compute_forest(self):
        cells = self._cells
        n_top = cells.n_top
        top_facets = cells.top_facets.tolist()
        incidences = {1: signed_incidences(cells, 1), -1: signed_incidences(cells, -1)}
        facet_leaf_ids = {}
        ancestors = {}
        for cell, filt_val in zip(reversed(cells.event_cells.tolist()), reversed(cells.event_values.tolist())):
            if cell < n_top:
                orientation = cells.orientation(cell)
                boundary = tuple(zip(top_facets[cell], incidences[orientation]))
                new_node = self._add_leaf(cell, filt_val, orientation)
                for facet, _ in boundary:
                    facet_leaf_ids.setdefault(facet, []).append(new_node.id)
            else:
                leaf_ids = facet_leaf_ids.pop(cell, [])
                active = [self.nodes[_cached_root_id(self.nodes, ancestors, i)]
                          for i in leaf_ids if not self.nodes[i]._is_inactive_leaf]
                if len(active) == 0:
                    continue
                elif len(active) == 1:
                    self._make_root(active[0], filt_val)
                elif len(active) == 2 and active[0] != active[1]:
                    self._merge_nodes(active[0], active[1], cell, filt_val)
                elif len(active) == 2:
                    self._update_node(active[0], cell, filt_val)
                else:
                    raise ValueError(f"Facet {cell} has {len(active)} incident active cycles")

    # ----- reduction (former _forest/reduction.py) -----

    def _collapse_parent_child(self, parent, child):
        parent.children.remove(child.id)
        parent.children.update(child.children)
        for gcid in child.children:
            self.nodes[gcid].parent = parent.id
        del self.nodes[child.id]
        parent._interior_diff = merge_owned_sets(parent._interior_diff, child._interior_diff)
        parent._codim1_simplex_diff = merge_owned_sets(parent._codim1_simplex_diff, child._codim1_simplex_diff)
        if len(parent.children) == 0 and parent.parent is None:
            del self.nodes[parent.id]
            self.roots.remove(parent.id)

    def _reduce_forest(self):
        for node_id in list(self.nodes):
            p = self.nodes.get(node_id)
            if p is None:
                continue
            for cid in p.children.copy():
                child = self.nodes[cid]
                if child.filt_val < p.filt_val:
                    raise ValueError("Invalid parent-child filtration order")
                if child.filt_val <= p.filt_val + self.filtration_tol:
                    self._collapse_parent_child(p, child)
        self._normalize_roots()

    def _normalize_roots(self):
        for root_id in sorted(self.roots):
            root = self.nodes[root_id]
            for index, child_id in enumerate(sorted(root.children)):
                child = self.nodes[child_id]
                if index == 0:
                    child_root = root
                else:
                    new_id = next(self._node_id)
                    child_root = PFNode(id=new_id, filt_val=root.filt_val, children={child_id})
                    self.nodes[new_id] = child_root
                    self.roots.add(new_id)
                child_root.children = {child_id}
                child_root._interior_diff = None
                child_root._codim1_simplex_diff = None
                child_root._barcode_interior_diff = None
                child_root._barcode_codim1_simplex_diff = None
                child.parent = child_root.id

    # ----- barcode (former _forest/compute_barcode.py) -----

    def _compute_barcode_diff(self):
        ancestors = {}
        for nid, node in reversed(self.nodes.items()):
            if node.children:
                continue
            death = node.filt_val
            progression = [nid]
            node._barcode_covered += 1
            is_max_tree_bar = True
            root_id = _cached_root_id(self.nodes, ancestors, nid)
            sources = [(node._interior_diff, node._codim1_simplex_diff)]
            if node.parent is None:
                raise ValueError("Leaf has no Parent, this should not happen")
            parent = self.nodes[node.parent]
            while parent.parent is not None:
                if parent._barcode_covered < len(parent.children) - 1:
                    parent._barcode_covered += 1
                    is_max_tree_bar = False
                    break
                sources.append((parent._interior_diff, parent._codim1_simplex_diff))
                if parent._barcode_covered != 0:
                    sources.append((parent._barcode_interior_diff, parent._barcode_codim1_simplex_diff))
                progression.append(parent.id)
                parent._barcode_covered += 1
                parent = self.nodes[parent.parent]
            if not is_max_tree_bar:
                for interior_diff, codim1_simplex_diff in sources:
                    if interior_diff is not None:
                        if parent._barcode_interior_diff is None:
                            parent._barcode_interior_diff = set()
                        parent._barcode_interior_diff.update(interior_diff)
                    if codim1_simplex_diff is not None:
                        if parent._barcode_codim1_simplex_diff is None:
                            parent._barcode_codim1_simplex_diff = set()
                        parent._barcode_codim1_simplex_diff.update(codim1_simplex_diff)
            self.barcode.add(ReferenceBar(parent.filt_val, death, tuple(reversed(progression)),
                                          is_max_tree_bar, root_id))

    # ----- queries (former _forest/representatives.py) -----

    def parent_bar(self, bar):
        return self._parent[bar]

    def _keyed_diffs(self, diffs):
        ids = [cell for pair in diffs for diff in pair if diff is not None for cell, _ in diff]
        if not ids:
            return diffs
        keys = iter(self._cells.cell_keys(ids))
        return tuple(tuple(None if diff is None else {(next(keys), sign) for _, sign in diff} for diff in pair)
                     for pair in diffs)

    def _diff_steps(self, bar):
        for nid in reversed(bar._node_progression):
            node = self.nodes[nid]
            parent = self.nodes[node.parent]
            yield node, parent.filt_val, node.filt_val, (
                (node._barcode_interior_diff, node._barcode_codim1_simplex_diff),
                (node._interior_diff, node._codim1_simplex_diff),
            )

    def iter_bar_cycle_reps(self, bar):
        cells = self._cells
        interior, simplices = set(), set()
        for _node, active_start, active_end, diffs in self._diff_steps(bar):
            for interior_diff, codim1_simplex_diff in self._keyed_diffs(diffs):
                if interior_diff is not None:
                    for top, orientation in interior_diff:
                        simplices.update((facet, orientation * incidence) for facet, incidence in cells.boundary(top))
                    interior.update(interior_diff)
                if codim1_simplex_diff is not None:
                    for oriented in codim1_simplex_diff:
                        simplices.remove(oriented)
            yield SignedChain(signed_simplices=simplices.copy(), active_start=active_start, active_end=active_end,
                              interior_available=True, interior=interior.copy())

    def interior_simplex_activity(self):
        activity = defaultdict(list)
        for bar in self.barcode:
            active_end_by_cell = {}
            for node_id in reversed(bar._node_progression):
                node = self.nodes[node_id]
                for interior_diff in (node._interior_diff, node._barcode_interior_diff):
                    if interior_diff is not None:
                        for cell, _orientation in interior_diff:
                            active_end_by_cell.setdefault(cell, node.filt_val)
            for cell, active_end in active_end_by_cell.items():
                activity[cell].append((bar, bar.birth, active_end))
        keys = self._cells.cell_keys(list(activity))
        return dict(zip(keys, activity.values()))
