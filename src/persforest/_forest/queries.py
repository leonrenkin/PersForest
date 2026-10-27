"""_forest.queries: mechanically extracted reference implementation."""

from persforest._forest.graph_model import PFNode
from persforest.chains import SignedChain
from typing import List

def active_nodes_at(self, filt_val: float) -> List[PFNode]:
    """
        Return nodes active at a given filtration value.

        A non-root node is active at ``r`` if ``node.filt_val > r`` and its
        parent has filtration value ``<= r``.

        Parameters
        ----------
        filt_val : float
            Filtration value to query.

        Returns
        -------
        list[PFNode]
            Active nodes sorted by decreasing node filtration value and then id.
        """
    nodes = self.nodes

    active: List[PFNode] = []
    for n in nodes.values():
        if n.filt_val <= filt_val:
            continue
        if n.parent == None:
        # all children must exist and be strictly above alpha
            continue
        else:
            parent = nodes[n.parent]
            if parent.filt_val > filt_val:
                continue

        active.append(n)
    
    # deterministic order: higher filt_val first, then id
    active.sort(key=lambda n: (-n.filt_val, n.id))
    return active

def active_cycles_at(self, filt_val: float) -> List[SignedChain]:
    """Return active cycle representatives at ``filt_val``."""
    if self.diff_only_mode:
        if not self._barcode_computed:
            raise RuntimeError("Call compute_barcode() before querying cycles in diff_only_mode")
        return [
            self.cycle_for_bar_at(bar, filt_val)
            for bar in self.active_bars_at(filt_val)
        ]

    active_nodes = self.active_nodes_at(filt_val=filt_val)
    return [node.cycle for node in active_nodes]

def leaves_below_node(self, node: PFNode) -> set[int]:
    """Return ids of all descendant leaves below ``node``."""
    leaf_ids: set[int] = set()
    stack = [node.id]

    while stack:
        vid = stack.pop()
        children_ids = self.nodes[vid].children

        if not children_ids:
            leaf_ids.add(vid)
        else:
            stack.extend(children_ids)

    return leaf_ids

def leaf_to_node_path(self, leaf: PFNode, node: PFNode) -> List[int]:
    """Return node ids on the path from ``leaf`` up to ancestor ``node``."""
    path = [leaf.id]

    active_node = leaf
    while active_node.parent != None:
        active_node = self.nodes[active_node.parent]
        path.append(active_node.id)
        if active_node.id == node.id:
            return path

    if active_node.id != node.id:
        raise ValueError(f"Node {node} is not above leaf {leaf}")

    return path

def node_to_leaf_path(self, leaf: PFNode, node: PFNode) -> List[int]:
    """Return node ids on the path from ancestor ``node`` down to ``leaf``."""
    return list(reversed(self.leaf_to_node_path(leaf=leaf, node=node)))

def get_root(self, node: PFNode) -> PFNode:
    """Return the highest ancestor of ``node`` with no parent."""
    while node.parent != None:
                pid = node.parent
                node = self.nodes[pid]

    return node

def _node_activity(self, node):
    """Return ``(active_start, active_end)`` for a node's cycle."""
    if node.parent == None:
        return node.filt_val, node.filt_val

    parent = self.nodes[node.parent]
    active_start = parent.filt_val
    active_end = node.filt_val
    return active_start, active_end
