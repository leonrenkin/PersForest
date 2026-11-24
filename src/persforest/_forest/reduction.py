"""_forest.reduction: mechanically extracted reference implementation."""

from persforest._forest.graph_model import PFNode
from typing import Optional
from typing import Set
import time

def merge_owned_sets(kept: Optional[Set], absorbed: Optional[Set]) -> Optional[Set]:
    """
    Return the union of two diff sets, reusing the larger set in place.

    Parameters
    ----------
    kept : set or None
        Diff set of the node that survives the collapse.
    absorbed : set or None
        Diff set of the node that is removed.

    Returns
    -------
    set or None
        The union, stored in one of the input sets, or None if both are None.

    Notes
    -----
    Either input may be mutated. This is safe during reduction because every
    diff set has exactly one live owner and the absorbing node takes ownership
    of the result. Adding the smaller set to the larger keeps chains of
    collapses on flat filtration regions linear; building a new union per
    collapse was quadratic in time and memory.
    """
    if kept is None:
        return absorbed
    if absorbed is None:
        return kept
    if len(kept) < len(absorbed):
        kept, absorbed = absorbed, kept
    kept |= absorbed
    return kept

def _collapse_parent_child(self, parent: PFNode, child: PFNode):
    """
        Collapse a parent-child pair into the parent node.

        Intended for pairs whose filtration values agree up to
        ``self.filtration_tol``. The parent node is kept; the child is removed
        and its children are re-parented to ``parent``.
        """

    parent.children.remove(child.id)
    parent.children.update(child.children)

    #re-parent grandchildren
    for gcid in child.children:
        self.nodes[gcid].parent = parent.id

    #remove child node from forest
    del self.nodes[child.id]

    #update node simplex diffs
    parent._interior_diff = merge_owned_sets(parent._interior_diff, child._interior_diff)
    parent._codim1_simplex_diff = merge_owned_sets(parent._codim1_simplex_diff, child._codim1_simplex_diff)

    #if parent is now isolated point in forest, delete it from forest completely
    if len(parent.children)==0 and parent.parent == None:
        del self.nodes[parent.id]
        self.roots.remove(parent.id)

    return

def _reduce_forest(self, print_info: bool = False):
    """
        Reduce the forest by collapsing every parent-child pair whose
        filtration values differ by at most ``self.filtration_tol``.

        Collapse rule for an edge (parent p, child c) with
        p.filt_val <= c.filt_val <= p.filt_val + self.filtration_tol:
        - Keep the *parent* node p (its cycle stays as-is).
        - Remove the child node c from the forest.
        - The parent of the resulting (collapsed) node remains p.parent (i.e., the
            parent of the original parent), if any.
        - Children of the resulting node are the union of p.children and c.children,
            minus the removed child c itself.
        - For every grandchild g in c.children, set g.parent = p.id.
        Repeats until no collapsible edges remain.

        Finally, normalize roots so that branches born simultaneously belong
        to separate trees, each with its child's cycle as root representative.

        Parameters
        ----------
        print_info : bool
            If True, print timing information and node counts.
        """
    if print_info:
        print("Reducing the forest")
    reduction_start = time.perf_counter()

    collapses = 0

    if print_info:
        print(f"Number of nodes before reduction: {len(self.nodes.keys())}")

    # Iterate over a snapshot of node ids: the nodes dict changes during the
    # loop, and holding ids rather than nodes lets removed nodes be freed.
    for node_id in list(self.nodes):

        # p might have been deleted as a child in a previous iteration of this outer loop
        p = self.nodes.get(node_id)
        if p is None:
            continue

        for cid in p.children.copy():
            child = self.nodes[cid]
            if child.filt_val < p.filt_val:
                raise ValueError(
                    f"Invalid parent-child filtration order: "
                    f"parent {p.id} has filt_val={p.filt_val}, "
                    f"child {child.id} has filt_val={child.filt_val}"
                )

            if child.filt_val <= p.filt_val + self.filtration_tol:
                self._collapse_parent_child(parent=p, child=child)
                collapses += 1

    self._normalize_roots()

    reduction_time = time.perf_counter() - reduction_start
    if print_info:
        print(f"Reduction complete in {reduction_time} sec")
        print(f"Number of nodes after reduction: {len(self.nodes.keys())}")
    
    return

def _normalize_roots(self) -> None:
    """Give each surviving root one child and that child's representative.

        Notes
        -----
        Called after edge collapse and before activity/barcode computation.
        A branching root represents a split at birth, so each child receives
        a separate root at the original birth value. The smallest child ID
        retains the existing root; other children receive fresh root IDs.
        Non-root node order is preserved for barcode tie-breaking.

        Root diffs are empty because a root represents exactly its child,
        including with ``storage="diffs"``, where both stored cycles are ``None``.
        Existing unary roots are normalized too, since collapse can change
        their child. Repeating this operation leaves the forest unchanged.
        """
    for root_id in sorted(self.roots):
        root = self.nodes[root_id]
        child_ids = sorted(root.children)
        for index, child_id in enumerate(child_ids):
            child = self.nodes[child_id]
            if index == 0:
                child_root = root
            else:
                new_id = next(self._node_id)
                child_root = PFNode(
                    id=new_id, filt_val=root.filt_val,
                    cycle=child.cycle, children={child_id},
                )
                self.nodes[new_id] = child_root
                self.roots.add(new_id)

            child_root.children = {child_id}
            child_root.cycle = child.cycle
            child_root._simplex_diff_available = True
            child_root._interior_diff = None
            child_root._codim1_simplex_diff = None
            child_root._barcode_interior_diff = None
            child_root._barcode_codim1_simplex_diff = None
            child.parent = child_root.id
