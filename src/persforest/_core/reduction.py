
def union_optional_sets(set1: Optional[Set], set2: Optional[Set]) -> Optional[Set]:
    """
    Return the union of two sets, where either set may be None.

    Parameters
    ----------
    set1 : set or None
        First set to union.
    set2 : set or None
        Second set to union.

    Returns
    -------
    set or None
        Union of the two sets, or None if both inputs are None.
    """
    if set1 is None and set2 is None:
        return None
    elif set1 is None:
        return set2
    elif set2 is None:
        return set1
    else:
        return set1 | set2

def collapse_parent_child(self, parent: PFNode, child: PFNode):
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
    if self.keep_simplex_diff:

        parent._interior_diff = union_optional_sets(parent._interior_diff, child._interior_diff)
        parent._codim1_simplex_diff = union_optional_sets(parent._codim1_simplex_diff, child._codim1_simplex_diff)

    #if parent is now isolated point in forest, delete it from forest completely
    if len(parent.children)==0 and parent.parent == None:
        del self.nodes[parent.id]
        self.roots.remove(parent.id)

    return

def split_and_normalize_roots(forest) -> None:
    """Give each surviving root one child and that child's representative.

    Notes
    -----
    Called after edge collapse and before activity/barcode computation.
    A branching root represents a split at birth, so each child receives
    a separate root at the original birth value. The smallest child ID
    retains the existing root; other children receive fresh root IDs.
    Non-root node order is preserved for barcode tie-breaking.

    Root diffs are empty because a root represents exactly its child,
    including in diff-only mode, where both stored cycles are ``None``.
    Existing unary roots are normalized too, since collapse can change
    their child. Repeating this operation leaves the forest unchanged.
    """
    for root_id in sorted(forest.roots):
        root = forest.nodes[root_id]
        child_ids = sorted(root.children)
        for index, child_id in enumerate(child_ids):
            child = forest.nodes[child_id]
            if index == 0:
                child_root = root
            else:
                new_id = next(forest._node_id)
                child_root = PFNode(
                    id=new_id, filt_val=root.filt_val,
                    cycle=child.cycle, children={child_id},
                )
                forest.nodes[new_id] = child_root
                forest.roots.add(new_id)

            child_root.children = {child_id}
            child_root.cycle = child.cycle
            child_root._simplex_diff_available = forest.keep_simplex_diff
            child_root._interior_diff = None
            child_root._codim1_simplex_diff = None
            child_root._barcode_interior_diff = None
            child_root._barcode_codim1_simplex_diff = None
            child.parent = child_root.id

    return


def reduce_forest(forest, print_info: bool = False):
    """
    Reduce the forest by collapsing every parent-child pair whose
    filtration values differ by at most ``forest.filtration_tol``.

    Collapse rule for an edge (parent p, child c) with
    p.filt_val <= c.filt_val <= p.filt_val + forest.filtration_tol:
    - Keep the *parent* node p (its cycle stays as-is).
    - Remove the child node c from the forest.
    - The parent of the resulting (collapsed) node remains p.parent (i.e., the
        parent of the original parent), if any.
    - Children of the resulting node are the union of p.children and c.children,
        minus the removed child c itforest.
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
        print(f"Number of nodes before reduction: {len(forest.nodes.keys())}")

    #iterate over snapshot of the nodes in the tree, the nodes dict might be changed each iteration
    node_list = list(forest.nodes.values())

    for p in node_list:

        # p might have been deleted as a child in a previous iteration of this outer loop
        if p.id not in forest.nodes.keys():
            print("continue case")
            continue

        for cid in p.children.copy():
            child = forest.nodes[cid]
            if child.filt_val < p.filt_val:
                raise ValueError(
                    f"Invalid parent-child filtration order: "
                    f"parent {p.id} has filt_val={p.filt_val}, "
                    f"child {child.id} has filt_val={child.filt_val}"
                )

            if child.filt_val <= p.filt_val + forest.filtration_tol:
                forest._collapse_parent_child(parent=p, child=child)
                collapses += 1

    split_and_normalize_roots(forest)

    reduction_time = time.perf_counter() - reduction_start
    if print_info:
        print(f"Reduction complete in {reduction_time} sec")
        print(f"Number of nodes after reduction: {len(forest.nodes.keys())}")
    
    return

