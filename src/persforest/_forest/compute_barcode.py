"""_forest.compute_barcode: barcode extraction from node diffs."""

from persforest.barcode import PFBar
import time

def are_dict_keys_sorted(d):
    """
    Return True if dict keys are in ascending order (linear time).

    Parameters
    ----------
    d : dict
        Dictionary whose key insertion order is inspected.

    Returns
    -------
    bool
        True if keys appear in ascending order, False otherwise.
    """
    it = iter(d)  # iterates over keys in insertion order
    try:
        prev = next(it)
    except StopIteration:
        return True  # empty dict is trivially sorted

    for k in it:
        if k < prev:
            return False
        prev = k
    return True

def compute_barcode_diff(self, print_info: bool = False):
    """
        Compute the barcode from the forest structure using stored node diffs.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.

        Notes
        -----
        Iterates over leaves in reverse node order, walks to the root or to a
        merge node with an undiscovered branch, and records the sequence of
        representatives. Accumulated barcode diffs are stored on merge nodes
        for later reconstruction of cycle representatives with interiors. Bars are stored in ``self.barcode``.
        """
    if self._barcode_computed:
        raise RuntimeError("Barcode computation has already been called for this forest")
    
    if print_info:
        print("Computing Barcode")
    barcode_start = time.perf_counter()

    #dict should be ordered with filtration values decreasing since nodes are added in that order
    if not are_dict_keys_sorted(self.nodes):
        raise ValueError("Node dict keys are not sorted. This should not happen. Easy fix: sort keys in compute_barcode function (currently not implemented)")


    for id, node in reversed(self.nodes.items()):
        #every barcode starts at leaf
        if node.children:
            continue

        death = node.filt_val
        node_id_progession = [id]
        cycle_progression = [node.cycle]
        node._barcode_covered += 1
        is_max_tree_bar = True
        root_id = self.get_root(node).id

        # Defer unions until we know whether this bar ends at a merge.
        # Root-reaching bars never need an accumulated diff.
        barcode_diff_sources = [(node._interior_diff, node._codim1_simplex_diff)]

        if node.parent == None:
            raise ValueError("Leaf has no Parent, this should not happen")
        else:
            parent = self.nodes[node.parent]

        #walk up forest until a root or a merge node with node._barcode_covered < number of children
        while parent.parent is not None:
            #check if parent node still has undiscovered leaves
            if parent._barcode_covered < len(parent.children) - 1:
                parent._barcode_covered += 1
                is_max_tree_bar = False

                break

            barcode_diff_sources.append((parent._interior_diff, parent._codim1_simplex_diff))

            if parent._barcode_covered != 0:
                barcode_diff_sources.append((parent._barcode_interior_diff, parent._barcode_codim1_simplex_diff))

            node_id_progession.append(parent.id)
            cycle_progression.append(parent.cycle)
            parent._barcode_covered += 1

            #move to parent of parent
            parent = self.nodes[parent.parent]

        birth = parent.filt_val

        if not is_max_tree_bar:
            # Each merge owns its accumulators: update them without copying
            # growing sets or mutating any source node/descendant merge.
            for interior_diff, codim1_simplex_diff in barcode_diff_sources:
                if interior_diff is not None:
                    if parent._barcode_interior_diff is None:
                        parent._barcode_interior_diff = set()
                    parent._barcode_interior_diff.update(interior_diff)
                if codim1_simplex_diff is not None:
                    if parent._barcode_codim1_simplex_diff is None:
                        parent._barcode_codim1_simplex_diff = set()
                    parent._barcode_codim1_simplex_diff.update(codim1_simplex_diff)


        #reverse lists to get progression which is ascending with respect to filtration value
        bar = PFBar(birth=birth,
                  death=death, 
                  _node_progression = tuple(reversed(node_id_progession)), 
                  cycle_reps=list(reversed(cycle_progression)), 
                  is_max_tree_bar=is_max_tree_bar,
                  root_id=root_id)
        self.barcode.add(bar)
 

    barcode_time = time.perf_counter() - barcode_start
    if print_info:
        print(f"Barcode computation completed in {barcode_time} sec")
    self._barcode_diffs_ready = True
    self._barcode_computed = True
    return
