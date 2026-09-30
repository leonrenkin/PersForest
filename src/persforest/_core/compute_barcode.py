#move things like "_is_barcode_covered" to this file,
#This does not need to stay on the PF Node

    # ----- Compute barcode sequence ---------

def compute_barcode_cycles(forest, print_info: bool = False):
    """
    Compute the barcode from the forest structure.

    Parameters
    ----------
    print_info : bool
        If True, print timing information.

    Notes
    -----
    Iterates over leaves, walks to the root (or first covered node), and
    records the sequence of representatives. Bars are stored in
    ``forest.barcode``.
    """
    
    if forest._barcode_computed:
        raise RuntimeError("Barcode computation has already been called for this forest")
    if print_info:
        print("Computing Barcode")
    barcode_start = time.perf_counter()

    #dict should be ordered with filtration values decreasing since nodes are added in that order
    if not are_dict_keys_sorted(forest.nodes):
        raise ValueError("Node dict keys are not sorted. This should not happen. Easy fix: sort keys in compute_barcode function (currently not implemented)")


    for id, node in forest.nodes.items():
        #every barcode starts at leaf
        if len(node.children)>0:
            continue

        death = node.filt_val
        node_id_progession = [id]
        cycle_progression = [node.cycle]
        node._barcode_covered += 1
        is_max_tree_bar = True
        root_id = forest.get_root(node).id

        if node.parent == None:
            raise ValueError("Leaf has no Parent, this should not happen")
        else:
            parent = forest.nodes[node.parent]

        #walk up forest until a root or an already _barcode_covered node is discovered
        while parent.parent is not None:
            #check if parent node has already been covered by leaf with larger filtration value
            if parent._barcode_covered != 0:
                is_max_tree_bar = False
                break

            node_id_progession.append(parent.id)
            cycle_progression.append(parent.cycle)
            parent._barcode_covered += 1

            #move to parent of parent
            parent = forest.nodes[parent.parent]

        birth = parent.filt_val


        #reverse lists to get progression which is ascending with respect to filtration value
        bar = PFBar(birth=birth,
                    death=death, 
                    _node_progression = tuple(reversed(node_id_progession)), 
                    cycle_reps=list(reversed(cycle_progression)), 
                    is_max_tree_bar=is_max_tree_bar,
                    root_id=root_id)
        forest.barcode.add(bar)


    barcode_time = time.perf_counter() - barcode_start
    if print_info:
        print(f"Barcode computation completed in {barcode_time} sec")
    forest._barcode_computed = True
    return

def compute_barcode_diff(forest, print_info: bool = False):
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
    representatives. When ``keep_simplex_diff=True``, accumulated barcode
    diffs are stored on merge nodes for later reconstruction of cycle
    representatives with interiors. Bars are stored in ``forest.barcode``.
    """
    if forest._barcode_computed:
        raise RuntimeError("Barcode computation has already been called for this forest")
    
    if print_info:
        print("Computing Barcode")
    barcode_start = time.perf_counter()

    #dict should be ordered with filtration values decreasing since nodes are added in that order
    if not are_dict_keys_sorted(forest.nodes):
        raise ValueError("Node dict keys are not sorted. This should not happen. Easy fix: sort keys in compute_barcode function (currently not implemented)")


    for id, node in reversed(forest.nodes.items()):
        #every barcode starts at leaf
        if node.children:
            continue

        death = node.filt_val
        node_id_progession = [id]
        cycle_progression = [node.cycle]
        node._barcode_covered += 1
        is_max_tree_bar = True
        root_id = forest.get_root(node).id

        # Defer unions until we know whether this bar ends at a merge.
        # Root-reaching bars never need an accumulated diff.
        barcode_diff_sources = [(node._interior_diff, node._codim1_simplex_diff)]

        if node.parent == None:
            raise ValueError("Leaf has no Parent, this should not happen")
        else:
            parent = forest.nodes[node.parent]

        #walk up forest until a root or a merge node with node._barcode_covered < number of children
        while parent.parent is not None:
            #check if parent node still has undiscovered leaves
            if parent._barcode_covered < len(parent.children) - 1:
                parent._barcode_covered += 1
                is_max_tree_bar = False

                break

            if forest.keep_simplex_diff:
                barcode_diff_sources.append((parent._interior_diff, parent._codim1_simplex_diff))

                if parent._barcode_covered != 0:
                    barcode_diff_sources.append((parent._barcode_interior_diff, parent._barcode_codim1_simplex_diff))

            node_id_progession.append(parent.id)
            cycle_progression.append(parent.cycle)
            parent._barcode_covered += 1

            #move to parent of parent
            parent = forest.nodes[parent.parent]

        birth = parent.filt_val

        if forest.keep_simplex_diff and not is_max_tree_bar:
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
        forest.barcode.add(bar)


    barcode_time = time.perf_counter() - barcode_start
    if print_info:
        print(f"Barcode computation completed in {barcode_time} sec")
    forest._barcode_computed = True
    return