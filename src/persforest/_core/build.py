def key(simplex):
    """
    Return a canonical, orientation-free key for a simplex.

    Parameters
    ----------
    simplex : iterable[int]
        Vertex ids of the simplex.

    Returns
    -------
    tuple
        Sorted tuple of vertex ids, suitable for hashing regardless of
        orientation.
    """
    return tuple(sorted(simplex))

def merge_at_simplex(cycle1: Chain, cycle2: Chain, simplex: list[int]) -> Chain:
    """
    Union two chains and remove the specified simplex if it
    appears with opposite orientation in the two chains.

    Parameters
    ----------
    cycle1 : SignedChain
        Chain to merge.
    cycle2 : SignedChain
        Chain to merge.
    simplex : list[int]
        Simplex that is being collapsed when the chains merge.

    Returns
    -------
    SignedChain
        New chain representing the merged cycles.
    """
    union = cycle1.signed_simplices | cycle2.signed_simplices
    return SignedChain(signed_simplices= union.difference({(tuple(simplex),1),(tuple(simplex),-1)}) )

def add_leaf(forest, simplex: List[int], filt_val: float, orientation: int):
    """
    Create a new leaf node for a full-dimensional simplex.

    In the reverse-filtration algorithm this corresponds to the death of a
    codimension-one homology class; the new cycle is the signed boundary of
    ``simplex``.

    Parameters
    ----------
    simplex : list[int]
        Vertices of the full-dimensional simplex being added. It has
        length ``forest.dim + 1``.
    filt_val : float
        Filtration value where the simplex enters.
    orientation : int
        Orientation (+/-1) of the simplex in the complex.

    Returns
    -------
    PFNode
        Newly created leaf node.
    """

    nid = next(forest._node_id)

    if not forest.diff_only_mode:
        new_cycle = SignedChain(signed_simplices=signed_boundary(simplex=simplex,orientation=orientation))
    else:
        new_cycle = None

    if forest.keep_simplex_diff:
        new_node = PFNode(id=nid,
                            filt_val=filt_val, 
                            children=set(), 
                            cycle=new_cycle, 
                            _simplex_diff_available = True,
                            _interior_diff = {(key(simplex=simplex), orientation)},
                            _codim1_simplex_diff = None)
    else: 
        new_node = PFNode(id=nid,filt_val=filt_val, children=set(), cycle=new_cycle, _simplex_diff_available = False)

    forest.nodes[nid]=new_node
    forest._active_node_ids.add(nid)  # roots are active nodes; at termination they are the roots of the forest

    forest.levels.append(filt_val)

    return new_node

def make_root(forest, node: PFNode, filt_val: float):
    """
    End a tree by creating a root parent for an active node.

    In the reverse-filtration algorithm this corresponds to the birth of a
    codimension-one homology class.

    Parameters
    ----------
    node : PFNode
        Child node that becomes the child of the new root.
    filt_val : float
        Filtration value at which the root is created.
    """
    nid = next(forest._node_id)

    forest._active_node_ids.remove(node.id)
    node.parent = nid #new root node is parent of input node

    if forest.keep_simplex_diff:
        root_node = PFNode(id=nid, 
                            filt_val=filt_val, 
                            cycle=node.cycle, 
                            children={node.id},
                            _simplex_diff_available = True,
                            _interior_diff = None,
                            _codim1_simplex_diff = None)
    else:
        root_node = PFNode(id=nid, filt_val=filt_val, cycle=node.cycle, children={node.id}, _simplex_diff_available = False)


    forest.nodes[nid]=root_node
    forest.roots.add(root_node.id)

    #set all leaves below root as inactive
    leaf_ids = forest.leaves_below_node(node=root_node)
    for leaf_id in leaf_ids:
        forest.nodes[leaf_id]._is_inactive_leaf=True

    forest.levels.append(filt_val)

    return

def merge_nodes(forest, node1: PFNode, node2: PFNode, simplex: List[int], filt_val: float):
    """ 
    Create a parent node by merging two active child nodes.

    The parent cycle is obtained by unioning the child cycles and removing
    both orientations of ``simplex``. In the reverse-filtration algorithm,
    this records a split event and creates a new barcode branch.

    Parameters
    ----------
    node1, node2 : PFNode
        Child nodes that are merged.
    simplex : list[int]
        Codimension-one simplex at which the two children cycles merge.
    filt_val : float
        Filtration value of the merge event.
    """

    nid = next(forest._node_id)
    node1.parent = nid
    node2.parent = nid

    if not forest.diff_only_mode:
        parent_cycle = merge_at_simplex(cycle1 = node1.cycle, cycle2=node2.cycle,  simplex=simplex) 
    else:
        parent_cycle = None

    if forest.keep_simplex_diff:
        parent_node = PFNode(id=nid, 
                                filt_val=filt_val, 
                                children={node1.id,node2.id}, 
                                cycle=parent_cycle,
                                _simplex_diff_available = True,
                                _interior_diff = None,
                                _codim1_simplex_diff = {(key(simplex),1),(key(simplex),-1)})
    else:
        parent_node = PFNode(id=nid, filt_val=filt_val, children={node1.id,node2.id}, cycle=parent_cycle, _simplex_diff_available = False)
    forest.nodes[nid]=parent_node
    
    forest._active_node_ids.add(nid)
    forest._active_node_ids.remove(node1.id)
    forest._active_node_ids.remove(node2.id)

    forest.levels.append(filt_val)

    return

def update_node(forest, node: PFNode, simplex: List[int], filt_val:float):
    """
    Create a parent node for an active node whose cycle changes.

    The updated cycle is obtained by removing both orientations of
    ``simplex`` from ``node.cycle``.

    Parameters
    ----------
    node : PFNode
        Node whose representative is being updated.
    simplex : list[int]
        Codimension-one simplex to remove from the representative.
    filt_val : float
        Filtration value of the update event.
    """

    if not forest.diff_only_mode:
        updated_cycle = node.cycle.cancel_simplex(simplex=simplex)
    else: updated_cycle = None

    nid = next(forest._node_id)
    node.parent=nid

    if forest.keep_simplex_diff:
        update_node = PFNode(id=nid, 
                                filt_val=filt_val, 
                                children={node.id},
                                cycle = updated_cycle,
                                _simplex_diff_available = True,
                                _interior_diff = None,
                                _codim1_simplex_diff = {(key(simplex),1),(key(simplex),-1)})
    else:
        update_node = PFNode(id=nid, filt_val=filt_val, children={node.id},cycle = updated_cycle, _simplex_diff_available = False)

    forest.nodes[nid]=update_node

    forest._active_node_ids.add(nid)
    forest._active_node_ids.remove(node.id)

    forest.levels.append(filt_val)

    return

def update_node_list(forest, node_id_list: List[int]) -> List[PFNode]:
    """Return current root ancestors for ``node_id_list``."""
    L=[]
    for id in node_id_list:
        if not forest.nodes[id]._is_inactive_leaf:
            active_node = forest.get_root(forest.nodes[id])
            L.append(active_node)
    return L

def build_forest(forest):
    """ 
    Compute the persistence forest from the alpha-complex filtration.
    """

    face_cycle_dict = {}
    Filtration = forest.Filtration
    Filtration.remove_low_dim_simplices(dim=forest.dim)

    #simplices is already ordered in ascending order by number of simplices 
    for simplex, filt_val in reversed(Filtration.filtration_list):
        #print("simplex", simplex)
        #print("active_node_ids",forest._active_node_ids)
        
        if Filtration.dim(simplex) == forest.dim:
            orientation = Filtration.simplex_orientation(simplex=simplex, point_cloud=forest.point_cloud)

            new_node = add_leaf(forest=forest, simplex=simplex, filt_val=filt_val, orientation = orientation)

            faces = Filtration.faces(simplex=simplex)
            for face in faces:
                if key(face) in face_cycle_dict:
                    face_cycle_dict[key(face)].append(new_node.id)
                else:
                    face_cycle_dict[key(face)] = [new_node.id]

        elif Filtration.dim(simplex) == forest.dim-1:
            #face_cycle_dict[simplex] contains nodes induced by cofaces of simplex
            # can be of the form [],[l1], [l1,l2], [l1,l1]

            #If key exists, get its value and remove it
            #if key does not exists, get []
            L_tmp_ids = face_cycle_dict.pop(key(simplex), [])

            # find nodes corresponding to cofaces, find active nodes above leaves induced by cofaces, 
            # we remove construction roots since they are not active anymore
            L = update_node_list(forest=forest, node_id_list=L_tmp_ids)


            #if no cycle contains simplex, nothing happens
            if len(L) == 0:
                continue

            #if cycle is only contained in a single loop and appears only once in that loop once, remove that loop from the active loops 
            elif len(L) == 1:
                make_root(forest=forest, node=L[0],filt_val=filt_val)

                continue

            elif len(L) == 2 and L[0]!=L[1]: 
                merge_nodes(forest=forest, node1=L[0], node2=L[1], simplex =simplex, filt_val=filt_val)
                """if not loop_in_filtration_check(parent_loop.vertex_list, simplex_tree=forest.simplex_tree, filt_value=filt_val):
                        print('edge', simplex)
                        print('edge dict entry')
                        print('filtration value', filt_val)
                        print(f'first loop', L[0].cycle)
                        print(f'second loop', L[1].cycle)
                        raise ValueError("Loop not in simplex, Loop concat Case")"""


            elif len(L) == 2 and L[0]==L[1]:
                #Same simplex is contained in a cycle in both orientations -> we remove it from the the cycle
                update_node(forest=forest,node=L[0], simplex=simplex, filt_val=filt_val)
                """if not loop_in_filtration_check(vertex_loop, simplex_tree=forest.simplex_tree, filt_value=filt_val):
                            print('edge', simplex)
                            print(f'starting loop', L[0].loop)
                            print(f'outer loop', vertex_loop)
                            raise ValueError("Loop not in simplex, Tiebreak Case")"""
                
            else:
                print(L)
                print(simplex)
                raise ValueError("Error, L is of the wrong form")

    return