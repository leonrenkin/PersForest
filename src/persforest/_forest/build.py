"""_forest.build: mechanically extracted reference implementation."""

from persforest._forest.graph_model import PFNode
from persforest.chains import SignedChain
from persforest.chains import merge_at_simplex
from persforest.filtrations.simplicial import key
from persforest.filtrations.simplicial import signed_boundary
from persforest.geometry.simplicial import simplex_orientation
from typing import List
import itertools
import time

def add_leaf(self, simplex: List[int], filt_val: float, orientation: int):
    """
        Create a new leaf node for a full-dimensional simplex.

        In the reverse-filtration algorithm this corresponds to the death of a
        codimension-one homology class; the new cycle is the signed boundary of
        ``simplex``.

        Parameters
        ----------
        simplex : list[int]
            Vertices of the full-dimensional simplex being added. It has
            length ``self.dim + 1``.
        filt_val : float
            Filtration value where the simplex enters.
        orientation : int
            Orientation (+/-1) of the simplex in the complex.

        Returns
        -------
        PFNode
            Newly created leaf node.
        """

    nid = next(self._node_id)

    if not self.diff_only_mode:
        new_cycle = SignedChain(signed_simplices=signed_boundary(simplex=simplex,orientation=orientation))
    else:
        new_cycle = None

    if self.keep_simplex_diff:
        new_node = PFNode(id=nid,
                          filt_val=filt_val, 
                          children=set(), 
                          cycle=new_cycle, 
                          _simplex_diff_available = True,
                          _interior_diff = {(key(simplex=simplex), orientation)},
                          _codim1_simplex_diff = None)
    else: 
        new_node = PFNode(id=nid,filt_val=filt_val, children=set(), cycle=new_cycle, _simplex_diff_available = False)

    self.nodes[nid]=new_node
    self._active_node_ids.add(nid)  # roots are active nodes; at termination they are the roots of the forest

    self.levels.append(filt_val)

    return new_node

def make_root(self, node: PFNode, filt_val: float):
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
    nid = next(self._node_id)

    self._active_node_ids.remove(node.id)
    node.parent = nid #new root node is parent of input node

    if self.keep_simplex_diff:
        root_node = PFNode(id=nid, 
                           filt_val=filt_val, 
                           cycle=node.cycle, 
                           children={node.id},
                           _simplex_diff_available = True,
                           _interior_diff = None,
                           _codim1_simplex_diff = None)
    else:
        root_node = PFNode(id=nid, filt_val=filt_val, cycle=node.cycle, children={node.id}, _simplex_diff_available = False)


    self.nodes[nid]=root_node
    self.roots.add(root_node.id)

    #set all leaves below root as inactive
    leaf_ids = self.leaves_below_node(node=root_node)
    for leaf_id in leaf_ids:
        self.nodes[leaf_id]._is_inactive_leaf=True

    self.levels.append(filt_val)

    return

def merge_nodes(self, node1: PFNode, node2: PFNode, simplex: List[int], filt_val: float):
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

    nid = next(self._node_id)
    node1.parent = nid
    node2.parent = nid

    if not self.diff_only_mode:
        parent_cycle = merge_at_simplex(cycle1 = node1.cycle, cycle2=node2.cycle,  simplex=simplex) 
    else:
        parent_cycle = None

    if self.keep_simplex_diff:
        parent_node = PFNode(id=nid, 
                             filt_val=filt_val, 
                             children={node1.id,node2.id}, 
                             cycle=parent_cycle,
                             _simplex_diff_available = True,
                             _interior_diff = None,
                             _codim1_simplex_diff = {(key(simplex),1),(key(simplex),-1)})
    else:
        parent_node = PFNode(id=nid, filt_val=filt_val, children={node1.id,node2.id}, cycle=parent_cycle, _simplex_diff_available = False)
    self.nodes[nid]=parent_node
    
    self._active_node_ids.add(nid)
    self._active_node_ids.remove(node1.id)
    self._active_node_ids.remove(node2.id)

    self.levels.append(filt_val)

    return

def update_node(self, node: PFNode, simplex: List[int], filt_val:float):
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

    if not self.diff_only_mode:
        updated_cycle = node.cycle.cancel_simplex(simplex=simplex)
    else: updated_cycle = None

    nid = next(self._node_id)
    node.parent=nid

    if self.keep_simplex_diff:
        update_node = PFNode(id=nid, 
                             filt_val=filt_val, 
                             children={node.id},
                             cycle = updated_cycle,
                             _simplex_diff_available = True,
                             _interior_diff = None,
                             _codim1_simplex_diff = {(key(simplex),1),(key(simplex),-1)})
    else:
        update_node = PFNode(id=nid, filt_val=filt_val, children={node.id},cycle = updated_cycle, _simplex_diff_available = False)

    self.nodes[nid]=update_node

    self._active_node_ids.add(nid)
    self._active_node_ids.remove(node.id)

    self.levels.append(filt_val)

    return

def _update_node_list(self, node_id_list: List[int]) -> List[PFNode]:
    """Return current root ancestors for ``node_id_list``."""
    L=[]
    for id in node_id_list:
        if not self.nodes[id]._is_inactive_leaf:
            active_node = self.get_root(self.nodes[id])
            L.append(active_node)
    return L

def _compute_forest(self, print_info: bool = False):
    """ 
        Compute the persistence forest from the alpha-complex filtration.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.
        """

    loop_forest_start = time.perf_counter()

    face_cycle_dict = {}

    #simplices is already ordered in ascending order by number of simplices 
    for simplex, filt_val in reversed(self.filtration):
        #print("simplex", simplex)
        #print("active_node_ids",self._active_node_ids)
        
        if len(simplex) == self.dim+1:
            orientation = simplex_orientation(simplex=simplex, point_cloud=self.point_cloud)

            new_node = self.add_leaf( simplex=simplex, filt_val=filt_val, orientation = orientation)

            faces = list(itertools.combinations(simplex, len(simplex)-1 ) )
            for face in faces:
                if key(face) in face_cycle_dict:
                   face_cycle_dict[key(face)].append(new_node.id)
                else:
                    face_cycle_dict[key(face)] = [new_node.id]

        elif len(simplex) == self.dim:
            #face_cycle_dict[simplex] contains nodes induced by cofaces of simplex
            # can be of the form [],[l1], [l1,l2], [l1,l1]

            #If key exists, get its value and remove it
            #if key does not exists, get []
            L_tmp_ids = face_cycle_dict.pop(key(simplex), [])

            # find nodes corresponding to cofaces, find active nodes above leaves induced by cofaces, 
            # we remove construction roots since they are not active anymore
            L = self._update_node_list(L_tmp_ids)


            #if no cycle contains simplex, nothing happens
            if len(L) == 0:
                continue

            #if cycle is only contained in a single loop and appears only once in that loop once, remove that loop from the active loops 
            elif len(L) == 1:
                self.make_root(node=L[0],filt_val=filt_val)

                continue

            elif len(L) == 2 and L[0]!=L[1]: 
                self.merge_nodes( node1=L[0], node2=L[1], simplex =simplex, filt_val=filt_val)
                """if not loop_in_filtration_check(parent_loop.vertex_list, simplex_tree=self.simplex_tree, filt_value=filt_val):
                            print('edge', simplex)
                            print('edge dict entry')
                            print('filtration value', filt_val)
                            print(f'first loop', L[0].cycle)
                            print(f'second loop', L[1].cycle)
                            raise ValueError("Loop not in simplex, Loop concat Case")"""


            elif len(L) == 2 and L[0]==L[1]:
                #Same simplex is contained in a cycle in both orientations -> we remove it from the the cycle
                self.update_node(node=L[0], simplex=simplex, filt_val=filt_val)
                """if not loop_in_filtration_check(vertex_loop, simplex_tree=self.simplex_tree, filt_value=filt_val):
                                print('edge', simplex)
                                print(f'starting loop', L[0].loop)
                                print(f'outer loop', vertex_loop)
                                raise ValueError("Loop not in simplex, Tiebreak Case")"""
                
            else:
                print(L)
                print(simplex)
                raise ValueError("Error, L is of the wrong form")

    loop_forest_time = time.perf_counter() - loop_forest_start
    if print_info:
        print(f"Forest succesfully computed in {loop_forest_time} sec")

    return
