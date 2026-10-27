"""_forest.build: reverse sweep over a cell filtration."""

from collections.abc import Hashable
from persforest._forest.graph_model import PFNode
from persforest._forest.queries import _cached_root_id
from persforest.chains import SignedChain
from persforest.chains import merge_at_simplex
from persforest.filtrations._interface import oriented_boundary
from typing import List
import time

def add_leaf(self, cell: Hashable, boundary: tuple, filt_val: float, orientation: int):
    """
        Create a new leaf node for a top cell.

        In the reverse-filtration algorithm this corresponds to the death of a
        codimension-one homology class; the new cycle is the signed boundary of
        ``cell``.

        Parameters
        ----------
        cell : hashable
            Top cell being added.
        boundary : tuple of (cell, int)
            Boundary of ``cell`` with ``orientation`` applied.
        filt_val : float
            Filtration value where the cell enters.
        orientation : int
            Orientation (+/-1) of the cell in the complex.

        Returns
        -------
        PFNode
            Newly created leaf node.
        """

    nid = next(self._node_id)

    if self.storage == "cycles":
        new_cycle = SignedChain(signed_simplices=set(boundary))
    else:
        new_cycle = None

    new_node = PFNode(id=nid,
                      filt_val=filt_val, 
                      children=set(), 
                      cycle=new_cycle, 
                      _simplex_diff_available = True,
                      _interior_diff = {(cell, orientation)},
                      _codim1_simplex_diff = None)

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

    root_node = PFNode(id=nid, 
                       filt_val=filt_val, 
                       cycle=node.cycle, 
                       children={node.id},
                       _simplex_diff_available = True,
                       _interior_diff = None,
                       _codim1_simplex_diff = None)


    self.nodes[nid]=root_node
    self.roots.add(root_node.id)

    #set all leaves below root as inactive
    leaf_ids = self.leaves_below_node(node=root_node)
    for leaf_id in leaf_ids:
        self.nodes[leaf_id]._is_inactive_leaf=True

    self.levels.append(filt_val)

    return

def merge_nodes(self, node1: PFNode, node2: PFNode, cell: Hashable, filt_val: float):
    """ 
        Create a parent node by merging two active child nodes.

        The parent cycle is obtained by unioning the child cycles and removing
        both orientations of ``cell``. In the reverse-filtration algorithm,
        this records a split event and creates a new barcode branch.

        Parameters
        ----------
        node1, node2 : PFNode
            Child nodes that are merged.
        cell : hashable
            Codimension-one cell at which the two children cycles merge.
        filt_val : float
            Filtration value of the merge event.
        """

    nid = next(self._node_id)
    node1.parent = nid
    node2.parent = nid

    if self.storage == "cycles":
        parent_cycle = merge_at_simplex(cycle1 = node1.cycle, cycle2=node2.cycle,  simplex=cell) 
    else:
        parent_cycle = None

    parent_node = PFNode(id=nid, 
                         filt_val=filt_val, 
                         children={node1.id,node2.id}, 
                         cycle=parent_cycle,
                         _simplex_diff_available = True,
                         _interior_diff = None,
                         _codim1_simplex_diff = {(cell,1),(cell,-1)})
    self.nodes[nid]=parent_node
    
    self._active_node_ids.add(nid)
    self._active_node_ids.remove(node1.id)
    self._active_node_ids.remove(node2.id)

    self.levels.append(filt_val)

    return

def update_node(self, node: PFNode, cell: Hashable, filt_val:float):
    """
        Create a parent node for an active node whose cycle changes.

        The updated cycle is obtained by removing both orientations of
        ``cell`` from ``node.cycle``.

        Parameters
        ----------
        node : PFNode
            Node whose representative is being updated.
        cell : hashable
            Codimension-one cell to remove from the representative.
        filt_val : float
            Filtration value of the update event.
        """

    if self.storage == "cycles":
        updated_cycle = node.cycle.cancel_simplex(simplex=cell)
    else: updated_cycle = None

    nid = next(self._node_id)
    node.parent=nid

    update_node = PFNode(id=nid, 
                         filt_val=filt_val, 
                         children={node.id},
                         cycle = updated_cycle,
                         _simplex_diff_available = True,
                         _interior_diff = None,
                         _codim1_simplex_diff = {(cell,1),(cell,-1)})

    self.nodes[nid]=update_node

    self._active_node_ids.add(nid)
    self._active_node_ids.remove(node.id)

    self.levels.append(filt_val)

    return

def _update_node_list(self, node_id_list: List[int], ancestors: dict[int, int]) -> List[PFNode]:
    """Return current root ancestors for ``node_id_list``, skipping inactive leaves."""
    L=[]
    for id in node_id_list:
        if not self.nodes[id]._is_inactive_leaf:
            L.append(self.nodes[_cached_root_id(self.nodes, ancestors, id)])
    return L

def _compute_forest(self, print_info: bool = False):
    """ 
        Compute the persistence forest by sweeping ``self._cells`` in reverse.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.
        """

    loop_forest_start = time.perf_counter()

    cells = self._cells
    top_dimension = cells.top_dimension

    # Facet -> leaves created by its incident top cells, in sweep order.
    facet_leaf_ids = {}
    # Node -> some ancestor, for root lookups; valid only during the sweep.
    ancestors: dict[int, int] = {}

    for cell, filt_val in reversed(cells.events):
        dimension = cells.dimension(cell)

        if dimension == top_dimension:
            orientation = cells.orientation(cell)
            boundary = oriented_boundary(cells, cell, orientation)

            new_node = add_leaf(self, cell=cell, boundary=boundary, filt_val=filt_val, orientation=orientation)

            for facet, _ in boundary:
                if facet in facet_leaf_ids:
                    facet_leaf_ids[facet].append(new_node.id)
                else:
                    facet_leaf_ids[facet] = [new_node.id]

        elif dimension == top_dimension - 1:
            # Leaves of the cofaces of this facet: [], [l1], [l1, l2] or [l1, l1].
            L_tmp_ids = facet_leaf_ids.pop(cell, [])

            # Active nodes above those leaves; leaves below a construction root are no longer active.
            L = _update_node_list(self, L_tmp_ids, ancestors)

            # No active cycle contains the facet.
            if len(L) == 0:
                continue

            # One active cycle contains the facet once: the class is born here.
            elif len(L) == 1:
                make_root(self, node=L[0], filt_val=filt_val)

            elif len(L) == 2 and L[0]!=L[1]: 
                merge_nodes(self, node1=L[0], node2=L[1], cell=cell, filt_val=filt_val)

            elif len(L) == 2 and L[0]==L[1]:
                # The same cycle contains the facet in both orientations; remove it.
                update_node(self, node=L[0], cell=cell, filt_val=filt_val)

            else:
                raise ValueError(f"Facet {cell} has {len(L)} incident active cycles; expected at most 2")

    loop_forest_time = time.perf_counter() - loop_forest_start
    if print_info:
        print(f"Forest succesfully computed in {loop_forest_time} sec")

    return
