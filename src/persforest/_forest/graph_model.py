"""_forest.graph model: mechanically extracted reference implementation."""

from dataclasses import dataclass
from persforest.chains import SignedChain
from typing import Optional

@dataclass(slots=True)
class PFNode:
    """
    Node in the PersistenceForest graph.

    Each node stores a cycle representative and links to its parent and
    children in the forest. Optional diff fields store how to reconstruct cycle
    representatives and interiors.
    """
    id: int # unique node identifier
    filt_val: float
    cycle: SignedChain | None
    children: set[int]                                    #ids of children
    parent: Optional[int] = None
    _is_inactive_leaf: bool = False #keeps track of leaves where root above leaves have been removed from active cycles
    _barcode_covered: int = 0

    _simplex_diff_available: bool = False       # True when simplex diffs were stored for this node.
    _interior_diff: set[tuple] | None = None     # oriented full-dimensional simplices, stored as (simplex, orientation)
    _codim1_simplex_diff: set[tuple] | None = None  # oriented codimension-one simplices to remove, stored as (simplex, orientation)
    # Cycle at node: union of child cycles plus boundaries from _interior_diff,
    # minus _codim1_simplex_diff.
    _barcode_interior_diff: set[tuple] | None = None  
    _barcode_codim1_simplex_diff: set[tuple] | None = None  

    def __repr__(self) -> str:
        return f"Node(id={self.id}, f={self.filt_val})"
