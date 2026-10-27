"""measurements.basic: mechanically extracted reference implementation."""

from numpy.typing import NDArray
from persforest.chains import SignedChain
from persforest.geometry.planar import polygon_area
from persforest.geometry.planar import signed_chain_to_polyhedral_paths
from typing import Dict
from typing import Set
import math
import numpy as np

def signed_chain_edge_length(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return the total Euclidean length of the chain's edges.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Sum of edge lengths, ignoring orientation. The zero chain has length zero.
    """
    if not signed_chain.signed_simplices:
        return 0.0
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    return math.fsum(
        float(np.linalg.norm(point_cloud[simplex[1]] - point_cloud[simplex[0]]))
        for simplex, _ in signed_chain.signed_simplices
        if len(simplex) == 2
    )

def constant_one_functional(signed_chain: SignedChain, point_cloud = None) -> float:
    """
    Return the constant value 1 for all non-zero chains.

    Parameters
    ----------
    signed_chain : SignedChain
    point_cloud : ndarray, optional
        Ignored.

    Returns
    -------
    float:
        1 if chain is non-zero, 
        0 for zero-chain
        
    """
    if not signed_chain.signed_simplices:
        return 0
    else:
        return 1

def _support_graph_component_count(signed_chain: SignedChain) -> int:
    """Count components of the underlying edge support, ignoring orientation.
    Edges of opposing support do not cancel.
    Empty support has zero components.

    Parameters
    ----------
    signed_chain : SignedChain
        Chain containing only edges.

    Returns
    -------
    int
        Number of connected support components.

    Raises
    ------
    ValueError
        If any simplex is not an edge.
    """
    adjacency: Dict[int, Set[int]] = {}
    for simplex, _ in signed_chain.signed_simplices:
        if len(simplex) != 2:
            raise ValueError("Function only defined for 1-dimensional chains")
        first, second = simplex
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)

    visited: Set[int] = set()
    component_count = 0
    for vertex in adjacency:
        if vertex in visited:
            continue
        component_count += 1
        visited.add(vertex)
        pending = [vertex]
        while pending:
            for neighbor in adjacency[pending.pop()]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
    return component_count

def signed_chain_connected_components(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> int:
    """
    Count connected components of a signed 1-chain's edge support.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Unused; connectivity is determined by shared vertex indices.

    Returns
    -------
    int
        Number of support components, or zero for an empty chain.
    """
    return _support_graph_component_count(signed_chain)

def signed_chain_excess_connected_components(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> int:
    """
    Count support components beyond the first one.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    int
        Number of support components minus one, or zero for an empty chain.
    """
    return max(0, _support_graph_component_count(signed_chain) - 1)

def signed_chain_connected_components_only_signed_simplices(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> int:
    """
    Count support components formed only by doubled simplices.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    int
        Number of support components formed by doubled simplices.
    """
    
    purely_signed_chain = signed_chain.only_double_simplices()
    if len(purely_signed_chain.signed_simplices)==0:
        return 0
    
    return signed_chain_connected_components( signed_chain=purely_signed_chain, point_cloud=point_cloud)

def signed_chain_area(signed_chain: SignedChain, point_cloud:  NDArray[np.float64]) -> float:
    """
    Return the area enclosed by a planar signed 1-cycle.

    Intended for cycles produced by the persistence forest, whose boundary
    paths consist of one outer boundary and possibly hole boundaries.
    Stored interior data is not required. General cycles are unsupported;
    for disconnected interiors, the result is not guaranteed to equal the
    enclosed area.

    Parameters
    ----------
    signed_chain : SignedChain
        Planar signed 1-cycle with the boundary structure described above.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Outer area minus inner path areas. The zero chain has area zero.
    """
    if not signed_chain.signed_simplices:
        return 0.0
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    paths = signed_chain_to_polyhedral_paths(signed_chain=signed_chain, point_cloud=point_cloud)
    x_max_list = np.array([point_cloud[path, 0].max() for path in paths])
    index_max = np.argmax(x_max_list)

    signed_areas = []

    for index, path in enumerate(paths):
        if len(path) < 2:
            print(paths)
            raise ValueError("Paths too short")
        path_area = polygon_area(point_cloud[path])
        signed_areas.append(path_area if index == index_max else -path_area)
    return math.fsum(signed_areas)

def _require_chain_interior(signed_chain: SignedChain) -> None:
    """Raise an actionable error when a chain has no interior data.

    Parameters
    ----------
    signed_chain : SignedChain
        Chain whose interior availability is checked.

    Raises
    ------
    ValueError
        If the interior was not stored or reconstructed.
    """
    if not signed_chain.interior_available or signed_chain.interior is None:
        raise ValueError(
            "Interior is unavailable. Obtain the chain through "
            "forest.iter_bar_cycle_reps(bar) or forest.cycle_for_bar_at(bar, value)."
        )

def signed_chain_interior_volume(
    signed_chain: SignedChain, point_cloud: NDArray[np.float64]
) -> float:
    """Return enclosed volume by summing absolute interior simplex volumes.

    Computes area in 2D and volume in 3D (or d-volume in dimension d).
    Orientation is ignored; an available but empty interior has volume zero.

    Raises
    ------
    ValueError
        If the interior is unavailable or simplices are not full-dimensional.
    """
    _require_chain_interior(signed_chain)
    if not signed_chain.interior:
        return 0.0

    points = np.asarray(point_cloud, dtype=float)
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError("point_cloud must have shape (n_points, dim) with dim >= 1")
    dim = points.shape[1]
    simplices = [simplex for simplex, _ in signed_chain.interior]
    if any(len(simplex) != dim + 1 for simplex in simplices):
        raise ValueError("Interior simplices must be full-dimensional (dim + 1 vertices)")

    vertices = points[np.asarray(simplices, dtype=int)]
    edges = vertices[:, 1:, :] - vertices[:, :1, :]
    volumes = np.abs(np.linalg.det(edges)) / math.factorial(dim)
    return float(math.fsum(volumes))

def signed_chain_volume(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return the summed simplex volume of a signed chain.

    Parameters
    ----------
    signed_chain : SignedChain
        Chain of simplices whose vertices index into ``point_cloud``.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Sum of nonnegative intrinsic simplex volumes. Opposite orientations
        contribute separately. An empty chain has volume zero.

    Notes
    -----
    For a k-simplex, edge vectors relative to its first vertex form the
    rows of E. Its volume is sqrt(det(E @ E.T)) / k!, including when the
    simplex is embedded in a higher-dimensional space. This measures the
    chain itself, not its enclosed volume.
    """

    if not signed_chain.signed_simplices:
        return 0.0

    simplices = np.asarray(
        [simplex for simplex, _ in signed_chain.signed_simplices], dtype=int
    )
    vertices = np.asarray(point_cloud, dtype=float)[simplices]
    edges = vertices[:, 1:, :] - vertices[:, :1, :]
    simplex_dim = edges.shape[1]

    gram = edges @ edges.swapaxes(-1, -2)
    # Roundoff can make the determinant of a singular Gram matrix negative.
    volumes = np.sqrt(np.maximum(np.linalg.det(gram), 0.0)) / math.factorial(simplex_dim)

    return float(math.fsum(volumes))
