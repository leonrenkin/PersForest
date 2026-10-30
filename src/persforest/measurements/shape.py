"""measurements.shape: shape and tendril measurements of planar simplicial 1-chains.

The shape functionals measure the raw polygon loops of the chain with the
functionals of ``persforest.measurements.loops``.
"""

from numpy.typing import NDArray
from persforest.chains import SignedChain
from persforest.geometry.loops import _polygon_loops
from persforest.measurements.basic import _reject_cubical_geometry
from persforest.measurements.basic import _support_graph_component_count
from persforest.measurements.basic import signed_chain_connected_components_only_signed_simplices
from persforest.measurements.basic import signed_chain_edge_length
from persforest.measurements.loops import loops_circularity
from persforest.measurements.loops import loops_circularity_complement
from persforest.measurements.loops import loops_circularity_complement_squared
from persforest.measurements.loops import loops_circularity_squared_complement
from persforest.measurements.loops import loops_convex_hull_area_deficit
from persforest.measurements.loops import loops_convex_hull_perimeter_deficit
from persforest.measurements.loops import loops_excess_curvature
from persforest.measurements.loops import loops_max_convexity_defect_depth
from persforest.measurements.loops import loops_non_circularity
from typing import Dict
import math
import numpy as np

def signed_chain_avg_tendril_length(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return the average length of doubled-edge components.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Average geometric length of doubled-edge support components. Each
        underlying edge is counted once, preserving the factor of two used
        to convert signed length to geometric length.
    """
    _reject_cubical_geometry(point_cloud, "cubical_chain_avg_tendril_length")
    tendril_chain = signed_chain.only_double_simplices()
    if len(tendril_chain.signed_simplices)==0:
        return 0
    length = signed_chain_edge_length(signed_chain=tendril_chain, point_cloud=point_cloud)
    tendril_num = _support_graph_component_count(tendril_chain)
    
    return length / (tendril_num*2)

def signed_chain_num_of_branching_points(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Count vertices incident to more than two signed edges.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices. Unused, kept for the
        vectorisation functional signature.

    Returns
    -------
    int
        Number of vertices contained in more than two signed 1-simplices.
    """
    _reject_cubical_geometry(point_cloud, "cubical_chain_num_of_branching_points")
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    vertex_counts: Dict[int, int] = {}
    for simplex, sign in signed_chain.signed_simplices:

        for vertex in simplex:
            vertex_counts[vertex] = vertex_counts.get(vertex, 0) + 1

    return sum(1 for count in vertex_counts.values() if count > 2)

def signed_chain_num_of_branching_points_only_signed_simplices(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    _reject_cubical_geometry(point_cloud, "cubical_chain_num_of_branching_points_only_signed_simplices")
    purely_signed_chain = signed_chain.only_double_simplices()
    if len(purely_signed_chain.signed_simplices)==0:
        return 0
    
    return signed_chain_num_of_branching_points(signed_chain=purely_signed_chain, point_cloud=point_cloud)

def signed_chain_tendril_branching_ratio(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    _reject_cubical_geometry(point_cloud, "cubical_chain_tendril_branching_ratio")
    tendril_branching_count = signed_chain_num_of_branching_points_only_signed_simplices(signed_chain=signed_chain, point_cloud=point_cloud)
    tendril_count = signed_chain_connected_components_only_signed_simplices(signed_chain=signed_chain, point_cloud=point_cloud)
    if tendril_count == 0:
        return 0
    return tendril_branching_count / tendril_count

def signed_chain_convex_hull_area_deficit(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
) -> float:
    """Return the relative area missing from the chain's convex hull.

    ``loops_convex_hull_area_deficit`` of the chain's polygon loops:
    ``1 - enclosed_area / convex_hull_area``, zero for a convex enclosed region
    and approaching one as the enclosed area becomes small relative to its
    convex hull. One global hull is computed from all chain vertices,
    including hole boundaries and signed tendrils.

    Parameters
    ----------
    signed_chain : SignedChain
        Planar signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Area deficit in ``[0, 1]``.
    """
    return loops_convex_hull_area_deficit(_polygon_loops(signed_chain, point_cloud))

def signed_chain_convex_hull_perimeter_deficit(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
) -> float:
    """Return the relative perimeter excess over the chain's convex hull.

    ``loops_convex_hull_perimeter_deficit`` of the chain's polygon loops:
    ``max(1 - convex_hull_perimeter / chain_edge_length, 0)``, zero for a
    convex polygonal cycle. Doubled signed edges contribute to the chain
    length and therefore increase the deficit.

    Parameters
    ----------
    signed_chain : SignedChain
        Planar signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Perimeter deficit in ``[0, 1]``.
    """
    return loops_convex_hull_perimeter_deficit(_polygon_loops(signed_chain, point_cloud))

def signed_chain_max_convexity_defect_depth(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
    normalize: bool = True,
) -> float:
    """Return the maximum hull-chord indentation depth of the chain.

    ``loops_max_convexity_defect_depth`` of the chain's polygon loops:
    defects are computed separately on each closed polygonal path, and the
    largest depth is returned, by default divided by the diameter of the
    convex hull of the entire chain.

    Parameters
    ----------
    signed_chain : SignedChain
        Planar signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.
    normalize : bool, optional
        If True, divide the depth by the global convex-hull diameter. If
        False, return depth in the coordinate units of ``point_cloud``.

    Returns
    -------
    float
        Maximum normalized or absolute convexity-defect depth.
    """
    return loops_max_convexity_defect_depth(_polygon_loops(signed_chain, point_cloud), normalize=normalize)

def signed_chain_excess_curvature(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return total excess curvature over the chain's closed paths.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Sum of unnormalized excess curvature values; zero for the zero chain.
    """
    if not signed_chain.signed_simplices:
        return 0.0
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    return loops_excess_curvature(_polygon_loops(signed_chain, point_cloud))

def signed_chain_excess_curvature_diff_to_unsigned(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return excess curvature lost when opposite orientations are cancelled.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Difference between signed and unsigned excess curvature.
    """
    diff = signed_chain_excess_curvature(signed_chain=signed_chain,point_cloud=point_cloud) - signed_chain_excess_curvature(signed_chain=signed_chain.unsigned(),point_cloud=point_cloud)
    return diff

def signed_chain_excess_curvature_normalized(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return total excess curvature normalized by 2π.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Normalized excess curvature.
    """
    return signed_chain_excess_curvature(signed_chain, point_cloud) / (2.0*math.pi)

def _nonzero_planar_loops(signed_chain: SignedChain, point_cloud: NDArray[np.float64]):
    """Return the polygon loops of a non-zero chain, or None for the zero chain."""
    if not signed_chain.signed_simplices:
        return None
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    return _polygon_loops(signed_chain, point_cloud)

def signed_chain_circularity(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return the circularity score ``4π * area / perimeter**2``.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Circularity score; 1 is circular, lower values are less circular.
        The zero chain has score zero by convention.
    """
    loops = _nonzero_planar_loops(signed_chain, point_cloud)
    return 0.0 if loops is None else loops_circularity(loops)

def signed_chain_circularity_complement(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return ``1 - signed_chain_circularity(...)``; zero for the zero chain.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.
    """
    loops = _nonzero_planar_loops(signed_chain, point_cloud)
    return 0.0 if loops is None else loops_circularity_complement(loops)

def signed_chain_circularity_squared_complement(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
) -> float:
    """Return ``1 - signed_chain_circularity(...)**2``; zero for the zero chain.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.
    """
    loops = _nonzero_planar_loops(signed_chain, point_cloud)
    return 0.0 if loops is None else loops_circularity_squared_complement(loops)

def signed_chain_circularity_complement_squared(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
) -> float:
    """Return ``(1 - signed_chain_circularity(...))**2``; zero for the zero chain.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.
    """
    loops = _nonzero_planar_loops(signed_chain, point_cloud)
    return 0.0 if loops is None else loops_circularity_complement_squared(loops)

def signed_chain_non_circularity(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return the non-circularity score ``perimeter**2 / (4π * area) - 1``.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Non-circularity score; 0 is circular, higher values are less circular.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    return loops_non_circularity(_polygon_loops(signed_chain, point_cloud))
