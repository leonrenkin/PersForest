"""measurements.shape: mechanically extracted reference implementation."""

from numpy.typing import NDArray
from persforest.chains import SignedChain
from persforest.geometry.planar import _convex_hull_indices
from persforest.geometry.planar import _convex_polygon_diameter
from persforest.geometry.planar import _planar_signed_chain_vertices
from persforest.geometry.planar import _polygon_max_convexity_defect_depth
from persforest.geometry.planar import curvature_excess
from persforest.geometry.planar import polygon_area
from persforest.geometry.planar import polygon_length
from persforest.geometry.planar import signed_chain_to_polyhedral_paths
from persforest.measurements.basic import _support_graph_component_count
from persforest.measurements.basic import signed_chain_area
from persforest.measurements.basic import signed_chain_connected_components_only_signed_simplices
from persforest.measurements.basic import signed_chain_edge_length
from typing import Dict
import math
import numpy as np

_UNIT_INTERVAL_TOL = 64 * np.finfo(np.float64).eps

def _require_unit_interval(value: float, measurement_name: str) -> float:
    """Validate a normalized geometric measurement."""
    if not math.isfinite(value):
        raise ValueError(f"{measurement_name} must be finite, got {value}")
    if -_UNIT_INTERVAL_TOL <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + _UNIT_INTERVAL_TOL:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{measurement_name} must lie in [0, 1], got {value}. "
            "This indicates inconsistent cycle and convex-hull geometry."
        )
    return float(value)

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
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    vertex_counts: Dict[int, int] = {}
    for simplex, sign in signed_chain.signed_simplices:

        for vertex in simplex:
            vertex_counts[vertex] = vertex_counts.get(vertex, 0) + 1

    return sum(1 for count in vertex_counts.values() if count > 2)

def signed_chain_num_of_branching_points_only_signed_simplices(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    purely_signed_chain = signed_chain.only_double_simplices()
    if len(purely_signed_chain.signed_simplices)==0:
        return 0
    
    return signed_chain_num_of_branching_points(signed_chain=purely_signed_chain, point_cloud=point_cloud)

def signed_chain_tendril_branching_ratio(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
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

    The measurement is

    ``1 - enclosed_area / convex_hull_area``.

    It is zero for a convex enclosed region and approaches one as the enclosed
    area becomes small relative to its convex hull. A chain with zero convex
    hull area is treated as convex and returns zero.

    Intended for cycles with connected interior, as produced by the
    persistence forest. The enclosed-area computation assumes one outer
    boundary and possibly hole boundaries. Stored interior data is not
    required. Disconnected interiors are outside this functional's supported
    scope. One global hull is computed from all chain vertices, including
    hole boundaries and signed tendrils; it is not computed componentwise.

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
    vertices = _planar_signed_chain_vertices(signed_chain, point_cloud)
    if len(vertices) == 0:
        return 0.0

    hull = vertices[_convex_hull_indices(vertices)]
    hull_area = polygon_area(hull)
    enclosed_area = signed_chain_area(signed_chain, point_cloud)
    if hull_area == 0.0:
        if enclosed_area != 0.0:
            raise ValueError(
                "Convex hull has zero area but the signed chain encloses "
                f"area {enclosed_area}"
            )
        return 0.0

    return _require_unit_interval(
        1.0 - enclosed_area / hull_area, "area deficit"
    )

def signed_chain_convex_hull_perimeter_deficit(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
) -> float:
    """Return the relative perimeter excess over the chain's convex hull.

    The measurement is

    ``max(1 - convex_hull_perimeter / chain_edge_length, 0)``.

    It is zero for a convex polygonal cycle. Doubled signed edges contribute
    to the chain length and therefore increase the deficit. A zero-length
    chain is treated as convex and returns zero.

    Intended for cycles with connected interior, as produced by the
    persistence forest. Stored interior data is not required. Disconnected
    interiors are outside this functional's supported scope; the hull is not
    computed componentwise. The clipping implements the nonnegative deficit
    definition when the global hull perimeter exceeds the chain length;
    a zero result alone does not establish convexity for disconnected input.

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
    vertices = _planar_signed_chain_vertices(signed_chain, point_cloud)
    if len(vertices) == 0:
        return 0.0

    hull = vertices[_convex_hull_indices(vertices)]
    hull_perimeter = polygon_length(hull)
    chain_length = signed_chain_edge_length(signed_chain, point_cloud)
    if chain_length == 0.0:
        if hull_perimeter != 0.0:
            raise ValueError(
                "Signed chain has zero length but its convex hull has perimeter "
                f"{hull_perimeter}"
            )
        return 0.0

    return _require_unit_interval(
        max(1.0 - hull_perimeter / chain_length, 0.0), "perimeter deficit"
    )

def signed_chain_max_convexity_defect_depth(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
    normalize: bool = True,
) -> float:
    """Return the maximum hull-chord indentation depth of the chain.

    Convexity defects are computed separately on each closed polygonal path,
    and the largest depth is returned. With the default normalization, depth
    is divided by the diameter of the convex hull of the entire chain. Thus a
    convex or degenerate chain returns zero and the normalized result lies in
    ``[0, 1]``.

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
    vertices = _planar_signed_chain_vertices(signed_chain, point_cloud)
    if len(vertices) == 0:
        return 0.0

    coordinates = np.asarray(point_cloud, dtype=float)
    paths = signed_chain_to_polyhedral_paths(signed_chain, coordinates)
    if not paths:
        raise ValueError("Non-empty signed chain produced no polygonal paths")
    maximum_depth = max(
        _polygon_max_convexity_defect_depth(coordinates[path])
        for path in paths
    )

    if not normalize:
        return maximum_depth

    hull = vertices[_convex_hull_indices(vertices)]
    hull_diameter = _convex_polygon_diameter(hull)
    if hull_diameter == 0.0:
        if maximum_depth != 0.0:
            raise ValueError(
                "Convex hull has zero diameter but the chain has "
                f"convexity-defect depth {maximum_depth}"
            )
        return 0.0
    return _require_unit_interval(
        maximum_depth / hull_diameter, "normalized convexity-defect depth"
    )

def signed_chain_excess_curvature(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return total excess curvature over the chain's closed paths.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Sum of unnormalized excess curvature values; zero for the zero chain.
    """
    if not signed_chain.signed_simplices:
        return 0.0
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    paths = signed_chain_to_polyhedral_paths(signed_chain=signed_chain, point_cloud=point_cloud)
    total = 0
    for path in paths:
        if len(path) < 2:
            print(paths)
            raise ValueError("Paths too short")
        total += curvature_excess(point_cloud[path], normalize=False)

    return total

def signed_chain_excess_curvature_diff_to_unsigned(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return excess curvature lost when opposite orientations are cancelled.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

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
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Normalized excess curvature.
    """
    return signed_chain_excess_curvature(signed_chain, point_cloud) / (2.0*math.pi)

def signed_chain_circularity(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return the circularity score ``4π * area / perimeter**2``.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Circularity score; 1 is circular, lower values are less circular.
        The zero chain has score zero by convention.
    """
    if not signed_chain.signed_simplices:
        return 0.0
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    
    length = signed_chain_edge_length(signed_chain=signed_chain, point_cloud=point_cloud)
    area = signed_chain_area(signed_chain=signed_chain,point_cloud=point_cloud)
    circularity = (4.0*math.pi*area)/length**2

    return circularity

def signed_chain_circularity_complement(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return ``1 - signed_chain_circularity(...)``.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Complement of the circularity score; zero for the zero chain.
    """
    if not signed_chain.signed_simplices:
        return 0.0
    return 1- signed_chain_circularity(signed_chain=signed_chain, point_cloud=point_cloud)

def signed_chain_circularity_squared_complement(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
) -> float:
    """Return ``1 - signed_chain_circularity(...)**2``.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Complement of the squared circularity score; zero for the zero chain.
    """
    if not signed_chain.signed_simplices:
        return 0.0
    circularity = signed_chain_circularity(signed_chain, point_cloud)
    return 1 - circularity**2

def signed_chain_circularity_complement_squared(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
) -> float:
    """Return ``(1 - signed_chain_circularity(...))**2``.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Squared complement of the circularity score; zero for the zero chain.
    """
    if not signed_chain.signed_simplices:
        return 0.0
    complement = signed_chain_circularity_complement(signed_chain, point_cloud)
    return complement**2

def signed_chain_non_circularity(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return the non-circularity score ``perimeter**2 / (4π * area) - 1``.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Non-circularity score; 0 is circular, higher values are less circular.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    length = signed_chain_edge_length(signed_chain=signed_chain, point_cloud=point_cloud)
    area = signed_chain_area(signed_chain=signed_chain,point_cloud=point_cloud)
    non_circularity = length**2/(4.0*math.pi*area) - 1

    return non_circularity
