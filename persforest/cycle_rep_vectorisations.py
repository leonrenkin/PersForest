import math
from typing import Iterable, Tuple, List, Dict, Set, Optional
from numpy.typing import NDArray
import numpy as np
from .PersistenceForest import SignedChain

_EPS = 1e-12
_UNIT_INTERVAL_TOL = 64 * np.finfo(np.float64).eps

def _to_xy(points: Iterable[Tuple[float, float]]) -> np.ndarray:
    """
    Convert input points to an ``(N, 2)`` float64 array, dropping a duplicated
    closing point when the first and last coordinates coincide.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices in order; a trailing copy of the first vertex is
        tolerated and removed.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with dtype float64.

    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be an (N,2) array-like of x,y pairs")
    if np.linalg.norm(P[0] - P[-1]) < _EPS:  # drop repeated closure point
        P = P[:-1].copy()
    return P

def _convex_hull_indices(points: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return counterclockwise convex-hull vertex indices without closure."""
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("points must contain only finite coordinates")

    # Lexicographic ordering with the original index as a deterministic
    # tie-breaker. Keep one representative of each repeated coordinate.
    order = np.lexsort(
        (np.arange(len(coordinates)), coordinates[:, 1], coordinates[:, 0])
    )
    unique_indices: List[int] = []
    for index in order:
        index = int(index)
        if not unique_indices or not np.array_equal(
            coordinates[index], coordinates[unique_indices[-1]]
        ):
            unique_indices.append(index)

    if len(unique_indices) <= 2:
        return np.asarray(unique_indices, dtype=np.int64)

    def cross(origin_index: int, first_index: int, second_index: int) -> float:
        first = coordinates[first_index] - coordinates[origin_index]
        second = coordinates[second_index] - coordinates[origin_index]
        return float(first[0] * second[1] - first[1] * second[0])

    lower: List[int] = []
    for index in unique_indices:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], index) <= 0.0:
            lower.pop()
        lower.append(index)

    upper: List[int] = []
    for index in reversed(unique_indices):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], index) <= 0.0:
            upper.pop()
        upper.append(index)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.int64)

def _convex_polygon_diameter(hull: NDArray[np.float64]) -> float:
    """Return the Euclidean diameter of a counterclockwise convex polygon."""
    hull_coordinates = np.asarray(hull, dtype=float)
    n_vertices = len(hull_coordinates)
    if n_vertices <= 1:
        return 0.0
    if n_vertices == 2:
        return float(np.linalg.norm(hull_coordinates[1] - hull_coordinates[0]))

    def doubled_triangle_area(i: int, j: int, k: int) -> float:
        edge = hull_coordinates[j] - hull_coordinates[i]
        offset = hull_coordinates[k] - hull_coordinates[i]
        return abs(float(edge[0] * offset[1] - edge[1] * offset[0]))

    # Rotating calipers: each hull vertex advances at most once around the
    # polygon, so the diameter computation is linear in the hull size.
    antipodal_index = 1
    maximum_squared_distance = 0.0
    for index in range(n_vertices):
        next_index = (index + 1) % n_vertices
        while doubled_triangle_area(
            index,
            next_index,
            (antipodal_index + 1) % n_vertices,
        ) > doubled_triangle_area(index, next_index, antipodal_index):
            antipodal_index = (antipodal_index + 1) % n_vertices

        for first_index in (index, next_index):
            for second_index in (
                antipodal_index,
                (antipodal_index + 1) % n_vertices,
            ):
                difference = (
                    hull_coordinates[first_index] - hull_coordinates[second_index]
                )
                squared_distance = float(np.dot(difference, difference))
                maximum_squared_distance = max(
                    maximum_squared_distance,
                    squared_distance,
                )

    return math.sqrt(maximum_squared_distance)

def _point_to_segment_distances(
    points: NDArray[np.float64],
    segment_start: NDArray[np.float64],
    segment_end: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return distances from points to a closed line segment."""
    segment = segment_end - segment_start
    squared_length = float(np.dot(segment, segment))
    if squared_length == 0.0:
        raise ValueError("Convex-hull edges must have positive length")

    parameters = ((points - segment_start) @ segment) / squared_length
    parameters = np.clip(parameters, 0.0, 1.0)
    projections = segment_start + parameters[:, None] * segment
    return np.linalg.norm(points - projections, axis=1)

def _polygon_max_convexity_defect_depth(points: NDArray[np.float64]) -> float:
    """Return the deepest hull-chord indentation of an ordered polygon."""
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if len(polygon) < 3:
        return 0.0

    hull_indices = _convex_hull_indices(polygon)
    if len(hull_indices) < 3:
        return 0.0

    # For a simple polygon, sorting hull vertices by their occurrence on the
    # boundary makes each consecutive pair delimit exactly one hull pocket.
    boundary_positions = sorted(int(index) for index in hull_indices)
    maximum_depth = 0.0
    for position_index, start in enumerate(boundary_positions):
        end = boundary_positions[(position_index + 1) % len(boundary_positions)]
        if end > start:
            pocket_points = polygon[start + 1:end]
        else:
            pocket_points = np.concatenate(
                (polygon[start + 1:], polygon[:end]), axis=0
            )

        if len(pocket_points) == 0:
            continue

        distances = _point_to_segment_distances(
            pocket_points,
            polygon[start],
            polygon[end],
        )
        maximum_depth = max(maximum_depth, float(distances.max()))

    return maximum_depth

def _planar_signed_chain_vertices(
    signed_chain: SignedChain,
    point_cloud: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate a planar signed 1-chain and return its support vertices."""
    coordinates = np.asarray(point_cloud, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("point_cloud must have shape (n_points, 2)")
    if not signed_chain.signed_simplices:
        return np.empty((0, 2), dtype=float)
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    return signed_chain.vertex_coordinates(coordinates, signed=True)

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

def polygon_length(points: Iterable[Tuple[float, float]]) -> float:
    """
    Perimeter (sum of edge lengths) of the closed polygonal loop.

    Parameters
    ----------
    points : iterable of (float, float)
        Vertices of the polygon; a trailing duplicate of the first point
        is allowed.

    Returns
    -------
    float
        Total Euclidean length of all edges.
    """
    P = _to_xy(points)
    diffs = np.roll(P, -1, axis=0) - P
    return math.fsum(np.linalg.norm(diffs, axis=1))

def polygon_area(points: Iterable[Tuple[float, float]], signed: bool = False) -> float:
    """
    Shoelace area of the polygon; optionally return the signed value.

    Parameters
    ----------
    points : iterable of (float, float)
        Vertices of the polygon; a trailing duplicate of the first point
        is allowed.
    signed : bool, optional
        If True, return the signed area (CCW positive, CW negative).

    Returns
    -------
    float
        Polygon area (signed or absolute depending on ``signed``).
    """
    P = _to_xy(points)
    x, y = P[:, 0], P[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    a = 0.5 * math.fsum(x * ys - y * xs)
    return a if signed else abs(a)

def polygon_area_length_ratio(points: Iterable[Tuple[float, float]]) -> float:
    """
    Area divided by squared perimeter (isoperimetric ratio).

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.

    Returns
    -------
    float
        ``area / perimeter**2``.
    """
    return polygon_area(points=points)/(polygon_length(points=points)**2)

def polygon_area_length_squared_ratio(points: Iterable[Tuple[float, float]]) -> float:
    """
    Area divided by perimeter.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.

    Returns
    -------
    float
        ``area / perimeter``.
    """
    return polygon_area(points=points)/polygon_length(points=points)

def polygon_length_area_ratio(points: Iterable[Tuple[float, float]], tol = 1e-8) -> float:
    """
    Perimeter divided by area; returns 0 when area is below ``tol``.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.
    tol : float, optional
        If area < ``tol``, return 0 to avoid division by tiny values.

    Returns
    -------
    float
        ``perimeter / area`` or 0 when the area threshold is not met.
    """
    if polygon_area(points=points) < tol:
        return 0
    else:
        return polygon_length(points=points)/polygon_area(points=points)

def polygon_length_squared_area_ratio_normalized(points: Iterable[Tuple[float, float]], tol = 1e-8) -> float:
    """
    Isoperimetric deficit: (perimeter^2 / area) - 4π.

    Returns 0 when area is below ``tol``.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.
    tol : float, optional
        If area < ``tol``, return 0 to avoid division by tiny values.

    Returns
    -------
    float
        ``perimeter**2 / area - 4π`` or 0 when the area threshold is not met.
    """
    if polygon_area(points=points) < tol:
        return 0
    else:
        return polygon_length(points=points)**2/polygon_area(points=points) - 4*np.pi

def polygon_length_squared_area_ratio(points: Iterable[Tuple[float, float]]) -> float:
    """
    Perimeter squared divided by area (isoperimetric quotient).

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.

    Returns
    -------
    float
        ``perimeter**2 / area``.
    """
    return (polygon_length(points=points)**2)/polygon_area(points=points)

def total_curvature(points: Iterable[Tuple[float, float]]) -> float:
    """
    Total curvature of a simple closed polygonal loop:
        K_total = sum_i |kappa_i|
    where kappa_i is the signed exterior turning angle at vertex i.

    Definition at vertex i:
        a = P[i]   - P[i-1]   (incoming edge)
        b = P[i+1] - P[i]     (outgoing edge)
        kappa_i = atan2( cross(a,b), dot(a,b) ) ∈ (-π, π]

    Properties
    ----------
    - For a simple CCW convex polygon (e.g., regular n-gon), sum(kappa_i) ≈ +2π and
      total_curvature ≈ 2π.
    - For non-convex/star-shaped polygons (some kappa_i < 0), total_curvature > 2π.
    - Invariant to translation and rotation; scale-invariant (angles only).

    Parameters
    ----------
    points : sequence of (x,y) defining a simple loop (last connects to first).

    Returns
    -------
    K : float
        Total curvature (radians).
    """
    P = _to_xy(points)

    prevP = np.roll(P, 1, axis=0)
    nextP = np.roll(P, -1, axis=0)
    a = P - prevP
    b = nextP - P

    a_len = np.linalg.norm(a, axis=1)
    b_len = np.linalg.norm(b, axis=1)
    bad = (a_len < _EPS) | (b_len < _EPS)
    if np.any(bad):
        raise ValueError("Zero-length edge detected near vertices: "
                         f"{np.where(bad)[0].tolist()}")

    cross = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]     # z-component
    dot   = a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1]
    kappa = np.arctan2(cross, dot)                # signed exterior angle ∈ (-π, π]
    abs_kappa = np.abs(kappa)

    K = float(abs_kappa.sum())
    return K

def curvature_excess(points: Iterable[Tuple[float, float]], normalize: bool = True) -> float:
    """
    Excess curvature: total_curvature - 2π

    Returns 0 for a convex loop (total curvature ≈ 2π) and positive values
    for star-shaped or non-convex loops.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.
    normalize : bool, optional
        If True, return excess curvature normalized by 2π.    

    Returns
    -------
    float
        Excess curvature normalized by 2π.
    """
    K = total_curvature(points) - 2.0*math.pi
    if normalize:
        K /= (2.0*math.pi)
    return K 

# -------- Signed Chains --------------

def signed_chain_to_polyhedral_paths(signed_chain: SignedChain, point_cloud: NDArray) -> List[NDArray[np.int32]]:
    """
    Convert a 1-dimensional signed chain into closed vertex paths.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain whose simplices are oriented edges.
    point_cloud : ndarray, shape (n_points, 2)
        Planar coordinates indexed by the chain vertices.

    Returns
    -------
    list of ndarray
        Closed paths as vertex-index arrays. The closing vertex is not repeated.
    """
    if not signed_chain.signed_simplices:
        return []

    if signed_chain.dim() !=1:
        raise ValueError(f"Polyhedral path methods only works for Signed 1-chains. Dimension of chain: {signed_chain.dim()}")
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 2:
        raise ValueError("point_cloud must have shape (n_points, 2)")

    from collections import defaultdict

    def _start_end(simplex: Tuple[int, ...], orientation: int) -> Tuple[int, int]:
        """Return (start, end) vertex of an oriented 1-simplex."""
        if len(simplex) != 2:
            raise ValueError(
                "signed_chain_to_polyhedral_paths is only implemented for 1-chains (edges)."
            )
        a, b = simplex
        if orientation == 1:
            return a, b
        elif orientation == -1:
            return b, a
        else:
            raise ValueError("Orientation must be ±1.")

    def _angle_ccw(prev_vec: np.ndarray, next_vec: np.ndarray) -> float:
        """
        Signed angle from prev_vec to next_vec in [0, 2π),
        measured counterclockwise.
        """
        # Skip zero-length directions at caller level
        cross = prev_vec[0] * next_vec[1] - prev_vec[1] * next_vec[0]
        dot   = prev_vec[0] * next_vec[0] + prev_vec[1] * next_vec[1]
        angle = math.atan2(cross, dot)  # ∈ (-π, π]
        if np.all(prev_vec == -next_vec):
            angle = -math.pi
        return angle

    # Precompute adjacency: start vertex -> list of (signed_simplex, end_vertex)
    edges_by_start: Dict[int, List[Tuple[tuple, int]]] = defaultdict(list)
    for signed_simplex in signed_chain.signed_simplices:
        simplex, orientation = signed_simplex
        start, end = _start_end(simplex, orientation)
        edges_by_start[start].append((signed_simplex, end))

    visited: Set[tuple] = set()
    paths: List[NDArray[np.int32]] = []

    for signed_simplex in signed_chain.signed_simplices:
        if signed_simplex in visited:
            continue

        simplex, orientation = signed_simplex
        start, end = _start_end(simplex, orientation)

        # Start a new path with this edge
        path_vertices: List[int] = [int(start), int(end)]
        visited.add(signed_simplex)

        prev_vertex = start
        cur_vertex = end

        p_prev = np.asarray(point_cloud[prev_vertex], dtype=float)
        p_cur  = np.asarray(point_cloud[cur_vertex],  dtype=float)
        prev_vec = p_cur - p_prev

        while True:
            candidates = edges_by_start.get(cur_vertex, [])
            if not candidates:
                # No outgoing edges from this vertex
                raise ValueError("No outgoing edges, this should not happen")

            best_edge: Optional[tuple] = None
            best_next_vertex: Optional[int] = None
            best_angle: Optional[float] = None

            for edge_key, next_vertex in candidates:
                p_next = np.asarray(point_cloud[next_vertex], dtype=float)
                next_vec = p_next - p_cur

                # Ignore degenerate directions
                if np.allclose(next_vec, 0.0):
                    continue

                angle = _angle_ccw(prev_vec, next_vec)

                if best_angle is None or angle > best_angle:
                    best_angle = angle
                    best_edge = edge_key
                    best_next_vertex = int(next_vertex)

            if best_edge is None or best_next_vertex is None:
                # Only degenerate candidates
                raise ValueError("Only degenerate candidates, this should not happen")

            # Stop if we would traverse an already covered signed simplex
            if best_edge in visited:
                break

            # Advance along the chosen leftmost edge
            path_vertices.append(best_next_vertex)
            visited.add(best_edge)

            prev_vertex, cur_vertex = cur_vertex, best_next_vertex
            p_prev, p_cur = p_cur, np.asarray(point_cloud[cur_vertex], dtype=float)
            prev_vec = p_cur - p_prev

        paths.append(np.array(path_vertices[:-1], dtype=np.int32)) #last vertex is repeated, remove it

    for path in paths:
        if len(path)<2:
            print(paths)
            raise ValueError("Path too short in signed_chain_to_polyhedral_paths()")

    return paths

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
        Sum of edge lengths, ignoring orientation.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    return math.fsum(
        float(np.linalg.norm(point_cloud[simplex[1]] - point_cloud[simplex[0]]))
        for simplex, _ in signed_chain.signed_simplices
        if len(simplex) == 2
    )

def constant_one_functional(signed_chain = None, point_cloud = None) -> float:
    """
    Return the constant value 1, ignoring all inputs.

    Parameters
    ----------
    signed_chain : SignedChain, optional
        Ignored.
    point_cloud : ndarray, optional
        Ignored.

    Returns
    -------
    float
        Always 1.
    """
    return 1

def signed_chain_connected_components(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> int:
    """
    Count the closed paths represented by a signed 1-chain.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    int
        Number of closed paths.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    return len(signed_chain_to_polyhedral_paths(signed_chain=signed_chain, point_cloud=point_cloud))

def signed_chain_excess_connected_components(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> int:
    """
    Count closed paths beyond the first one.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    int
        Number of closed paths minus one.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    return len(signed_chain_to_polyhedral_paths(signed_chain=signed_chain, point_cloud=point_cloud)) - 1

def signed_chain_connected_components_only_signed_simplices(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> int:
    """
    Count closed paths formed only by doubled simplices.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    int
        Number of closed paths formed by doubled simplices.
    """
    
    purely_signed_chain = signed_chain.only_double_simplices()
    if len(purely_signed_chain.signed_simplices)==0:
        return 0
    
    return signed_chain_connected_components( signed_chain=purely_signed_chain, point_cloud=point_cloud)

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
        Average doubled-edge component length, divided by 2.
    """
    tendril_chain = signed_chain.only_double_simplices()
    if len(tendril_chain.signed_simplices)==0:
        return 0
    length = signed_chain_edge_length(signed_chain=tendril_chain, point_cloud=point_cloud)
    tendril_num = len(signed_chain_to_polyhedral_paths(signed_chain=tendril_chain, point_cloud=point_cloud))
    
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

def signed_chain_area(signed_chain: SignedChain, point_cloud:  NDArray[np.float64]) -> float:
    """
    Return the area enclosed by a signed 1-chain.

    Parameters
    ----------
    signed_chain : SignedChain
        Signed 1-chain.
    point_cloud : ndarray, shape (n_points, dim)
        Coordinates indexed by the chain vertices.

    Returns
    -------
    float
        Outer area minus inner path areas.
    """
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

    ``1 - convex_hull_perimeter / chain_edge_length``.

    It is zero for a convex polygonal cycle. Doubled signed edges contribute
    to the chain length and therefore increase the deficit. A zero-length
    chain is treated as convex and returns zero.

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
        1.0 - hull_perimeter / chain_length, "perimeter deficit"
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
        Sum of unnormalized excess curvature values.
    """
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
    """
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
        Complement of the circularity score.
    """
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
        Complement of the squared circularity score.
    """
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
        Squared complement of the circularity score.
    """
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
        Sum of absolute simplex volumes, ignoring orientation.
    """

    simplices = np.asarray([simplex for simplex, sign in signed_chain.signed_simplices])

    # Gather all matrices at once: shape (m, d, d)
    mats = point_cloud[simplices]

    # Batched determinant: shape (m,)
    dets = np.linalg.det(mats)

    return float(np.abs(dets).sum())
