"""geometry.planar: mechanically extracted reference implementation."""

from numpy.typing import NDArray
from persforest.chains import SignedChain
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import math
import numpy as np

_EPS = 1e-12

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
