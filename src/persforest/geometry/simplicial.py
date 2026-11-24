"""geometry.simplicial: mechanically extracted reference implementation."""

from collections import defaultdict
from numpy.typing import NDArray
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Set
from typing import Tuple
import numpy as np

from persforest import _engine

def simplex_orientations(simplices: Sequence[Sequence[int]] | NDArray, point_cloud: NDArray) -> NDArray[np.int8]:
    """
    Compute the orientations of simplices with respect to the ambient point cloud.

    Parameters
    ----------
    simplices : array-like of int, shape (n_simplices, dim + 1)
        Vertex ids of each simplex, in any order.
    point_cloud : array-like, shape (n_points, dim)
        Coordinates of all vertices.

    Returns
    -------
    ndarray of int8, shape (n_simplices,)
        +1 or -1: the sign of the determinant of the edge vectors
        ``point_cloud[v_k] - point_cloud[v_0]``, k = 1 .. dim.

    Raises
    ------
    ValueError
        If the shapes do not match or a vertex id is out of range, and for the
        first simplex that is degenerate or fails the numerical check below.
        The error identifies its vertex ids.

    Notes
    -----
    The edge vectors are divided by their largest absolute entry before the
    determinant is taken, and scaled determinants of magnitude <= 1e-12 are
    rejected. This conservative numerical cutoff is invariant under uniform
    scaling; it is not an exact degeneracy predicate. No higher-precision
    fallback is attempted. The compiled engine applies the rule to all
    simplices in one pass.
    """
    points = np.ascontiguousarray(point_cloud, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError("point_cloud must have shape (n_points, dim) with dim >= 1")
    vertices = np.ascontiguousarray(simplices, dtype=np.int64)
    if vertices.ndim == 1 and vertices.size == 0:
        vertices = vertices.reshape(0, points.shape[1] + 1)
    if vertices.ndim != 2 or vertices.shape[1] != points.shape[1] + 1:
        raise ValueError(f"simplices in R^{points.shape[1]} need {points.shape[1] + 1} vertices each")
    return _engine.simplex_orientations(points, vertices)

def simplex_orientation(simplex: Sequence[int], point_cloud: NDArray) -> int:
    """
    Compute the orientation of a simplex with respect to the ambient point cloud.

    Parameters
    ----------
    simplex : sequence[int]
        Vertex ids of the simplex.
    point_cloud : ndarray
        Coordinates of all vertices.

    Returns
    -------
    int
        +1 for positive orientation, -1 for negative orientation.

    Raises
    ------
    ValueError
        If the simplex is degenerate or its orientation fails the numerical
        check of ``simplex_orientations``. The error identifies its vertex ids.
    """
    return int(simplex_orientations([simplex], point_cloud)[0])

def boundary_faces_from_tetrahedra(
    tetrahedra: Iterable[Sequence[int]],
    point_cloud: NDArray,
) -> List[Tuple[int, int, int]]:
    """
    Compute boundary triangular faces of a tetrahedral complex.

    Tetrahedra are put in increasing vertex order and oriented using
    ``point_cloud``. Their signed boundaries are summed, and faces with net zero
    coefficient are removed. This cancels interior faces while preserving
    boundary faces.

    Parameters
    ----------
    tetrahedra : iterable[sequence[int]]
        Tetrahedra represented by vertex indices.
    point_cloud : ndarray
        Ambient coordinates for simplex vertices.

    Returns
    -------
    list[tuple[int, int, int]]
        Boundary triangular faces with canonical vertex order.

    Raises
    ------
    ValueError
        If a tetrahedron is degenerate or numerically near-degenerate.
    """
    face_coeffs: Dict[Tuple[int, int, int], int] = defaultdict(int)

    for tet in tetrahedra:
        tet_tuple = tuple(sorted(tet))
        orientation = simplex_orientation(simplex=tet_tuple, point_cloud=point_cloud)
        for i in range(len(tet_tuple)):
            face_coeffs[tet_tuple[:i] + tet_tuple[i + 1:]] += orientation * (-1) ** i

    return [face for face, coeff in face_coeffs.items() if coeff != 0]

def chain_segments(self, point_cloud: NDArray):
    """
        Convert oriented edges in the chain to geometric segments.

        Parameters
        ----------
        point_cloud : ndarray
            Ambient coordinates for simplex vertices.

        Returns
        -------
        list[np.ndarray]
            List of 2xD arrays representing oriented edge segments.
        """
    segments = []
    for signed_simplex in self.signed_simplices:
        if signed_simplex[1]==1:
            segments.append(np.array(point_cloud[list(signed_simplex[0])]))
        else:
            segments.append(np.array(point_cloud[list(reversed(signed_simplex[0]))]))
    return segments

def chain_vertex_coordinates(self, point_cloud: NDArray, signed = True) -> NDArray[np.float64]:
    """
        Extract the coordinates of the vertices in this SignedChain.

        Parameters
        ----------
        point_cloud : ndarray, shape (n_points, dim)
            Ambient point cloud.
        signed : bool
            If True, use the stored signed simplices. If False, first cancel
            opposite-oriented copies via ``unsigned()``.

        Returns
        -------
        ndarray, shape (n_vertices, dim)
            Coordinates of the vertices in the chain.
        """
    vertex_indices: Set[int] = set()
    if signed: 
        signed_simplices = self.signed_simplices
    else:
        signed_simplices = self.unsigned().signed_simplices

    for signed_simplex in signed_simplices:
        simplex, _ = signed_simplex
        vertex_indices.update(simplex)

    coords = np.array([point_cloud[i] for i in sorted(vertex_indices)], dtype=float)
    return coords
