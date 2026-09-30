
def sign_of_determinant(vectors: Sequence[Sequence[float]] | NDArray) -> int:
    """
    Computes the sign of the determinant of d vectors in R^d.

    Parameters
    ----------
    vectors : array-like, shape (d, d)
        Collection of d vectors of length d.

    Returns
    -------
    int
        +1 if det > 0, -1 if det < 0.

    Raises
    ------
    ValueError
        If the matrix is invalid, degenerate, or numerically near-degenerate.

    Notes
    -----
    Divide by the largest absolute entry before computing the determinant.
    Reject normalized determinant magnitudes <= 1e-12. This conservative
    numerical cutoff is invariant under uniform scaling; it is not an exact
    degeneracy predicate. No higher-precision fallback is attempted.
    """
    A = np.array(vectors, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("Orientation requires a non-empty square matrix")
    scale = np.max(np.abs(A))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Orientation requires finite, nonzero edge vectors")
    determinant = np.linalg.det(A / scale)
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError("Degenerate or numerically near-degenerate edge matrix")
    return 1 if determinant > 0 else -1

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
        check in ``sign_of_determinant``. The error identifies its vertex ids.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        vectors = [point_cloud[i]-point_cloud[simplex[0]] for i in simplex[1:]]
    try:
        return sign_of_determinant(vectors=vectors)
    except ValueError as error:
        raise ValueError(
            f"Cannot determine orientation reliably for simplex {tuple(simplex)}: "
            "degenerate, numerically near-degenerate, or non-finite edge vectors."
        ) from error

def boundary_faces_from_tetrahedra(
    tetrahedra: Iterable[Sequence[int]],
    point_cloud: NDArray,
) -> List[Tuple[int, int, int]]:
    """
    Compute boundary triangular faces of a tetrahedral complex.

    Tetrahedra are oriented using ``point_cloud``. Their signed boundaries are
    summed, and faces with net zero coefficient are removed. This cancels
    interior faces while preserving boundary faces.

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
        tet_tuple = tuple(tet)
        orientation = simplex_orientation(simplex=tet_tuple, point_cloud=point_cloud)
        for face, face_orientation in signed_boundary(
            simplex=list(tet_tuple),
            orientation=orientation,
        ):
            face_coeffs[key(face)] += int(face_orientation)

    return [face for face, coeff in face_coeffs.items() if coeff != 0]