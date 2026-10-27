"""filtrations.simplicial: mechanically extracted reference implementation."""

from persforest.filtrations._validation import _validate_filtration
from collections.abc import Iterable, Sequence

def key(simplex):
    """
    Return a canonical, orientation-free key for a simplex.

    Parameters
    ----------
    simplex : iterable[int]
        Vertex ids of the simplex.

    Returns
    -------
    tuple
        Sorted tuple of vertex ids, suitable for hashing regardless of
        orientation.
    """
    return tuple(sorted(simplex))

def signed_boundary(simplex, orientation: int): 
    """
    Compute the oriented boundary of a simplex.

    Parameters
    ----------
    simplex : list[int]
        Vertices of the simplex whose boundary is computed.
    orientation : int
        Orientation (+1 or -1) of the simplex.

    Returns
    -------
    set[tuple]
        Set of oriented faces stored as (simplex_tuple, orientation).
    """
    return {(tuple(simplex[:i] + simplex[i+1:]), orientation* (-1)**i) for i in range(len(simplex))}

def prepare_simplicial_filtration(
    filtration: Iterable[tuple[Sequence[int], float]], n_points: int,
) -> list[tuple[tuple[int, ...], float]]:
    """Validate custom input without changing its values or caller tie order.

    Parameters
    ----------
    filtration : iterable of (sequence of int, float)
        Existing ordered simplicial input.
    n_points : int
        Number of available vertices.

    Returns
    -------
    list
        The existing validator's materialized simplex-value pairs.
    """
    return _validate_filtration(filtration, n_points)
