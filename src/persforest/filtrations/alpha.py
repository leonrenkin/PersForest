"""GUDHI alpha preparation in radius units."""
import gudhi as gd
from collections.abc import Iterator
from numpy.typing import NDArray


def prepare_alpha_filtration(
    point_cloud: NDArray,
) -> tuple[gd.SimplexTree, Iterator[tuple[list[int], float]]]:
    """Prepare the existing alpha tree and its ordered event iterator.

    Parameters
    ----------
    point_cloud : ndarray
        Validated point coordinates supplied by the facade.

    Returns
    -------
    tuple
        The original simplex tree and its unmaterialized filtration iterator.
    """
    simplex_tree = gd.AlphaComplex(points=point_cloud).create_simplex_tree(output_squared_values=False)
    filtration = simplex_tree.get_filtration()
    return simplex_tree, filtration
