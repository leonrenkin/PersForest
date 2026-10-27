"""GUDHI alpha preparation in radius units."""
import gudhi as gd
from numpy.typing import NDArray

from persforest.filtrations.simplicial import SimplicialFiltration, simplicial_events


def prepare_alpha_filtration(
    point_cloud: NDArray,
) -> tuple[gd.SimplexTree, SimplicialFiltration]:
    """Prepare the alpha simplex tree and its simplicial adapter.

    Parameters
    ----------
    point_cloud : ndarray
        Validated point coordinates supplied by the facade.

    Returns
    -------
    tuple
        The simplex tree and an adapter over its top simplices and their facets,
        in GUDHI's filtration order. GUDHI lists vertices in increasing order.
    """
    simplex_tree = gd.AlphaComplex(points=point_cloud).create_simplex_tree(output_squared_values=False)
    events = simplicial_events(simplex_tree.get_filtration(), point_cloud.shape[1])
    return simplex_tree, SimplicialFiltration(point_cloud, events)
