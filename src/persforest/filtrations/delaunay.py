"""Delaunay-based filtrations of point clouds from GUDHI, in radius units."""
import gudhi as gd
from numpy.typing import NDArray

from persforest.filtrations.simplicial import SimplicialFiltration

POINT_FILTRATION_TYPES = ("alpha", "delaunay_cech")


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
    return simplex_tree, SimplicialFiltration.from_events(point_cloud, simplex_tree.get_filtration())


def prepare_delaunay_cech_filtration(
    point_cloud: NDArray,
) -> tuple[gd.SimplexTree, SimplicialFiltration]:
    """Prepare the Delaunay–Čech simplex tree and its simplicial adapter.

    The complex is the Delaunay triangulation, as for alpha; a simplex enters at
    the radius of its minimal enclosing ball.

    Parameters
    ----------
    point_cloud : ndarray
        Validated point coordinates supplied by the facade.

    Returns
    -------
    tuple
        The simplex tree and an adapter over its top simplices and their facets,
        in GUDHI's filtration order.
    """
    simplex_tree = gd.DelaunayCechComplex(points=point_cloud).create_simplex_tree(output_squared_values=False)
    return simplex_tree, SimplicialFiltration.from_events(point_cloud, simplex_tree.get_filtration())


def prepare_point_filtration(
    point_cloud: NDArray, filtration_type: str,
) -> tuple[gd.SimplexTree, SimplicialFiltration]:
    """Dispatch on ``filtration_type`` (``"alpha"`` or ``"delaunay_cech"``)."""
    if filtration_type == "alpha":
        return prepare_alpha_filtration(point_cloud)
    if filtration_type == "delaunay_cech":
        return prepare_delaunay_cech_filtration(point_cloud)
    raise ValueError(f"filtration_type must be one of {POINT_FILTRATION_TYPES}, not {filtration_type!r}")
