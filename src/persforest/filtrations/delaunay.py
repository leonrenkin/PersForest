"""Delaunay-based filtrations of point clouds, in radius units: compiled alpha builder and GUDHI."""
import gudhi as gd
import numpy as np
from numpy.typing import NDArray

from persforest import _engine
from persforest.filtrations.simplicial import SimplicialFiltration

POINT_FILTRATION_TYPES = ("alpha", "delaunay_cech")


def alpha_filtration(point_cloud: NDArray) -> SimplicialFiltration:
    """Alpha filtration of a point cloud in R^2 or R^3 from the compiled builder.

    The complex is geogram's Delaunay triangulation, restricted to top simplices
    and their facets; see ``cpp/alpha.hpp`` for the exact definition.

    Parameters
    ----------
    point_cloud : ndarray, shape (n_points, 2) or (n_points, 3)
        Validated point coordinates.

    Returns
    -------
    SimplicialFiltration
        Cells numbered in event order, events ordered by value (radius units),
        then facets before top simplices, then by vertex tuple. Of identical
        points only the first is a vertex. Without dim + 1 affinely independent
        points the filtration is empty. Top orientations are exact.

    Raises
    ------
    ValueError
        For another dimension, and for coordinates the builder cannot handle
        exactly: a nonzero coordinate below 2^-100 times the largest absolute
        coordinate, or radii beyond the double range.
    """
    result = _engine.alpha_filtration(np.ascontiguousarray(point_cloud, dtype=np.float64))
    return SimplicialFiltration(
        point_cloud=point_cloud,
        n_top=int(result["n_top"]),
        n_cells=len(result["event_cells"]),
        top_vertices=result["top_vertices"],
        facet_vertices=result["facet_vertices"],
        event_cells=result["event_cells"],
        event_values=result["event_values"],
        top_facets=result["top_facets"],
        top_orientation=result["top_orientation"],
    )


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
