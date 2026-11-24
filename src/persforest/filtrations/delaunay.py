"""Delaunay-based filtrations of point clouds, in radius units.

In R^2 and R^3 the compiled builders (``cpp/alpha.hpp``, on geogram's Delaunay
triangulation) compute alpha and Delaunay–Čech filtrations; other dimensions
use GUDHI.
"""
import gudhi as gd
import numpy as np
from numpy.typing import NDArray

from persforest import _engine
from persforest.filtrations.simplicial import SimplicialFiltration

POINT_FILTRATION_TYPES = ("alpha", "delaunay_cech")
_GUDHI_COMPLEXES = {"alpha": gd.AlphaComplex, "delaunay_cech": gd.DelaunayCechComplex}


def _compiled_filtration(build, point_cloud: NDArray) -> SimplicialFiltration:
    result = build(np.ascontiguousarray(point_cloud, dtype=np.float64))
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


def alpha_filtration(point_cloud: NDArray) -> SimplicialFiltration:
    """Alpha filtration of a point cloud in R^2 or R^3 from the compiled builder.

    The complex is geogram's Delaunay triangulation, restricted to top simplices
    and their facets; a top simplex enters at its circumradius, a facet at the
    radius of its smallest circumsphere or, if attached, with its earliest
    coface. See ``cpp/alpha.hpp`` for the exact definition.

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
    return _compiled_filtration(_engine.alpha_filtration, point_cloud)


def delaunay_cech_filtration(point_cloud: NDArray) -> SimplicialFiltration:
    """Delaunay–Čech filtration of a point cloud in R^2 or R^3 from the compiled builder.

    The complex is the same as for ``alpha_filtration``; every top simplex and
    facet enters at the radius of its minimal enclosing ball.

    Parameters
    ----------
    point_cloud : ndarray, shape (n_points, 2) or (n_points, 3)
        Validated point coordinates.

    Returns
    -------
    SimplicialFiltration
        As for ``alpha_filtration``.

    Raises
    ------
    ValueError
        As for ``alpha_filtration``.
    """
    return _compiled_filtration(_engine.delaunay_cech_filtration, point_cloud)


_COMPILED_FILTRATIONS = {"alpha": alpha_filtration, "delaunay_cech": delaunay_cech_filtration}


def _gudhi_filtration(complex_type, point_cloud: NDArray) -> SimplicialFiltration:
    """Adapter over GUDHI's filtration of a Delaunay-based complex, in GUDHI's order.

    GUDHI lists vertices in increasing order. Its filtration iterator does not
    keep the simplex tree alive, so the tree stays referenced while it is read.
    """
    simplex_tree = complex_type(points=point_cloud).create_simplex_tree(output_squared_values=False)
    return SimplicialFiltration.from_events(point_cloud, simplex_tree.get_filtration())


def prepare_point_filtration(point_cloud: NDArray, filtration_type: str) -> SimplicialFiltration:
    """Prepare the simplicial adapter of an ``"alpha"`` or ``"delaunay_cech"`` filtration.

    Point clouds in R^2 and R^3 go through the compiled builders, other
    dimensions through GUDHI's ``AlphaComplex`` and ``DelaunayCechComplex``.

    Parameters
    ----------
    point_cloud : ndarray
        Validated point coordinates supplied by the facade.
    filtration_type : {"alpha", "delaunay_cech"}

    Returns
    -------
    SimplicialFiltration
        An adapter over the top simplices and their facets.
    """
    if filtration_type not in POINT_FILTRATION_TYPES:
        raise ValueError(f"filtration_type must be one of {POINT_FILTRATION_TYPES}, not {filtration_type!r}")
    if point_cloud.shape[1] in (2, 3):
        return _COMPILED_FILTRATIONS[filtration_type](point_cloud)
    return _gudhi_filtration(_GUDHI_COMPLEXES[filtration_type], point_cloud)
