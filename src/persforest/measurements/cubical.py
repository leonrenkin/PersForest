"""Measurements of cycle representatives on cubical grids.

Shape measurements of 2D representatives (circularity, convexity, curvature)
are loop functionals: build loops with ``persforest.geometry.loops`` (e.g.
``taut_loops``) or wrap a functional with ``persforest.measurements.loops``
(e.g. ``on_taut_loops``).
"""

import math

from persforest.chains import SignedChain
from persforest.geometry.cubical import CubicalGeometry
from persforest.geometry.cubical import cubical_chain_to_vertex_chain
from persforest.measurements.basic import _require_chain_interior
from persforest.measurements.basic import signed_chain_connected_components
from persforest.measurements.basic import signed_chain_connected_components_only_signed_simplices
from persforest.measurements.basic import signed_chain_excess_connected_components
from persforest.measurements.shape import signed_chain_avg_tendril_length
from persforest.measurements.shape import signed_chain_num_of_branching_points
from persforest.measurements.shape import signed_chain_num_of_branching_points_only_signed_simplices
from persforest.measurements.shape import signed_chain_tendril_branching_ratio


def _require_cubical_geometry(geometry) -> None:
    if not isinstance(geometry, CubicalGeometry):
        raise TypeError(
            "Cubical measurements need the CubicalGeometry of a forest built with "
            "PersistenceForest.from_cubical_grid; use the simplicial measurements "
            "in persforest.measurements.basic for point clouds.")


def cubical_chain_volume(signed_chain: SignedChain, geometry: CubicalGeometry) -> float:
    """Return the summed measure of the chain's cells.

    This is the boundary length of a 2D representative and its boundary area in
    3D, measured along grid cells: a pixel loop has its staircase (L1) length,
    not the Euclidean length of a smooth curve it approximates. Opposite
    orientations contribute separately, as in ``signed_chain_volume``; pass
    ``chain.unsigned()`` to cancel them.

    Parameters
    ----------
    signed_chain : SignedChain
        Chain of cubical cells (doubled coordinates).
    geometry : CubicalGeometry
        Grid geometry of the forest (``forest.geometry``).

    Returns
    -------
    float
        Sum of cell measures; zero for the empty chain.
    """
    _require_cubical_geometry(geometry)
    return math.fsum(geometry.cell_measure(cell) for cell, _ in signed_chain.signed_simplices)


def cubical_chain_interior_volume(signed_chain: SignedChain, geometry: CubicalGeometry) -> float:
    """Return the enclosed area (2D) or volume (3D) of a representative.

    Orientation is ignored, as in ``signed_chain_interior_volume``; an available
    but empty interior has volume zero.

    Parameters
    ----------
    signed_chain : SignedChain
        Representative with interior data.
    geometry : CubicalGeometry
        Grid geometry of the forest (``forest.geometry``).

    Raises
    ------
    ValueError
        If the interior is unavailable.
    """
    _require_cubical_geometry(geometry)
    _require_chain_interior(signed_chain)
    return math.fsum(geometry.cell_measure(cell) for cell, _ in signed_chain.interior)


# The functionals below rewrite a 2D grid chain as a chain on grid vertices
# and apply their simplicial counterparts; tendrils (edges present in both
# orientations) occur with filtration_type="vertices".

def _vertex_chain(signed_chain: SignedChain, geometry: CubicalGeometry):
    _require_cubical_geometry(geometry)
    return cubical_chain_to_vertex_chain(signed_chain, geometry)


def cubical_chain_connected_components(signed_chain: SignedChain, geometry: CubicalGeometry) -> int:
    """Count connected components of the edge support of a 2D grid chain.

    Edges sharing a grid vertex are connected, as in
    ``signed_chain_connected_components``; zero for the empty chain.
    """
    return signed_chain_connected_components(*_vertex_chain(signed_chain, geometry))


def cubical_chain_excess_connected_components(signed_chain: SignedChain, geometry: CubicalGeometry) -> int:
    """Count support components beyond the first one, as in ``signed_chain_excess_connected_components``."""
    return signed_chain_excess_connected_components(*_vertex_chain(signed_chain, geometry))


def cubical_chain_connected_components_only_signed_simplices(
        signed_chain: SignedChain, geometry: CubicalGeometry) -> int:
    """Count components formed by doubled edges (tendrils) of a 2D grid chain."""
    return signed_chain_connected_components_only_signed_simplices(*_vertex_chain(signed_chain, geometry))


def cubical_chain_num_of_branching_points(signed_chain: SignedChain, geometry: CubicalGeometry) -> int:
    """Count grid vertices incident to more than two signed edges.

    Besides tendril branchings, this counts vertices where two pieces of the
    boundary touch diagonally, as pixels meeting at a corner do.
    """
    return signed_chain_num_of_branching_points(*_vertex_chain(signed_chain, geometry))


def cubical_chain_num_of_branching_points_only_signed_simplices(
        signed_chain: SignedChain, geometry: CubicalGeometry) -> int:
    """Count branching points of the doubled edges (tendrils) of a 2D grid chain."""
    return signed_chain_num_of_branching_points_only_signed_simplices(*_vertex_chain(signed_chain, geometry))


def cubical_chain_avg_tendril_length(signed_chain: SignedChain, geometry: CubicalGeometry) -> float:
    """Return the average physical length of the tendrils of a 2D grid chain.

    See ``signed_chain_avg_tendril_length``; tendrils run along grid edges, so
    their grid length is their Euclidean length.
    """
    return signed_chain_avg_tendril_length(*_vertex_chain(signed_chain, geometry))


def cubical_chain_tendril_branching_ratio(signed_chain: SignedChain, geometry: CubicalGeometry) -> float:
    """Return tendril branching points per tendril component of a 2D grid chain."""
    return signed_chain_tendril_branching_ratio(*_vertex_chain(signed_chain, geometry))
