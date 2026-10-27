"""Measurements of cycle representatives on cubical grids."""

import math

from persforest.chains import SignedChain
from persforest.geometry.cubical import CubicalGeometry
from persforest.measurements.basic import _require_chain_interior


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
