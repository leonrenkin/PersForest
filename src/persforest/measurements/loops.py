"""Shape measurements of closed planar loops.

Every functional takes ``PlanarLoops`` from ``persforest.geometry.loops``. To
measure representatives, turn them into loops first::

    loops = smoothed_loops(chain, forest.geometry, sigma=2.0)
    loops_circularity(loops)

or wrap the functional for profiles and landscapes, which call measurements
as ``(chain, geometry)``::

    on_smoothed_loops(loops_circularity, sigma=2.0)

The simplicial ``signed_chain_*`` shape functionals are these functionals
applied to the raw polygon loops.
"""

from types import MappingProxyType
from typing import Any
from typing import Callable
import math
import warnings

import numpy as np

from persforest.chains import SignedChain
from persforest.geometry.loops import PlanarLoops
from persforest.geometry.loops import _validated_passes
from persforest.geometry.loops import _validated_sigma
from persforest.geometry.loops import averaged_loops
from persforest.geometry.loops import midpoint_loops
from persforest.geometry.loops import raw_loops
from persforest.geometry.loops import smoothed_loops
from persforest.geometry.loops import taut_loops
from persforest.geometry.planar import _convex_hull_indices
from persforest.geometry.planar import _convex_polygon_diameter
from persforest.geometry.planar import _polygon_max_convexity_defect_depth
from persforest.geometry.planar import curvature_excess
from persforest.geometry.planar import polygon_area
from persforest.geometry.planar import polygon_length

_UNIT_INTERVAL_TOL = 64 * np.finfo(np.float64).eps

def _require_unit_interval(value: float, measurement_name: str) -> float:
    """Validate a normalized geometric measurement."""
    if not math.isfinite(value):
        raise ValueError(f"{measurement_name} must be finite, got {value}")
    if -_UNIT_INTERVAL_TOL <= value < 0.0:
        return 0.0
    if 1.0 < value <= 1.0 + _UNIT_INTERVAL_TOL:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{measurement_name} must lie in [0, 1], got {value}. "
            "This indicates inconsistent cycle and convex-hull geometry."
        )
    return float(value)

def loops_length(loops: PlanarLoops) -> float:
    """
    Return the total length of all loops.

    Parameters
    ----------
    loops : PlanarLoops

    Returns
    -------
    float
        Sum of segment lengths; zero for no loops.
    """
    if not loops.loops:
        return 0.0
    return math.fsum(
        np.concatenate([np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1) for loop in loops])
    )

def loops_area(loops: PlanarLoops) -> float:
    """
    Return the area enclosed by the loops.

    Representatives bound one connected component of the complement, so their
    loops are one counterclockwise outer boundary and clockwise hole
    boundaries. The absolute sum of signed loop areas is then the outer area
    minus the hole areas. It stays exact when a boundary is split into several
    loops at a vertex it touches twice. A loop collapsed to one vertex, such as
    the taut loop of a single pixel, encloses nothing.

    Parameters
    ----------
    loops : PlanarLoops

    Returns
    -------
    float
        Enclosed area; zero for no loops.
    """
    return abs(math.fsum(polygon_area(loop, signed=True) for loop in loops if len(loop) > 1))

def loops_circularity(loops: PlanarLoops) -> float:
    """
    Return the circularity score ``4π * area / length**2``.

    Parameters
    ----------
    loops : PlanarLoops

    Returns
    -------
    float
        Circularity score; 1 is circular, lower values are less circular.
        No loops, or loops of zero length, have score zero by convention.
    """
    length = loops_length(loops)
    if length == 0.0:
        return 0.0
    area = loops_area(loops)
    return (4.0*math.pi*area)/length**2

def loops_circularity_complement(loops: PlanarLoops) -> float:
    """Return ``1 - loops_circularity(loops)``; zero for no loops."""
    if not loops.loops:
        return 0.0
    return 1 - loops_circularity(loops)

def loops_circularity_squared_complement(loops: PlanarLoops) -> float:
    """Return ``1 - loops_circularity(loops)**2``; zero for no loops."""
    if not loops.loops:
        return 0.0
    return 1 - loops_circularity(loops)**2

def loops_circularity_complement_squared(loops: PlanarLoops) -> float:
    """Return ``(1 - loops_circularity(loops))**2``; zero for no loops."""
    if not loops.loops:
        return 0.0
    return loops_circularity_complement(loops)**2

def loops_non_circularity(loops: PlanarLoops) -> float:
    """
    Return the non-circularity score ``length**2 / (4π * area) - 1``.

    Parameters
    ----------
    loops : PlanarLoops

    Returns
    -------
    float
        Non-circularity score; 0 is circular, higher values are less circular.
        Undefined (division by zero) for zero enclosed area.
    """
    length = loops_length(loops)
    area = loops_area(loops)
    return length**2/(4.0*math.pi*area) - 1

def loops_convex_hull_area_deficit(loops: PlanarLoops) -> float:
    """Return the relative area missing from the loops' convex hull.

    The measurement is ``1 - enclosed_area / convex_hull_area``.

    It is zero for a convex enclosed region and approaches one as the enclosed
    area becomes small relative to its convex hull. Loops with zero convex
    hull area are treated as convex and return zero. One global hull is
    computed from all loop vertices, including hole boundaries and tendrils.

    Parameters
    ----------
    loops : PlanarLoops

    Returns
    -------
    float
        Area deficit in ``[0, 1]``.
    """
    vertices = loops.points
    if len(vertices) == 0:
        return 0.0

    hull = vertices[_convex_hull_indices(vertices)]
    hull_area = polygon_area(hull)
    enclosed_area = loops_area(loops)
    if hull_area == 0.0:
        if enclosed_area != 0.0:
            raise ValueError(
                "Convex hull has zero area but the loops enclose "
                f"area {enclosed_area}"
            )
        return 0.0

    return _require_unit_interval(
        1.0 - enclosed_area / hull_area, "area deficit"
    )

def loops_convex_hull_perimeter_deficit(loops: PlanarLoops) -> float:
    """Return the relative perimeter excess over the loops' convex hull.

    The measurement is ``max(1 - convex_hull_perimeter / loops_length, 0)``.

    It is zero for a convex loop. Tendrils and hole boundaries add to the loop
    length and therefore increase the deficit. Loops of zero length are
    treated as convex and return zero. The clipping implements the
    nonnegative deficit definition when the global hull perimeter exceeds the
    loop length.

    Parameters
    ----------
    loops : PlanarLoops

    Returns
    -------
    float
        Perimeter deficit in ``[0, 1]``.
    """
    vertices = loops.points
    if len(vertices) == 0:
        return 0.0

    hull = vertices[_convex_hull_indices(vertices)]
    hull_perimeter = polygon_length(hull)
    length = loops_length(loops)
    if length == 0.0:
        if hull_perimeter != 0.0:
            raise ValueError(
                "Loops have zero length but their convex hull has perimeter "
                f"{hull_perimeter}"
            )
        return 0.0

    return _require_unit_interval(
        max(1.0 - hull_perimeter / length, 0.0), "perimeter deficit"
    )

def loops_max_convexity_defect_depth(loops: PlanarLoops, normalize: bool = True) -> float:
    """Return the maximum hull-chord indentation depth of the loops.

    Convexity defects are computed separately on each loop, and the largest
    depth is returned. With the default normalization, depth is divided by the
    diameter of the convex hull of all loops. Thus convex or degenerate loops
    return zero and the normalized result lies in ``[0, 1]``.

    Parameters
    ----------
    loops : PlanarLoops
    normalize : bool, optional
        If True, divide the depth by the global convex-hull diameter. If
        False, return depth in physical units.

    Returns
    -------
    float
        Maximum normalized or absolute convexity-defect depth.
    """
    vertices = loops.points
    if len(vertices) == 0:
        return 0.0

    maximum_depth = max(_polygon_max_convexity_defect_depth(loop) for loop in loops)

    if not normalize:
        return maximum_depth

    hull = vertices[_convex_hull_indices(vertices)]
    hull_diameter = _convex_polygon_diameter(hull)
    if hull_diameter == 0.0:
        if maximum_depth != 0.0:
            raise ValueError(
                "Convex hull has zero diameter but the loops have "
                f"convexity-defect depth {maximum_depth}"
            )
        return 0.0
    return _require_unit_interval(
        maximum_depth / hull_diameter, "normalized convexity-defect depth"
    )

def loops_excess_curvature(loops: PlanarLoops) -> float:
    """
    Return total excess curvature ``sum(total_curvature - 2π)`` over the loops.

    Staircase and midpoint loops turn at every grid corner, so their excess
    curvature grows with the size of the cycle even for a disk; a warning
    recommends a reconstruction. Averaged loops keep part of these turns.
    A loop collapsed to one vertex does not turn and contributes zero.

    Parameters
    ----------
    loops : PlanarLoops

    Returns
    -------
    float
        Sum of unnormalized excess curvature values; zero for no loops.
    """
    if loops.loops and loops.kind in ("staircase", "midpoint"):
        warnings.warn(
            f"Excess curvature of {loops.kind} loops counts the turns of the grid; "
            "use taut or smoothed loops to measure the curvature of the cycle's shape.",
            UserWarning, stacklevel=2)
    total = 0.0
    for loop in loops:
        if len(loop) > 1:
            total += curvature_excess(loop, normalize=False)
    return total

def loops_excess_curvature_normalized(loops: PlanarLoops) -> float:
    """Return ``loops_excess_curvature(loops) / 2π``."""
    return loops_excess_curvature(loops) / (2.0*math.pi)


def on_raw_loops(functional: Callable[..., float], **kwargs: Any) -> Callable[[SignedChain, Any], float]:
    """Return the measurement ``functional(raw_loops(chain, geometry), **kwargs)``.

    The ``on_*_loops`` functions turn a loop functional into a
    ``(chain, geometry)`` measurement for profiles and landscapes, e.g.
    ``forest.compute_measurement_profiles(on_raw_loops(loops_area), "area")``.
    The returned function carries ``measurement_name``, such as
    ``"loops_area[raw]"``, and ``measurement_params``, which profiles record
    unless given other ones.

    Parameters
    ----------
    functional : callable
        ``(PlanarLoops, **kwargs) -> float``, e.g. ``loops_circularity``.
    **kwargs
        Keyword arguments of ``functional``.
    """
    return _on_loops(functional, raw_loops, "raw", {}, kwargs)

def on_midpoint_loops(functional: Callable[..., float], **kwargs: Any) -> Callable[[SignedChain, Any], float]:
    """Return the measurement ``functional(midpoint_loops(chain, geometry), **kwargs)``.

    Cubical forests only; see ``on_raw_loops``.
    """
    return _on_loops(functional, midpoint_loops, "midpoint", {}, kwargs)

def on_smoothed_loops(functional: Callable[..., float], sigma: float, **kwargs: Any) -> Callable[[SignedChain, Any], float]:
    """Return the measurement ``functional(smoothed_loops(chain, geometry, sigma), **kwargs)``.

    ``sigma`` is the Gaussian standard deviation in physical length along the
    loop; see ``on_raw_loops``.
    """
    return _on_loops(functional, smoothed_loops, "smoothed", {"sigma": _validated_sigma(sigma)}, kwargs)

def on_averaged_loops(functional: Callable[..., float], passes: int = 4, **kwargs: Any) -> Callable[[SignedChain, Any], float]:
    """Return the measurement ``functional(averaged_loops(chain, geometry, passes), **kwargs)``.

    See ``on_raw_loops``.
    """
    return _on_loops(functional, averaged_loops, "averaged", {"passes": _validated_passes(passes)}, kwargs)

def on_taut_loops(functional: Callable[..., float], **kwargs: Any) -> Callable[[SignedChain, Any], float]:
    """Return the measurement ``functional(taut_loops(chain, geometry), **kwargs)``.

    Cubical forests only; see ``on_raw_loops``.
    """
    return _on_loops(functional, taut_loops, "taut", {}, kwargs)

def _on_loops(functional, build, curve: str, curve_params: dict, kwargs: dict) -> Callable[[SignedChain, Any], float]:
    if not callable(functional):
        raise TypeError("functional must be callable")

    def measurement(signed_chain: SignedChain, geometry) -> float:
        return functional(build(signed_chain, geometry, **curve_params), **kwargs)

    params = {**curve_params, **kwargs}
    details = "".join(f", {key}={value!r}" for key, value in params.items())
    name = f"{getattr(functional, '__name__', repr(functional))}[{curve}{details}]"
    measurement.__name__ = measurement.__qualname__ = name
    measurement.measurement_name = name
    measurement.measurement_params = MappingProxyType({"curve": curve, **params})
    return measurement
