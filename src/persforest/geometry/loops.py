"""Closed planar loops built from 1-cycle representatives.

Each function turns a representative into ``PlanarLoops``, the oriented closed
polylines measured by ``persforest.measurements.loops``:

``raw_loops``
    The cycle as given: the polygon of a simplicial cycle, the staircase of a
    cubical one, doubled edges included.
``midpoint_loops`` (cubical only)
    Joins the midpoints of consecutive grid edges, cutting every staircase
    corner. Each loop loses ``spacing[0] * spacing[1] / 2`` of area per full
    turn, and length remains a few percent above the smooth curve.
``smoothed_loops``
    Gaussian smoothing with arc-length standard deviation ``sigma`` of the
    midpoint loops (cubical) or of the polygon (simplicial). Features smaller
    than ``sigma`` are removed: corners round off, and a loop much shorter
    than ``sigma`` becomes an ellipse, a circle for a single pixel.
``averaged_loops``
    ``passes`` rounds of replacing every vertex of the midpoint loops
    (cubical) or of the polygon (simplicial) by ``(previous + 2 * vertex +
    next) / 4``. Each pass moves the curve only locally, so it stays close to
    the cycle; the reach is counted in vertices, not in length.
``taut_loops`` (cubical only)
    The shortest closed curve that crosses every staircase edge between the
    centers of the two pixels it separates: the minimum-length polygon of the
    region, at most half a pixel from the staircase along each axis. Its
    corners are pixel centers, and a convex region gives a convex polygon.
    Regions one pixel wide collapse: a single pixel to its center, a bar to a
    segment.

Smoothed and averaged loops are scaled about their centroid to the signed area
of their raw loop, so they change the shape but keep the enclosed area. All
reconstructions describe the region bounded by the cycle, so they are built
from ``chain.unsigned()``; tendrils (opposite copies of an edge) are measured
on the chain itself, for example by the tendril functionals.
"""

from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from typing import Any
from typing import Mapping
import math

import numpy as np
from numpy.typing import NDArray

from persforest.chains import SignedChain
from persforest.geometry.cubical import CubicalGeometry
from persforest.geometry.cubical import cubical_chain_to_vertex_chain
from persforest.geometry.planar import polygon_area
from persforest.geometry.planar import signed_chain_to_polyhedral_paths

LOOP_KINDS = ("polygon", "staircase", "midpoint", "smoothed", "averaged", "taut")

# Smoothing samples each loop uniformly in arc length, _SAMPLES_PER_SIGMA
# times per sigma; a Gaussian of that width leaves no detail finer than the
# sampling.
_SAMPLES_PER_SIGMA = 8
_MIN_SAMPLES = 64
_MAX_SAMPLES = 2**22


@dataclass(frozen=True)
class PlanarLoops:
    """Oriented closed polylines in physical coordinates.

    Attributes
    ----------
    loops : tuple of ndarray, shape (n_i, 2)
        Read-only vertex arrays; each loop closes from its last vertex back to
        its first, which is not repeated.
    kind : {"polygon", "staircase", "midpoint", "smoothed", "averaged", "taut"}
        How the loops were built: the raw simplicial or cubical cycle, or one
        of the reconstructions of this module.
    params : mapping
        Construction parameters, e.g. ``{"sigma": 2.0}`` for smoothed loops.
    """

    loops: tuple[NDArray[np.float64], ...]
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in LOOP_KINDS:
            raise ValueError(f"kind must be one of {LOOP_KINDS}, got {self.kind!r}")
        loops = []
        for loop in self.loops:
            array = np.array(loop, dtype=float)
            if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
                raise ValueError("Each loop must be a non-empty array of shape (n, 2)")
            if not np.isfinite(array).all():
                raise ValueError("Loop coordinates must be finite")
            array.setflags(write=False)
            loops.append(array)
        object.__setattr__(self, "loops", tuple(loops))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __len__(self) -> int:
        return len(self.loops)

    def __iter__(self):
        return iter(self.loops)

    @property
    def points(self) -> NDArray[np.float64]:
        """All loop vertices stacked, shape ``(n, 2)``."""
        if not self.loops:
            return np.empty((0, 2), dtype=float)
        return np.concatenate(self.loops, axis=0)


def raw_loops(signed_chain: SignedChain, geometry) -> PlanarLoops:
    """Return the closed paths of a planar 1-cycle as given.

    Every oriented edge, including both copies of a doubled edge, is traversed
    once. Cubical paths keep every grid vertex, also where they run straight
    on, and are split at vertices shared by several boundary pieces with the
    turning rule of the simplicial paths.

    Parameters
    ----------
    signed_chain : SignedChain
        Planar 1-cycle representative.
    geometry : ndarray or CubicalGeometry
        ``forest.geometry``: the point array of a 2D simplicial forest or the
        geometry of a 2D cubical grid.

    Returns
    -------
    PlanarLoops
        Polygon (simplicial) or staircase (cubical) loops in physical
        coordinates; empty for the zero chain.
    """
    if isinstance(geometry, CubicalGeometry):
        return _staircase_loops(signed_chain, geometry)
    return _polygon_loops(signed_chain, geometry)


def midpoint_loops(signed_chain: SignedChain, geometry: CubicalGeometry) -> PlanarLoops:
    """Join the midpoints of consecutive staircase edges of a cubical 1-cycle.

    Parameters
    ----------
    signed_chain : SignedChain
        Cycle of edges of a 2D grid; built from ``signed_chain.unsigned()``.
    geometry : CubicalGeometry
        Grid geometry of the forest (``forest.geometry``).
    """
    _require_planar_grid(geometry, "midpoint_loops")
    return _midpoints(_staircase_loops(signed_chain.unsigned(), geometry))


def smoothed_loops(signed_chain: SignedChain, geometry, sigma: float) -> PlanarLoops:
    """Smooth the cycle with a Gaussian of arc-length standard deviation ``sigma``.

    Each midpoint loop (cubical) or polygon (simplicial) is resampled
    uniformly in arc length from its lexicographically smallest vertex,
    filtered as a periodic signal, and scaled about its sample mean to the
    signed area of its raw loop. Harmonics are damped relative to the first
    one; the rescaling makes this equivalent to plain Gaussian filtering while
    avoiding underflow for loops much shorter than ``sigma``.

    Parameters
    ----------
    signed_chain : SignedChain
        Planar 1-cycle; built from ``signed_chain.unsigned()``.
    geometry : ndarray or CubicalGeometry
        ``forest.geometry`` of a 2D simplicial or cubical forest.
    sigma : float
        Positive standard deviation in physical length along the loop.

    Raises
    ------
    ValueError
        If ``sigma`` is not positive and finite, too small to sample a loop,
        or smoothing reverses a loop.
    """
    sigma = _validated_sigma(sigma)
    raw, curves = _region_curves(signed_chain, geometry)
    smoothed = [_gaussian_smoothed_loop(curve, sigma) for curve in curves]
    return PlanarLoops(_area_matched(smoothed, raw, f"Smoothing with sigma {sigma}"),
                       "smoothed", {"sigma": sigma})


def averaged_loops(signed_chain: SignedChain, geometry, passes: int = 4) -> PlanarLoops:
    """Average every vertex with its neighbours, ``passes`` times.

    Each pass replaces every vertex of the midpoint loops (cubical) or of the
    polygon (simplicial) by ``(previous + 2 * vertex + next) / 4``. Each loop
    is then scaled about its centroid to the signed area of its raw loop.
    On a grid the vertices are about one pixel apart, so ``passes`` sets the
    reach in pixels; four passes keep the curve within about a third of a
    pixel of the staircase. Polygon vertices can be unevenly spaced, so the
    reach varies along the loop.

    Parameters
    ----------
    signed_chain : SignedChain
        Planar 1-cycle; built from ``signed_chain.unsigned()``.
    geometry : ndarray or CubicalGeometry
        ``forest.geometry`` of a 2D simplicial or cubical forest.
    passes : int, default 4
        Positive number of averaging passes.
    """
    passes = _validated_passes(passes)
    raw, curves = _region_curves(signed_chain, geometry)
    averaged = []
    for curve in curves:
        for _ in range(passes):
            curve = (np.roll(curve, 1, axis=0) + 2 * curve + np.roll(curve, -1, axis=0)) / 4
        averaged.append(curve)
    return PlanarLoops(_area_matched(averaged, raw, f"Averaging with {passes} passes"),
                       "averaged", {"passes": passes})


def taut_loops(signed_chain: SignedChain, geometry: CubicalGeometry) -> PlanarLoops:
    """Return the minimum-length polygons of the regions bounded by a cubical 1-cycle.

    Every staircase edge separates two pixels; the taut loop is the shortest
    closed curve, in the same homotopy class along the staircase, that crosses
    each edge on the segment joining their centers. It stays within half a
    pixel of the staircase along each axis, turns only at pixel centers, and
    is convex for a convex digital region, so curvature measures see shape,
    not grid steps. The loop is not area matched: a region one pixel wide
    collapses, a single pixel to a one-vertex loop at its center and a bar to
    a two-vertex segment, both of zero area.

    Parameters
    ----------
    signed_chain : SignedChain
        Cycle of edges of a 2D grid; built from ``signed_chain.unsigned()``.
    geometry : CubicalGeometry
        Grid geometry of the forest (``forest.geometry``).
    """
    _require_planar_grid(geometry, "taut_loops")
    origin, spacing = np.asarray(geometry.origin), np.asarray(geometry.spacing)
    taut = []
    for loop in _staircase_loops(signed_chain.unsigned(), geometry):
        doubled = np.rint(2 * (loop - origin) / spacing).astype(np.int64)
        taut.append(origin + _taut_doubled_loop(doubled) / 2 * spacing)
    return PlanarLoops(tuple(taut), "taut")


def _polygon_loops(signed_chain: SignedChain, point_cloud: NDArray[np.float64]) -> PlanarLoops:
    coordinates = np.asarray(point_cloud, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("point_cloud must have shape (n_points, 2)")
    if not signed_chain.signed_simplices:
        return PlanarLoops((), "polygon")
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    paths = signed_chain_to_polyhedral_paths(signed_chain, coordinates)
    return PlanarLoops(tuple(coordinates[path] for path in paths), "polygon")


def _staircase_loops(signed_chain: SignedChain, geometry: CubicalGeometry) -> PlanarLoops:
    _require_planar_grid(geometry, "Staircase loops")
    if not signed_chain.signed_simplices:
        return PlanarLoops((), "staircase")
    vertex_chain, points = cubical_chain_to_vertex_chain(signed_chain, geometry)
    paths = signed_chain_to_polyhedral_paths(vertex_chain, points)
    return PlanarLoops(tuple(points[path] for path in paths), "staircase")


def _require_planar_grid(geometry, name: str) -> None:
    if not isinstance(geometry, CubicalGeometry):
        raise TypeError(f"{name} needs the CubicalGeometry of a cubical forest; "
                        "use raw_loops, smoothed_loops or averaged_loops for point clouds")
    if geometry.dim != 2:
        raise ValueError(f"Cubical 1-cycles live on 2D grids; this grid has {geometry.dim} axes")


def _midpoints(staircase: PlanarLoops) -> PlanarLoops:
    return PlanarLoops(tuple((loop + np.roll(loop, -1, axis=0)) / 2 for loop in staircase), "midpoint")


def _region_curves(signed_chain: SignedChain, geometry) -> tuple[PlanarLoops, list[NDArray[np.float64]]]:
    """Raw loops of the unsigned chain and the curves to smooth: midpoint loops on grids."""
    raw = raw_loops(signed_chain.unsigned(), geometry)
    curves = _midpoints(raw) if raw.kind == "staircase" else raw
    return raw, list(curves)


def _area_matched(curves: list[NDArray[np.float64]], reference: PlanarLoops, what: str) -> tuple:
    """Scale each curve about its centroid to the signed area of its reference loop."""
    matched = []
    for curve, target in zip(curves, reference):
        target_area = polygon_area(target, signed=True)
        area = polygon_area(curve, signed=True)
        if target_area != 0.0:
            if area * target_area <= 0.0:
                raise ValueError(f"{what} reversed or collapsed a loop; smooth less")
            center = curve.mean(axis=0)
            curve = center + (curve - center) * math.sqrt(target_area / area)
        matched.append(curve)
    return tuple(matched)


def _validated_sigma(sigma) -> float:
    if isinstance(sigma, (bool, np.bool_)) or not np.isscalar(sigma):
        raise TypeError("sigma must be a real number")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be positive and finite, got {sigma}")
    return sigma


def _validated_passes(passes) -> int:
    if isinstance(passes, (bool, np.bool_)) or not isinstance(passes, (int, np.integer)):
        raise TypeError("passes must be an integer")
    if passes < 1:
        raise ValueError(f"passes must be positive, got {passes}")
    return int(passes)


def _gaussian_smoothed_loop(loop: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    # Starting at a canonical vertex makes the samples independent of where
    # path extraction happened to start the loop.
    loop = np.roll(loop, -int(np.lexsort((loop[:, 1], loop[:, 0]))[0]), axis=0)
    closed = np.vstack((loop, loop[:1]))
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    length = float(arc[-1])
    if length == 0.0:
        raise ValueError("Cannot smooth a loop of zero length")
    n_samples = max(_MIN_SAMPLES, math.ceil(_SAMPLES_PER_SIGMA * length / sigma))
    if n_samples > _MAX_SAMPLES:
        raise ValueError(
            f"sigma {sigma} is too small for a loop of length {length:.6g}; "
            f"it would need {n_samples} samples")
    positions = np.linspace(0.0, length, n_samples, endpoint=False)
    samples = np.column_stack([np.interp(positions, arc, closed[:, axis]) for axis in range(2)])
    frequencies = np.fft.rfftfreq(n_samples, d=length / n_samples)
    # Damp relative to the first harmonic: the area rescaling about the mean
    # (the zeroth harmonic) cancels a common factor, which underflows when the
    # loop is much shorter than sigma.
    transfer = np.ones_like(frequencies)
    transfer[1:] = np.exp(-0.5 * (2 * np.pi * sigma) ** 2 * (frequencies[1:] ** 2 - length**-2))
    return np.fft.irfft(np.fft.rfft(samples, axis=0) * transfer[:, None], n=n_samples, axis=0)


def _cross(origin, a, b) -> int:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _funnel(start, lefts, rights) -> list:
    """Corners of the shortest path from ``start`` through the portals ``(lefts[i], rights[i])``.

    The simple funnel algorithm: the path is pulled tight against portal
    endpoints, which become corners where the funnel of visible portal
    segments closes. Returns ``(point, portal index)`` pairs; orientation
    tests on integer points are exact.
    """
    corners = []
    apex = left = right = start
    apex_index = left_index = right_index = -1
    index = 0
    while index < len(lefts):
        new_left, new_right = lefts[index], rights[index]
        if _cross(apex, right, new_right) >= 0:
            if apex == right or _cross(apex, left, new_right) < 0:
                right, right_index = new_right, index
            else:
                corners.append((left, left_index))
                apex = right = left
                apex_index = right_index = left_index
                index = apex_index + 1
                continue
        if _cross(apex, left, new_left) <= 0:
            if apex == left or _cross(apex, right, new_left) > 0:
                left, left_index = new_left, index
            else:
                corners.append((right, right_index))
                apex = left = right
                apex_index = left_index = right_index
                index = apex_index + 1
                continue
        index += 1
    return corners


def _taut_doubled_loop(staircase: NDArray[np.int64]) -> NDArray[np.float64]:
    """Minimum-length closed polygon of a staircase given in doubled grid coordinates.

    Staircase edge ``i`` is crossed on its portal, the segment between the
    centers of the pixels to its left and right. The shortest closed path
    turns at some portal endpoint, so it is the shortest path from that
    corner around the loop back to itself. Starting from an inner portal
    endpoint, the search restarts from a corner of the current path until the
    corners repeat; each restart is no longer than the path it starts on.
    """
    n = len(staircase)
    edges = np.roll(staircase, -1, axis=0) - staircase
    midpoints = staircase + edges // 2
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) // 2
    lefts = [tuple(point) for point in (midpoints + normals).tolist()]
    rights = [tuple(point) for point in (midpoints - normals).tolist()]
    counterclockwise = polygon_area(staircase.astype(float), signed=True) > 0
    start, start_index = (lefts[0] if counterclockwise else rights[0]), 0
    previous = None
    for _ in range(n + 1):
        order = [(start_index + 1 + step) % n for step in range(n - 1)]
        path = _funnel(start, [lefts[i] for i in order] + [start], [rights[i] for i in order] + [start])
        corners = [(start, start_index)]
        for point, index in path:
            if point != corners[-1][0]:
                corners.append((point, order[index] if index < n - 1 else start_index))
        while len(corners) > 1 and corners[-1][0] == start:
            corners.pop()
        points = frozenset(point for point, _ in corners)
        if points == previous:
            break
        previous = points
        start, start_index = corners[len(corners) // 2]
    return np.array(_without_straight_corners([point for point, _ in corners]), dtype=float)


def _without_straight_corners(points: list) -> list:
    """Drop vertices the loop passes straight through, e.g. on the way back along a bar."""
    changed = True
    while changed and len(points) > 2:
        changed = False
        for index in range(len(points)):
            before, point, after = points[index - 1], points[index], points[(index + 1) % len(points)]
            forward = (point[0] - before[0]) * (after[0] - point[0]) + (point[1] - before[1]) * (after[1] - point[1])
            if _cross(before, point, after) == 0 and forward > 0:
                del points[index]
                changed = True
                break
    return points
