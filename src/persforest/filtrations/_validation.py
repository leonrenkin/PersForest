"""filtrations. validation: mechanically extracted reference implementation."""

from typing import Iterable
from typing import Sequence
import itertools
import math
import numpy as np
import operator

def _validate_filtration(
    filtration: Iterable[tuple[Sequence[int], float]],
    n_points: int,
) -> list[tuple[tuple[int, ...], float]]:
    """Validate filtration and retain its simplex and tie order.

    A face must already have appeared when its coface is encountered. Checking
    immediate faces therefore verifies closure and inclusion order in one pass.

    Parameters
    ----------
    filtration : iterable of (sequence of int, float)
        Simplex-filtration pairs in ascending filtration order.
    n_points : int
        Number of available point-cloud vertices.

    Returns
    -------
    list of (tuple of int, float)
        Validated entries in their original order.

    Raises
    ------
    ValueError
        If an entry is degenerate, duplicated, out of order, or lacks an
        earlier face, or if its value or vertex index is invalid.
    """
    entries = []
    seen = set()
    previous_value = -math.inf

    for simplex, value in filtration:
        try:
            vertices = tuple(operator.index(vertex) for vertex in simplex)
        except TypeError as exc:
            raise ValueError("Filtration vertices must be integer indices") from exc
        if not vertices or len(set(vertices)) != len(vertices):
            raise ValueError(f"Filtration simplex {vertices} is empty or has repeated vertices")
        if any(vertex < 0 or vertex >= n_points for vertex in vertices):
            raise ValueError(f"Filtration simplex {vertices} has an invalid vertex index")

        try:
            filtration_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid filtration value for simplex {vertices}") from exc
        if not math.isfinite(filtration_value):
            raise ValueError(f"Filtration value for simplex {vertices} must be finite")
        if filtration_value < previous_value:
            raise ValueError("Filtration values must be nondecreasing")

        simplex_key = tuple(sorted(vertices))
        if simplex_key in seen:
            raise ValueError(f"Filtration simplex {vertices} appears more than once")
        for face in itertools.combinations(simplex_key, len(simplex_key) - 1):
            if face and face not in seen:
                raise ValueError(f"Face {face} must precede filtration simplex {vertices}")

        entries.append((vertices, filtration_value))
        seen.add(simplex_key)
        previous_value = filtration_value

    return entries

def validate_point_cloud(point_cloud):
    """Return a finite numeric point array with shape ``(n_points, dim)``."""
    try:
        points = np.asarray(point_cloud, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("point_cloud must be a finite 2D numeric array") from error
    if points.ndim != 2 or points.shape[1] < 1 or not np.isfinite(points).all():
        raise ValueError("point_cloud must be a finite 2D numeric array")
    return points
