"""Small input checks at scientific computation boundaries."""

import numpy as np


def validate_point_cloud(point_cloud):
    """Return a finite numeric point array with shape ``(n_points, dim)``."""
    try:
        points = np.asarray(point_cloud, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("point_cloud must be a finite 2D numeric array") from error
    if points.ndim != 2 or points.shape[1] < 1 or not np.isfinite(points).all():
        raise ValueError("point_cloud must be a finite 2D numeric array")
    return points


def validate_landscape_request(max_k, num_grid_points, mode, x_grid):
    """Return a valid explicit grid, or ``None`` for an inferred grid."""
    if mode not in ("raw", "pyramid"):
        raise ValueError("mode must be 'raw' or 'pyramid'")
    if isinstance(max_k, (bool, np.bool_)) or not isinstance(max_k, (int, np.integer)) or max_k < 1:
        raise ValueError("max_k must be a positive integer")
    if x_grid is None:
        if (isinstance(num_grid_points, (bool, np.bool_))
                or not isinstance(num_grid_points, (int, np.integer))
                or num_grid_points < 2):
            raise ValueError("num_grid_points must be an integer at least 2")
        return None
    grid = np.asarray(x_grid, dtype=float)
    if (grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)):
        raise ValueError("x_grid must be finite and strictly increasing with at least 2 points")
    return grid


def validate_landscape_kernel_input(bar, step_function):
    """Require finite bar and profile endpoints and nonnegative values."""
    if not np.isfinite((bar.birth, bar.death)).all():
        raise ValueError("Landscape bar endpoints must be finite")
    if (not np.isfinite(step_function.starts).all()
            or not np.isfinite(step_function.ends).all()):
        raise ValueError("Landscape profile endpoints must be finite")
    if not np.isfinite(step_function.vals).all() or np.any(step_function.vals < 0):
        raise ValueError("Landscape measurements must be finite and nonnegative")
