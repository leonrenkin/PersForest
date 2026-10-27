"""landscapes. validation: mechanically extracted reference implementation."""

import numpy as np

def require_computed_barcode(forest: object) -> None:
    """Reject an explicitly uncomputed barcode before measurement work.

    Parameters
    ----------
    forest : object
        Forest or forest-like input. Objects without ``_barcode_computed``
        continue to supply their barcode directly; its presence is not inferred
        to mean computation has completed when the flag is available.

    Raises
    ------
    RuntimeError
        If the input declares that its barcode has not been computed.
    """
    if not getattr(forest, "_barcode_computed", True):
        raise RuntimeError("Barcode has not been computed; call compute_barcode() first")

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
    grid = np.array(x_grid, dtype=float, copy=True)
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
    if step_function.baseline != 0.0:
        raise ValueError("Landscape profiles must have baseline=0; nonzero baselines are not convolved")


def validate_tolerance(tol: float) -> float:
    """Return a finite nonnegative convolution tolerance."""
    if isinstance(tol, (bool, np.bool_)) or not np.isscalar(tol):
        raise ValueError("tol must be a finite nonnegative number")
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite nonnegative number")
    return tol


def sample_grid(grid: object) -> np.ndarray:
    """Validate a nonempty one-dimensional, finite, nondecreasing query grid."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("grid must be a non-empty one-dimensional array")
    if not np.isfinite(grid).all():
        raise ValueError("grid contains non-finite values")
    if np.any(np.diff(grid) < 0):
        raise ValueError("grid must be sorted in non-decreasing order")
    return grid


def positive_integer(value: object, name: str) -> int:
    """Reject coercion of booleans and nonintegral level/index requests."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
