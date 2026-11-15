"""Reproducible, bounded-memory benchmark inputs (no global RNG).

Point clouds and cubical grids come in three matching families: unstructured
noise, one large codimension-one cycle, and a fixed number of random holes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


POINT_SAMPLERS = ("uniform", "noisy_sphere", "holes")
GRID_SAMPLERS = ("uniform", "single_hole", "holes")

HOLES = {"num_balls": 30, "radius_range": [0.0, 0.05]}
SAMPLER_PARAMETERS = {
    "points": {
        "uniform": {"domain": "[0, 1]^dim"},
        "noisy_sphere": {"radius": 1.0, "noise_std": 0.05},
        "holes": {"domain": "[0, 1]^dim", **HOLES, "batch_size": 8192},
    },
    "cubical": {
        "uniform": {"values": "iid uniform [0, 1)", "filtration_type": "top_cells"},
        "single_hole": {"value": "| |x - c| - radius | + noise", "center": 0.5, "radius": 0.3,
                        "noise_std": 0.03, "domain": "cell centers in [0, 1]^dim",
                        "filtration_type": "top_cells"},
        "holes": {"value": "max_i max(0, 1 - |x - c_i| / r_i) + noise", **HOLES,
                  "noise_std": 0.03, "domain": "cell centers in [0, 1]^dim",
                  "filtration_type": "top_cells"},
    },
}


def _hole_balls(rng, dim: int):
    """Draw the excluded balls shared by the point and grid hole families."""
    radii = rng.uniform(*HOLES["radius_range"], size=HOLES["num_balls"])
    centers = rng.uniform(radii[:, None], 1.0 - radii[:, None], (HOLES["num_balls"], dim))
    return centers, radii


def sample_points(name: str, n_points: int, dim: int, seed: int) -> np.ndarray:
    """Sample a cloud using an independent local generator.

    Parameters
    ----------
    name : str
        ``uniform``, ``noisy_sphere`` (a circle in 2D), or ``holes``.
    n_points, dim : int
        Positive point count and ambient dimension (2 or 3).
    seed : int
        Seed for NumPy's default_rng.

    Returns
    -------
    ndarray
        Exactly ``(n_points, dim)`` points, with bounded rejection batches.
    """
    import numpy as np

    if n_points < 1 or dim not in (2, 3):
        raise ValueError("Require positive n_points and dim in (2, 3)")
    rng = np.random.default_rng(seed)
    if name == "uniform":
        return rng.random((n_points, dim))
    if name == "noisy_sphere":
        points = rng.normal(size=(n_points, dim))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points + rng.normal(0.0, 0.05, size=points.shape)
    if name != "holes":
        raise ValueError(f"Unknown point sampler: {name}")
    centers, radii = _hole_balls(rng, dim)
    points = np.empty((n_points, dim))
    filled = trials = 0
    while filled < n_points:
        batch_size = min(8192, max(128, 2 * (n_points - filled)))
        candidates = rng.random((batch_size, dim))
        valid = np.ones(batch_size, dtype=bool)
        for center, radius in zip(centers, radii):
            valid &= np.sum((candidates - center) ** 2, axis=1) > radius ** 2
        accepted = candidates[valid][:n_points - filled]
        points[filled:filled + len(accepted)] = accepted
        filled += len(accepted)
        trials += batch_size
        if trials > max(8192, 1000 * n_points) and filled < n_points:
            raise RuntimeError("Rejection sampler exceeded its trial budget")
    return points


def sample_grid(name: str, side: int, dim: int, seed: int) -> np.ndarray:
    """Sample top-cell values on a ``side**dim`` grid with a local generator.

    With sublevel (``top_cells``) filtrations, ``single_hole`` has a low-valued
    noisy shell, so one large cycle (2D) or void (3D) is born early and dies
    when the enclosed region fills. ``holes`` has a noisy low background with
    up to 30 high-valued cones, each enclosing one hole unless they overlap.

    Parameters
    ----------
    name : str
        ``uniform``, ``single_hole``, or ``holes``.
    side, dim : int
        Grid side length (at least 2) and dimension (2 or 3).
    seed : int
        Seed for NumPy's default_rng.

    Returns
    -------
    ndarray
        Float64 array of shape ``(side,) * dim``.
    """
    import numpy as np

    if side < 2 or dim not in (2, 3):
        raise ValueError("Require side >= 2 and dim in (2, 3)")
    rng = np.random.default_rng(seed)
    shape = (side,) * dim
    if name == "uniform":
        return rng.random(shape)
    # Cell centers in [0, 1]; open grids keep peak memory at a few grid copies.
    axes = np.ogrid[tuple(slice(0, side) for _ in range(dim))]
    axes = [(axis + 0.5) / side for axis in axes]
    if name == "single_hole":
        parameters = SAMPLER_PARAMETERS["cubical"]["single_hole"]
        squared = sum((axis - parameters["center"]) ** 2 for axis in axes)
        values = np.abs(np.sqrt(squared) - parameters["radius"])
    elif name == "holes":
        centers, radii = _hole_balls(rng, dim)
        values = np.zeros(shape)
        for center, radius in zip(centers, radii):
            if radius <= 0:
                continue
            distance = np.sqrt(sum((axis - c) ** 2 for axis, c in zip(axes, center)))
            np.maximum(values, 1.0 - distance / radius, out=values)
    else:
        raise ValueError(f"Unknown grid sampler: {name}")
    values += rng.normal(0.0, SAMPLER_PARAMETERS["cubical"][name]["noise_std"], size=shape)
    return values
