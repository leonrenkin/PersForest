"""Reproducible, bounded-memory benchmark distributions (no global RNG)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


SAMPLER_PARAMETERS = {
    "uniform": {"domain": "[0, 1]^dim"},
    "noisy_sphere": {"radius": 1.0, "noise_std": 0.05},
    "holes": {"domain": "[0, 1]^dim", "num_balls": 30,
              "radius_range": [0.0, 0.05], "batch_size": 8192},
}


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
        raise ValueError(f"Unknown sampler: {name}")
    radii = rng.uniform(0.0, 0.05, size=30)
    centers = rng.uniform(radii[:, None], 1.0 - radii[:, None], (30, dim))
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
