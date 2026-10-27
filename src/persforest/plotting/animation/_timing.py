"""Frame-time calculation for the PNG-based 3D animation renderer.

Other animation paths retain their existing time-window rules.
"""

from typing import Optional
import numpy as np

def _compute_frame_times(
    forest,
    *,
    frames: int,
    t_min: Optional[float],
    t_max: Optional[float],
) -> np.ndarray:
    """
    Return a monotone time grid used for filtration animations.

    Parameters
    ----------
    forest : object
        Forest-like object exposing ``filtration`` and ``barcode``.
    frames : int
        Number of frame times to generate.
    t_min, t_max : float | None
        Optional lower and upper bounds.

    Returns
    -------
    np.ndarray
        Array of shape (frames,) with filtration values.
    """
    if frames <= 0:
        raise ValueError("frames must be a positive integer.")

    if t_min is None:
        t0 = 0.0
    else:
        t0 = float(t_min)

    if t_max is None:
        finite_deaths = [
            float(bar.death)
            for bar in getattr(forest, "barcode", [])
            if np.isfinite(float(bar.death))
        ]
        if finite_deaths:
            t1 = max(finite_deaths)
        else:
            finite_filtration = [float(f) for _, f in forest.filtration if np.isfinite(float(f))]
            if not finite_filtration:
                raise ValueError("Could not infer finite t_max from barcode or filtration.")
            t1 = max(finite_filtration)
    else:
        t1 = float(t_max)

    if t1 < t0:
        raise ValueError(f"Invalid time window: t_max ({t1}) is smaller than t_min ({t0}).")

    return np.linspace(t0, t1, int(frames))
