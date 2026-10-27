"""Animate a filtration beside a barcode measurement profile."""

from typing import Optional, Tuple
from .._dependencies import require_plotting

require_plotting()

import matplotlib.pyplot as plt
import numpy as np

from persforest.landscapes.profiles import CycleValueFunc, MeasurementProfiles, _build_step_function_data
from persforest.landscapes._validation import require_computed_barcode, positive_integer
from ._export import _save_measurement_animation

def animate_barcode_measurement_generic(
        forest,
        cycle_func: CycleValueFunc | None = None,
        bar=None,
        *,
        profiles: MeasurementProfiles | None = None,
        filename: Optional[str] = None,
        fps: int = 20,
        frames: int = 200,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        dpi: int = 200,
        total_figsize: Tuple[float, float] = (12.0, 5.0),
        filtration_kwargs: Optional[dict] = None,
        measurement_kwargs: Optional[dict] = None,
    ):
    """
    Animate the filtration of a forest together with a barcode measurement.

    The left panel shows ``forest.plot_at_filtration`` at a moving
    filtration value :math:`alpha`. The right panel shows the step
    function obtained from ``_build_step_function_data`` for a single
    bar together with a vertical line that tracks the current value of
    :math:`alpha`.

    Parameters
    ----------
    forest :
        Forest-like object with attributes

        - ``point_cloud`` (``np.ndarray``),
        - ``barcode`` (iterable of Bar objects),
        - ``filtration`` (iterable of ``(simplex, filt_val)``),
        - a method ``plot_at_filtration(filt_val, ax=None, **kwargs)``.

    cycle_func : CycleValueFunc
        Callable ``(cycle_rep, point_cloud) -> float`` used to assign a
        scalar to each cycle representative of ``bar``.
    bar :
        Bar object in ``forest.barcode``. If ``None``, ``forest.max_bar()``
        is used.
    profiles : MeasurementProfiles, optional
        Reuse this explicit result for the measurement panel. Supply either
        profiles or cycle_func, never both. Source/bar association is checked.
    filename : str or None, optional
        If given, the animation is written to this path. The file extension
        must be ".mp4". Export requires ffmpeg on PATH.
    fps : int, optional
        Frames per second for the saved animation.
    frames : int, optional
        Number of time steps (frames) in the animation.
    t_min, t_max : float or None, optional
        Optional lower/upper bounds on the filtration values to animate.
        If ``None``, ``t_min = 0`` and ``t_max`` is the maximum bar death
        in ``forest.barcode``.
    dpi : int, optional
        DPI used when saving the animation.
    total_figsize : (float, float), optional
        Overall figure size for the two-panel figure.
    filtration_kwargs : dict or None, optional
        Extra keyword arguments forwarded to
        ``forest.plot_at_filtration``. Useful keys include e.g.
        ``show_complex``, ``vertex_size``, ``coloring``.
    measurement_kwargs : dict or None, optional
        Extra keyword arguments forwarded to
        ``StepFunctionData.plot``, **except** for ``ax`` which is
        managed by this function. This is the place to pass e.g.
        ``x_range``, ``y_range``, ``show_baseline``, ``title`` or any
        valid ``matplotlib`` line style keyword.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The created animation. If ``filename`` is not ``None``, the
        animation is also saved to disk.
    fig : matplotlib.figure.Figure
        The figure on which the animation is drawn.
    """
    from matplotlib.animation import FuncAnimation

    require_computed_barcode(forest)
    frames = positive_integer(frames, "frames")
    if (profiles is None) == (cycle_func is None):
        raise ValueError("Supply exactly one of cycle_func or profiles")
    if profiles is not None and not isinstance(profiles, MeasurementProfiles):
        raise TypeError("profiles must be MeasurementProfiles")
    if not hasattr(forest, "filtration") or not forest.filtration:
        raise ValueError("Forest has no filtration data to animate.")

    # Choose bar if not specified
    if bar is None:
        bar = forest.max_bar()

    # Optional restriction to a time window
    if t_min is None:
        t_min = 0.0
    if t_max is None:
        # Use the largest death time in the barcode
        deaths = [b.death for b in forest.barcode]
        if not deaths:
            raise ValueError("Forest has an empty barcode.")
        t_max = max(deaths)

    if not np.isfinite((t_min, t_max)).all() or t_max <= t_min:
        raise ValueError("Animation bounds must be finite with t_max strictly larger than t_min.")

    # Uniformly spaced in filtration value → uniform speed
    frame_times = np.linspace(t_min, t_max, frames).tolist()  # type: ignore

    # ---- Common kwargs for plot_at_filtration (cloud panel) ----
    if filtration_kwargs is None:
        filtration_kwargs = {}
    # Reasonable defaults (only used if not explicitly overridden)
    filtration_kwargs = {
        "show_complex": True,
        "vertex_size": 3,
        "coloring": "forest",
        "title": "Alpha Filtration",
        "show": False,   # important: we manage the figure ourselves
        **filtration_kwargs,
    }

    # ---- Build the step function data once ----
    if profiles is not None:
        profiles.require_source(forest, bar)
        step_func = profiles[bar]
    else:
        step_func = _build_step_function_data(forest=forest, bar=bar, cycle_func=cycle_func)

    # ---- Figure and axes ----
    fig, (ax_cloud, ax_meas) = plt.subplots(
        1, 2,
        figsize=total_figsize,
        gridspec_kw={"width_ratios": [3, 4]},
    )

    # ---- Measurement panel: draw the static step function ----
    if measurement_kwargs is None:
        measurement_kwargs = {}

    # Do not let the caller override the axis here
    measurement_kwargs = {
        k: v for k, v in measurement_kwargs.items()
        if k not in {"ax"}
    }

    # Provide sensible defaults, but allow user overrides
    meas_defaults = {
        "x_range": (0, t_max*1.05), # type: ignore
        "show_baseline": True,
        "title": "Barcode measurement",
    }
    # Only fill in keys that the user did not specify
    for k, v in meas_defaults.items():
        measurement_kwargs.setdefault(k, v)

    step_func.plot(ax=ax_meas, **measurement_kwargs)

    # Vertical line that will move with the filtration
    current_t0 = frame_times[0]
    meas_line = ax_meas.axvline(current_t0, color="k", linewidth=2)

    # ---- Helper to draw a single frame on the cloud panel ----
    def _draw_frame_at_time(t: float):
        """Clear the cloud axis and redraw for filtration value t."""
        ax_cloud.clear()
        forest.plot_at_filtration(filt_val=t, ax=ax_cloud, **filtration_kwargs)

        # Optional: overlay a small text box with the current filtration value.
        ax_cloud.text(
            0.02, 0.98, rf"$\alpha = {t:.3g}$",
            transform=ax_cloud.transAxes,
            va="top", ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # ---- Animation callbacks ----
    def init():
        _draw_frame_at_time(frame_times[0])
        t0 = frame_times[0]
        meas_line.set_xdata([t0, t0])
        return []

    def update(frame_idx: int):
        t = frame_times[frame_idx]
        _draw_frame_at_time(t)
        meas_line.set_xdata([t, t])
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        init_func=init,
        blit=False,
    )

    # ---- Optionally write to disk ----
    if filename is not None:
        _save_measurement_animation(anim, filename, fps=fps, dpi=dpi)

    return anim, fig
