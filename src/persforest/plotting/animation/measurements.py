"""Animate a filtration beside the measurement profiles of its bars."""

from typing import Optional, Tuple
from .._dependencies import require_plotting

require_plotting()

import matplotlib.pyplot as plt
import numpy as np

from persforest.landscapes.profiles import CycleValueFunc, MeasurementProfiles, compute_measurement_profiles
from persforest.landscapes._validation import require_computed_barcode, positive_integer
from ._export import _save_measurement_animation
from ._layout import _default_t_min, _is_grid, _point_cloud_defaults

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
    Animate the filtration of a forest together with its measurement profiles.

    The left panel shows ``forest.plot_at_filtration`` at a moving
    filtration value :math:`alpha`. The right panel shows
    ``plot_measurement_profiles`` for the selected bars, a vertical line at
    the current value of :math:`alpha` and a dot on each active profile.

    Parameters
    ----------
    forest :
        Forest-like object with attributes

        - ``point_cloud`` (``np.ndarray``),
        - ``barcode`` (iterable of Bar objects),
        - ``filtration`` (iterable of ``(simplex, filt_val)``),
        - a method ``plot_at_filtration(filt_val, ax=None, **kwargs)``.

    cycle_func : CycleValueFunc
        Callable ``(cycle_rep, point_cloud) -> float`` evaluated on the
        stored (signed) cycle representatives. Its ``__name__`` labels the
        measurement axis unless it is a lambda.
    bar :
        Bar object in ``forest.barcode``. If given, only its profile is shown;
        otherwise all bars selected by ``measurement_kwargs`` are shown.
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
        ``show_complex``, ``vertex_size``, ``coloring``, ``min_bar_length``.
    measurement_kwargs : dict or None, optional
        Extra keyword arguments forwarded to ``plot_measurement_profiles``
        (``ax`` and ``bars`` are managed here), e.g. ``min_bar_length``,
        ``max_bars``, ``coloring``, ``extend_baseline``, ``legend``, ``title``
        or line styles.
        ``coloring`` and ``min_bar_length`` default to the filtration
        panel's values, so both panels show the same bars in the same colors.
        With ``cycle_func``, only bars passing ``min_bar_length`` are measured.

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

    # Optional restriction to a time window
    if t_min is None:
        t_min = _default_t_min(forest)
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
        **_point_cloud_defaults(forest, vertex_size=3),
        "coloring": "forest",
        "title": "Cubical filtration" if _is_grid(forest) else "Alpha Filtration",
        "show": False,   # important: we manage the figure ourselves
        **filtration_kwargs,
    }

    # ---- Measurement panel: same bars and colors as the filtration panel ----
    measurement_kwargs = {
        "coloring": filtration_kwargs["coloring"],
        # An explicitly chosen bar is shown even if the filtration panel hides it.
        "min_bar_length": 0.0 if bar is not None else filtration_kwargs.get("min_bar_length", 0.0),
        **{k: v for k, v in (measurement_kwargs or {}).items() if k not in {"ax", "bars", "x_include"}},
    }
    if profiles is None:
        # Measure only bars that can be shown; a single bar needs no shorter bars.
        min_length = measurement_kwargs["min_bar_length"]
        if bar is not None and np.isfinite(bar.lifespan()):
            min_length = max(min_length, bar.lifespan())
        name = getattr(cycle_func, "__name__", "<lambda>")
        profiles = compute_measurement_profiles(
            forest, cycle_func, "measurement" if name == "<lambda>" else name,
            signed=True, min_bar_length=min_length, cache=False)
    elif bar is not None:
        profiles.require_source(forest, bar)
    measurement_kwargs.setdefault("title", f"{profiles.label} profiles")

    # ---- Figure and axes ----
    fig, (ax_cloud, ax_meas) = plt.subplots(
        1, 2,
        figsize=total_figsize,
        gridspec_kw={"width_ratios": [3, 4]},
    )

    from persforest.plotting.landscapes import _draw_measurement_profiles
    try:
        _, drawn = _draw_measurement_profiles(
            forest, profiles, ax=ax_meas, bars=None if bar is None else [bar],
            x_include=(t_min, t_max), **measurement_kwargs)  # time line stays visible
    except Exception:
        plt.close(fig)  # e.g. foreign profiles or no bars left: do not leak the figure
        raise

    # Vertical line that will move with the filtration
    current_t0 = frame_times[0]
    meas_line = ax_meas.axvline(current_t0, color="k", linewidth=1.5, alpha=0.7)
    markers = ax_meas.scatter([], [], s=30, zorder=3, edgecolors="k", linewidths=0.5)

    def _update_measurement(t: float):
        """Move the time line and mark the current value of every active profile."""
        meas_line.set_xdata([t, t])
        # Bars are half-open [birth, death): a profile is active iff birth <= t < death.
        active = [(step, line) for bar_, step, line in drawn if bar_.birth <= t < bar_.death]
        offsets = [(t, step.eval_on_grid(np.array([t]))[0]) for step, _ in active]
        markers.set_offsets(np.reshape(offsets, (-1, 2)))
        markers.set_facecolors([line.get_color() for _, line in active])

    # ---- Helper to draw a single frame on the cloud panel ----
    def _draw_frame_at_time(t: float):
        """Clear the cloud axis and redraw for filtration value t."""
        ax_cloud.clear()
        forest.plot_at_filtration(filt_val=t, ax=ax_cloud, **filtration_kwargs)

        # Optional: overlay a small text box with the current filtration value.
        symbol = "t" if _is_grid(forest) else r"\alpha"
        ax_cloud.text(
            0.02, 0.98, rf"${symbol} = {t:.3g}$",
            transform=ax_cloud.transAxes,
            va="top", ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # ---- Animation callbacks ----
    def init():
        _draw_frame_at_time(frame_times[0])
        _update_measurement(frame_times[0])
        return []

    def update(frame_idx: int):
        t = frame_times[frame_idx]
        _draw_frame_at_time(t)
        _update_measurement(t)
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
