"""Matplotlib filtration animations and paired forest views."""

from .._dependencies import require_plotting

require_plotting()

from numbers import Real
from pathlib import Path
from typing import Any, Literal, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import warnings

from ._export import (
    _assemble_mp4_with_ffmpeg,
    _save_filtration_animation,
    _save_paired_animation,
)
from ._layout import (
    _add_radius_box,
    _default_t_min,
    _filtration_axis_label,
    _filtration_symbol,
    _is_grid,
    _point_cloud_defaults,
    _parse_camera_eye,
    _resolve_figure_margins,
    _resolve_matplotlib_figsize,
    _validate_radius_box_corner,
)
from ._timing import _compute_frame_times

def _animate_filtration_generic(
        forest,
        filename: Optional[str] = None,
        *,
        fps: int = 20,
        frames: int = 200,
        coloring: Literal["forest", "bars"] = "forest",
        with_barcode: bool = False,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        dpi: int = 300,
        figsize: Optional[tuple[float, float]] = None,
        pixel_size: Optional[tuple[int, int]] = None,
        panel_width_ratios: tuple[float, float] = (3.5, 2.0),
        panel_spacing: float = 0.08,
        figure_margins: Optional[dict[str, float]] = None,
        filtration_kwargs: Optional[dict] = None,
        barcode_kwargs: Optional[dict] = None,
        # Deprecated aliases, kept for backward compatibility
        cloud_figsize: Optional[tuple[float, float]] = None,
        total_figsize: Optional[tuple[float, float]] = None,
        plot_kwargs: Optional[dict] = None,
        alpha_digits: Optional[int] = None,
        radius_box_corner: Literal["upper right", "upper left", "lower right", "lower left"] = "upper right",
    ):
        """
        Create a Matplotlib animation across filtration values.

        This is the generic target for ``PersistenceForest.animate_filtration``
        when using Matplotlib-backed animation paths.

        Parameters
        ----------
        forest : Forest
            Forest exposing ``filtration``, ``barcode``, and ``plot_at_filtration``.
        filename : str | None, optional
            If given, the animation is written to this path.
        fps : int, optional
            Frames per second for the saved animation.
        frames : int, optional
            Number of time steps (frames) sampled between ``t_min`` and ``t_max``.
        coloring : {"forest","bars"}, optional
            Which color scheme to apply consistently to both the cloud panel and
            barcode panel.
        with_barcode : bool, optional
            If True, show a second panel with the barcode and a moving vertical
            line indicating the current filtration value.
        t_min, t_max : float | None, optional
            Optional lower/upper bounds on the filtration values to animate.
        dpi : int, optional
            DPI for saving the animation.
        figsize : (float, float) | None, optional
            Figure size in inches. If None, defaults to (6,6) without barcode
            and (10,5) with barcode.
        pixel_size : (int, int) | None, optional
            Figure size in pixels. If provided, takes precedence over
            ``figsize`` and is converted using ``dpi``.
        panel_width_ratios : (float, float), optional
            Width ratios for cloud and barcode panels when
            ``with_barcode=True``.
        panel_spacing : float, optional
            Horizontal spacing between cloud and barcode panels (GridSpec
            ``wspace``) when ``with_barcode=True``.
        figure_margins : dict[str, float] | None, optional
            Outer margins applied via ``fig.subplots_adjust`` with keys
            ``{"left","right","bottom","top"}``.
        filtration_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``plot_at_filtration``.
            The keys ``ax``, ``show`` and ``filt_val`` are reserved and are
            managed by this helper. If ``coloring`` is supplied here and
            conflicts with the top-level ``coloring``, the top-level value
            wins.
            Example::
                filtration_kwargs=dict(
                    show_complex=True,
                    vertex_size=3,
                    coloring="forest",
                )
        barcode_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``_plot_barcode`` **except**
            ``ax`` and ``coloring``, which are managed by this method.
            Example::
                barcode_kwargs=dict(
                    max_bars=150,
                    min_bar_length=1e-3,
                    sort="length",
                )
        cloud_figsize, total_figsize : tuple[float, float] | None, optional
            Deprecated aliases used only when ``figsize`` is omitted.
        plot_kwargs : dict | None, optional
            Deprecated alias for ``filtration_kwargs``.
        alpha_digits : int | None, optional
            Number of decimal places in the filtration-value overlay. If None,
            uses compact formatting.

        radius_box_corner : {"upper right", "upper left", "lower right", "lower left"}
            Radius label corner (default: "upper right"), with an 8 pt inset.

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The created animation. If ``filename`` is not None, the animation
            is also saved to disk.
        fig : matplotlib.figure.Figure
            The figure on which the animation is drawn.
        """
        from matplotlib.animation import FuncAnimation

        _validate_radius_box_corner(radius_box_corner)

        if not hasattr(forest, "filtration") or not forest.filtration:
            raise ValueError("Forest has no filtration data to animate.")
        is_3d = (getattr(forest, "dim", None) == 3)
        if int(dpi) <= 0:
            raise ValueError("dpi must be a positive integer.")

        if len(panel_width_ratios) != 2:
            raise ValueError("panel_width_ratios must have exactly two entries.")
        if float(panel_width_ratios[0]) <= 0 or float(panel_width_ratios[1]) <= 0:
            raise ValueError("panel_width_ratios entries must be positive.")
        if not isinstance(panel_spacing, Real):
            raise ValueError("panel_spacing must be a numeric value.")
        panel_spacing = float(panel_spacing)
        if panel_spacing < 0.0:
            raise ValueError("panel_spacing must be non-negative.")

        if cloud_figsize is not None or total_figsize is not None:
            warnings.warn(
                "`cloud_figsize` and `total_figsize` are deprecated in animate_filtration; "
                "use `figsize=(w, h)`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if figsize is None:
                if with_barcode and total_figsize is not None:
                    figsize = total_figsize
                elif (not with_barcode) and cloud_figsize is not None:
                    figsize = cloud_figsize

        if plot_kwargs is not None:
            warnings.warn(
                "`plot_kwargs` is deprecated in animate_filtration; "
                "use `filtration_kwargs`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filtration_kwargs is None:
                filtration_kwargs = dict(plot_kwargs)
            else:
                merged = dict(plot_kwargs)
                merged.update(filtration_kwargs)
                filtration_kwargs = merged

        # Optional restriction to a time window
        if t_min is None:
            t_min = _default_t_min(forest)
        if t_max is None:
            finite_deaths = [
                float(bar.death)
                for bar in getattr(forest, "barcode", [])
                if np.isfinite(float(bar.death))
            ]
            if finite_deaths:
                t_max = max(finite_deaths)
            else:
                finite_filtration = [float(f) for _, f in forest.filtration if np.isfinite(float(f))]
                if not finite_filtration:
                    raise ValueError("Could not infer finite t_max from barcode or filtration.")
                t_max = max(finite_filtration)

        # Uniformly spaced in filtration value → uniform speed
        frame_times = np.linspace(t_min, t_max, frames).tolist()  # pyright: ignore[reportCallIssue, reportArgumentType]

        # ---- Common kwargs for plot_at_filtration (cloud panel) ----
        if filtration_kwargs is None:
            filtration_kwargs = {}
        else:
            filtration_kwargs = dict(filtration_kwargs)

        banned_filtration_keys = {"ax", "show", "filt_val"}
        bad_keys = [k for k in banned_filtration_keys if k in filtration_kwargs]
        if bad_keys:
            raise ValueError(
                "filtration_kwargs contains reserved keys that are managed by animate_filtration: "
                f"{sorted(bad_keys)}"
            )
        if "coloring" in filtration_kwargs and filtration_kwargs["coloring"] != coloring:
            warnings.warn(
                "Ignoring `filtration_kwargs['coloring']` because top-level `coloring` controls both panels.",
                UserWarning,
                stacklevel=2,
            )
            filtration_kwargs.pop("coloring", None)

        # Reasonable defaults (only used if not explicitly overridden)
        if is_3d:
            filtration_kwargs = {
                "show_complex": True,
                **_point_cloud_defaults(forest, vertex_size=3),
                "coloring": coloring,
                "show": False,
                **filtration_kwargs,
            }
        else:
            filtration_kwargs = {
                **_point_cloud_defaults(forest, vertex_size=3),
                "coloring": coloring,
                "show": False,
                **filtration_kwargs,
            }
        filtration_kwargs["show"] = False

        # ---- Barcode kwargs & shared color dict ----
        rendered_figsize, _ = _resolve_matplotlib_figsize(
            with_barcode=with_barcode,
            dpi=int(dpi),
            figsize=figsize,
            pixel_size=pixel_size,
        )
        margins = _resolve_figure_margins(
            with_barcode=with_barcode,
            figure_margins=figure_margins,
        )
        if with_barcode:
            fig = plt.figure(figsize=rendered_figsize, constrained_layout=False)
            if is_3d:
                gs = fig.add_gridspec(
                    1,
                    2,
                    width_ratios=panel_width_ratios,
                    wspace=panel_spacing,
                )
                ax_cloud = fig.add_subplot(gs[0, 0], projection="3d")
                ax_bar = fig.add_subplot(gs[0, 1])
            else:
                gs = fig.add_gridspec(
                    1,
                    2,
                    width_ratios=panel_width_ratios,
                    wspace=panel_spacing,
                )
                ax_cloud = fig.add_subplot(gs[0, 0])
                ax_bar = fig.add_subplot(gs[0, 1])
            fig.subplots_adjust(**margins)

            # Draw the (static) barcode once
            if not getattr(forest, "barcode", None):
                raise ValueError("`with_barcode=True` but `forest.barcode` is empty.")

            if barcode_kwargs is None:
                barcode_kwargs = {}
            else:
                barcode_kwargs = dict(barcode_kwargs)
            if "ax" in barcode_kwargs:
                raise ValueError("barcode_kwargs cannot contain 'ax'; axis is managed internally.")
            if "coloring" in barcode_kwargs and barcode_kwargs["coloring"] != coloring:
                warnings.warn(
                    "Ignoring `barcode_kwargs['coloring']` because top-level `coloring` controls both panels.",
                    UserWarning,
                    stacklevel=2,
                )
                barcode_kwargs.pop("coloring", None)
            # Do not let the caller override ax here
            # Defaults for the barcode panel – user can override sort/title/xlabel
            barcode_kwargs = {
                "sort": "length",
                "xlabel": _filtration_axis_label(forest),
                "tight_layout": False,
                "coloring": coloring,
                **barcode_kwargs,
            }


            forest.plot_barcode(
                ax=ax_bar,
                **barcode_kwargs,
            )

            # Vertical line that will move with the filtration
            current_t0 = frame_times[0]
            barcode_line = ax_bar.axvline(current_t0, color="k", linewidth=2)
        else:
            if is_3d:
                fig = plt.figure(figsize=rendered_figsize, constrained_layout=False)
                ax_cloud = fig.add_subplot(111, projection="3d")
            else:
                fig = plt.figure(figsize=rendered_figsize, constrained_layout=False)
                ax_cloud = fig.add_subplot(111)
            fig.subplots_adjust(**margins)
            ax_bar = None
            barcode_line = None

        # ---- Helper to draw a single frame on the cloud panel ----
        def _draw_frame_at_time(t: float, frame_idx: int = 0):
            """Helper: clear the cloud axis and redraw for filtration value t."""
            ax_cloud.clear()
            local_plot_kwargs = filtration_kwargs
            if is_3d:
                local_plot_kwargs = dict(filtration_kwargs)
                camera_mode = local_plot_kwargs.pop("camera_mode", "fixed")
                style_3d = dict(local_plot_kwargs.get("style_3d", {}) or {})
                camera_eye = style_3d.get("camera_eye", None)
                if camera_mode not in {"fixed", "orbit"}:
                    raise ValueError("camera_mode must be 'fixed' or 'orbit'.")
                if camera_mode == "orbit":
                    base_elev = 22.0
                    base_azim = -55.0
                    if isinstance(camera_eye, dict):
                        base_elev = float(camera_eye.get("elev", base_elev))
                        base_azim = float(camera_eye.get("azim", base_azim))
                    elif isinstance(camera_eye, (tuple, list)) and len(camera_eye) >= 2:
                        base_elev = float(camera_eye[0])
                        base_azim = float(camera_eye[1])
                    denom = max(1, len(frame_times) - 1)
                    style_3d["camera_eye"] = (
                        base_elev,
                        base_azim + 360.0 * float(frame_idx) / float(denom),
                    )
                    local_plot_kwargs["style_3d"] = style_3d
            # Delegate the heavy lifting to the existing helper
            forest.plot_at_filtration(filt_val=t, ax=ax_cloud, **local_plot_kwargs)

            # Overlay the current radius with a fixed inset from the axes.
            symbol = _filtration_symbol(forest)
            if alpha_digits is None:
                radius_text = rf"${symbol} = {t:.3g}$"
            else:
                radius_text = rf"${symbol} = {t:.{alpha_digits}f}$"
            _add_radius_box(ax_cloud, radius_text, radius_box_corner)

        # ---- Animation callbacks ----
        def init():
            _draw_frame_at_time(frame_times[0], frame_idx=0)
            if with_barcode and barcode_line is not None:
                t0 = frame_times[0]
                barcode_line.set_xdata([t0, t0])
            return []

        def update(frame_idx: int):
            t = frame_times[frame_idx]
            _draw_frame_at_time(t, frame_idx=frame_idx)
            if with_barcode and barcode_line is not None:
                barcode_line.set_xdata([t, t])
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
            # Force a draw so text/ticks/layout are finalized
            _save_filtration_animation(anim, fig, filename, fps=fps, dpi=dpi)
        return anim, fig

def _animate_filtration_generic_3d_matplotlib(
    forest,
    filename: str,
    *,
    with_barcode: bool = False,
    fps: int = 20,
    frames: int = 200,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    coloring: Literal["forest", "bars"] = "forest",
    show_cycles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0.0,
    show_complex: bool = True,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    vertex_size: float = 3.0,
    cloud_figsize: tuple[float, float] = (6.0, 6.0),
    total_figsize: Optional[tuple[float, float]] = None,
    barcode_kwargs: Optional[dict] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    camera_mode: Literal["fixed", "orbit"] = "fixed",
    camera_eye: Optional[Any] = None,
    dpi: int = 200,
    alpha_digits: Optional[int] = None,
    radius_box_corner: Literal["upper right", "upper left", "lower right", "lower left"] = "upper right",
) -> None:
    """
    Animate a 3D filtration and export as MP4 using matplotlib.

    This implementation avoids browser-based rendering and instead
    generates frames directly via matplotlib, which are then assembled
    into a video using ffmpeg.

    Parameters
    ----------
    forest : PersistenceForest
        Forest instance with 3D point cloud data.
    filename : str
        Output MP4 path.
    with_barcode : bool
        Whether to render a second panel with barcode and moving time marker.
    fps : int
        Frames per second for the resulting video.
    frames : int
        Number of animation frames.
    t_min, t_max : float | None
        Filtration interval. If omitted, inferred from forest data.
    coloring : {"forest", "bars"}
        Bar color map strategy used for cycle surfaces.
    show_cycles : bool
        If True, show active cycle representatives.
    signed : bool
        If False, cancel opposite-oriented duplicate simplices in cycle chains.
    min_bar_length : float
        Exclude bars shorter than this threshold.
    show_complex : bool
        If True, draw complex edges and boundary triangles.
    complex_opacity : float
        Opacity used for complex boundary surfaces.
    cycle_opacity : float
        Opacity used for cycle surfaces.
    vertex_size : float
        Marker size for vertices.
    cloud_figsize : tuple[float, float]
        Figure size used when ``with_barcode=False``.
    total_figsize : tuple[float, float] | None
        Total figure size used when ``with_barcode=True``.
    barcode_kwargs : dict | None
        Extra kwargs forwarded to ``forest.plot_barcode`` (except ``ax``).
    width, height : int | None
        Optional figure size in pixels.
    camera_mode : {"fixed", "orbit"}
        Camera behavior. Orbit rotates azimuth uniformly across frames.
    camera_eye : Any
        Camera specification as ``(elev, azim)`` or ``{"elev": ..., "azim": ...}``.
    dpi : int
        PNG rendering DPI before ffmpeg assembly.
    alpha_digits : int | None
        Number of digits shown in the filtration value overlay.

    radius_box_corner : {"upper right", "upper left", "lower right", "lower left"}
        Radius label corner (default: "upper right"), with an 8 pt inset.

    Returns
    -------
    None
        Writes the MP4 to ``filename``.
    """
    _validate_radius_box_corner(radius_box_corner)
    from matplotlib import colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if int(fps) <= 0:
        raise ValueError("fps must be a positive integer.")
    if forest.dim != 3:
        raise ValueError("_animate_filtration_generic_3d_matplotlib requires ambient dimension 3.")
    if camera_mode not in {"fixed", "orbit"}:
        raise ValueError("camera_mode must be 'fixed' or 'orbit'.")

    frame_times = _compute_frame_times(
        forest=forest,
        frames=frames,
        t_min=t_min,
        t_max=t_max,
    )
    color_map = forest._get_color_map(coloring=coloring)

    pts = np.asarray(forest.point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("point_cloud must be an (n_points, 3) array-like.")

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    pad = 0.05 * np.max(spans)
    xlim = (mins[0] - pad, maxs[0] + pad)
    ylim = (mins[1] - pad, maxs[1] + pad)
    zlim = (mins[2] - pad, maxs[2] + pad)
    box_aspect = (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])

    elev0, azim0 = _parse_camera_eye(camera_eye)
    figsize, _frame_px = _resolve_matplotlib_figsize(
        with_barcode=with_barcode,
        width=width,
        height=height,
        dpi=int(dpi),
        cloud_figsize=cloud_figsize,
        total_figsize=total_figsize,
    )
    out_path = Path(filename).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if barcode_kwargs is None:
        barcode_kwargs = {}
    else:
        barcode_kwargs = dict(barcode_kwargs)
    barcode_kwargs.pop("ax", None)
    barcode_kwargs = {
        "sort": "length",
        "xlabel": "Radius r",
        "tight_layout": False,
        "coloring": coloring,
        "min_bar_length": min_bar_length,
        **barcode_kwargs,
    }

    def _draw_scene(ax_scene, t: float, frame_idx: int) -> None:
        ax_scene.cla()
        snapshot = forest._complex_snapshot_at_filtration(float(t))
        frame_pts = snapshot["points"]

        if show_complex:
            edges = snapshot.get("edges", [])
            if edges:
                segments = [frame_pts[list(edge)] for edge in edges]
                edge_coll = Line3DCollection(
                    segments,
                    colors="0.35",
                    linewidths=0.7,
                    alpha=max(0.2, float(complex_opacity)),
                )
                ax_scene.add_collection3d(edge_coll)

            triangles = snapshot.get("triangles", [])
            if triangles:
                tri_polys = [frame_pts[list(tri)] for tri in triangles]
                tri_coll = Poly3DCollection(
                    tri_polys,
                    facecolors=mcolors.to_rgba("lightblue", alpha=complex_opacity),
                    edgecolors="none",
                )
                ax_scene.add_collection3d(tri_coll)

        ax_scene.scatter(
            frame_pts[:, 0],
            frame_pts[:, 1],
            frame_pts[:, 2],
            s=vertex_size,
            c="black",
            depthshade=False,
        )

        if show_cycles:
            active = forest._active_bars_with_cycles_at(
                filt_val=float(t),
                min_bar_length=min_bar_length,
            )
            active = sorted(active, key=lambda bc: bc[0].lifespan(), reverse=True)

            for bar, cycle in active:
                if cycle.dim() != 2:
                    raise ValueError(
                        f"3D filtration animation expected a 2-chain, got dim={cycle.dim()}"
                    )
                tri_faces = cycle.simplices(signed=signed)
                if not tri_faces:
                    continue
                cycle_polys = [frame_pts[list(face)] for face in tri_faces]
                color = color_map.get(bar, "#d62728")
                cycle_coll = Poly3DCollection(
                    cycle_polys,
                    facecolors=mcolors.to_rgba(color, alpha=cycle_opacity),
                    edgecolors=mcolors.to_rgba(color, alpha=min(1.0, cycle_opacity + 0.25)),
                    linewidths=0.2,
                )
                ax_scene.add_collection3d(cycle_coll)

        ax_scene.set_xlim(*xlim)
        ax_scene.set_ylim(*ylim)
        ax_scene.set_zlim(*zlim)
        ax_scene.set_box_aspect(box_aspect)
        ax_scene.set_xlabel("x")
        ax_scene.set_ylabel("y")
        ax_scene.set_zlabel("z")
        ax_scene.set_title(f"Filtration value r = {float(t):.4g}")
        if alpha_digits is None:
            radius_text = rf"$r = {float(t):.3g}$"
        else:
            radius_text = rf"$r = {float(t):.{alpha_digits}f}$"
        _add_radius_box(ax_scene, radius_text, radius_box_corner)

        if camera_mode == "orbit":
            denom = max(1, len(frame_times) - 1)
            azim = azim0 + 360.0 * float(frame_idx) / float(denom)
        else:
            azim = azim0
        ax_scene.view_init(elev=elev0, azim=azim)

    with tempfile.TemporaryDirectory(prefix="persforest_3d_frames_") as tmp_dir:
        frame_dir = Path(tmp_dir)
        for idx, t in enumerate(frame_times):
            fig = plt.figure(figsize=figsize)
            if with_barcode:
                gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 2.0])
                ax_scene = fig.add_subplot(gs[0, 0], projection="3d")
                ax_bar = fig.add_subplot(gs[0, 1])

                forest.plot_barcode(
                    ax=ax_bar,
                    **barcode_kwargs,
                )
                ax_bar.set_xlim(float(frame_times[0]), float(frame_times[-1]))
                ax_bar.axvline(float(t), color="k", linewidth=2.0)
            else:
                ax_scene = fig.add_subplot(111, projection="3d")

            _draw_scene(ax_scene=ax_scene, t=float(t), frame_idx=idx)

            frame_file = frame_dir / f"frame_{idx:04d}.png"
            fig.savefig(frame_file, dpi=dpi, facecolor=fig.get_facecolor())
            plt.close(fig)

        _assemble_mp4_with_ffmpeg(
            frame_dir=frame_dir,
            fps=int(fps),
            output_file=out_path,
        )

def animate_filtration_pair(
    forest1,
    forest2,
    filename: Optional[str] = None,
    *,
    fps: int = 20,
    frames: int = 200,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    dpi: int = 200,
    total_figsize: Optional[Tuple[float, float]] = None,
    plot_kwargs_forest1: Optional[dict] = None,
    plot_kwargs_forest2: Optional[dict] = None,
    barcode_kwargs_forest1: Optional[dict] = None,
    barcode_kwargs_forest2: Optional[dict] = None,
    radius_box_corner: Literal["upper right", "upper left", "lower right", "lower left"] = "upper right",
):
    """
    Animate two Forests side-by-side: for each forest, show the evolving
    cycle representatives in the point cloud together with its barcode.

    The filtration panels and barcodes are styled via `plot_at_filtration` and
    `_plot_barcode`, so they match what `animate_filtration` produces.

    Parameters
    ----------
    forest1, forest2 : Forest
        The two forests to animate.
    filename : str or None, optional
        If not None, the animation is also written to this file.
        The extension must be '.mp4'. Export requires ffmpeg on PATH.
    fps : int, optional
        Frames per second for the saved animation.
    frames : int, optional
        Number of frames in the animation (shared time grid).
    t_min, t_max : float or None, optional
        Filtration time window. If None, t_min = 0 and t_max is the max
        filtration value across both forests.
    dpi : int, optional
        DPI for saving.
    total_figsize : (float, float) or None, optional
        Overall figure size. If None, a reasonable default is used.
    plot_kwargs_forest1, plot_kwargs_forest2 : dict or None, optional
        Extra kwargs forwarded to `plot_at_filtration` for each forest.
        Used on the *cloud/loop* panels.
    barcode_kwargs_forest1, barcode_kwargs_forest2 : dict or None, optional
        Extra kwargs forwarded to `_plot_barcode` for each forest.

    radius_box_corner : {"upper right", "upper left", "lower right", "lower left"}
        Radius label corner (default: "upper right"), with an 8 pt inset.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    fig : matplotlib.figure.Figure
    """
    from matplotlib.animation import FuncAnimation

    _validate_radius_box_corner(radius_box_corner)

    # --- 1) Sanity checks -----------------------------------------------------
    for forest, name in ((forest1, "forest1"), (forest2, "forest2")):
        if not hasattr(forest, "filtration") or not forest.filtration:
            raise ValueError(f"{name} has no filtration data to animate.")
        if not getattr(forest, "barcode", None):
            raise ValueError(f"{name} has an empty barcode.")

    # --- 2) Time grid ---------------------------------------------------------
    if t_min is None:
        t_start: float = min(_default_t_min(forest1), _default_t_min(forest2))
    else:
        t_start = float(t_min)

    if t_max is None:
        max1 = max(node.filt_val for node in forest1.nodes.values())
        max2 = max(node.filt_val for node in forest2.nodes.values())
        t_end: float = float(max(max1, max2))
    else:
        t_end = float(t_max)
    # Radii end 5% past t_end; grid values, which may be negative, 5% of the range.
    if _is_grid(forest1) or _is_grid(forest2):
        t_end += 0.05 * max(t_end - t_start, np.finfo(float).eps)
    else:
        t_end *= 1.05

    n_frames: int = int(frames)

    frame_times = np.linspace(start = t_start, stop = t_end, num = n_frames).tolist()

    # --- 3) Figure layout: 2x2 (clouds on top, barcodes below) ----------------
    if total_figsize is None:
        total_figsize = (12.0, 8.0)

    fig, ((ax_cloud_1, ax_cloud_2),
          (ax_bar_1,   ax_bar_2)) = plt.subplots(
        2,
        2,
        figsize=total_figsize,
        gridspec_kw={"height_ratios": [5, 2]},
    )

    # --- 4) Barcode plotting --------------------------------------------------
    # Base kwargs to resemble your existing barcode style
    base_barcode_kwargs = {
        "sort": "length",
        "xlabel": _filtration_axis_label(forest1),
        "tight_layout": False,
    }

    if barcode_kwargs_forest1 is None:
        barcode_kwargs_forest1 = {}
    if barcode_kwargs_forest2 is None:
        barcode_kwargs_forest2 = {}

    kwargs_bar_1 = {**base_barcode_kwargs, **barcode_kwargs_forest1}
    kwargs_bar_2 = {**base_barcode_kwargs, **barcode_kwargs_forest2}

    forest1.plot_barcode(ax=ax_bar_1, **kwargs_bar_1)
    forest2.plot_barcode(ax=ax_bar_2, **kwargs_bar_2)

    # Force both barcodes to share the same x-range as the animation time
    ax_bar_1.set_xlim(t_start, t_end)
    ax_bar_2.set_xlim(t_start, t_end)

    # Vertical lines tracking the current filtration value
    t0 = frame_times[0]
    barcode_line_1 = ax_bar_1.axvline(t0, color="k", linewidth=2)
    barcode_line_2 = ax_bar_2.axvline(t0, color="k", linewidth=2)

    # --- 5) Filtration plot kwargs (match animate_filtration defaults) --------
    base_plot_kwargs = {
        "coloring": "forest",  # uses the forest's color dict, shared with barcode
        "show": False,         # we manage figure/axes ourselves
    }

    if plot_kwargs_forest1 is None:
        plot_kwargs_forest1 = {}
    if plot_kwargs_forest2 is None:
        plot_kwargs_forest2 = {}

    kwargs_cloud_1 = {**base_plot_kwargs, **_point_cloud_defaults(forest1, vertex_size=3), **plot_kwargs_forest1}
    kwargs_cloud_2 = {**base_plot_kwargs, **_point_cloud_defaults(forest2, vertex_size=3), **plot_kwargs_forest2}

    if "title" not in plot_kwargs_forest1:
        plot_kwargs_forest1["title"] = "Acitve Loops"
    if "title" not in plot_kwargs_forest2:
        plot_kwargs_forest2["title"] = "Acitve Loops"

    # --- 6) Helpers to draw a single frame -----------------------------------
    def _draw_frame_at_time(t: float):
        # First forest
        ax_cloud_1.clear()
        forest1.plot_at_filtration(filt_val=t, ax=ax_cloud_1, **kwargs_cloud_1)
        _add_radius_box(ax_cloud_1, rf"${_filtration_symbol(forest1)} = {t:.3g}$", radius_box_corner)

        # Second forest
        ax_cloud_2.clear()
        forest2.plot_at_filtration(filt_val=t, ax=ax_cloud_2, **kwargs_cloud_2)
        _add_radius_box(ax_cloud_2, rf"${_filtration_symbol(forest2)} = {t:.3g}$", radius_box_corner)

    def _align_barcode_axes():
        # Make barcode axes have same left/right as their corresponding cloud axes
        for cloud_ax, bar_ax in ((ax_cloud_1, ax_bar_1), (ax_cloud_2, ax_bar_2)):
            cloud_pos = cloud_ax.get_position()
            bar_pos = bar_ax.get_position()
            # Keep bar's vertical position/height, but match horizontal start and width
            bar_ax.set_position([
                cloud_pos.x0,    # left
                bar_pos.y0,      # bottom
                cloud_pos.width, # width
                bar_pos.height,  # height
            ])

    # --- 7) Animation callbacks -----------------------------------------------
    def init():
        _draw_frame_at_time(frame_times[0])
        _align_barcode_axes()
        t_init = frame_times[0]
        barcode_line_1.set_xdata([t_init, t_init])
        barcode_line_2.set_xdata([t_init, t_init])
        return []

    def update(frame_idx: int):
        t = frame_times[frame_idx]
        _draw_frame_at_time(t)
        _align_barcode_axes()
        barcode_line_1.set_xdata([t, t])
        barcode_line_2.set_xdata([t, t])
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        init_func=init,
        blit=False,
    )

    # --- 8) Optional save to disk ---------------------------------------------
    if filename is not None:
        _save_paired_animation(anim, filename, fps=fps, dpi=dpi)

    return anim, fig
