"""Forest animation dispatch and compatibility argument translation."""

from typing import Any, Literal, Optional
import numpy as np
import warnings

def animate_filtration(
    self,
    filename: Optional[str] = None,
    format: Optional[Literal["mp4", "html"]] = None,
    with_barcode: bool = False,
    fps: int = 20,
    frames: int = 200,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    coloring: Literal["forest", "bars"] = "forest",
    signed: bool = False,
    show_cycles: bool = True,
    show_complex: bool = True,
    dpi: int = 200,
    figsize: Optional[tuple[float, float]] = None,
    pixel_size: Optional[tuple[int, int]] = None,
    panel_width_ratios: tuple[float, float] = (1.0, 1.0),
    panel_spacing: float = 0.08,
    figure_margins: Optional[dict[str, float]] = None,
    filtration_kwargs: Optional[dict] = None,
    barcode_kwargs: Optional[dict] = None,
    camera_mode: Literal["fixed", "orbit"] = "fixed",
    # Deprecated aliases (kept for backward compatibility)
    cloud_figsize: Optional[tuple[float, float]] = None,
    total_figsize: Optional[tuple[float, float]] = None,
    plot_kwargs: Optional[dict] = None,
    alpha_digits: Optional[int] = None,
    radius_box_corner: Literal["upper right", "upper left", "lower right", "lower left"] = "upper right",
):
    """
        Animate the filtration with automatic 2D/3D backend dispatch.

        Dispatch rules:
        - 2D: uses the existing matplotlib animation pipeline.
        - 3D + MP4: uses a matplotlib frame renderer and ffmpeg assembly.
        - 3D + HTML: uses the existing Plotly interactive pipeline.

        Parameters
        ----------
        filename : str | None
            Output path. If omitted, a default filename is chosen for exports.
        format : {"mp4", "html"} | None
            Output format. If None, inferred from ``filename``; defaults to
            ``"mp4"`` when not inferable.
        with_barcode : bool
            If True, show a barcode panel with a moving filtration marker.
        fps : int
            Frames per second for MP4 output.
        frames : int
            Number of sampled filtration values.
        t_min, t_max : float | None
            Optional filtration interval.
        coloring : {"forest", "bars"}
            Color map used for cycle overlays and barcode.
        signed : bool
            If False, cancel opposite-oriented duplicate simplices first.
        show_cycles : bool
            Whether to display active cycle representatives.
        show_complex : bool
            Whether to render complex geometry.
        dpi : int
            DPI used for matplotlib frame rendering and video export.
        figsize : tuple[float, float] | None
            Matplotlib figure size in inches.
            Defaults to (6, 6) without barcode and (10, 5) with barcode.
        pixel_size : tuple[int, int] | None
            Figure size in pixels. If provided, overrides ``figsize``.
        panel_width_ratios : tuple[float, float]
            Relative widths of cloud/barcode panels when ``with_barcode=True``.
        panel_spacing : float
            Horizontal spacing between cloud and barcode panels.
        figure_margins : dict[str, float] | None
            Outer figure margins with keys ``left``, ``right``, ``bottom``,
            and ``top`` passed to ``subplots_adjust``.
        filtration_kwargs : dict | None
            Extra kwargs forwarded to ``plot_at_filtration`` (except ``ax``,
            ``show`` and ``filt_val``). For rendering controls, use top-level
            ``show_cycles``/``show_complex``; keep this dict for values like
            ``vertex_size`` and 3D style values such as
            ``style_3d={'camera_eye': ..., 'complex_face_alpha': ..., 'cycle_face_alpha': ...}``.
            Backward-compatible keys are translated before forwarding:
            ``fill_triangles`` to ``show_complex``, ``remove_double_edges`` to
            ``signed``, ``linewidth_filt`` and ``linewidth_cycle`` to 2D style
            widths, ``show_orientation_arrows`` to 2D style, and
            ``camera_eye``/``remove_axes`` to 3D style.
        barcode_kwargs : dict | None
            Extra kwargs forwarded to barcode plotting for matplotlib paths.
            For animations with a barcode panel, the generic helper manages
            ``ax`` and uses the top-level ``coloring`` for both panels.
        camera_mode : {"fixed", "orbit"}
            3D camera mode.
        cloud_figsize, total_figsize : tuple[float, float] | None
            Deprecated aliases for ``figsize``. ``total_figsize`` applies when
            ``with_barcode=True``; ``cloud_figsize`` applies otherwise.
        plot_kwargs : dict | None
            Deprecated alias for ``filtration_kwargs``.
        radius_box_corner : {"upper right", "upper left", "lower right", "lower left"}
            Radius label corner for Matplotlib animations (default: "upper right").
            Uses an 8 pt inset. Does not affect the interactive 3D HTML output.
        alpha_digits : int | None
            Number of digits shown in the filtration value overlay (matplotlib paths).

        Returns
        -------
        object
            - 2D path: ``(anim, fig)`` from matplotlib.
            - 3D MP4 path: output filename string.
            - 3D HTML path: Plotly figure.
        """
    if self.diff_only_mode:
        raise ValueError(
            "Animations are unavailable with diff_only_mode=True because "
            "reconstructing cycles for every frame can be slow. "
            "Build the forest with diff_only_mode=False to animate it."
        )
    from pathlib import Path
    from persforest.plotting.animation.filtration import _animate_filtration_generic
    from persforest.plotting.animation._layout import _validate_radius_box_corner

    _validate_radius_box_corner(radius_box_corner)

    out_format = format
    if out_format is None and filename is not None:
        suffix = Path(filename).suffix.lower()
        if suffix in {".mp4", ".html"}:
            out_format = suffix[1:]
    if out_format is None:
        out_format = "mp4"
    if out_format not in {"mp4", "html"}:
        raise ValueError(f"Unsupported animation format: {out_format!r}. Use 'mp4' or 'html'.")
    if filename is not None and Path(filename).suffix.lower() not in {"", f".{out_format}"}:
        raise ValueError(f"Animation export requires a .{out_format} filename.")

    if plot_kwargs is not None:
        warnings.warn(
            "`plot_kwargs` is deprecated; use `filtration_kwargs`.",
            DeprecationWarning,
            stacklevel=3,
        )
        if filtration_kwargs is None:
            filtration_kwargs = dict(plot_kwargs)
        else:
            merged = dict(plot_kwargs)
            merged.update(filtration_kwargs)
            filtration_kwargs = merged

    if cloud_figsize is not None or total_figsize is not None:
        warnings.warn(
            "`cloud_figsize` and `total_figsize` are deprecated; use `figsize=(w, h)`.",
            DeprecationWarning,
            stacklevel=3,
        )
        if figsize is None:
            if with_barcode and total_figsize is not None:
                figsize = total_figsize
            elif (not with_barcode) and cloud_figsize is not None:
                figsize = cloud_figsize

    base_filtration_kwargs: dict[str, Any] = {}
    if filtration_kwargs is not None:
        base_filtration_kwargs.update(filtration_kwargs)

    style_2d = dict(base_filtration_kwargs.pop("style_2d", {}) or {})
    style_3d = dict(base_filtration_kwargs.pop("style_3d", {}) or {})

    if "fill_triangles" in base_filtration_kwargs:
        base_filtration_kwargs.setdefault(
            "show_complex",
            bool(base_filtration_kwargs.pop("fill_triangles")),
        )

    if "remove_double_edges" in base_filtration_kwargs:
        base_filtration_kwargs.setdefault(
            "signed",
            not bool(base_filtration_kwargs.pop("remove_double_edges")),
        )

    if "linewidth_filt" in base_filtration_kwargs:
        style_2d.setdefault(
            "complex_edge_width",
            float(base_filtration_kwargs.pop("linewidth_filt")),
        )
    if "linewidth_cycle" in base_filtration_kwargs:
        style_2d.setdefault(
            "cycle_edge_width",
            float(base_filtration_kwargs.pop("linewidth_cycle")),
        )

    if "show_orientation_arrows" in base_filtration_kwargs:
        style_2d.setdefault(
            "show_orientation_arrows",
            bool(base_filtration_kwargs.pop("show_orientation_arrows")),
        )

    if "camera_eye" in base_filtration_kwargs:
        style_3d.setdefault("camera_eye", base_filtration_kwargs.pop("camera_eye"))
    if "remove_axes" in base_filtration_kwargs:
        style_3d.setdefault(
            "remove_axes",
            bool(base_filtration_kwargs.pop("remove_axes")),
        )

    base_filtration_kwargs["signed"] = bool(signed)
    base_filtration_kwargs["show_cycles"] = bool(show_cycles)
    base_filtration_kwargs["show_complex"] = bool(show_complex)

    if style_2d:
        base_filtration_kwargs["style_2d"] = style_2d
    if style_3d:
        base_filtration_kwargs["style_3d"] = style_3d

    if self.dim == 2:
        if out_format == "html":
            raise ValueError("HTML animation export is only implemented for ambient dimension 3.")

        resolved_filename = filename
        if resolved_filename is not None and Path(resolved_filename).suffix.lower() == "":
            resolved_filename = f"{resolved_filename}.mp4"

        filtration_panel_kwargs = dict(base_filtration_kwargs)
        barcode_panel_kwargs = {}
        if barcode_kwargs is not None:
            barcode_panel_kwargs.update(barcode_kwargs)
        t_max_2d = t_max
        if t_max_2d is None:
            finite_deaths = [float(bar.death) for bar in self.barcode if np.isfinite(float(bar.death))]
            if finite_deaths:
                t_max_2d = max(finite_deaths)
            else:
                finite_filtration = [float(f) for _, f in self.filtration if np.isfinite(float(f))]
                if finite_filtration:
                    t_max_2d = max(finite_filtration)

        return _animate_filtration_generic(
            self,
            filename=resolved_filename,
            fps=fps,
            frames=frames,
            coloring=coloring,
            with_barcode=with_barcode,
            t_min=t_min,
            t_max=t_max_2d,
            dpi=dpi,
            figsize=figsize,
            pixel_size=pixel_size,
            panel_width_ratios=panel_width_ratios,
            panel_spacing=panel_spacing,
            figure_margins=figure_margins,
            filtration_kwargs=filtration_panel_kwargs,
            barcode_kwargs=barcode_panel_kwargs,
            alpha_digits=alpha_digits,
            radius_box_corner=radius_box_corner,
        )

    if self.dim == 3:
        if out_format == "html":
            html_width = None
            html_height = None
            if pixel_size is not None:
                if len(pixel_size) != 2:
                    raise ValueError("pixel_size must be a 2-tuple (width, height).")
                html_width = int(pixel_size[0])
                html_height = int(pixel_size[1])
            elif figsize is not None:
                html_width = int(round(float(figsize[0]) * float(dpi)))
                html_height = int(round(float(figsize[1]) * float(dpi)))

            filtration_panel_kwargs = dict(base_filtration_kwargs)
            style_3d = dict(filtration_panel_kwargs.get("style_3d", {}) or {})

            fig = self.plot_filtration_interactive(
                coloring=coloring,
                show_cycles=filtration_panel_kwargs.get("show_cycles", True),
                signed=filtration_panel_kwargs["signed"],
                filt_max=t_max,
                show_complex=filtration_panel_kwargs.get("show_complex", True),
                complex_opacity=float(style_3d.get("complex_face_alpha", 0.20)),
                cycle_opacity=float(style_3d.get("cycle_face_alpha", 0.55)),
                resolution=frames,
                show=False,
                vertex_size=float(filtration_panel_kwargs.get("vertex_size", 3.0)),
                width=html_width,
                height=html_height,
            )
            resolved_filename = filename if filename is not None else "filtration_animation_3d.html"
            if Path(resolved_filename).suffix.lower() == "":
                resolved_filename = f"{resolved_filename}.html"
            fig.write_html(str(resolved_filename))
            return fig

        resolved_filename = filename if filename is not None else "filtration_animation_3d.mp4"
        if Path(resolved_filename).suffix.lower() == "":
            resolved_filename = f"{resolved_filename}.mp4"

        filtration_panel_kwargs = {"camera_mode": camera_mode, **base_filtration_kwargs}

        barcode_panel_kwargs = {}
        if barcode_kwargs is not None:
            barcode_panel_kwargs.update(barcode_kwargs)

        t_max_3d = t_max
        if t_max_3d is None:
            finite_deaths = [float(bar.death) for bar in self.barcode if np.isfinite(float(bar.death))]
            if finite_deaths:
                t_max_3d = max(finite_deaths)
            else:
                finite_filtration = [float(f) for _, f in self.filtration if np.isfinite(float(f))]
                if finite_filtration:
                    t_max_3d = max(finite_filtration)

        return _animate_filtration_generic(
            self,
            filename=resolved_filename,
            with_barcode=with_barcode,
            fps=fps,
            frames=frames,
            t_min=t_min,
            t_max=t_max_3d,
            coloring=coloring,
            dpi=dpi,
            figsize=figsize,
            pixel_size=pixel_size,
            panel_width_ratios=panel_width_ratios,
            panel_spacing=panel_spacing,
            figure_margins=figure_margins,
            filtration_kwargs=filtration_panel_kwargs,
            barcode_kwargs=barcode_panel_kwargs,
            alpha_digits=alpha_digits,
            radius_box_corner=radius_box_corner,
        )

    raise ValueError("animate_filtration is only implemented for ambient dimensions 2 and 3.")

def animate_barcode_measurement(
            self,
            cycle_func,
            signed: bool = True,
            bar = None,
            *args,
            **kwargs,
    ):
    """
        Animate the filtration together with a barcode measurement.

        This is the animated analogue of :meth:`plot_barcode_measurement`.
        The left panel shows :meth:`plot_at_filtration` at a moving
        filtration value, while the right panel shows the associated
        barcode measurement with a vertical line indicating the current
        filtration value.

        Parameters
        ----------
        cycle_func :
            Callable ``(chain, point_cloud) -> float`` that assigns a scalar
            to each cycle representative associated to the chosen bar.
            See :meth:`plot_barcode_measurement` for details.
        signed : bool, optional
            If ``True`` (default), ``cycle_func`` is called on the signed
            cycle representatives. If ``False``, the chains are first
            passed through ``chain.unsigned()``.
        bar :
            Bar to use for the measurement. If ``None``, the method uses
            ``self.max_bar()`` inside the generic helper.
        *args, **kwargs :
            Forwarded to
            :func:`persforest.plotting.animation.measurements.animate_barcode_measurement_generic`.
            Typical keyword arguments include ``filename``, ``fps``,
            ``frames``, ``t_min``, ``t_max``, ``dpi``, ``total_figsize``,
            ``filtration_kwargs`` and ``measurement_kwargs``.
            ``filtration_kwargs`` is copied and normalized before forwarding:
            ``fill_triangles``, ``remove_double_edges``, ``linewidth_filt``,
            ``linewidth_cycle``, ``show_orientation_arrows``, ``camera_eye``
            and ``remove_axes`` are translated to the current
            ``plot_at_filtration`` style keys. ``plot_kwargs`` is no longer
            accepted.

        Returns
        -------
        anim, fig :
            A pair ``(anim, fig)`` where ``anim`` is a
            :class:`matplotlib.animation.FuncAnimation` and ``fig`` is the
            underlying figure.
        """
    if self.diff_only_mode:
        raise ValueError(
            "Animations are unavailable with diff_only_mode=True because "
            "reconstructing cycles for every frame can be slow. "
            "Build the forest with diff_only_mode=False to animate it."
        )
    from persforest.plotting.animation.measurements import animate_barcode_measurement_generic
    from copy import deepcopy

    kwargs = dict(kwargs)
    if "plot_kwargs" in kwargs:
        raise TypeError(
            "`plot_kwargs` is no longer supported; use `filtration_kwargs`."
        )
    raw_filtration_kwargs = kwargs.get("filtration_kwargs", None)
    if raw_filtration_kwargs is not None:
        if not isinstance(raw_filtration_kwargs, dict):
            raise TypeError("filtration_kwargs must be a dict when provided.")
        filtration_kwargs = deepcopy(raw_filtration_kwargs)
    else:
        filtration_kwargs = {}

    style_2d = dict(filtration_kwargs.pop("style_2d", {}) or {})
    style_3d = dict(filtration_kwargs.pop("style_3d", {}) or {})

    if "fill_triangles" in filtration_kwargs:
        filtration_kwargs["show_complex"] = bool(filtration_kwargs.pop("fill_triangles"))

    if "remove_double_edges" in filtration_kwargs:
        filtration_kwargs["signed"] = not bool(filtration_kwargs.pop("remove_double_edges"))

    if "linewidth_filt" in filtration_kwargs:
        style_2d["complex_edge_width"] = float(filtration_kwargs.pop("linewidth_filt"))
    if "linewidth_cycle" in filtration_kwargs:
        style_2d["cycle_edge_width"] = float(filtration_kwargs.pop("linewidth_cycle"))

    if "show_orientation_arrows" in filtration_kwargs:
        style_2d["show_orientation_arrows"] = bool(
            filtration_kwargs.pop("show_orientation_arrows")
        )

    if "camera_eye" in filtration_kwargs:
        style_3d["camera_eye"] = filtration_kwargs.pop("camera_eye")
    if "remove_axes" in filtration_kwargs:
        style_3d["remove_axes"] = bool(filtration_kwargs.pop("remove_axes"))

    # `alpha_digits` is not a `plot_at_filtration` kwarg in this animation path.
    filtration_kwargs.pop("alpha_digits", None)

    if style_2d:
        filtration_kwargs["style_2d"] = style_2d
    if style_3d:
        filtration_kwargs["style_3d"] = style_3d

    kwargs["filtration_kwargs"] = filtration_kwargs

    if signed:
        def _cycle_value(chain, point_cloud):
            # `chain` is a SignedChain
            return float(cycle_func(chain, point_cloud))
    else:
        def _cycle_value(chain, point_cloud):
            # `chain` is a SignedChain
            return float(cycle_func(chain.unsigned(), point_cloud))

    return animate_barcode_measurement_generic(
        forest=self,
        cycle_func=_cycle_value,
        bar=bar,
        *args,
        **kwargs,
    )
