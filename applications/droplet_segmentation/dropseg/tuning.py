"""Interactive views of the droplet pipeline, one per step, on a window of the image.

``Tuner`` holds the current ``Parameters`` and a ``stages.Stages``
cache of the window. ``Tuner.step(name)`` shows the view of one step with
sliders for its parameters; moving a slider updates ``Tuner.parameters``, so
the later steps start from the values chosen for the earlier ones. Only the
changed step and the later ones are recomputed. ``Tuner.compare(name, values)``
draws the view of the step owning a parameter for several of its values, side
by side.

Coordinates are fine pixels of the segmented window (with its margin);
``Stages.view`` gives the window itself.
"""

from dataclasses import replace
import io
import math
import time

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

from .parameters import STEPS
from .parameters import Parameters
from .pipeline import peak
from .pipeline import search_start
from .plots import draw_cycle
from .plots import outline_segments
from .plots import show_image
from .plots import show_labels
from .plots import zoom
from .segments import outline_mask
from .stages import Stages

_SLIDER = dict(continuous_update=False, readout_format=".2f", style={"description_width": "120px"},
               layout=widgets.Layout(width="420px"))
_OFF = 0.0  # slider value meaning None, for the optional steps

# One control per parameter; None-able parameters use 0 for None.
CONTROLS = {
    "upsample": lambda value: widgets.Dropdown(options=[1, 2, 4], value=value, description="upsample",
                                               style=_SLIDER["style"]),
    "kernel_width": lambda value: widgets.FloatSlider(value, min=0.2, max=0.7, step=0.05, **_SLIDER),
    "min_bar_length": lambda value: widgets.FloatSlider(value, min=0.02, max=0.5, step=0.01, **_SLIDER),
    "split_level": lambda value: widgets.FloatSlider(value, min=0.05, max=0.8, step=0.01, **_SLIDER),
    "min_level": lambda value: widgets.FloatSlider(value, min=0.0, max=0.5, step=0.01, **_SLIDER),
    "convexity_tol": lambda value: widgets.Dropdown(options=[("off", None), 0.3, 0.4, 0.5, 0.6], value=value,
                                                    style=_SLIDER["style"]),
    "birth_offset": lambda value: widgets.FloatSlider(value, min=0.0, max=0.5, step=0.05, **_SLIDER),
    "split_lobes": lambda value: widgets.Checkbox(value),
    "lobe_waist": lambda value: widgets.FloatSlider(value, min=0.3, max=1.0, step=0.05, **_SLIDER),
    "smooth_radius": lambda value: widgets.FloatSlider(value or _OFF, min=0.0, max=2.0, step=0.05, **_SLIDER),
    "min_area": lambda value: widgets.FloatSlider(value, min=0.0, max=20.0, step=0.5, **_SLIDER),
}
_OPTIONAL = ("smooth_radius",)
# The controls shown with each step; the derived parameters are left out unless asked for.
MAIN_CONTROLS = {
    "peaks": ("min_bar_length",),
    "branches": ("split_level",),
    "outlines": ("min_level",),
    "lobes": ("split_lobes", "lobe_waist"),
    "flooding": (),
    "smoothing": ("smooth_radius",),
    "filter": ("min_area",),
}


def _png(fig) -> bytes:
    """The figure as PNG bytes on a white background, cropped to its content, and closed.

    The views are shown as images rather than through the inline backend,
    which would draw a figure made while the cell runs a second time, below
    the widget, and leaves transparent margins dark in a dark editor theme.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _show(image: widgets.Image, png: bytes):
    """Put PNG bytes into an image widget at their natural width (read from the PNG header)."""
    image.value = png
    if png:
        image.layout.width = f"{int.from_bytes(png[16:20], 'big')}px"


def step_of(name: str) -> str:
    """The step reading parameter ``name``."""
    return next(step for step, fields in STEPS.items() if name in fields)


# Colors that stand out on a grayscale image, with a black outline where they are thin.
_DROPPED = "#ff1a1a"
_OUTLINE = [patheffects.withStroke(linewidth=3.2, foreground="black")]


def _color(index: int) -> str:
    # tab10 without its gray (a background outline), pink, and red (rejected lobes, dropped segments)
    return ("C0", "C1", "C2", "C4", "C5", "C8", "C9")[index % 7]


def _window(ax, stages: Stages, parameters: Parameters):
    rows, columns = stages.view(parameters)
    ax.set_xlim(columns.start, columns.stop)
    ax.set_ylim(rows.stop, rows.start)


def _in_view(pixel, view, margin=0):
    return all(s.start - margin <= x < s.stop + margin for x, s in zip(pixel, view))


def _base(ax, stages, parameters, vmax, title):
    show_image(ax, stages.image(parameters), title, vmax)
    _window(ax, stages, parameters)


# The views: one per step, each drawing on one axes.


def view_peaks(ax, stages: Stages, parameters: Parameters, vmax=None):
    """Step 1: kept peaks (green) and bars at least a quarter as long (red crosses)."""
    forest, kept = stages.forest(parameters), set(stages.peaks(parameters))
    view = stages.view(parameters)
    near = [bar for bar in forest.barcode
            if math.isfinite(bar.death) and bar.lifespan() >= parameters.min_bar_length / 4]
    shown = [(bar, peak(bar, forest)) for bar in near]
    shown = [(bar, pixel) for bar, pixel in shown if _in_view(pixel, view)]
    n_kept = sum(bar in kept for bar, _ in shown)
    _base(ax, stages, parameters, vmax, f"step 1: {n_kept} peaks (green), {len(shown) - n_kept} shorter bars (red)")
    for bar, (row, column) in shown:
        if bar in kept:
            ax.plot(column + 0.5, row + 0.5, "o", color="#3ddc84", markersize=5, markeredgecolor="black",
                    markeredgewidth=0.5)
        else:
            ax.plot(column + 0.5, row + 0.5, "x", color="#ff5050", markersize=5)


def view_diagram(ax, stages: Stages, parameters: Parameters):
    """The bars of the window as (birth, death) with the cutoffs of steps 1 and 2."""
    forest, kept = stages.forest(parameters), set(stages.peaks(parameters))
    view = stages.view(parameters)
    bars = [bar for bar in forest.barcode if math.isfinite(bar.death) and bar.lifespan() >= 0.01
            and _in_view(peak(bar, forest), view)]
    births = np.array([bar.birth for bar in bars])
    deaths = np.array([bar.death for bar in bars])
    is_kept = np.array([bar in kept for bar in bars], dtype=bool)
    ax.scatter(births[~is_kept], deaths[~is_kept], s=6, color="0.6", label="shorter bars")
    ax.scatter(births[is_kept], deaths[is_kept], s=12, color="#2a9d5c", label="peaks")
    top = max(1.0, deaths.max() if len(deaths) else 1.0)
    ax.plot([0, top], [parameters.min_bar_length, top + parameters.min_bar_length], color="#ff5050", linewidth=1,
            label=f"min_bar_length {parameters.min_bar_length:.2f}")
    ax.axvline(parameters.split_level, color="C0", linewidth=1, linestyle="--",
               label=f"split_level {parameters.split_level:.2f}")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top * 1.02)
    ax.set_xlabel("birth (level where the peak separates)")
    ax.set_ylabel("death (peak brightness)")
    ax.set_title("bars with a peak in the window")
    ax.legend(fontsize=8, loc="lower right")


def view_branches(ax, stages: Stages, parameters: Parameters, vmax=None):
    """Step 2: droplets (dots) and peaks merged into the droplet they split off (diamonds of its color, joined by a line)."""
    forest = stages.forest(parameters)
    branches, outlined, late = stages.branches(parameters)
    view = stages.view(parameters)
    outlined = [bar for bar in outlined if _in_view(peak(bar, forest), view, margin=8)]
    late = [bar for bar in late if _in_view(peak(bar, forest), view)]
    _base(ax, stages, parameters, vmax, f"step 2: {len(outlined)} droplets (dots), {len(late)} merged peaks (diamonds)")
    colors = {bar: _color(index) for index, bar in enumerate(branches.bars)}
    marker = dict(markersize=7, markeredgewidth=1)
    for bar in late:
        row, column = peak(bar, forest)
        elder = branches.parent[bar]
        while elder is not None and elder in late:
            elder = branches.parent[elder]
        color = colors[elder] if elder is not None else "white"
        if elder is not None:
            elder_row, elder_column = peak(elder, forest)
            ax.plot([column + 0.5, elder_column + 0.5], [row + 0.5, elder_row + 0.5], color=color, linewidth=1.8,
                    path_effects=_OUTLINE)
        ax.plot(column + 0.5, row + 0.5, "D", color=color, markeredgecolor="black", **marker)
    for bar in outlined:
        row, column = peak(bar, forest)
        ax.plot(column + 0.5, row + 0.5, "o", color=colors[bar], markeredgecolor="white", **marker)


def search_starts(stages: Stages, parameters: Parameters) -> dict:
    """Where the outline search of every outlined bar starts, and why.

    Returns ``{bar: (start, reason)}`` with reason ``"min_level"``, ``"split"``
    (after a split of the bar below ``split_level``) or ``"birth offset"``.
    """
    settings = stages.settings(parameters)
    branches, outlined, _ = stages.branches(parameters)
    starts = {}
    for bar in outlined:
        start = search_start(branches, bar, settings)
        if start > parameters.min_level and math.isclose(start, bar.birth + settings.birth_offset):
            reason = "birth offset"
        elif start > parameters.min_level and start == branches.search_start(bar, until=settings.split_level):
            reason = "split"
        else:
            reason = "min_level"
        starts[bar] = (start, reason)
    return starts


def view_outlines(ax, stages: Stages, parameters: Parameters, vmax=None):
    """Step 3: the chosen outlines; where the birth offset raised the start, the outline at the birth in gray.

    Outlines are solid if taken at ``min_level``, dash-dotted if taken after
    a split, and dashed if the convexity search reached ``split_level``.
    """
    forest = stages.forest(parameters)
    selections = stages.outlines(parameters)
    view = stages.view(parameters)
    starts = search_starts(stages, parameters)
    shown = [bar for bar in selections if _in_view(peak(bar, forest), view, margin=8)]
    counts = {reason: sum(starts[bar][1] == reason for bar in shown) for reason in ("min_level", "split", "birth offset")}
    title = (f"step 3: {len(shown)} outlines from min_level {counts['min_level']}, after a split (-.) "
             f"{counts['split']}, birth offset {counts['birth offset']}")
    if parameters.convexity_tol is not None:
        title += f", {sum(selections[bar].fallback for bar in shown)} reached split_level (--)"
    _base(ax, stages, parameters, vmax, title)
    ax.title.set_fontsize(9)
    for index, bar in enumerate(shown):
        selection = selections[bar]
        _, reason = starts[bar]
        if reason == "birth offset":
            draw_cycle(ax, forest.cycle_for_bar_at(bar, bar.birth), forest.geometry, "0.8", linewidth=0.7)
        style = "-." if reason == "split" else "-"
        if selection.fallback and parameters.convexity_tol is not None:
            style = "--"
        draw_cycle(ax, selection.cycle, forest.geometry, _color(index), linewidth=1.3, linestyle=style)


def view_lobes(ax, stages: Stages, parameters: Parameters, vmax=None):
    """Step 4: the outlines after splitting; lobes are thick, dim lobes (removed after flooding) dashed red."""
    forest = stages.forest(parameters)
    selections, splits = stages.outlines(parameters), stages.lobes(parameters)
    view = stages.view(parameters)
    # The outlines in the order and colors of the step-3 view.
    shown = [bar for bar in selections if _in_view(peak(bar, forest), view, margin=8)]
    n_split = sum(bar in splits and bool(splits[bar].lobes) for bar in shown)
    n_dim = sum(bar in splits and bool(splits[bar].dim_lobes) for bar in shown)
    _base(ax, stages, parameters, vmax, f"step 4: {n_split} outlines split, {n_dim} with a dim lobe (red dashes)")
    for index, bar in enumerate(shown):
        selection, split = selections[bar], splits.get(bar)
        if split is not None and split.lobes:
            draw_cycle(ax, selection.cycle, forest.geometry, "white", linewidth=0.6)
            for lobe_index, lobe in enumerate(split.lobes):
                if any(lobe is dim for dim in split.dim_lobes):
                    draw_cycle(ax, lobe.cycle, split.points, _DROPPED, linewidth=1.6, linestyle="--")
                else:
                    draw_cycle(ax, lobe.cycle, split.points, _color(index + lobe_index), linewidth=2.2)
        else:
            draw_cycle(ax, selection.cycle, forest.geometry, _color(index), linewidth=1)


def _weakest_lobe(split):
    """The split-off lobe bar with the widest waist relative to its radius (the elder lobe has no split)."""
    split_off = [bar for bar in split.lobe_bars if split.branches.parent[bar] is not None] or split.lobe_bars
    return max(split_off, key=lambda bar: bar.birth / bar.death)


def lobe_gallery(stages: Stages, parameters: Parameters, vmax=None, columns=6, size=2.6, max_panels=12):
    """Every split of the window, the weakest first: the waist ratio s / rho and radius rho of the split-off lobe.

    Splits rejected because a lobe rises less than ``min_bar_length`` above the
    bar's birth (removed after the flooding) are red dashed, with the dimmest lobe's rise.
    """
    forest = stages.forest(parameters)
    view = stages.view(parameters)
    image = stages.image(parameters)
    splits = [(bar, split) for bar, split in stages.lobes(parameters).items()
              if split.lobes and _in_view(peak(bar, forest), view)]
    splits = sorted(splits, key=lambda item: -_weakest_lobe(item[1]).birth / _weakest_lobe(item[1]).death)[:max_panels]
    if not splits:
        return None
    rows = -(-len(splits) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * (size + 0.45)), squeeze=False)
    for ax, (bar, split) in zip(axes.ravel(), splits):
        show_image(ax, image, None, vmax)
        draw_cycle(ax, split.selection.cycle, forest.geometry, "white", linewidth=1)
        bright = [lobe for lobe in split.lobes if not any(lobe is dim for dim in split.dim_lobes)]
        for index, lobe in enumerate(bright):
            draw_cycle(ax, lobe.cycle, split.points, f"C{index}", linewidth=2)
        for lobe in split.dim_lobes:
            draw_cycle(ax, lobe.cycle, split.points, _DROPPED, linewidth=1.5, linestyle="--")
        zoom(ax, split.points, margin=3 * parameters.upsample)
        weakest = _weakest_lobe(split)
        title = f"s/ρ {weakest.birth / weakest.death:.2f}, ρ {weakest.death / parameters.upsample:.2f} px"
        if split.dim_lobes:
            rise = min(image[outline_mask(lobe.cycle, split.points, image.shape)].max(initial=0.0)
                       for lobe in split.dim_lobes) - bar.birth
            title += f"\ndim lobe (rises {rise:.2f}): removed"
        ax.set_title(title, fontsize=9, color=_DROPPED if split.dim_lobes else "black")
    for ax in axes.ravel()[len(splits):]:
        ax.axis("off")
    fig.suptitle("splits in the window, weakest first: split radius s (half the waist) over the radius ρ of the "
                 f"split-off lobe, ρ in original px; split if s/ρ ≤ lobe_waist = {parameters.lobe_waist}", fontsize=10)
    fig.tight_layout()
    return fig


def _labels(ax, stages, parameters, labels, vmax, title, edge_color="yellow"):
    view = stages.view(parameters)
    show_labels(ax, stages.image(parameters), labels, title, vmax=vmax, alpha=0.35)
    outline_segments(ax, labels, color=edge_color, linewidth=0.8)
    _window(ax, stages, parameters)
    return len(np.unique(labels[view])) - 1


def _count(labels, view):
    return len(np.unique(labels[view])) - 1


def view_outline_regions(ax, stages: Stages, parameters: Parameters, vmax=None):
    """Before step 5: the regions of the outlines and lobes."""
    result = stages.result(parameters)
    _labels(ax, stages, parameters, result.region_labels, vmax,
            f"before flooding: {_count(result.region_labels, stages.view(parameters))} regions")


def view_flooding(ax, stages: Stages, parameters: Parameters, vmax=None):
    """Step 5: the segments after flooding down to ``min_level``."""
    result = stages.result(parameters)
    _labels(ax, stages, parameters, result.grown_labels, vmax, f"step 5: grown to {parameters.min_level:.2f}: "
            f"{_count(result.grown_labels, stages.view(parameters))} segments\n(red: dim lobes removed)")
    ax.title.set_fontsize(10)
    outline_segments(ax, result.dimmed.astype(int), color=_DROPPED, linewidth=1.4)


def view_smoothing(ax, stages: Stages, parameters: Parameters, vmax=None):
    """Step 6: the segments after the alpha opening."""
    result = stages.result(parameters)
    radius = "off" if not parameters.smooth_radius else f"{parameters.smooth_radius:.2f} px"
    _labels(ax, stages, parameters, result.smoothed_labels, vmax,
            f"step 6: smoothed, radius {radius}: {_count(result.smoothed_labels, stages.view(parameters))} segments")
    if result.broken:
        ax.set_title(f"{ax.get_title()}\nwarning: {len(result.broken)} cut into large pieces, radius too large",
                     color=_DROPPED)
    ax.title.set_fontsize(10)


def view_filter(ax, stages: Stages, parameters: Parameters, vmax=None):
    """Step 7: the kept segments; those dropped for an area below ``min_area`` are outlined in red."""
    result = stages.result(parameters)
    dropped = np.where(result.labels == 0, result.smoothed_labels, 0)
    n_kept = _labels(ax, stages, parameters, result.labels, vmax, None)
    outline_segments(ax, dropped, color=_DROPPED, linewidth=1.6)
    title = f"step 7: {n_kept} kept, pieces dropped (red): area < {parameters.min_area:g} px"
    if result.broken:
        title += f"\nwarning: smoothing cut {len(result.broken)} segment(s) into several large pieces"
    ax.set_title(title, color=_DROPPED if result.broken else "black")


VIEWS = {
    "peaks": view_peaks,
    "branches": view_branches,
    "outlines": view_outlines,
    "lobes": view_lobes,
    "flooding": view_flooding,
    "smoothing": view_smoothing,
    "filter": view_filter,
}


class Tuner:
    """Step-by-step tuning of the parameters on a window of the image.

    Parameters
    ----------
    image : ndarray
        The whole image, original pixels, intensities in [0, 1].
    window : tuple of slice
        ``(rows, columns)`` of the window, original pixels.
    parameters : Parameters
        Starting values.
    vmax : float, optional
        Brightness shown as white.
    size : float
        Panel size in inches.
    """

    def __init__(self, image, window, parameters: Parameters = Parameters(), vmax=None, size=6.0):
        self.stages = Stages(image, window)
        self.parameters = parameters
        self.vmax = vmax
        self.size = size

    def figure(self, step: str, parameters: Parameters | None = None):
        """The figure of one step: its view, with the view of the step before it or a diagram for comparison."""
        parameters = parameters or self.parameters
        stages, vmax, size = self.stages, self.vmax, self.size
        if step == "lobes":
            fig, axes = plt.subplots(1, 2, figsize=(2 * size, size + 0.4), sharex=True, sharey=True)
            view_outlines(axes[0], stages, parameters, vmax)
            view_lobes(axes[1], stages, parameters, vmax)
        elif step in ("peaks", "branches"):
            fig, axes = plt.subplots(1, 2, figsize=(2 * size, size + 0.4), gridspec_kw={"width_ratios": [1, 0.85]})
            VIEWS[step](axes[0], stages, parameters, vmax)
            view_diagram(axes[1], stages, parameters)
        elif step in ("flooding", "smoothing", "filter"):
            before = {"flooding": view_outline_regions, "smoothing": view_flooding, "filter": view_smoothing}[step]
            fig, axes = plt.subplots(1, 2, figsize=(2 * size, size + 0.4), sharex=True, sharey=True)
            before(axes[0], stages, parameters, vmax)
            VIEWS[step](axes[1], stages, parameters, vmax)
        else:
            fig, ax = plt.subplots(figsize=(size * 1.3, size * 1.3 + 0.4))
            VIEWS[step](ax, stages, parameters, vmax)
        fig.tight_layout()
        return fig

    def step(self, step: str, extra: tuple[str, ...] = ()):
        """Sliders for the parameters of ``step`` (and ``extra`` ones) above its view.

        Every change is stored in ``self.parameters``. ``extra`` adds controls
        for the derived parameters, e.g. ``("lobe_waist",)``.
        """
        names = MAIN_CONTROLS[step] + tuple(extra)
        if step == "peaks" and "upsample" not in names:
            names = ("upsample", "kernel_width") + names
        controls = {name: CONTROLS[name](getattr(self.parameters, name)) for name in names}
        for name, control in controls.items():
            control.description = name
        # Shown at the figure's own pixel width, shrunk only if the notebook is narrower.
        view = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
        gallery = widgets.Image(format="png", layout=widgets.Layout(max_width="100%", height="auto"))
        status = widgets.HTML()

        def update(_=None):
            values = {name: control.value for name, control in controls.items()}
            for name in _OPTIONAL:
                if name in values and values[name] == _OFF:
                    values[name] = None
            try:
                parameters = replace(self.parameters, **values)
            except ValueError as error:
                status.value = f"<span style='color:#c00'>{error}</span>"
                return
            self.parameters = parameters
            clock = time.perf_counter()
            _show(view, _png(self.figure(step, parameters)))
            message = ""
            if step == "lobes":
                splits = lobe_gallery(self.stages, parameters, self.vmax) if parameters.split_lobes else None
                _show(gallery, b"" if splits is None else _png(splits))
                gallery.layout.display = None if splits is not None else "none"
                if parameters.split_lobes and splits is None:
                    message = "; no outline split in the window"
            status.value = f"{time.perf_counter() - clock:.1f} s{message}"

        for control in controls.values():
            control.observe(update, names="value")
        update()
        children = [*controls.values(), status, view] + ([gallery] if step == "lobes" else [])
        return widgets.VBox(children)

    def compare(self, name: str, values: list, columns: int = 4, size: float = 4.5):
        """The view of the step owning parameter ``name``, for each of ``values``; ``self.parameters`` otherwise."""
        step = step_of(name)
        if step not in VIEWS:  # upsample and kernel_width change the image itself: show the peaks
            step = "peaks"
        columns = min(columns, len(values))
        rows = -(-len(values) // columns)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * (size + 0.4)), squeeze=False)
        for ax, value in zip(axes.ravel(), values):
            parameters = replace(self.parameters, **{name: value})
            VIEWS[step](ax, self.stages, parameters, self.vmax)
            ax.set_title(f"{name} = {value}\n{ax.get_title()}", fontsize=9)
        for ax in axes.ravel()[len(values):]:
            ax.axis("off")
        fig.tight_layout()
        return fig
