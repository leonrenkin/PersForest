"""plotting.landscapes: mechanically extracted reference implementation."""

from ._dependencies import require_plotting

require_plotting()

from matplotlib.colors import Colormap
from persforest.landscapes.functions import StepFunctionData
from persforest.landscapes.profiles import CycleValueFunc
from persforest.landscapes.profiles import _build_step_function_data
from persforest.landscapes.family import MeasurementLandscapeFamily
from persforest.landscapes._validation import positive_integer
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

def plot_barcode_measurement_generic(
        forest,
        cycle_func: CycleValueFunc,
        bar=None,
        *,
        ax: Optional["matplotlib.axes.Axes"] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        label: Optional[str] = None,
        show_baseline: bool = True,
        show: bool = False,
        **kwargs,
    ) -> Tuple["matplotlib.axes.Axes", StepFunctionData]:
    """
    Plot the cycle measurement step function for a single bar.

    Parameters
    ----------
    forest :
        Forest-like object with ``barcode``, ``point_cloud`` and ``max_bar``.
    cycle_func : CycleValueFunc
        Callable (cycle_rep, point_cloud) -> float, used to assign a scalar
        value to each cycle representative of the bar.
    bar :
        Bar object from `forest.barcode`. If None, `forest.max_bar()` is used.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure and axis are created.
    x_range : (float, float), optional
        If given, set the x-limits to this range.
    y_range : (float, float), optional
        If given, set the y-limits to this range.
    title : str, optional
        Title for the axes. If None, a default title is constructed.
    label : str, optional
        Label used in the fallback title when ``title`` is omitted.
    show_baseline : bool, default True
        Whether to draw the baseline of the step function.
    show : bool, default False
        Whether to call `plt.show()` after plotting.
    **kwargs
        Forwarded to ``StepFunctionData.plot`` as line style options.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot.
    step_func : StepFunctionData
        The piecewise-constant function induced by `cycle_func`.
    """

    if bar is None:
        bar = forest.max_bar()

    step_func = _build_step_function_data(forest=forest, bar=bar,cycle_func=cycle_func)

    if ax is None:
        _, ax = plt.subplots()

    step_func.plot(x_range=x_range, y_range=y_range, ax= ax, title = title,show_baseline=show_baseline, **kwargs)

    if title is None:
        ax.set_title(f"{label} progression in max bar")

    if show:
        plt.show()

    return (ax, step_func)

def plot_landscape_family(
        family: MeasurementLandscapeFamily,
        ks: Optional[List[int]] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
        title: Optional[str] = None,
        show: bool = True,
        *,
        show_legend: Optional[bool] = None,
        linewidth: Optional[float] = None,
        cmap: Optional[Union[str, Colormap]] = None,
        higher_layers_on_top: bool = False,
    ):
    """
    Plot stored landscape samples from an explicit family result.

    Parameters
    ----------
    family : MeasurementLandscapeFamily
        Explicit result; no forest or label lookup is needed.
    ks : list of int, optional
        Which landscape levels to plot; defaults to all available.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new one is created if omitted.
    title : str, optional
        Custom plot title.
    show : bool
        If True, call ``plt.show()`` after plotting.
    show_legend : bool, optional
        Whether to show the legend. Defaults to showing it for fewer than
        10 plotted landscapes and hiding it otherwise.
    linewidth : float, optional
        Line width for the landscape plots. If omitted, Matplotlib's default
        line width is used.
    cmap : str or matplotlib.colors.Colormap | None
        Sequential colormap used across the selected landscape levels.
        Colors follow increasing ``k``.
        Named reversed colormaps such as ``"viridis_r"`` are supported.
        If omitted, Matplotlib's default color cycle is used.
    higher_layers_on_top : bool
        If True, landscape levels with larger ``k`` are drawn in front of
        levels with smaller ``k``. The default keeps lower levels, including
        the first landscape, in front.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """

    if not isinstance(family, MeasurementLandscapeFamily):
        raise TypeError("family must be MeasurementLandscapeFamily")
    if ks is None:
        ks = sorted(family.landscapes)
    else:
        ks = [positive_integer(k, "landscape level") for k in ks]
    family.evaluate_on_grid(levels=ks)  # Validate before creating any artists.

    if ax is None:
        fig, ax = plt.subplots()

    colors_by_k = {}
    if cmap is not None:
        colormap = matplotlib.colormaps.get_cmap(cmap)
        ordered_ks = sorted(set(ks))
        if len(ordered_ks) == 1:
            color_positions = np.array([0.5])
        else:
            color_positions = np.linspace(0.0, 1.0, len(ordered_ks))
        colors_by_k = dict(zip(ordered_ks, colormap(color_positions)))

    zorders = {
        k: zorder
        for zorder, k in enumerate(
            sorted(ks, reverse=not higher_layers_on_top),
            start=1,
        )
    }
    for k in ks:
        plf = family.landscapes[k]
        plot_kwargs = {
            "label": fr"$\lambda_{{{k}}}$",
            "zorder": zorders[k],
        }
        if linewidth is not None:
            plot_kwargs["linewidth"] = linewidth
        if cmap is not None:
            plot_kwargs["color"] = colors_by_k[k]
        ax.plot(plf.xs, plf.ys, **plot_kwargs)

    ax.set_xlabel("filtration value")
    ax.set_ylabel("landscape value")
    if title is None:
        title = f"Measurement landscapes of {family.label}"
    ax.set_title(title)
    if show_legend is None:
        show_legend = len(ks) < 10
    if show_legend:
        ax.legend()
    ax.set_ylim(bottom=0)

    if show:
        plt.show()

    return ax

def plot_landscape_comparison(
    families: List[MeasurementLandscapeFamily], k: int = 1,
    ax: Optional["matplotlib.axes.Axes"] = None,
    labels: Optional[List[str]] = None, title: Optional[str] = None, **kwargs,
):
    """Compare one stored level across explicit family results.

    Parameters
    ----------
    families : list of MeasurementLandscapeFamily
        Results to compare, including across sources or measurements.
    k : int, default 1
        Positive computed rank. Missing ranks raise rather than skipping curves.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; created after input validation when omitted.
    labels : list of str, optional
        One label per family, defaulting to each result label.
    title : str, optional
        Custom title.
    **kwargs
        Line options forwarded to ax.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing one line per input family.
    """
    k = positive_integer(k, "k")
    families = list(families)
    if any(not isinstance(family, MeasurementLandscapeFamily) for family in families):
        raise TypeError("families must contain MeasurementLandscapeFamily results")
    labels = [family.label for family in families] if labels is None else list(labels)
    if len(labels) != len(families):
        raise ValueError("labels must contain exactly one label per family")
    for family in families:
        family.evaluate_on_grid(levels=[k])
    if ax is None:
        _, ax = plt.subplots()
    for family, label in zip(families, labels):
        ax.plot(family.x_grid, family.landscapes[k].ys, label=label, **kwargs)
    ax.set_xlabel("filtration value")
    ax.set_ylabel(fr"$\lambda_{k}$")
    ax.set_title(fr"Comparison of $\lambda_{k}$" if title is None else title)
    if families:
        ax.legend()
    return ax


def _plot_step_function(
    self,
    ax: Optional[matplotlib.axes.Axes] = None,
    *,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    show_baseline: bool = False,
    title: Optional[str] = None,
    baseline_kwargs: Optional[Dict[str, Any]] = None,
    **line_kwargs: Any,
) -> matplotlib.axes.Axes:
    """
        Plot the step function represented by this object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, a new figure and axis are created.
        x_range : (float, float), optional
            Restrict the x-axis to the given range; defaults to the stored domain.
        y_range : (float, float), optional
            Restrict the y-axis to the given range; auto-scales when omitted.
        show_baseline : bool, default False
            Whether to draw a dashed line at `baseline`.
        title : str, optional
            Title for the plot.
        baseline_kwargs : dict, optional
            Extra keyword arguments forwarded to the baseline line plot.
        **line_kwargs :
            Keyword arguments forwarded to `ax.plot` for the step curve.

        Returns
        -------
        matplotlib.axes.Axes
            The axis with the rendered step function.
        """
    if ax is None:
        _, ax = plt.subplots()

    starts = np.asarray(self.starts, dtype=float).ravel()
    ends = np.asarray(self.ends, dtype=float).ravel()
    vals = np.asarray(self.vals, dtype=float).ravel()

    if not (starts.size == ends.size == vals.size):
        raise ValueError("starts, ends and vals must have the same length.")

    # Determine plotting domain
    if x_range is not None:
        xmin, xmax = x_range
    else:
        xmin, xmax = self.domain

    if xmax <= xmin:
        raise ValueError("x_range / domain must satisfy xmin < xmax.")

    # Build breakpoints inside [xmin, xmax]
    xs_candidates = [xmin, xmax]
    xs_candidates.extend(starts.tolist())
    xs_candidates.extend(ends.tolist())
    xs_unique = sorted({x for x in xs_candidates if xmin <= x <= xmax})

    if len(xs_unique) < 2:
        xs_unique = [xmin, xmax]

    # Use the same validated interval semantics as numerical evaluation.
    edges = np.asarray(xs_unique, dtype=float)
    mids = edges[:-1] + (edges[1:] - edges[:-1]) / 2
    ys_plot = np.repeat(self.eval_on_grid(mids), 2)
    xs_plot = np.column_stack((edges[:-1], edges[1:])).ravel()

    # Defaults that can be overridden by **line_kwargs
    ax.plot(xs_plot, ys_plot, **line_kwargs)

    if show_baseline:
        bl_kwargs = {
            "linewidth": 1.0,
            "linestyle": "--",
        }
        if baseline_kwargs is not None:
            bl_kwargs.update(baseline_kwargs)
        ax.hlines(self.baseline, xmin, xmax, **bl_kwargs)

    # ax.grid(True, alpha=0.3)

    ax.set_xlim(xmin, xmax)
    if y_range is not None:
        ax.set_ylim(*y_range)

    if title is not None:
        ax.set_title(title)

    return ax
