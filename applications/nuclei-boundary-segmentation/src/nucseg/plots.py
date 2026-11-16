"""Matplotlib figures of 2D and 3D segmentations, in µm.

A nucleus has the same color in every figure that colors by label
(``outline_color``), so 2D overviews and 3D sections can be compared.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ._grid import bounding_box
from .io import Stack
from .segment2d import Region
from .segment2d import Segmentation
from .segment2d import by_label
from .segment2d import reference_image
from .segment3d import Segmentation3D

# tab10 without its grey, which disappears on grayscale images
OUTLINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]


def outline_color(label: int) -> str:
    """Color of nucleus ``label`` (1, 2, ...)."""
    return OUTLINE_COLORS[(label - 1) % len(OUTLINE_COLORS)]


def _display_range(image: NDArray) -> tuple[float, float]:
    low, high = np.percentile(image, [0.5, 99.8])
    return float(low), float(high)


def _contour(ax, mask: NDArray[np.bool_], um_per_bin: float, color, linewidth: float = 1.5,
             linestyle: str = "-", offset: tuple[int, int] = (0, 0)) -> None:
    """Outline of a binned mask, pixel centres at ``(index + 0.5) * um_per_bin``."""
    h, w = mask.shape
    xs = (offset[1] + np.arange(w) + 0.5) * um_per_bin
    ys = (offset[0] + np.arange(h) + 0.5) * um_per_bin
    ax.contour(xs, ys, mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth, linestyles=linestyle)


def _centroid_um(region: Region, shape: tuple[int, int], um_per_bin: float) -> tuple[float, float]:
    rr, cc = np.unravel_index(region.pixels, shape)
    return (cc.mean() + 0.5) * um_per_bin, (rr.mean() + 0.5) * um_per_bin


def _scale_bar(ax, length_um: float, extent: tuple[float, float, float, float]) -> None:
    x1, y1 = extent[1], extent[2]
    pad = 0.04 * x1
    ax.plot([x1 - pad - length_um, x1 - pad], [y1 - pad] * 2, color="white", linewidth=4, solid_capstyle="butt")
    ax.text(x1 - pad - length_um / 2, y1 - pad - 0.015 * y1, f"{length_um:g} µm", color="white",
            ha="center", va="bottom", fontsize=11)


# ----- 2D -----

def plot_regions(ax, segmentation: Segmentation, regions: list[Region] | None = None, *,
                 image: NDArray | None = None, annotate: bool = True, title: str | None = None,
                 color: str | None = None, linewidth: float = 1.2) -> None:
    """Draw region outlines on a prepared projection, in µm.

    Without ``color``, regions reaching the image edge are drawn in magenta,
    other merged regions (more than one piece) in orange, and other single
    loops in lime. Labels are numbered as in ``Segmentation.labels`` of the
    same regions. The default image is the mean of the prepared projections.
    """
    regions = segmentation.nuclei if regions is None else by_label(regions)
    image = reference_image(segmentation) if image is None else image
    h, w = image.shape
    s = segmentation.um_per_bin
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, extent=(0, w * s, h * s, 0))
    for label, region in enumerate(regions, start=1):
        edge = color or ("magenta" if region.border else "orange" if region.pieces > 1 else "lime")
        _contour(ax, region.mask((h, w)), s, edge, linewidth)
        if annotate:
            x, y = _centroid_um(region, (h, w), s)
            ax.text(x, y, str(label), color="yellow", fontsize=9, ha="center", va="center", weight="bold",
                    clip_on=True)
    ax.set_xlim(0, w * s)
    ax.set_ylim(h * s, 0)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    if title is not None:
        ax.set_title(title)


def figure_overview(segmentation: Segmentation, stack: Stack, scale_bar_um: float = 20.0,
                    title: str | None = None):
    """Outlines and labels on the maximum projection and on the mean of the slices used.

    Shown at full resolution with a scale bar. Every outline comes from its
    own slice window (``z_window`` in the table), so it matches the mean
    projection better than the maximum projection, which also shows the
    staining above and below the nuclei. Outlines of nuclei at the image edge
    are thin. Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    first, last = segmentation.parameters.z_range(len(stack.data))
    images = {"maximum projection (all slices)": stack.data.max(axis=0),
              f"mean of slices {first}-{last}": stack.data[first:last + 1].mean(axis=0)}
    h, w = segmentation.shape
    extent = (0, w * stack.um_per_pixel, h * stack.um_per_pixel, 0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4))
    binned = segmentation.binned_shape
    for ax, (panel, image) in zip(axes, images.items()):
        low, high = _display_range(image)
        ax.imshow(image, cmap="gray", extent=extent, vmin=low, vmax=high)
        for label, region in enumerate(segmentation.nuclei, start=1):
            _contour(ax, region.mask(binned), segmentation.um_per_bin, outline_color(label),
                     linewidth=1.0 if region.border else 1.8)
            x, y = _centroid_um(region, binned, segmentation.um_per_bin)
            ax.text(x, y, str(label), color="white", fontsize=10, ha="center", va="center", weight="bold")
        _scale_bar(ax, scale_bar_um, extent)
        ax.set_title(panel)
        ax.set_axis_off()
    if title is not None:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title is not None else 1))
    return fig


def figure_gallery(segmentation: Segmentation, stack: Stack, columns: int = 6, margin_um: float = 3.0):
    """One crop per nucleus, from the slice window its outline was found in.

    Titles give label (as in the table), area and slices; outlines of nuclei
    at the image edge are magenta. Returns the matplotlib figure, or None
    without nuclei.
    """
    import matplotlib.pyplot as plt

    regions = segmentation.nuclei
    if not regions:
        return None
    rows = int(np.ceil(len(regions) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 3.0 * rows), squeeze=False)
    um, s = stack.um_per_pixel, segmentation.um_per_bin
    height, width = segmentation.shape
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, (label, region) in zip(axes.ravel(), enumerate(regions, start=1)):
        r0, r1, c0, c1 = bounding_box(region.pixels, segmentation.binned_shape)
        y0, y1 = max(r0 * s - margin_um, 0), min(r1 * s + margin_um, height * um)
        x0, x1 = max(c0 * s - margin_um, 0), min(c1 * s + margin_um, width * um)
        i0, i1, j0, j1 = int(y0 / um), int(np.ceil(y1 / um)), int(x0 / um), int(np.ceil(x1 / um))
        crop = stack.data[region.window[0]:region.window[1], i0:i1, j0:j1].mean(axis=0)
        low, high = _display_range(crop)
        ax.imshow(crop, cmap="gray", vmin=low, vmax=high, extent=(j0 * um, j1 * um, i1 * um, i0 * um))
        _contour(ax, region.mask(segmentation.binned_shape), s, "magenta" if region.border else "yellow")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(f"{label}: {region.area_um2:.0f} µm², z {region.window[0]}-{region.window[1] - 1}", fontsize=9)
    fig.tight_layout()
    return fig


# ----- 3D -----

def plot_sections(result: Segmentation3D, *, smooth: bool = True, ncols: int = 5, show_seed: bool = True,
                  slice_contrast: bool = True, panel_size: float = 4.0, title: str | None = None):
    """Every slice of the prepared stack with the outlines of the nuclei through it.

    Outlines are solid where the void is closed by the envelope and dashed on
    nuclei a cap closes. With ``show_seed`` the 2D outline is drawn dotted in
    white on the centre slice of its projection window. With
    ``slice_contrast`` every slice is shown up to its own 99.8th percentile,
    since slices far from the coverslip are often much dimmer.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    z, h, w = result.image.shape
    s = result.um_per_bin
    nrows = int(np.ceil(z / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= z:
            continue
        vmax = np.percentile(result.image[k], 99.8) if slice_contrast else 1.0
        ax.imshow(result.image[k], cmap="gray", vmin=0, vmax=vmax, extent=(0, w * s, h * s, 0))
        for nucleus in result.nuclei:
            if show_seed and nucleus.seed.centre_slice == k:
                _contour(ax, nucleus.seed.mask((h, w)), s, "white", 0.8, ":")
            if not nucleus.found:
                continue
            section = (nucleus.smooth_mask if smooth else nucleus.mask)[k]
            if not section.any():
                continue
            color = outline_color(nucleus.label)
            _contour(ax, section, s, color, 1.2, "--" if nucleus.capped else "-",
                     offset=(nucleus.rows[0], nucleus.cols[0]))
            rr, cc = np.nonzero(section)
            ax.text((nucleus.cols[0] + cc.mean() + 0.5) * s, (nucleus.rows[0] + rr.mean() + 0.5) * s,
                    str(nucleus.label), color=color, fontsize=8, ha="center", va="center", weight="bold")
        ax.set_xlim(0, w * s)
        ax.set_ylim(h * s, 0)
        ax.set_title(f"z = {k} ({k * result.z_step_um:.1f} µm)", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_orthogonal(result: Segmentation3D, label: int, *, smooth: bool = True, show_raw: bool = True):
    """Sections of one nucleus through its centroid: xy, xz and yz, in µm.

    The smoothed outline is drawn in lime, the raw one in orange. The dashed
    red lines in xz and yz mark the ends of the stack, where the caps are;
    z increases upwards in xz and to the right in yz.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    nucleus = next((n for n in result.nuclei if n.label == label), None)
    if nucleus is None:
        raise ValueError(f"no nucleus {label}")
    if not nucleus.found:
        raise ValueError(f"nucleus {label} has no 3D void")
    dz, s = result.z_step_um, result.um_per_bin
    crop = result.image[:, nucleus.rows[0]:nucleus.rows[1], nucleus.cols[0]:nucleus.cols[1]]
    shown = [(nucleus.smooth_mask, "lime")] if smooth else []
    if show_raw or not smooth:
        shown.append((nucleus.mask, "orange"))
    zc, rc, cc = (int(round(c)) for c in ndimage.center_of_mass(nucleus.mask))
    nz, nr, nc = crop.shape
    x0, y0 = nucleus.cols[0] * s, nucleus.rows[0] * s
    xs, ys, zs = x0 + (np.arange(nc) + 0.5) * s, y0 + (np.arange(nr) + 0.5) * s, (np.arange(nz) + 0.5) * dz
    # Pad the z extent by the caps' half slices so the cap lines are visible.
    z_lo, z_hi = -0.5 * dz, (nz + 0.5) * dz
    widths, heights = [nc * s, z_hi - z_lo], [nr * s, z_hi - z_lo]
    fig, axes = plt.subplots(2, 2, figsize=(9, 9 * sum(heights) / sum(widths)),
                             gridspec_kw={"width_ratios": widths, "height_ratios": heights})
    xy, yz, xz, empty = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    empty.axis("off")
    xy.imshow(crop[zc], cmap="gray", vmin=0, vmax=1, extent=(x0, x0 + nc * s, y0 + nr * s, y0))
    xz.imshow(crop[:, rc, :], cmap="gray", vmin=0, vmax=1, extent=(x0, x0 + nc * s, 0, nz * dz), origin="lower")
    yz.imshow(crop[:, :, cc].T, cmap="gray", vmin=0, vmax=1, extent=(0, nz * dz, y0 + nr * s, y0))
    xz.set_ylim(z_lo, z_hi)
    yz.set_xlim(z_lo, z_hi)
    for mask, color in shown:
        xy.contour(xs, ys, mask[zc].astype(float), levels=[0.5], colors=[color], linewidths=1.2)
        xz.contour(xs, zs, mask[:, rc, :].astype(float), levels=[0.5], colors=[color], linewidths=1.2)
        yz.contour(zs, ys, mask[:, :, cc].T.astype(float), levels=[0.5], colors=[color], linewidths=1.2)
    for ax, horizontal in ((xz, True), (yz, False)):
        for edge in (0, nz * dz):
            (ax.axhline if horizontal else ax.axvline)(edge, color="red", linestyle="--", linewidth=1)
    xy.axhline(ys[rc], color="cyan", linewidth=0.6)
    xy.axvline(xs[cc], color="cyan", linewidth=0.6)
    xy.set_title(f"nucleus {label}: xy at z = {zc}")
    xz.set_title(f"xz at y = {ys[rc]:.1f} µm")
    yz.set_title(f"yz at x = {xs[cc]:.1f} µm")
    xz.set_xlabel("x (µm)")
    xz.set_ylabel("z (µm)")
    yz.set_xlabel("z (µm)")
    xy.set_ylabel("y (µm)")
    fig.tight_layout()
    return fig
