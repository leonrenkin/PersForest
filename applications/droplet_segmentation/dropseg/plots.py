"""Figures of images, outlines and segments.

Images are drawn with rows downward: pixel ``(i, j)`` covers
``[j, j + 1] x [i, i + 1]`` in plot coordinates, so the ``(row, column)``
loop coordinates of ``persforest.geometry.loops`` are plotted as ``(column, row)``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from persforest.geometry.loops import midpoint_loops
from persforest.geometry.loops import raw_loops

from .results import boundary_overlay


def show_image(ax, image, title=None, vmax=None):
    """Grayscale image with rows downward; brightness above ``vmax`` is shown as white."""
    ax.imshow(image, cmap="gray", extent=(0, image.shape[1], image.shape[0], 0), interpolation="nearest",
              vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_cycle(ax, cycle, geometry, color, curve="raw", **kwargs):
    """Draw the loops of a cubical or alpha representative; ``curve`` is "raw" or "midpoint"."""
    kwargs = {"linewidth": 1.5, **kwargs}
    for loop in {"raw": raw_loops, "midpoint": midpoint_loops}[curve](cycle, geometry):
        closed = np.vstack((loop, loop[:1]))
        ax.plot(closed[:, 1], closed[:, 0], color=color, **kwargs)
        kwargs.pop("label", None)


def draw_points(ax, points, **kwargs):
    """Scatter ``(row, column)`` points."""
    ax.scatter(points[:, 1], points[:, 0], **{"s": 6, "color": "white", "zorder": 3, **kwargs})


def draw_alpha_complex(ax, split, value, lobe_colors=("C0", "C1")):
    """Draw the alpha complex of a split's points at radius ``value`` and its holes.

    Triangles and edges present at ``value`` are shaded; the representatives
    of the bars active at ``value`` are drawn in the color of their lobe, or
    gray if the bar is not a lobe.
    """
    import gudhi
    from matplotlib.collections import LineCollection
    from matplotlib.collections import PolyCollection

    simplex_tree = gudhi.AlphaComplex(points=split.points).create_simplex_tree(output_squared_values=False)
    present = [simplex for simplex, filtration in simplex_tree.get_filtration() if filtration <= value]
    xy = split.points[:, ::-1]
    ax.add_collection(PolyCollection([xy[simplex] for simplex in present if len(simplex) == 3],
                                     facecolor="lightsteelblue", edgecolor="none", alpha=0.55, zorder=2))
    ax.add_collection(LineCollection([xy[simplex] for simplex in present if len(simplex) == 2],
                                     colors="white", linewidths=0.3, alpha=0.7, zorder=2))
    draw_points(ax, split.points, s=4)
    lobe_bars = [lobe.bar for lobe in split.lobes]
    for bar in split.forest.active_bars_at(value):
        color = lobe_colors[lobe_bars.index(bar) % len(lobe_colors)] if bar in lobe_bars else "0.6"
        draw_cycle(ax, split.forest.cycle_for_bar_at(bar, value), split.points, color, linewidth=2, zorder=4)


def zoom(ax, points, margin=4):
    """Zoom an image axes onto ``(row, column)`` points."""
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    ax.set_xlim(low[1], high[1])
    ax.set_ylim(high[0], low[0])


def plot_waist(image, split, factor=1, vmax=None, size=4.5):
    """The alpha complex of an 8-shaped outline just before and just after its hole splits at the waist.

    ``split`` is a ``lobes.Split`` with lobes, on the grid of ``image``;
    ``factor`` is its upsampling, for the radii in original pixels.
    """
    split_radius = max(bar.birth for bar in split.lobe_bars)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size, size + 0.5))
    for ax, value, when in ((axes[0], 0.9 * split_radius, "before"), (axes[1], 1.1 * split_radius, "after")):
        show_image(ax, image, None, vmax)
        draw_alpha_complex(ax, split, value)
        zoom(ax, split.points, margin=2 * factor)
        ax.set_xlim(max(ax.get_xlim()[0], 0), min(ax.get_xlim()[1], image.shape[1]))
        ax.set_ylim(min(ax.get_ylim()[0], image.shape[0]), max(ax.get_ylim()[1], 0))
        ax.set_title(f"alpha radius {value / factor:.2f} px: {when} the split")
    fig.suptitle(f"the hole splits at s = {split_radius / factor:.2f} px, half the width of the waist")
    fig.tight_layout()
    return fig


def label_cmap(n_labels: int) -> ListedColormap:
    """Qualitative colors for labels 1..n_labels.

    tab20 pairs a dark and a light shade of each hue; consecutive labels, often
    the parts of one divided outline, get different hues instead.
    """
    colors = plt.get_cmap("tab20").colors
    colors = colors[0::2] + colors[1::2]
    return ListedColormap([colors[index % len(colors)] for index in range(n_labels)])


def show_labels(ax, image, labels, title=None, alpha=0.55, vmax=None):
    """Image with labels overlaid; label 0 is transparent."""
    show_image(ax, image, title, vmax)
    n_labels = max(int(labels.max()), 1)
    ax.imshow(np.ma.masked_equal(labels, 0), cmap=label_cmap(n_labels), vmin=0.5, vmax=n_labels + 0.5,
              alpha=alpha, interpolation="nearest", extent=(0, image.shape[1], image.shape[0], 0))


def outline_segments(ax, labels, color="white", linewidth=1.4):
    """Draw segment boundaries along pixel edges, between segments and towards the background."""
    from matplotlib.collections import LineCollection

    edges = []
    for first, second, edge in [(labels[1:, :], labels[:-1, :], lambda r, c: [(c, r + 1), (c + 1, r + 1)]),
                                (labels[:, 1:], labels[:, :-1], lambda r, c: [(c + 1, r), (c + 1, r + 1)])]:
        differs = (first != second) & ((first > 0) | (second > 0))
        edges += [edge(r, c) for r, c in zip(*np.nonzero(differs))]
    ax.add_collection(LineCollection(edges, colors=color, linewidths=linewidth, zorder=3))


def mark_window(ax, window, color="#ffcc00"):
    """Draw the rectangle of a ``(rows, columns)`` window."""
    rows, columns = window
    ax.add_patch(plt.Rectangle((columns.start, rows.start), columns.stop - columns.start, rows.stop - rows.start,
                               fill=False, edgecolor=color, linewidth=1.5))


def plot_segmentation(image, segmentation, window=None, vmax=None, size=7.0, title=None):
    """The image and its segments, whole and zoomed onto ``window`` (original pixels)."""
    factor = segmentation.factor
    panels = 3 if window is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(panels * size, size + 0.5))
    show_image(axes[0], image, f"image, {image.shape[0]} × {image.shape[1]} px", vmax)
    show_labels(axes[1], segmentation.image, segmentation.labels,
                f"{segmentation.n_segments} segments ({factor}×)", vmax=vmax)
    if window is not None:
        fine = tuple(slice(s.start * factor, s.stop * factor) for s in window)
        mark_window(axes[0], window)
        mark_window(axes[1], fine)
        overlay = boundary_overlay(image[window], segmentation.labels[fine], factor, 4 * factor, vmax,
                                   segmentation.groups[fine])
        axes[2].imshow(overlay, interpolation="nearest")
        axes[2].set_title("boundaries in the yellow box")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_resolutions(image, segmentations, windows, vmax=None, size=5.0):
    """Segment boundaries of several segmentations side by side, drawn on the original image, per window.

    ``segmentations`` maps a name to a ``Segmentation`` of ``image`` (at any
    upsampling); ``windows`` are ``(rows, columns)`` slices in original
    pixels. Yellow edges border the background, red edges separate segments
    cut apart by the forest, blue edges lobes of one 8-shaped outline.
    """
    factors = [segmentation.factor for segmentation in segmentations.values()]
    scale = 4 * int(np.lcm.reduce(factors))
    columns = 1 + len(segmentations)
    fig, axes = plt.subplots(len(windows), columns, figsize=(columns * size, len(windows) * (size + 0.3)),
                             squeeze=False)
    for row, window in zip(axes, windows):
        panels = [(boundary_overlay(image[window], np.zeros(image[window].shape, dtype=int), 1, scale, vmax),
                   f"image, rows {window[0].start}–{window[0].stop}, columns {window[1].start}–{window[1].stop}")]
        for name, segmentation in segmentations.items():
            factor = segmentation.factor
            fine = tuple(slice(s.start * factor, s.stop * factor) for s in window)
            labels = segmentation.labels[fine]
            panels.append((boundary_overlay(image[window], labels, factor, scale, vmax, segmentation.groups[fine]),
                           f"{name}: {len(np.unique(labels[labels > 0]))} segments"))
        for ax, (rgb, title) in zip(row, panels):
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_large_scene(image, labels, n_droplets, n_found, title, note=None, zoom_size=240, vmax=None, centers=None):
    """Image, segmentation and a zoom of a large scene, for readers new to the method.

    The zoom shows the ``zoom_size`` window holding the most ``centers``
    (``(row, column)`` points, e.g. the true droplet centers), or the center
    of the image without them. ``n_found`` of ``n_droplets`` is printed
    in the title; ``note`` small below the panels. ``vmax`` caps the gray
    scale of all panels; brighter pixels are white.
    """
    from matplotlib.patches import Rectangle

    corner = ((image.shape[0] - zoom_size) // 2, (image.shape[1] - zoom_size) // 2)
    if centers is not None:
        best = -1
        for row in range(0, image.shape[0] - zoom_size + 1, 10):
            for column in range(0, image.shape[1] - zoom_size + 1, 10):
                inside = ((centers[:, 0] >= row) & (centers[:, 0] < row + zoom_size)
                          & (centers[:, 1] >= column) & (centers[:, 1] < column + zoom_size)).sum()
                if inside > best:
                    best, corner = inside, (row, column)
    window = (slice(corner[0], corner[0] + zoom_size), slice(corner[1], corner[1] + zoom_size))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.6), gridspec_kw={"wspace": 0.22})
    show_image(axes[0], image, f"Simulated microscope image with {n_droplets} droplets", vmax)
    fig.colorbar(axes[0].images[0], cax=axes[0].inset_axes([1.015, 0.0, 0.025, 1.0]), label="brightness",
                 format="%.1f")
    show_labels(axes[1], image, labels, "Detected droplets (one color per droplet)", vmax=vmax)
    outline_segments(axes[1], labels, linewidth=0.6)
    axes[1].add_patch(Rectangle((corner[1], corner[0]), zoom_size, zoom_size, fill=False, edgecolor="#ffcc00",
                                linewidth=1.5))
    show_labels(axes[2], image[window], labels[window], "Close-up of the yellow box", vmax=vmax)
    outline_segments(axes[2], labels[window], linewidth=1.4)
    fig.suptitle(f"{title}\n{n_found} of {n_droplets} droplets found", fontsize=15)
    if note:
        fig.text(0.5, axes[1].get_position().y0 - 0.03, note, ha="center", va="top", fontsize=10, color="0.4")
    return fig
