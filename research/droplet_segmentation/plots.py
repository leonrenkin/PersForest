"""Figures for the droplet segmentation notebook.

Images are drawn with rows downward: pixel ``(i, j)`` covers
``[j, j + 1] x [i, i + 1]`` in plot coordinates, so the ``(row, column)``
loop coordinates of ``persforest.geometry.loops`` are plotted as ``(column, row)``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from persforest.geometry.loops import midpoint_loops
from persforest.geometry.loops import raw_loops
from persforest.geometry.loops import smoothed_loops

from selection import Profile
from selection import Stable
from selection import local_deviation


def show_image(ax, image, title=None, vmax=None):
    """Grayscale image with rows downward; brightness above ``vmax`` is shown as white."""
    ax.imshow(image, cmap="gray", extent=(0, image.shape[1], image.shape[0], 0), interpolation="nearest",
              vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_cycle(ax, cycle, geometry, color, curve="raw", scale=None, **kwargs):
    """Draw the loops of a cubical or alpha representative; ``scale`` is the smoothing sigma."""
    kwargs = {"linewidth": 1.5, **kwargs}
    if curve == "smoothed":
        loops = smoothed_loops(cycle, geometry, scale)
    else:
        loops = {"raw": raw_loops, "midpoint": midpoint_loops}[curve](cycle, geometry)
    for loop in loops:
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


def plot_profile(ax, profile: Profile, chosen_value=None, rule=None, search_start=None, title=None,
                 ylabel="convex-hull deficit", max_value=None):
    """Step plot of a defect profile with the rule and the chosen value."""
    edges = np.append(profile.starts, profile.ends[-1])

    def steps(values, **kwargs):
        ax.step(edges, np.append(values, values[-1]), where="post", **kwargs)

    steps(profile.values, color="C0", linewidth=1.2, label="defect")
    if isinstance(rule, Stable):
        steps(local_deviation(profile, rule.bandwidth), color="C1", linewidth=1,
              label=f"local deviation (bandwidth {rule.bandwidth:g})")
        ax.axhline(rule.tol, color="C1", linestyle="--", linewidth=0.8, label=f"tol {rule.tol:g}")
    max_defect = getattr(rule, "max_defect", None)
    if max_defect is not None and np.isfinite(max_defect):
        ax.axhline(max_defect, color="0.4", linestyle="--", linewidth=0.8, label=f"max defect {max_defect:g}")
    if chosen_value is not None:
        ax.axvline(chosen_value, color="C3", linewidth=1.2, label="chosen")
    if search_start is not None:
        ax.axvline(search_start, color="k", linestyle=":", linewidth=1.2, label="search start")
    if max_value is not None:
        ax.axvline(max_value, color="k", linestyle="-.", linewidth=1, label="max value")
    ax.set_xlabel("filtration value")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_lobe_diagram(ax, split, min_radius, min_lifespan, title=None):
    """Persistence diagram of an alpha forest with the lobe conditions shaded."""
    births = np.array([bar.birth for bar in split.forest.barcode])
    deaths = np.array([bar.death for bar in split.forest.barcode])
    top = deaths.max() * 1.08
    corner = max(min_radius - min_lifespan, 0)
    ax.fill_between([0, corner, top], [min_radius, min_radius, top + min_lifespan], top,
                    color="C2", alpha=0.12, label="lobe region")
    ax.scatter(births, deaths, s=10, color="0.5", label="alpha bars")
    ax.scatter([bar.birth for bar in split.lobe_bars], [bar.death for bar in split.lobe_bars],
               s=40, color="C3", label="lobes inside the outline")
    ax.plot([0, top], [0, top], color="k", linewidth=0.8)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("birth (alpha radius, px)")
    ax.set_ylabel("death (alpha radius, px)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")


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


def outline_truth(ax, truth, color="white", linewidth=0.6):
    """Contours of the true droplet labels."""
    rows, columns = np.mgrid[0:truth.shape[0], 0:truth.shape[1]] + 0.5
    for label in range(1, truth.max() + 1):
        ax.contour(columns, rows, (truth == label).astype(float), levels=[0.5], colors=color,
                   linewidths=linewidth)


def plot_benchmark_example(scene, result, scores, size=4.2):
    """Truth, stage-2 outlines, outlines after splitting and final labels of one scene."""
    fig, axes = plt.subplots(1, 4, figsize=(4 * size, size + 0.6))
    show_image(axes[0], scene.image, f"truth: {scores.droplets} droplets")
    fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.02)
    for number, droplet in enumerate(scene.droplets, start=1):
        if (scene.truth == number).any():  # droplets hidden under others are not counted
            axes[0].text(droplet.column, droplet.row, str(number), color="red", fontweight="bold",
                         ha="center", va="center", fontsize=8)
    show_image(axes[1], scene.image, f"stage 2: {len(result.selections)} outlines")
    show_image(axes[2], scene.image, "after splitting")
    n_regions = 0
    for index, (bar, selection) in enumerate(result.selections.items()):
        color = f"C{index % 10}"
        draw_cycle(axes[1], selection.cycle, result.forest.geometry, color, linewidth=1)
        split = result.splits.get(bar)
        if split is not None and split.lobes:
            for lobe in split.lobes:
                draw_cycle(axes[2], lobe.cycle, split.points, color, linewidth=1.2)
            n_regions += len(split.lobes)
        else:
            draw_cycle(axes[2], selection.cycle, result.forest.geometry, color, linewidth=1)
            n_regions += 1
    axes[2].set_title(f"after splitting: {n_regions} outlines")
    show_labels(axes[3], scene.image, result.labels, f"final: {scores.segments} segments")
    outline_segments(axes[3], result.labels)
    fig.suptitle(f"{scene.name}  |  matched {scores.matches}/{scores.droplets}, precision {scores.precision:.2f}, "
                 f"recall {scores.recall:.2f}, PQ {scores.pq:.2f}, mean best IoU {scores.mean_best_iou:.2f}, "
                 f"merges {scores.merges}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_large_scene(image, labels, droplets, scores, title, note=None, zoom_size=240, vmax=None):
    """Image, segmentation and a zoom of a large scene, for readers new to the method.

    The zoom shows the ``zoom_size`` window holding the most true droplet
    centers, chosen from the truth rather than from the result. The count of
    found droplets is out of all simulated droplets, including any hidden
    under brighter ones. ``note`` is printed small below the panels. ``vmax``
    caps the gray scale of all panels, so that the noise of the background
    shows; brighter pixels are white.
    """
    from matplotlib.patches import Rectangle

    centers = np.array([(droplet.row, droplet.column) for droplet in droplets])
    best, corner = -1, (0, 0)
    for row in range(0, image.shape[0] - zoom_size + 1, 10):
        for column in range(0, image.shape[1] - zoom_size + 1, 10):
            inside = ((centers[:, 0] >= row) & (centers[:, 0] < row + zoom_size)
                      & (centers[:, 1] >= column) & (centers[:, 1] < column + zoom_size)).sum()
            if inside > best:
                best, corner = inside, (row, column)
    window = (slice(corner[0], corner[0] + zoom_size), slice(corner[1], corner[1] + zoom_size))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.6), gridspec_kw={"wspace": 0.22})
    show_image(axes[0], image, f"Simulated microscope image with {len(droplets)} droplets", vmax)
    fig.colorbar(axes[0].images[0], cax=axes[0].inset_axes([1.015, 0.0, 0.025, 1.0]), label="brightness",
                 format="%.1f")
    show_labels(axes[1], image, labels, "Detected droplets (one color per droplet)", vmax=vmax)
    outline_segments(axes[1], labels, linewidth=0.6)
    axes[1].add_patch(Rectangle((corner[1], corner[0]), zoom_size, zoom_size, fill=False, edgecolor="#ffcc00",
                                linewidth=1.5))
    show_labels(axes[2], image[window], labels[window], "Close-up of the yellow box", vmax=vmax)
    outline_segments(axes[2], labels[window], linewidth=1.4)
    fig.suptitle(f"{title}\n{scores.matches} of {len(droplets)} droplets found", fontsize=15)
    if note:
        fig.text(0.5, axes[1].get_position().y0 - 0.03, note, ha="center", va="top", fontsize=10, color="0.4")
    return fig


def plot_segmentation(image, result, size=6.0, vmax=None, window=None, title=None, edge_color="yellow"):
    """Stage-2 outlines, outlines after splitting and final labels of an image without truth.

    Stage-2 outlines taken at the maximal search value are dashed; peaks born
    above it are white crosses. ``window`` is a ``(rows, columns)`` pair of
    slices to zoom onto; only outlines whose peak lies near it are drawn.
    Segment edges are drawn in ``edge_color``, which shows on saturated pixels.
    """
    from pipeline import peak

    forest = result.forest
    near = None
    if window is not None:
        rows, columns = window
        margin = 15
        near = lambda bar: (rows.start - margin <= peak(bar, forest)[0] < rows.stop + margin
                            and columns.start - margin <= peak(bar, forest)[1] < columns.stop + margin)
    fig, axes = plt.subplots(1, 3, figsize=(3 * size, size + 0.5), sharex=True, sharey=True)
    show_image(axes[0], image, f"stage 2: {len(result.selections)} outlines, {len(result.late)} late peaks", vmax)
    show_image(axes[1], image, None, vmax)
    n_regions = 0
    for index, (bar, selection) in enumerate(result.selections.items()):
        split = result.splits.get(bar)
        lobes = split.lobes if split is not None else []
        n_regions += len(lobes) or 1
        if near is not None and not near(bar):
            continue
        color = f"C{index % 10}"
        draw_cycle(axes[0], selection.cycle, forest.geometry, color, linewidth=1,
                   linestyle="--" if selection.fallback else "-")
        for lobe in lobes:
            draw_cycle(axes[1], lobe.cycle, split.points, color, linewidth=1.2)
        if not lobes:
            draw_cycle(axes[1], selection.cycle, forest.geometry, color, linewidth=1)
    for bar in result.late:
        if near is None or near(bar):
            row, column = peak(bar, forest)
            axes[0].plot(column + 0.5, row + 0.5, "w+", markersize=6, markeredgewidth=1)
    n_split = sum(bool(split.lobes) for split in result.splits.values())
    axes[1].set_title(f"after splitting: {n_regions} outlines, {n_split} split")
    show_labels(axes[2], image, result.labels, f"final: {len(np.unique(result.labels)) - 1} segments", vmax=vmax)
    outline_segments(axes[2], result.labels, color=edge_color, linewidth=0.8)
    if window is not None:
        axes[0].set_xlim(columns.start, columns.stop)
        axes[0].set_ylim(rows.stop, rows.start)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_label_grid(image, labels_by_name, window=None, vmax=None, columns=4, size=4.0, edge_color="yellow"):
    """Final labels for several settings side by side, zoomed onto ``window``."""
    rows = -(-len(labels_by_name) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * (size + 0.3)), squeeze=False)
    window = window or (slice(0, image.shape[0]), slice(0, image.shape[1]))
    for ax, (name, labels) in zip(axes.ravel(), labels_by_name.items()):
        shown = labels[window]
        show_labels(ax, image[window], shown, f"{name}: {len(np.unique(labels)) - 1} segments", vmax=vmax)
        outline_segments(ax, shown, color=edge_color, linewidth=0.8)
    for ax in axes.ravel()[len(labels_by_name):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_resolutions(image, labels, fine_labels, factor, windows, vmax=None, truth=None, size=5.0,
                     truth_color="cyan", groups=None, fine_groups=None):
    """Segment boundaries at the original resolution and on the upsampled grid, side by side per window.

    Both are drawn on the original image (``boundary_overlay``): yellow towards
    the background, red between segments of different outlines (cut by the
    forest) and, with ``groups`` and ``fine_groups``, blue between lobes of
    one outline. ``windows`` are ``(rows, columns)``
    slices in original pixels. ``truth``, on the fine grid, is outlined in
    every panel.
    """
    scale = 4 * factor
    fig, axes = plt.subplots(len(windows), 3, figsize=(3 * size, len(windows) * (size + 0.3)), squeeze=False)
    for row, window in zip(axes, windows):
        fine_window = tuple(slice(s.start * factor, s.stop * factor) for s in window)
        extent = (0, (window[1].stop - window[1].start) * factor, (window[0].stop - window[0].start) * factor, 0)
        panels = [(boundary_overlay(image[window], np.zeros_like(labels[window]), 1, scale, vmax),
                   f"image, rows {window[0].start}-{window[0].stop}, columns {window[1].start}-{window[1].stop}"),
                  (boundary_overlay(image[window], labels[window], 1, scale, vmax,
                                    None if groups is None else groups[window]),
                   f"original resolution: {len(np.unique(labels[window])) - 1} segments"),
                  (boundary_overlay(image[window], fine_labels[fine_window], factor, scale, vmax,
                                    None if fine_groups is None else fine_groups[fine_window]),
                   f"{factor}x upsampled: {len(np.unique(fine_labels[fine_window])) - 1} segments")]
        for ax, (rgb, title) in zip(row, panels):
            ax.imshow(rgb, extent=extent, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            if truth is not None:
                outline_segments(ax, truth[fine_window], color=truth_color, linewidth=0.6)
    fig.tight_layout()
    return fig


def boundary_overlay(image, labels, factor=1, scale=16, vmax=None, groups=None, edge_color=(255, 225, 0),
                     contact_color=(255, 40, 40), lobe_color=(0, 200, 255), width=3, rim_color=(20, 20, 20)):
    """RGB array of ``image`` with the segment boundaries drawn on it, for saving at high resolution.

    Every pixel of ``image`` becomes a ``scale`` x ``scale`` block, so the
    original pixels stay visible; ``labels`` lies on a grid ``factor`` times
    finer than ``image`` (``scale`` must be a multiple of ``factor``). Edges
    towards the background are drawn in ``edge_color``, edges between two
    segments in ``contact_color``, ``width`` output pixels wide and centered
    on the pixel edge. With ``groups`` (``segmentation.outline_groups``, on
    the grid of ``labels``), edges between two segments of one outline, cut
    apart by the lobe splitting, are drawn in ``lobe_color`` instead; edges
    between segments of different outlines are cuts of the forest. Every edge
    has a rim of one output pixel in ``rim_color`` on both sides, so that it
    also shows on saturated pixels; ``rim_color=None`` omits it. Interiors are
    left uncolored.
    """
    if scale % factor:
        raise ValueError(f"scale {scale} must be a multiple of factor {factor}")
    step = scale // factor
    gray = (np.clip(image / (vmax or image.max()), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)[..., None].repeat(3, axis=2)
    span = np.arange(step + 1)
    # Rims first, under all edges, then the edges themselves.
    passes = [(width + 2, True), (width, False)] if rim_color is not None else [(width, False)]
    for line_width, rim in passes:
        offsets = np.arange(line_width) - line_width // 2
        _draw_edges(rgb, labels, groups, step, span, offsets, rim_color if rim else None,
                    edge_color, contact_color, lobe_color)
    return rgb


def _draw_edges(rgb, labels, groups, step, span, offsets, rim_color, edge_color, contact_color, lobe_color):
    """Draw the boundary edges of ``labels`` into ``rgb``, all in ``rim_color`` if given; see ``boundary_overlay``."""
    width = len(offsets)
    for axis in (0, 1):
        first = labels[:-1, :] if axis == 0 else labels[:, :-1]
        second = labels[1:, :] if axis == 0 else labels[:, 1:]
        differs = (first != second) & ((first > 0) | (second > 0))
        contact = differs & (first > 0) & (second > 0)
        lobe = np.zeros_like(contact)
        if groups is not None:
            first_group = groups[:-1, :] if axis == 0 else groups[:, :-1]
            second_group = groups[1:, :] if axis == 0 else groups[:, 1:]
            lobe = contact & (first_group == second_group)
        for mask, color in ((differs & ~contact, edge_color), (contact & ~lobe, contact_color), (lobe, lobe_color)):
            color = rim_color or color
            rows, columns = np.nonzero(mask)
            # The edge lies after pixel (row, column) along the axis and spans the pixel across it.
            across = ((rows if axis == 0 else columns) + 1) * step
            along = (columns if axis == 0 else rows) * step
            lines = np.clip(across[:, None, None] + offsets[None, :, None], 0, None)
            extent = np.broadcast_to(along[:, None, None] + span[None, None, :], (len(along), width, step + 1))
            lines = np.broadcast_to(lines, extent.shape)
            keep = (lines < rgb.shape[axis]) & (extent < rgb.shape[1 - axis])
            if axis == 0:
                rgb[lines[keep], extent[keep]] = color
            else:
                rgb[extent[keep], lines[keep]] = color


def plot_splits(image, result, columns=6, size=3.2, vmax=None, max_panels=None):
    """Every split outline of a result: the stage-2 outline (white) and its lobes, weakest split first.

    The title gives each lobe bar's death (lobe radius) and lifespan (lobe
    radius minus half the waist width), in pixels of ``image``; the split is
    as pronounced as its shortest lobe bar, and the panels are sorted by it.
    """
    splits = sorted((split for split in result.splits.values() if split.lobes),
                    key=lambda split: min(bar.lifespan() for bar in split.lobe_bars))[:max_panels]
    if not splits:
        print("no outline was split")
        return None
    rows = -(-len(splits) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * (size + 0.3)), squeeze=False)
    for ax, split in zip(axes.ravel(), splits):
        show_image(ax, image, None, vmax)
        draw_cycle(ax, split.selection.cycle, result.forest.geometry, "white", linewidth=1)
        for index, lobe in enumerate(split.lobes):
            draw_cycle(ax, lobe.cycle, split.points, f"C{index}", linewidth=2)
        zoom(ax, split.points, margin=4)
        bars = sorted(split.lobe_bars, key=lambda bar: bar.lifespan())
        ax.set_title("  ".join(f"r {bar.death:.1f}, life {bar.lifespan():.1f}" for bar in bars[:2]), fontsize=9)
    for ax in axes.ravel()[len(splits):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def touching_pairs(result):
    """Pairs of touching segments, each with ``"divided"`` if both lie in one stage-2 outline, else ``"neighbors"``.

    A divided pair comes from late peaks seeded inside one outline or from the
    lobes of a split; neighbors had outlines of their own (their bars split
    below the maximal search value) and met when the segments grew.
    """
    labels, stage2 = result.labels, result.stage2_labels
    pairs = set()
    for first, second in [(labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])]:
        touching = (first != second) & (first > 0) & (second > 0)
        pairs |= {(min(a, b), max(a, b)) for a, b in zip(first[touching], second[touching])}

    def outline(label):
        values = stage2[labels == label]
        values = values[values > 0]
        return np.bincount(values).argmax() if len(values) else 0

    return [(a, b, "divided" if outline(a) and outline(a) == outline(b) else "neighbors") for a, b in sorted(pairs)]


def plot_touching(image, result, kind="divided", columns=6, size=3.0, vmax=None, max_panels=24, radius=7):
    """Touching pairs of one kind (see ``touching_pairs``), the most compact first, as boundary overlays."""
    labels = result.labels
    pairs = [(a, b) for a, b, found in touching_pairs(result) if found == kind]
    pairs = sorted(pairs, key=lambda pair: np.isin(labels, pair).sum())[:max_panels]
    if not pairs:
        print(f"no {kind} pairs")
        return None
    rows = -(-len(pairs) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * (size + 0.3)), squeeze=False)
    for ax, pair in zip(axes.ravel(), pairs):
        rows_, columns_ = np.nonzero(np.isin(labels, pair))
        center = int(rows_.mean()), int(columns_.mean())
        window = tuple(slice(max(c - radius, 0), c + radius + 1) for c in center)
        ax.imshow(boundary_overlay(image[window], labels[window], 1, 16, vmax), interpolation="nearest")
        ax.set_title(f"{kind} at {center}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
