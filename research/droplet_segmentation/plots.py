"""Figures for the droplet segmentation notebook.

Images are drawn with rows downward: pixel ``(i, j)`` covers
``[j, j + 1] x [i, i + 1]`` in plot coordinates, so the ``(row, column)``
loop coordinates of ``cycle_loops`` are plotted as ``(column, row)``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from persforest.geometry.loops import cycle_loops

from selection import Profile
from selection import Stable
from selection import local_deviation


def show_image(ax, image, title=None):
    """Grayscale image with rows downward."""
    ax.imshow(image, cmap="gray", extent=(0, image.shape[1], image.shape[0], 0), interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_cycle(ax, cycle, geometry, color, curve="raw", scale=None, **kwargs):
    """Draw the loops of a cubical or alpha representative."""
    kwargs = {"linewidth": 1.5, **kwargs}
    for loop in cycle_loops(cycle, geometry, curve, scale):
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
    if max_defect is not None:
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
    """Qualitative colors for labels 1..n_labels."""
    colors = plt.get_cmap("tab20").colors
    return ListedColormap([colors[index % len(colors)] for index in range(n_labels)])


def show_labels(ax, image, labels, title=None, alpha=0.55):
    """Image with labels overlaid; label 0 is transparent."""
    show_image(ax, image, title)
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


def plot_large_scene(image, labels, droplets, scores, title, note=None, zoom_size=240):
    """Image, segmentation and a zoom of a large scene, for readers new to the method.

    The zoom shows the ``zoom_size`` window holding the most true droplet
    centers, chosen from the truth rather than from the result. The count of
    found droplets is out of all simulated droplets, including any hidden
    under brighter ones. ``note`` is printed small below the panels.
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
    show_image(axes[0], image, f"Simulated microscope image with {len(droplets)} droplets")
    fig.colorbar(axes[0].images[0], cax=axes[0].inset_axes([1.015, 0.0, 0.025, 1.0]), label="brightness",
                 format="%.1f")
    show_labels(axes[1], image, labels, "Detected droplets (one color per droplet)")
    outline_segments(axes[1], labels, linewidth=0.6)
    axes[1].add_patch(Rectangle((corner[1], corner[0]), zoom_size, zoom_size, fill=False, edgecolor="#ffcc00",
                                linewidth=1.5))
    show_labels(axes[2], image[window], labels[window], "Close-up of the yellow box")
    outline_segments(axes[2], labels[window], linewidth=1.4)
    fig.suptitle(f"{title}\n{scores.matches} of {len(droplets)} droplets found", fontsize=15)
    if note:
        fig.text(0.5, axes[1].get_position().y0 - 0.03, note, ha="center", va="top", fontsize=10, color="0.4")
    return fig
