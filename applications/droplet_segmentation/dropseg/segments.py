"""Steps 5 to 7: from outlines to a label image.

A segment starts as the set of pixels whose centers lie inside the outer
loop of a cycle: a staircase for cubical cycles, a polygon on grid vertices
for alpha lobes. Taking the outer loop fills holes, the dim pixels a noisy
outline leaves out inside a droplet. A *region* is an outline or one of its
lobes.

The lobes of a split outline leave a strip along the outline and across the
waist uncovered. ``fill_in`` hands each of these pixels to one of the lobes
by flooding: pixels are visited from bright to dark, and each joins the
segment of an already labelled neighbor. This continues the superlevel
filtration below the saddle, so the boundary between two lobes follows the
valley.

An outline is one level set of its droplet, so it leaves out the flanks below
that level. ``grow`` extends every segment by the same flood down to a
*floor* intensity. Segments compete for pixels, so two growing droplets meet
in the valley between them. ``merge_small_lobes`` then hands lobes too
small to be droplets to a neighboring lobe of their outline.
``open_labels`` smooths the boundaries with an alpha opening and
``drop_segments`` removes small pieces.
"""

import heapq

import numpy as np
from scipy import ndimage
from matplotlib.path import Path
from numpy.typing import NDArray

from persforest import PersistenceForest

from .lobes import outer_loop

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def outline_mask(cycle, geometry, shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Pixels whose centers lie inside the outer loop of ``cycle``.

    ``geometry`` is the forest geometry of the cycle: the ``CubicalGeometry``
    of the image or the point array of an alpha forest on grid vertices of a
    grid with unit spacing and zero origin.
    """
    loop = outer_loop(cycle, geometry)
    low = np.maximum(np.floor(loop.min(axis=0)).astype(int), 0)
    high = np.minimum(np.ceil(loop.max(axis=0)).astype(int), shape)
    rows, columns = np.mgrid[low[0]:high[0], low[1]:high[1]]
    centers = np.column_stack([rows.ravel() + 0.5, columns.ravel() + 0.5])
    mask = np.zeros(shape, dtype=bool)
    mask[rows.ravel(), columns.ravel()] = Path(loop).contains_points(centers)
    return mask


def paint(masks: list[NDArray[np.bool_]], shape: tuple[int, int]) -> NDArray[np.int_]:
    """Label image with label ``k`` for ``masks[k - 1]``; earlier masks win on overlaps."""
    labels = np.zeros(shape, dtype=int)
    for label, mask in reversed(list(enumerate(masks, start=1))):
        labels[mask] = label
    return labels


def _flood(labels: NDArray[np.int_], image: NDArray[np.float64], allowed: NDArray[np.bool_]) -> NDArray[np.int_]:
    """Extend the segments into unlabelled ``allowed`` pixels from bright to dark."""
    labels = labels.copy()
    heap = []

    def push_neighbors(row, column, label):
        for d_row, d_column in _NEIGHBORS:
            r, c = row + d_row, column + d_column
            if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1] and allowed[r, c] and not labels[r, c]:
                heapq.heappush(heap, (-image[r, c], r, c, label))

    for row, column in np.argwhere(labels > 0):
        push_neighbors(row, column, labels[row, column])
    while heap:
        _, row, column, label = heapq.heappop(heap)
        if not labels[row, column]:
            labels[row, column] = label
            push_neighbors(row, column, label)
    return labels


def fill_in(labels: NDArray[np.int_], container: NDArray[np.bool_], image: NDArray[np.float64]) -> NDArray[np.int_]:
    """Label the unlabelled pixels of ``container`` by flooding from its labelled ones.

    Pixels the flood cannot reach inside the container go to the segment with
    the nearest pixel.
    """
    if not (labels[container] > 0).any():
        return labels.copy()
    inside = np.where(container, labels, 0)
    flooded = _flood(inside, image, container)
    labels = np.where(container & (labels == 0), flooded, labels)
    pending = np.argwhere(container & (labels == 0))
    if len(pending):
        seeds = np.argwhere(container & (labels > 0))
        distances = ((pending[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
        labels[tuple(pending.T)] = labels[tuple(seeds[distances.argmin(axis=1)].T)]
    return labels


def grow(labels: NDArray[np.int_], image: NDArray[np.float64], floor: float) -> NDArray[np.int_]:
    """Extend every segment into unlabelled pixels at least as bright as ``floor``, from bright to dark.

    A pixel joins the segment whose flood reaches it first, i.e. the segment
    with the brightest path to it; segments never shrink.
    """
    return _flood(labels, image, image >= floor)


def merge_small_lobes(labels: NDArray[np.int_], image: NDArray[np.float64], siblings: list[list[int]],
                      min_area: float) -> NDArray[np.int_]:
    """Hand lobes smaller than ``min_area`` pixels to a touching lobe of the same outline.

    Lobes are cut apart by the shape of their outline, not by a valley, so a
    lobe too small to be a droplet is part of its neighbor rather than
    background. ``siblings`` lists the labels of the lobes of each split
    outline. A small lobe joins the sibling with the brightest shared
    boundary (the highest pass between them), smallest lobes first; a small
    lobe touching no sibling is left to the final filter.
    """
    labels = labels.copy()
    for group in siblings:
        while True:
            areas = {label: int((labels == label).sum()) for label in group}
            small = sorted((area, label) for label, area in areas.items() if 0 < area < min_area)
            merged = False
            for _, label in small:
                target = _brightest_neighbor(labels, image, label, [other for other in group if areas[other]])
                if target is not None:
                    labels[labels == label] = target
                    merged = True
                    break
            if not merged:
                break
    return labels


def _brightest_neighbor(labels: NDArray[np.int_], image: NDArray[np.float64], label: int,
                        candidates: list[int]) -> int | None:
    """The candidate segment sharing the brightest edge with segment ``label``, or None if none touches it."""
    best, best_value = None, -np.inf
    for axis in (0, 1):
        first = labels[:-1, :] if axis == 0 else labels[:, :-1]
        second = labels[1:, :] if axis == 0 else labels[:, 1:]
        # An edge is as bright as its darker pixel.
        value = np.minimum(image[:-1, :], image[1:, :]) if axis == 0 else np.minimum(image[:, :-1], image[:, 1:])
        for own, other in ((first, second), (second, first)):
            for candidate in candidates:
                if candidate == label:
                    continue
                shared = (own == label) & (other == candidate)
                if shared.any() and value[shared].max() > best_value:
                    best, best_value = candidate, value[shared].max()
    return best


def boundary_vertices(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Grid vertices on the outer boundary of a mask, as ``(row, column)`` points.

    Holes of the mask are filled first, so only the outer boundary counts.
    """
    padded = np.pad(mask, 1)
    outside = np.zeros_like(padded)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    while True:  # grow the outside through non-mask pixels
        grown = outside.copy()
        grown[1:] |= outside[:-1]
        grown[:-1] |= outside[1:]
        grown[:, 1:] |= outside[:, :-1]
        grown[:, :-1] |= outside[:, 1:]
        grown &= ~padded
        if (grown == outside).all():
            break
        outside = grown
    filled = ~outside
    # A grid vertex lies on the boundary if its four pixels are neither all in nor all out.
    corners = filled[:-1, :-1].astype(int) + filled[1:, :-1] + filled[:-1, 1:] + filled[1:, 1:]
    rows, columns = np.nonzero((corners > 0) & (corners < 4))
    return np.column_stack([rows, columns]).astype(float)  # padding offsets the vertex index by one pixel


def open_mask(mask: NDArray[np.bool_], radius: float) -> NDArray[np.bool_]:
    """Smooth a mask with the alpha filtration of its boundary vertices.

    The holes of the alpha complex at ``radius`` inside the mask are, roughly,
    the union of the empty disks larger than ``radius``: the mask opened by a
    disk. Protrusions narrower than about ``2 * radius`` are cut off and corners
    rounded; the result is a subset of the mask with the hole pixels' centers.
    """
    if not mask.any():
        return mask
    points = boundary_vertices(mask)
    if len(points) < 4:
        return mask
    forest = PersistenceForest.from_points(points, storage="cycles")
    opened = np.zeros_like(mask)
    for bar in forest.active_bars_at(radius):
        hole = outline_mask(forest.cycle_for_bar_at(bar, radius), points, mask.shape)
        if (hole & mask).sum() > hole.sum() / 2:
            opened |= hole
    return opened & mask


def open_labels(labels: NDArray[np.int_], radius: float) -> NDArray[np.int_]:
    """Apply ``open_mask`` to every segment; removed pixels become background."""
    opened = np.zeros_like(labels)
    for label in np.unique(labels[labels > 0]):
        rows, columns = np.nonzero(labels == label)
        low = np.maximum([rows.min() - 2, columns.min() - 2], 0)
        high = np.minimum([rows.max() + 3, columns.max() + 3], labels.shape)
        window = labels[low[0]:high[0], low[1]:high[1]] == label
        result = open_mask(window, radius)
        opened[low[0]:high[0], low[1]:high[1]][result] = label
    return opened


def drop_segments(labels: NDArray[np.int_], min_area: float = 0.0) -> tuple[NDArray[np.int_], list[int]]:
    """Remove pieces smaller than ``min_area`` pixels; their pixels become background.

    A piece is a connected component of a segment: the whole segment, or one
    of the parts the smoothing (an opening) broke it into. Returns the labels
    and the segments left with several pieces of at least ``min_area``, which
    the smoothing has cut into droplet-sized parts.
    """
    labels = labels.copy()
    broken = []
    for label, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        mask = labels[window] == label
        pieces, n_pieces = ndimage.label(mask)
        sizes = np.bincount(pieces.ravel(), minlength=n_pieces + 1)
        small = sizes < min_area
        small[0] = False
        labels[window][small[pieces] & mask] = 0
        if (sizes[1:] >= min_area).sum() > 1:
            broken.append(label)
    return labels, broken


def outline_groups(labels: NDArray[np.int_], region_labels: NDArray[np.int_],
                   outline_labels: NDArray[np.int_]) -> NDArray[np.int_]:
    """Label image of the outline each segment comes from.

    Segments of one group were divided inside one outline (lobes of an
    8-shaped outline); segments of different groups had outlines of their
    own, separated by the forest. ``region_labels`` are the segments before
    growing, ``outline_labels`` the outlines; a segment lies in the outline
    holding most of its region, or forms a group of its own if it lies in none.
    """
    n_labels = max(int(labels.max()), int(region_labels.max())) + 1
    n_outlines = int(outline_labels.max()) + 1
    inside = (region_labels > 0) & (outline_labels > 0)
    counts = np.zeros((n_labels, n_outlines), dtype=int)
    np.add.at(counts, (region_labels[inside], outline_labels[inside]), 1)
    group = counts.argmax(axis=1)
    alone = counts.max(axis=1) == 0
    group[alone] = n_outlines + np.arange(alone.sum())
    return np.where(labels > 0, group[labels], 0)
