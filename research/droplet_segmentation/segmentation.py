"""From selected cycles to a label image, and its comparison with the truth.

A segment is the set of pixels whose centers lie inside the outer loop of a
cycle: a staircase for cubical cycles, a polygon on grid vertices for alpha
lobes. Taking the outer loop fills holes, the dim pixels a noisy outline
leaves out inside a droplet.

A *region* is a stage-2 outline or one of its lobes. A region holding at most
one candidate peak is one segment. A region holding several peaks (droplets
whose bars split off above the maximal search value) is seeded with one
*core* per peak, the peak's cycle just above its last split, and the rest of
it is filled in. The fill-in hands each uncovered pixel of a *container*, a
region with several peaks or a split outline whose lobes leave a strip
uncovered, to one of its segments:

``"flood"``
    Pixels are visited from bright to dark, and each joins the segment of an
    already labelled neighbor. This continues the superlevel filtration below
    the saddle, so the boundary between two droplets follows the valley.
``"nearest"``
    Each pixel joins the segment with the nearest pixel.
"""

import heapq

import numpy as np
from matplotlib.path import Path
from numpy.typing import NDArray

from persforest import PersistenceForest

from splitting import outer_loop

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


def label_regions(regions: list[NDArray[np.bool_]], peaks: dict, cores: dict,
                  shape: tuple[int, int]) -> tuple[NDArray[np.int_], list, list[NDArray[np.bool_]]]:
    """Label regions, splitting those that hold several peaks into seeded cores.

    Parameters
    ----------
    regions : list of ndarray of bool
        Disjoint regions.
    peaks : dict
        Peak pixel ``(row, column)`` of every candidate.
    cores : dict
        Core mask of every candidate, used where a region holds several peaks.
    shape : tuple of int

    Returns
    -------
    labels : ndarray of int
        Single-peak regions fully labelled, crowded regions only at their cores.
    owners : list
        ``owners[label - 1]`` is the candidate of the label, or None for a region
        without a peak (a droplet without a maximum of its own).
    crowded : list of ndarray of bool
        Regions holding several peaks, to be filled in.
    """
    labels = np.zeros(shape, dtype=int)
    owners, crowded = [], []
    for region in regions:
        inside = [candidate for candidate, pixel in peaks.items() if region[pixel]]
        if len(inside) <= 1:
            owners.append(inside[0] if inside else None)
            labels[region & (labels == 0)] = len(owners)
            continue
        for candidate in inside:
            owners.append(candidate)
            labels[cores[candidate] & region] = len(owners)
        crowded.append(region)
    return labels, owners, crowded


def fill_in(labels: NDArray[np.int_], container: NDArray[np.bool_], image: NDArray[np.float64],
            method: str = "flood") -> NDArray[np.int_]:
    """Label the unlabelled pixels of ``container`` from its labelled ones.

    Parameters
    ----------
    labels : ndarray of int
        Current labels; 0 is unlabelled.
    container : ndarray of bool
        Pixels to fill.
    image : ndarray
        Brightness, which orders the flooding.
    method : {"flood", "nearest"}
        Flooding leaves pixels unreachable from a segment inside the container
        to the nearest segment.
    """
    labels = labels.copy()
    if not (labels[container] > 0).any():
        return labels
    if method == "flood":
        heap = []

        def push_neighbors(row, column, label):
            for d_row, d_column in _NEIGHBORS:
                r, c = row + d_row, column + d_column
                if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1] and container[r, c] and not labels[r, c]:
                    heapq.heappush(heap, (-image[r, c], r, c, label))

        for row, column in np.argwhere(container & (labels > 0)):
            push_neighbors(row, column, labels[row, column])
        while heap:
            _, row, column, label = heapq.heappop(heap)
            if not labels[row, column]:
                labels[row, column] = label
                push_neighbors(row, column, label)
    elif method != "nearest":
        raise ValueError(f"method must be 'flood' or 'nearest', got {method!r}")
    pending = np.argwhere(container & (labels == 0))
    if len(pending):
        seeds = np.argwhere(container & (labels > 0))
        distances = ((pending[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
        labels[tuple(pending.T)] = labels[tuple(seeds[distances.argmin(axis=1)].T)]
    return labels



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
