"""Segmenting an image tile by tile, for upsampled images too large for one forest.

The image is cut into *cores* that partition it; each tile is its core with
a margin on every side. A tile is upsampled and segmented on its own, and a
segment is kept by the tile whose core holds its centroid, so every segment
is kept at most once. A droplet smaller than the margin, centered in a core,
lies inside that tile, away from its edge. Segments touching the tile's edge
inside the image are cut by the tile and dropped; they are counted, and a
tile that cut a segment it owns is segmented again with a wider margin.

Beyond the grid counts as brighter than everything, so within a margin of a
tile's edge the forest differs from that of the whole image; the margin must
also cover the background around the droplets near the core's edge.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from .pipeline import Settings
from .pipeline import segment


@dataclass(frozen=True)
class Tile:
    """A core of the partition and its tile, as ``(rows, columns)`` slices of the original image."""

    core: tuple[slice, slice]
    outer: tuple[slice, slice]


def make_tiles(shape: tuple[int, int], size: int, margin: int) -> list[Tile]:
    """Cores of at most ``size`` pixels a side, and tiles reaching ``margin`` beyond them."""
    tiles = []
    for row in range(0, shape[0], size):
        for column in range(0, shape[1], size):
            core = (slice(row, min(row + size, shape[0])), slice(column, min(column + size, shape[1])))
            outer = tuple(slice(max(s.start - margin, 0), min(s.stop + margin, length))
                          for s, length in zip(core, shape))
            tiles.append(Tile(core, outer))
    return tiles


@dataclass
class TileResult:
    """The segments kept from one tile, in fine pixels of the tile.

    Attributes
    ----------
    tile : Tile
    labels : ndarray of int
        Fine labels of the tile with only the kept segments.
    groups : ndarray of int
        The outline each kept segment comes from (``segments.outline_groups``).
    outlines, late, split, cut : int
        Outlines, merged peaks, split outlines, and segments centered in the
        core but dropped because they touch the tile's edge.
    seconds : float
    broken : int
        Kept segments the smoothing cut into several pieces of at least the
        minimal area.
    """

    tile: Tile
    labels: NDArray[np.int_]
    groups: NDArray[np.int_]
    outlines: int
    late: int
    split: int
    cut: int
    seconds: float
    broken: int = 0


def segment_tile(image: NDArray[np.float64], tile: Tile, settings: Settings, upsampler: Callable,
                 factor: int) -> TileResult:
    """Upsample and segment one tile, keeping the segments centered in its core."""
    result = segment(upsampler(image[tile.outer]), settings)
    labels = result.labels
    ids = np.unique(labels[labels > 0])
    centers = np.array(ndimage.center_of_mass(np.ones(labels.shape), labels, ids)).reshape(-1, 2)
    # Centroids in original pixels of the whole image; a fine pixel's center is (i + 0.5) / factor.
    centers = (centers + 0.5) / factor + [tile.outer[0].start, tile.outer[1].start]
    in_core = ((tile.core[0].start <= centers[:, 0]) & (centers[:, 0] < tile.core[0].stop)
               & (tile.core[1].start <= centers[:, 1]) & (centers[:, 1] < tile.core[1].stop))
    # Edges of the tile that lie inside the image cut segments; the image border does not.
    edge = np.zeros(labels.shape, dtype=bool)
    full = image.shape
    if tile.outer[0].start > 0:
        edge[0, :] = True
    if tile.outer[0].stop < full[0]:
        edge[-1, :] = True
    if tile.outer[1].start > 0:
        edge[:, 0] = True
    if tile.outer[1].stop < full[1]:
        edge[:, -1] = True
    touching = np.isin(ids, np.unique(labels[edge]))
    kept = ids[in_core & ~touching]
    seconds = sum(result.timings.values())
    keep = np.isin(labels, kept)
    return TileResult(tile, np.where(keep, labels, 0), np.where(keep, result.groups, 0), len(result.selections),
                      len(result.late), sum(bool(split.lobes) for split in result.splits.values()),
                      int((in_core & touching).sum()), seconds, int(np.isin(result.broken, kept).sum()))


def segment_tiled(image: NDArray[np.float64], settings: Settings, upsampler: Callable, factor: int,
                  size: int = 64, margin: int = 12, workers: int = 1,
                  retries: int = 2) -> tuple[NDArray[np.int_], list[TileResult]]:
    """Segment ``image`` tile by tile on the upsampled grid.

    Parameters
    ----------
    image : ndarray
        Image in original pixels, scaled to [0, 1].
    settings : Settings
        Settings for the upsampled tiles, with pixel lengths in fine pixels.
    upsampler : callable
        Maps a tile to its upsampled image ``factor`` times larger; it must be
        picklable for ``workers > 1``, e.g. a ``functools.partial`` of
        ``images.gaussian_upsample``.
    factor : int
    size, margin : int
        Core size and margin in original pixels. A tile that cuts a segment
        it owns (a droplet with its surroundings larger than the margin) is
        segmented again with the margin doubled, up to ``retries`` times.
    workers : int
        Processes segmenting tiles in parallel.
    retries : int
        Doublings of the margin for tiles that cut a segment.

    Returns
    -------
    labels : ndarray of int
        Fine labels of the whole image, numbered 1, 2, ... in tile order.
    tile_results : list of TileResult
    """
    def run(tiles):
        arguments = ([image] * len(tiles), tiles, [settings] * len(tiles), [upsampler] * len(tiles),
                     [factor] * len(tiles))
        if workers > 1 and len(tiles) > 1:
            with ProcessPoolExecutor(min(workers, len(tiles))) as pool:
                return list(pool.map(segment_tile, *arguments))
        return list(map(segment_tile, *arguments))

    tile_results = run(make_tiles(image.shape, size, margin))
    for retry in range(1, retries + 1):
        redo = [index for index, tile_result in enumerate(tile_results) if tile_result.cut]
        if not redo:
            break
        wider = margin * 2**retry
        tiles = [Tile(tile_results[index].tile.core,
                      tuple(slice(max(s.start - wider, 0), min(s.stop + wider, n))
                            for s, n in zip(tile_results[index].tile.core, image.shape))) for index in redo]
        for index, tile_result in zip(redo, run(tiles)):
            tile_results[index] = tile_result
    return assemble(tile_results, image.shape, factor), tile_results


def assemble(tile_results: list[TileResult], shape: tuple[int, int], factor: int) -> NDArray[np.int_]:
    """Fine labels of the whole image from the kept segments of its tiles, numbered in tile order."""
    labels = np.zeros((shape[0] * factor, shape[1] * factor), dtype=int)
    count = 0
    for tile_result in tile_results:
        window = tuple(slice(s.start * factor, s.stop * factor) for s in tile_result.tile.outer)
        target = labels[window]
        for label in np.unique(tile_result.labels[tile_result.labels > 0]):
            count += 1
            # Segments of neighboring tiles can overlap within the margin; the first one keeps the pixels.
            target[(tile_result.labels == label) & (target == 0)] = count
    return labels


def assemble_groups(tile_results: list[TileResult], shape: tuple[int, int], factor: int) -> NDArray[np.int_]:
    """Outline groups of the whole image, matching ``assemble``: groups of different tiles never coincide."""
    groups = np.zeros((shape[0] * factor, shape[1] * factor), dtype=int)
    offset = 0
    for tile_result in tile_results:
        window = tuple(slice(s.start * factor, s.stop * factor) for s in tile_result.tile.outer)
        target = groups[window]
        free = (tile_result.groups > 0) & (target == 0)
        target[free] = tile_result.groups[free] + offset
        offset += int(tile_result.groups.max())
    return groups
