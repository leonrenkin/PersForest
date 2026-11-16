"""Nuclear outlines in 2D: loops of the superlevel filtration of projections.

Where the staining also fills the nuclear envelope (ER markers, lamins, other
envelope proteins), a section through a nucleus is a bright ring around a
dark inside. Every nucleus is therefore a loop of the superlevel filtration of
the image: it is born when the threshold falls below the weakest point of the
ring and dies when it falls below the brightest point inside. The lifetime,
ring minus inside in normalized intensity, measures how clean the ring is.

Pipeline
--------
1. Several projections: the mean of ``window`` consecutive slices, for
   windows stepping through ``z_first`` .. ``z_last``. Different nuclei are
   sharpest at different depths.
2. Each projection is binned, smoothed, scaled to its ``normalize_percentile``
   and negated, and a cubical persistence forest is built on it.
3. Candidates: bars with lifetime at least ``min_lifetime``. The region is the
   inside of the bar's representative at ``cycle_position`` of the bar, holes
   filled (bright spots inside a nucleus are part of it). Candidates outside
   the area range, narrower than ``min_width_um``, or below ``min_solidity``
   (irregular dark pockets in the perinuclear ER) are dropped. A nucleus cut
   by the image edge is no loop, unless ``border_frame`` surrounds the image
   with a bright frame that closes it; such regions are marked ``border``.
4. Selection over all projections: candidates in decreasing lifetime times
   area, each kept unless it overlaps a kept one. Every nucleus is taken from
   the projection with the clean ring of the largest section, near its
   equator; lifetime alone would prefer small, high-contrast sections through
   the top or bottom of a nucleus.
5. Merging (optional): an envelope fold crossing a nucleus cuts its section
   into pieces. Pieces separated by a thin wall are merged unless the merged
   outline pinches in at both ends of the wall (a waist), the sign of two
   nuclei lying against each other.
6. Outline cleaning (optional, off by default): dark gaps in the ER next to
   the envelope attach spurs to some outlines. The same loops are found again
   on projections smoothed with ``clean_smooth_um``, where such gaps are
   closed but the outline lies further inside the ring. The cleaned outline is
   the smooth one, grown back towards the original outline by at most
   ``clean_reach_um``: spurs are cut to that length, the rest of the outline
   keeps its place.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from persforest import PersistenceForest

from ._grid import bin_xy
from ._grid import bounding_box
from ._grid import convex_hull_mask
from ._grid import disk
from ._grid import normalize
from ._grid import solidity
from ._grid import touches_edge
from ._grid import unbin_xy
from ._grid import width
from .io import Stack
from .parameters import Parameters


@dataclass
class Region:
    """A nucleus or nucleus candidate on the binned grid.

    Attributes
    ----------
    pixels : ndarray of int
        Flat indices of the region's pixels in the binned image.
    lifetime : float
        Lifetime of its loop; for a merged region, the smallest of its pieces.
    window : tuple of int
        Slice range ``(start, stop)`` (stop exclusive) of the projection it was
        found in; for a merged region, that of its longest-lived piece.
    birth, death : float
        Bar endpoints in negated normalized intensity (for merged regions,
        those of the longest-lived piece).
    area_um2 : float
    solidity : float
        Area over convex hull area.
    width_um : float
        Twice the radius of the largest disk inside the region.
    border : bool
        Whether the region reaches the image edge (only with ``border_frame``).
    pieces : int
        Number of loops merged into the region.
    notches : int
        For a merged region, the most pinches found at the ends of a wall
        between two of its pieces.
    """

    pixels: NDArray[np.intp]
    lifetime: float
    window: tuple[int, int]
    birth: float
    death: float
    area_um2: float
    solidity: float
    width_um: float
    border: bool = False
    pieces: int = 1
    notches: int = 0

    def mask(self, shape: tuple[int, int]) -> NDArray[np.bool_]:
        """The region as a boolean image of the binned ``shape``."""
        out = np.zeros(shape, dtype=bool)
        out.flat[self.pixels] = True
        return out

    @property
    def centre_slice(self) -> int:
        """Middle slice of the projection window."""
        return (self.window[0] + self.window[1] - 1) // 2


@dataclass
class Segmentation:
    """Result of ``segment_2d``.

    Attributes
    ----------
    parameters : Parameters
    um_per_pixel : float
        Pixel size of the original image.
    shape : tuple of int
        Shape ``(rows, cols)`` of the original image.
    images : dict
        Prepared (binned, smoothed, normalized) projection per window.
    candidates : list of Region
        All candidates of all projections.
    pieces : list of Region
        Selected loops before merging.
    nuclei : list of Region
        Final nuclei, ordered by label (decreasing area): ``pieces`` after
        merging and cleaning, as set in ``parameters``.
    smooth_candidates : list of Region
        Candidates on the strongly smoothed projections, for outline cleaning
        (empty without ``clean_outlines``).
    """

    parameters: Parameters
    um_per_pixel: float
    shape: tuple[int, int]
    images: dict[tuple[int, int], NDArray[np.float64]]
    candidates: list[Region]
    pieces: list[Region]
    nuclei: list[Region]
    smooth_candidates: list[Region]

    def __post_init__(self) -> None:
        self.nuclei = by_label(self.nuclei)

    @property
    def binned_shape(self) -> tuple[int, int]:
        return next(iter(self.images.values())).shape

    @property
    def um_per_bin(self) -> float:
        return self.um_per_pixel * self.parameters.bin_factor

    def labels(self, regions: list[Region] | None = None, full_resolution: bool = True) -> NDArray[np.int32]:
        """Label image, 0 for background and 1, 2, ... for the regions in decreasing area.

        At full resolution every binned pixel becomes a block of
        ``bin_factor`` pixels; pixels beyond the last full block are background.
        """
        regions = self.nuclei if regions is None else by_label(regions)
        binned = np.zeros(self.binned_shape, dtype=np.int32)
        for label, region in enumerate(regions, start=1):
            binned.flat[region.pixels] = label
        return unbin_xy(binned, self.parameters.bin_factor, self.shape) if full_resolution else binned

    def table(self, regions: list[Region] | None = None) -> list[dict]:
        """One row per region (labels as in ``labels``): area, centroid in µm, shape and bar data."""
        regions = self.nuclei if regions is None else by_label(regions)
        rows = []
        for label, region in enumerate(regions, start=1):
            rr, cc = np.unravel_index(region.pixels, self.binned_shape)
            rows.append({
                "label": label,
                "area_um2": round(float(region.area_um2), 2),
                "centroid_x_um": round(float((cc.mean() + 0.5) * self.um_per_bin), 2),
                "centroid_y_um": round(float((rr.mean() + 0.5) * self.um_per_bin), 2),
                "width_um": round(float(region.width_um), 2),
                "solidity": round(float(region.solidity), 3),
                "border": region.border,
                "lifetime": round(float(region.lifetime), 4),
                "z_window": f"{region.window[0]}-{region.window[1] - 1}",
                "pieces": region.pieces,
                "notches": region.notches,
            })
        return rows


def by_label(regions: list[Region]) -> list[Region]:
    """Regions in label order: decreasing area."""
    return sorted(regions, key=lambda r: -r.area_um2)


# ----- preparing projections -----

def prepare(projection: NDArray, um_per_pixel: float, parameters: Parameters,
            smooth_um: float | None = None) -> NDArray[np.float64]:
    """Bin, smooth (default ``parameters.smooth_um``) and scale a projection so its ``normalize_percentile`` is 1."""
    smooth_um = parameters.smooth_um if smooth_um is None else smooth_um
    binned = bin_xy(projection, parameters.bin_factor)
    if smooth_um > 0:
        binned = ndimage.gaussian_filter(binned, smooth_um / (um_per_pixel * parameters.bin_factor))
    return normalize(binned, parameters.normalize_percentile)


# ----- loops of one projection -----

def cycle_region(chain, shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Pixels enclosed by a cubical 1-cycle, holes filled.

    A grid edge with doubled coordinates ``(2r + 1, 2c)`` separates pixels
    ``(r, c - 1)`` and ``(r, c)``; along every row, a pixel is inside when an
    odd number of the cycle's edges lie to its left.
    """
    toggles = np.zeros((shape[0], shape[1] + 1), dtype=np.int8)
    for (a0, a1), _ in chain.unsigned().signed_simplices:
        if a0 & 1 and not a1 & 1:
            toggles[a0 // 2, a1 // 2] ^= 1
    inside = np.bitwise_xor.accumulate(toggles[:, :-1], axis=1).astype(bool)
    return ndimage.binary_fill_holes(inside)


def find_candidates(image: NDArray[np.float64], window: tuple[int, int], um_per_bin: float,
                    parameters: Parameters) -> list[Region]:
    """Loops of the bright structures of one prepared projection that may be nuclei."""
    frame = 1 if parameters.border_frame else 0
    values = np.pad(image, frame, constant_values=image.max()) if frame else image
    forest = PersistenceForest.from_cubical_grid(-values)
    pixel_area = um_per_bin**2
    candidates = []
    for bar in forest.barcode:
        lifetime = bar.death - bar.birth
        if lifetime < parameters.min_lifetime:
            continue
        chain = forest.cycle_for_bar_at(bar, bar.birth + parameters.cycle_position * lifetime)
        mask = cycle_region(chain, values.shape)[frame:values.shape[0] - frame, frame:values.shape[1] - frame]
        area = mask.sum() * pixel_area
        if not parameters.min_area_um2 <= area <= parameters.max_area_um2:
            continue
        region_width = width(mask) * um_per_bin
        if region_width < parameters.min_width_um:
            continue
        region_solidity = solidity(mask)
        if region_solidity < parameters.min_solidity:
            continue
        candidates.append(Region(np.flatnonzero(mask), lifetime, window, bar.birth, bar.death, area,
                                 region_solidity, region_width, touches_edge(mask)))
    return candidates


def select(candidates: list[Region], max_overlap: float) -> list[Region]:
    """Keep candidates in decreasing lifetime times area unless they overlap a kept one."""
    kept: list[Region] = []
    for candidate in sorted(candidates, key=lambda r: -r.lifetime * r.area_um2):
        if all(np.intersect1d(candidate.pixels, k.pixels, assume_unique=True).size
               <= max_overlap * min(candidate.pixels.size, k.pixels.size) for k in kept):
            kept.append(candidate)
    return kept


# ----- merging pieces cut by envelope folds -----

def _try_merge(a: NDArray[np.bool_], b: NDArray[np.bool_], wall_px: float,
               notch_px_per_radius: float) -> tuple[NDArray[np.bool_], int] | None:
    """The merged mask and its number of pinches, or None if ``a`` and ``b`` share no wall."""
    reach = max(1, int(np.ceil(wall_px)))
    structure = ndimage.generate_binary_structure(2, 1)
    near_a = ndimage.binary_dilation(a, structure, iterations=reach)
    near_b = ndimage.binary_dilation(b, structure, iterations=reach)
    wall = near_a & near_b & ~a & ~b
    if not wall.any():
        return None
    merged = ndimage.binary_fill_holes(a | b | wall)
    # A pinch is a dent of the merged outline next to the wall: a component of
    # convex hull minus region that touches the wall's surroundings and
    # reaches deep enough into the hull.
    rows, cols = np.nonzero(merged)
    pad = 2
    window = (slice(max(rows.min() - pad, 0), rows.max() + pad + 1),
              slice(max(cols.min() - pad, 0), cols.max() + pad + 1))
    sub = merged[window]
    hull = convex_hull_mask(sub)
    dents, n = ndimage.label(hull & ~sub)
    depth = ndimage.distance_transform_edt(np.pad(hull, 1))[1:-1, 1:-1]
    wall_zone = ndimage.binary_dilation(wall[window], structure, iterations=reach + 1)
    radius = np.sqrt(min(a.sum(), b.sum()) / np.pi)
    notches = 0
    for label in range(1, n + 1):
        dent = dents == label
        if (dent & wall_zone).any() and depth[dent].max() >= notch_px_per_radius * radius:
            notches += 1
    return merged, notches


def merge_pieces(pieces: list[Region], shape: tuple[int, int], um_per_bin: float,
                 parameters: Parameters) -> list[Region]:
    """Merge pieces sharing a thin wall unless the merged outline has a waist.

    Pairs are merged one at a time, the pair with the least wall area first,
    until no pair shares a wall without a waist (pinches at both wall ends).
    Only pairs whose bounding boxes come within the wall width are tested,
    on a window around the pair.
    """
    wall_px = parameters.wall_um / um_per_bin
    margin = int(np.ceil(wall_px)) + 4
    regions = list(pieces)
    boxes = [bounding_box(r.pixels, shape) for r in regions]
    tested: dict[tuple[int, int], tuple | None] = {}
    ids = list(range(len(regions)))

    def test(i: int, j: int):
        a, b = boxes[i], boxes[j]
        if a[0] > b[1] + margin or b[0] > a[1] + margin or a[2] > b[3] + margin or b[2] > a[3] + margin:
            return None
        window = (slice(max(min(a[0], b[0]) - margin, 0), min(max(a[1], b[1]) + margin, shape[0])),
                  slice(max(min(a[2], b[2]) - margin, 0), min(max(a[3], b[3]) + margin, shape[1])))
        result = _try_merge(regions[i].mask(shape)[window], regions[j].mask(shape)[window],
                            wall_px, parameters.notch_depth)
        if result is None:
            return None
        merged = np.zeros(shape, dtype=bool)
        merged[window] = result[0]
        return merged, result[1]

    while True:
        best = None
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                key = (ids[x], ids[y])
                if key not in tested:
                    tested[key] = test(*key)
                result = tested[key]
                if result is None or result[1] >= 2:
                    continue
                merged, notches = result
                area = merged.sum() * um_per_bin**2
                if area > parameters.max_area_um2:
                    continue
                gap = merged.sum() - regions[key[0]].pixels.size - regions[key[1]].pixels.size
                if best is None or gap < best[0]:
                    best = (gap, key, merged, notches, area)
        if best is None:
            return [regions[i] for i in ids]
        _, (i, j), merged, notches, area = best
        a, b = regions[i], regions[j]
        lead = a if a.lifetime >= b.lifetime else b
        regions.append(Region(np.flatnonzero(merged), min(a.lifetime, b.lifetime), lead.window, lead.birth,
                              lead.death, area, solidity(merged), width(merged) * um_per_bin,
                              a.border or b.border, a.pieces + b.pieces, max(notches, a.notches, b.notches)))
        boxes.append(bounding_box(regions[-1].pixels, shape))
        ids = [k for k in ids if k not in (i, j)] + [len(regions) - 1]


# ----- cleaning outlines -----

def clean_outlines(nuclei: list[Region], smooth_candidates: list[Region], shape: tuple[int, int],
                   um_per_bin: float, parameters: Parameters) -> list[Region]:
    """Replace each outline by its smooth counterpart grown back by at most ``clean_reach_um``.

    The smooth counterpart of a nucleus is the union of the candidates on the
    smoothed projections that lie mostly (70 %) inside it, from the nucleus's
    own projection if there are any there. Nuclei without one are kept.
    """
    reach = max(1, int(round(parameters.clean_reach_um / um_per_bin)))
    structure = disk(reach)
    cleaned = []
    for nucleus in nuclei:
        inside = [c for c in smooth_candidates
                  if np.intersect1d(c.pixels, nucleus.pixels, assume_unique=True).size >= 0.7 * c.pixels.size]
        own = [c for c in inside if c.window == nucleus.window]
        matches = own or inside
        if not matches:
            cleaned.append(nucleus)
            continue
        pixels = np.unique(np.concatenate([nucleus.pixels] + [c.pixels for c in matches]))
        r0, r1, c0, c1 = bounding_box(pixels, shape)
        window = (slice(max(r0 - reach - 1, 0), min(r1 + reach + 1, shape[0])),
                  slice(max(c0 - reach - 1, 0), min(c1 + reach + 1, shape[1])))
        original = nucleus.mask(shape)[window]
        smooth = np.zeros(shape, dtype=bool)
        for c in matches:
            smooth.flat[c.pixels] = True
        smooth = smooth[window]
        grown = ndimage.binary_fill_holes(smooth | (original & ndimage.binary_dilation(smooth, structure)))
        mask = np.zeros(shape, dtype=bool)
        mask[window] = grown
        cleaned.append(replace(nucleus, pixels=np.flatnonzero(mask), area_um2=mask.sum() * um_per_bin**2,
                               solidity=solidity(mask), width_um=width(mask) * um_per_bin,
                               border=nucleus.border and touches_edge(mask)))
    return cleaned


# ----- whole pipeline -----

def segment_2d(stack: Stack, parameters: Parameters = Parameters()) -> Segmentation:
    """Segment the nuclei of a z-stack (or a single image) in 2D.

    Parameters
    ----------
    stack : Stack
        From ``read_stack``, or ``Stack(array, um_per_pixel)`` for an array.
    parameters : Parameters

    Returns
    -------
    Segmentation
    """
    um_per_pixel = stack.um_per_pixel
    um_per_bin = um_per_pixel * parameters.bin_factor
    images, candidates, smooth_candidates = {}, [], []
    for window in parameters.windows(len(stack.data)):
        projection = stack.data[window[0]:window[1]].mean(axis=0)
        image = prepare(projection, um_per_pixel, parameters)
        images[window] = image
        candidates += find_candidates(image, window, um_per_bin, parameters)
        if parameters.clean_outlines:
            smooth = prepare(projection, um_per_pixel, parameters, smooth_um=parameters.clean_smooth_um)
            smooth_candidates += find_candidates(smooth, window, um_per_bin, parameters)
    pieces = select(candidates, parameters.max_overlap)
    segmentation = Segmentation(parameters, um_per_pixel, tuple(stack.data.shape[1:]), images, candidates,
                                pieces, [], smooth_candidates)
    return postprocess(segmentation)


def postprocess(segmentation: Segmentation, **changes) -> Segmentation:
    """Merge and clean the selected loops again, optionally with changed parameters.

    Only parameters of these two steps may change (``merge``, ``wall_um``,
    ``notch_depth``, ``clean_outlines``, ``clean_reach_um``); the forests are
    not rebuilt. Outlines can only be cleaned if the segmentation was made
    with ``clean_outlines=True``.
    """
    allowed = {"merge", "wall_um", "notch_depth", "clean_outlines", "clean_reach_um"}
    if not set(changes) <= allowed:
        raise ValueError(f"postprocess can change only {sorted(allowed)}, not {sorted(set(changes) - allowed)}")
    parameters = replace(segmentation.parameters, **changes)
    shape, um_per_bin = segmentation.binned_shape, segmentation.um_per_bin
    nuclei = list(segmentation.pieces)
    if parameters.merge:
        nuclei = merge_pieces(nuclei, shape, um_per_bin, parameters)
    if parameters.clean_outlines:
        if not segmentation.parameters.clean_outlines:
            raise ValueError("cleaning needs a segmentation made with clean_outlines=True")
        nuclei = clean_outlines(nuclei, segmentation.smooth_candidates, shape, um_per_bin, parameters)
    return replace(segmentation, parameters=parameters, nuclei=nuclei)


def reference_image(segmentation: Segmentation) -> NDArray[np.float64]:
    """Mean of the prepared projections, for display."""
    return np.mean(list(segmentation.images.values()), axis=0)
