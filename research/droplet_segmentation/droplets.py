"""The droplet pipeline with few parameters, the same at every resolution.

``Parameters`` holds one or two parameters per step; everything else of
``pipeline.Settings`` is derived from them or fixed. Lengths are in pixels of
the original image and intensities in [0, 1], so the same parameters segment
the image at its own resolution (``upsample=1``) or upsampled (``upsample=4``).

The steps and the parameters they read (``STEPS``):

1. **peaks**: the bars of the superlevel persistence forest at least
   ``min_bar_length`` long, one per candidate droplet.
2. **branches**: a bar born below ``split_level`` is a droplet of its own; a
   bar born above it (a peak that separates from its neighbor only near the
   top) belongs to the droplet it splits off.
3. **outlines**: the outline of each droplet is its first representative at or
   above ``min_level``, and at least ``birth_offset`` x ``min_bar_length``
   above the bar's birth. With ``convexity_tol``, the search
   climbs further, to the first outline whose convex-hull perimeter deficit is
   at most that tolerance.
4. **lobes**: 8-shaped outlines are split where the alpha filtration of the
   outline's vertices splits off a lobe whose waist is at most ``lobe_waist``
   times as wide as the lobe (birth / death of the alpha bar), and each
   lobe rises at least ``min_bar_length`` above the birth of the outline's bar
   (step 1 for the lobes, which have no bar of their own).
5. **flooding**: segments grow from bright to dark down to ``min_level``;
   touching segments meet in the valley between them.
6. **smoothing**: an alpha opening of radius ``smooth_radius`` smooths the
   boundaries.
7. **filter**: segments smaller than ``min_area`` are dropped.

``segment_image`` runs all steps on a whole image, in tiles when upsampled.
``Stages`` runs them on a window and caches every step, so that changing a
parameter recomputes only its step and the later ones; ``tuning`` builds the
interactive views of the notebook on it.
"""

from collections import OrderedDict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from functools import partial
import math
import warnings

import numpy as np
from numpy.typing import NDArray

from persforest import PersistenceForest

from pipeline import Result
from pipeline import Settings
from pipeline import find_lobes
from pipeline import find_outlines
from pipeline import find_peaks
from pipeline import outlines_result
from pipeline import relabel
from pipeline import segment
from pipeline import split_forest
from real_images import gaussian_upsample
from tiles import TileResult
from tiles import assemble_groups
from tiles import segment_tiled

STEPS = {
    "peaks": ("upsample", "kernel_width", "min_bar_length"),
    "branches": ("split_level",),
    "outlines": ("min_level", "convexity_tol", "birth_offset"),
    "lobes": ("split_lobes", "lobe_waist"),
    "flooding": (),
    "smoothing": ("smooth_radius",),
    "filter": ("min_area",),
}


@dataclass(frozen=True)
class Parameters:
    """Parameters of the droplet pipeline; lengths in original pixels, intensities in [0, 1].

    The first block is meant to be tuned; the second has defaults relative to
    the first and rarely needs changing.

    Attributes
    ----------
    upsample : int
        Segment on a grid this many times finer (1 or 4).
    kernel_width : float
        Width of the Gaussian upsampling kernel, original pixels.
    min_bar_length : float
        Step 1: minimal rise of a peak above the saddle to its neighbor or the background.
    split_level : float
        Step 2: peaks separating from their neighbor below this level are
        droplets of their own; above it, they are part of the neighbor.
    min_level : float
        Step 3: outlines are taken at this level, or after the last split of
        the droplet's bar below ``split_level`` if that is higher.
    convexity_tol : float or None
        Step 3: if set, the outline search climbs to the first outline whose
        convex-hull perimeter deficit is at most this (but not past
        ``split_level``); None takes the first outline. Meant against messy
        outlines on a noisy background: about 0.4 to 0.5, since a deficit of
        0.3 is still a reasonable punctum.
    split_lobes : bool
        Step 4: split 8-shaped outlines.
    smooth_radius : float or None
        Step 6: alpha radius smoothing the boundaries, original pixels; None to skip.
    min_area : float
        Step 7: smallest segment kept, original pixels.
    birth_offset : float
        Step 3, in intensity: outlines lie at least this fraction of
        ``min_bar_length`` above the superlevel bar's birth, past the wiggly
        outlines just after it.
    lobe_waist : float
        Step 4, geometric: in the alpha filtration of the outline's vertices,
        the bar of a split-off lobe is born at the split radius s (half the
        waist width) and dies at the lobe radius rho. The outline is split if
        s / rho <= ``lobe_waist``: the waist is at most that fraction of the
        lobe's width. Independent of the size of the droplet, the resolution
        and the intensities.
    """

    upsample: int = 1
    kernel_width: float = 0.35
    min_bar_length: float = 0.2
    split_level: float = 0.3
    min_level: float = 0.1
    convexity_tol: float | None = None
    split_lobes: bool = True
    smooth_radius: float | None = None
    min_area: float = 2.0

    birth_offset: float = 0.1
    lobe_waist: float = 0.75

    def __post_init__(self):
        if self.min_level >= self.split_level:
            raise ValueError(f"min_level {self.min_level} must lie below split_level {self.split_level}")
        if self.upsample < 1:
            raise ValueError(f"upsample must be a positive integer, got {self.upsample}")

    def settings(self, storage: str = "cycles") -> Settings:
        """The full settings, in pixels of the upsampled grid.

        ``storage="diffs"`` reconstructs representatives instead of storing
        them, for large grids; the segments are the same.
        """
        factor = self.upsample
        return Settings(
            storage=storage,
            min_bar_length=self.min_bar_length,
            selection="threshold", max_defect=math.inf if self.convexity_tol is None else self.convexity_tol,
            max_value=self.split_level, min_value=self.min_level,
            birth_offset=self.birth_offset * self.min_bar_length,
            split_defect=0.02 if self.split_lobes else math.inf,
            lobe_waist_ratio=self.lobe_waist,
            lobe_min_rise=self.min_bar_length,
            seed_late_peaks=False, fill_in="flood",
            # Segments grow down to the outline level and no further: below it the flood follows dim
            # haze around darker pockets, which leaves holes and sprawling segments.
            grow_floor=self.min_level,
            boundary_radius=None if not self.smooth_radius else self.smooth_radius * factor,
            min_area=self.min_area * factor**2,
        )

    def record(self) -> dict:
        """The parameters as a plain dict, for saving."""
        return asdict(self)


def upsampler(parameters: Parameters):
    """The upsampling of ``parameters`` as a picklable function of an image."""
    return partial(gaussian_upsample, factor=parameters.upsample, width=parameters.kernel_width)


@dataclass
class Segmentation:
    """A whole image segmented with ``segment_image``.

    Attributes
    ----------
    image : ndarray
        The image segmented: upsampled if ``factor > 1``.
    labels : ndarray of int
        Segments on the grid of ``image``; 0 is background.
    factor : int
        Upsampling factor.
    groups : ndarray of int
        The outline each segment comes from: touching segments of one group
        were cut apart by the lobe splitting, of different groups by the forest.
    result : Result or None
        All intermediate products, at factor 1 (the image is one forest).
    tiles : list of TileResult
        Per-tile summaries, when upsampled.
    """

    image: NDArray[np.float64]
    labels: NDArray[np.int_]
    factor: int
    groups: NDArray[np.int_]
    result: Result | None = None
    tiles: list[TileResult] = field(default_factory=list)


def segment_image(image: NDArray[np.float64], parameters: Parameters = Parameters(), workers: int = 1,
                  tile_size: int = 64, tile_margin: int = 12) -> Segmentation:
    """Segment a whole image; upsampled images are segmented in tiles.

    ``tile_size`` and ``tile_margin`` are in original pixels; the margin must
    exceed the largest droplet (see ``tiles``). Warns if the smoothing cut
    segments into several pieces of at least ``min_area``.
    """
    factor = parameters.upsample
    if factor == 1:
        result = segment(image, parameters.settings())
        segmentation = Segmentation(image, result.labels, 1, result.groups, result=result)
        n_broken = len(result.broken)
    else:
        labels, tile_results = segment_tiled(image, parameters.settings("diffs"), upsampler(parameters), factor,
                                             tile_size, tile_margin, workers)
        segmentation = Segmentation(upsampler(parameters)(image), labels, factor,
                                    assemble_groups(tile_results, image.shape, factor), tiles=tile_results)
        n_broken = sum(tile_result.broken for tile_result in tile_results)
    if n_broken:
        warn_broken(n_broken, parameters)
    return segmentation


def warn_broken(n_broken: int, parameters: Parameters):
    """Warn that the smoothing cut segments into several droplet-sized pieces."""
    warnings.warn(f"smoothing cut {n_broken} segment(s) into several pieces of at least min_area = "
                  f"{parameters.min_area:g} px; smooth_radius = {parameters.smooth_radius} is too large for this "
                  "image and these parameters", stacklevel=3)


class Stages:
    """The steps of the pipeline on a window of an image, each cached on the parameters it reads.

    The window is segmented with a margin around it, so that droplets at its
    edge are complete. Coordinates of the products are fine pixels of the
    upsampled window with margin; ``view`` is the window in those coordinates.

    Parameters
    ----------
    image : ndarray
        The whole image, original pixels.
    window : tuple of slice
        ``(rows, columns)`` in original pixels.
    margin : int
        Original pixels added on every side.
    cache_size : int
        Products kept per step, so that going back to earlier values is fast.
    """

    def __init__(self, image: NDArray[np.float64], window: tuple[slice, slice], margin: int = 12,
                 cache_size: int = 6):
        self.window = window
        self.outer = tuple(slice(max(s.start - margin, 0), min(s.stop + margin, n)) for s, n in zip(window, image.shape))
        self.original = image[self.outer]
        self.cache_size = cache_size
        self._cache = {step: OrderedDict() for step in ("image", "forest", *STEPS)}

    def _cached(self, step: str, key: tuple, compute):
        cache = self._cache[step]
        if key not in cache:
            cache[key] = compute()
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    @staticmethod
    def _key(parameters: Parameters, step: str) -> tuple:
        # A step depends on its own parameters and those of every earlier step.
        names = []
        for name, fields in STEPS.items():
            names += fields
            if name == step:
                break
        return tuple(getattr(parameters, name) for name in names)

    def view(self, parameters: Parameters) -> tuple[slice, slice]:
        """The window in fine pixels of the segmented image."""
        factor = parameters.upsample
        return tuple(slice((s.start - o.start) * factor, (s.stop - o.start) * factor)
                     for s, o in zip(self.window, self.outer))

    def image(self, parameters: Parameters) -> NDArray[np.float64]:
        """The window with margin, upsampled."""
        key = (parameters.upsample, parameters.kernel_width)
        return self._cached("image", key, lambda: upsampler(parameters)(self.original))

    def settings(self, parameters: Parameters) -> Settings:
        return parameters.settings()

    def forest(self, parameters: Parameters) -> PersistenceForest:
        key = (parameters.upsample, parameters.kernel_width)
        return self._cached("forest", key, lambda: PersistenceForest.from_cubical_grid(
            self.image(parameters), storage=self.settings(parameters).storage))

    def peaks(self, parameters: Parameters) -> list:
        """Step 1: the candidate bars."""
        return self._cached("peaks", self._key(parameters, "peaks"),
                            lambda: find_peaks(self.forest(parameters), self.settings(parameters)))

    def branches(self, parameters: Parameters) -> tuple:
        """Step 2: ``(branches, outlined bars, late bars)``."""
        return self._cached("branches", self._key(parameters, "branches"), lambda: split_forest(
            self.forest(parameters), self.peaks(parameters), self.settings(parameters)))

    def outlines(self, parameters: Parameters) -> dict:
        """Step 3: the selection of every outlined bar."""
        def compute():
            branches, outlined, _ = self.branches(parameters)
            return find_outlines(self.forest(parameters), branches, outlined, self.settings(parameters))
        return self._cached("outlines", self._key(parameters, "outlines"), compute)

    def lobes(self, parameters: Parameters) -> dict:
        """Step 4: the splits of the examined outlines."""
        return self._cached("lobes", self._key(parameters, "lobes"), lambda: find_lobes(
            self.forest(parameters), self.outlines(parameters), self.image(parameters), self.settings(parameters)))

    def result(self, parameters: Parameters) -> Result:
        """Steps 5 to 7: the segments, with every intermediate product."""
        def compute():
            branches, _, late = self.branches(parameters)
            image = self.image(parameters)
            partial_result = outlines_result(self.forest(parameters), self.peaks(parameters), branches,
                                             self.outlines(parameters), late, self.lobes(parameters), image.shape)
            return relabel(partial_result, image, self.settings(parameters))
        return self._cached("filter", self._key(parameters, "filter"), compute)
