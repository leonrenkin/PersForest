"""``segment_image``: the whole segmentation of one image, the main entry point."""

from dataclasses import dataclass
from dataclasses import field
from functools import partial
import warnings

import numpy as np
from numpy.typing import NDArray

from .images import downsample_labels
from .images import gaussian_upsample
from .inference import Background
from .inference import estimate_background
from .inference import infer_parameters
from .inference import needs_background
from .parameters import Parameters
from .pipeline import Result
from .pipeline import segment
from .tiles import TileResult
from .tiles import assemble_groups
from .tiles import segment_tiled

# Images with more pixels are segmented in tiles also at 1x: one forest of a large image needs gigabytes.
LARGE_IMAGE = 600 * 600


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
        Segments on the grid of ``image``; 0 is background, 1, 2, ... are segments.
    factor : int
        Upsampling factor.
    groups : ndarray of int
        The outline each segment comes from: touching segments of one group
        were cut apart by the lobe splitting, of different groups by the forest.
    result : Result or None
        All intermediate products, when the image was segmented as one forest
        (at factor 1 and at most ``LARGE_IMAGE`` pixels).
    tiles : list of TileResult
        Per-tile summaries, when segmented in tiles.
    parameters : Parameters or None
        The parameters used, with inferred intensity parameters filled in.
    background : Background or None
        The background estimate the intensity parameters were inferred from;
        None if all of them were given.
    """

    image: NDArray[np.float64]
    labels: NDArray[np.int_]
    factor: int
    groups: NDArray[np.int_]
    result: Result | None = None
    tiles: list[TileResult] = field(default_factory=list)
    parameters: Parameters | None = None
    background: Background | None = None

    @property
    def n_segments(self) -> int:
        return len(np.unique(self.labels[self.labels > 0]))

    @property
    def original_labels(self) -> NDArray[np.int_]:
        """The labels on the original pixels: the most frequent label of each block of fine pixels."""
        return downsample_labels(self.labels, self.factor)


def segment_image(image: NDArray[np.float64], parameters: Parameters = Parameters(), workers: int = 1,
                  tile_size: int = 64, tile_margin: int = 12) -> Segmentation:
    """Segment the bright droplets of a 2D image with intensities in [0, 1].

    Parameters
    ----------
    image : ndarray, shape (rows, columns)
        Intensities in [0, 1], e.g. from ``images.preprocess``.
    parameters : Parameters
        Intensity parameters left at None are inferred from ``image`` with
        ``infer_parameters``; ``Segmentation.parameters`` holds the values used.
    workers : int
        Processes segmenting tiles in parallel. In a script, call
        ``segment_image`` under ``if __name__ == "__main__":`` when
        ``workers > 1``.
    tile_size, tile_margin : int
        Upsampled images, and images of more than ``LARGE_IMAGE`` pixels, are
        segmented in tiles of ``tile_size`` original pixels with a margin of
        ``tile_margin`` original pixels, which should exceed the largest
        droplet with its bright surroundings. Tile edges inside the image are
        framed just below ``min_level``, so a droplet reaching one is
        detected as cut; a tile that cuts a droplet it owns is redone with
        the margin doubled, up to twice (``tiles``).

    Warns with ``InferredLevelsWarning`` when intensity levels are inferred.
    Warns if the smoothing cut segments into several pieces of at least
    ``min_area``: ``smooth_radius`` is then too large. Warns if tiles still
    cut droplets after the last widening of their margin: these droplets are
    dropped, and ``tile_margin`` is too small.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if image.min() < 0 or image.max() > 1:
        raise ValueError(f"intensities must lie in [0, 1], got {image.min():g} to {image.max():g}; "
                         "scale the image first, e.g. with dropseg.preprocess")
    # Inferred once from the whole image, so that every tile is segmented with the same levels.
    background = estimate_background(image) if needs_background(parameters) else None
    parameters = infer_parameters(image, parameters, background)
    factor = parameters.upsample
    if factor == 1 and image.size <= LARGE_IMAGE:
        result = segment(image, parameters.settings())
        segmentation = Segmentation(image, result.labels, 1, result.groups, result=result)
        n_broken = len(result.broken)
    else:
        labels, tile_results = segment_tiled(image, parameters.settings("diffs"), upsampler(parameters), factor,
                                             tile_size, tile_margin, workers)
        segmentation = Segmentation(upsampler(parameters)(image), labels, factor,
                                    assemble_groups(tile_results, image.shape, factor), tiles=tile_results)
        n_broken = sum(tile_result.broken for tile_result in tile_results)
        if n_cut := sum(tile_result.cut for tile_result in tile_results):
            warnings.warn(f"tiles cut {n_cut} droplet piece(s) even after their margin was widened from "
                          f"tile_margin = {tile_margin} px (a droplet counts once in each tile it reaches); "
                          "these droplets are missing from the labels. Pass a larger tile_margin (and "
                          "tile_size), above the largest droplet with its bright surroundings", stacklevel=2)
    if n_broken:
        warn_broken(n_broken, parameters)
    segmentation.parameters, segmentation.background = parameters, background
    return segmentation


def warn_broken(n_broken: int, parameters: Parameters):
    """Warn that the smoothing cut segments into several droplet-sized pieces."""
    warnings.warn(f"smoothing cut {n_broken} segment(s) into several pieces of at least min_area = "
                  f"{parameters.min_area:g} px; smooth_radius = {parameters.smooth_radius} is too large for this "
                  "image and these parameters", stacklevel=3)
