"""Microscope images for the segmentation: loading, preprocessing, and look-alike scenes.

The segmentation works on intensities in [0, 1] (the thresholds of
``Settings`` are in these units). ``load_image`` reads an image file as
floats; ``preprocess`` scales it to [0, 1] and optionally crops and blurs
it. Upsampling gives small droplets more pixels, so that the shape measures
and the alpha lobes see a smooth outline rather than a staircase of a few
pixels; ``downsample_labels`` maps the labels back.

``gaussian_upsample`` is a kernel average on the finer grid: every fine pixel
is the Gaussian-weighted mean of the original pixels around it. Its level
sets are smooth curves through the original samples' neighborhoods, so a
waist between two droplets or the roundness of a droplet can show below the
pixel size. The kernel width trades smoothness of the outlines against
blurring neighboring droplets together.

``camera_scene`` draws simulated scenes of small droplets rendered like an
8-bit camera image (quantized and saturating), for testing settings on data
with known truth; ``fine_truth`` draws their truth on an upsampled grid.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

from simulate import Droplet
from simulate import droplet_image
from simulate import gaussian_blur
from simulate import sample_droplets
from simulate import true_labels

IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def image_files(folder) -> list[Path]:
    """Image files in ``folder``, sorted by name."""
    return sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def load_image(path, channel: int | None = None) -> tuple[NDArray[np.float64], float]:
    """Read a 2D image as floats, with the largest value its type can hold.

    Color images need ``channel``; a gray image saved as RGB has equal channels
    and is read as gray.
    """
    raw = np.asarray(Image.open(path))
    if raw.ndim == 3:
        if channel is None:
            if not all((raw[..., 0] == raw[..., c]).all() for c in range(1, min(raw.shape[-1], 3))):
                raise ValueError(f"{path} has {raw.shape[-1]} channels; pass channel")
            channel = 0
        raw = raw[..., channel]
    full_scale = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else float(raw.max())
    return raw.astype(float), full_scale


@dataclass(frozen=True)
class Preprocessing:
    """How a raw image becomes the input of the segmentation.

    Attributes
    ----------
    normalize : {"full scale", "percentile"}
        Divide by the largest value of the image type, or map the
        ``low_percentile`` and ``high_percentile`` of the image to 0 and 1.
        Values outside [0, 1] are clipped.
    low_percentile, high_percentile : float
    crop : tuple of slice, optional
        ``(rows, columns)`` window of the raw image.
    blur : float
        Standard deviation in raw pixels of a Gaussian blur against noise;
        0 disables it.
    """

    normalize: str = "full scale"
    low_percentile: float = 1.0
    high_percentile: float = 99.9
    crop: tuple[slice, slice] | None = None
    blur: float = 0.0


def preprocess(raw: NDArray[np.float64], full_scale: float, options: Preprocessing) -> NDArray[np.float64]:
    """Scale ``raw`` to [0, 1], then crop and blur it."""
    if options.normalize == "full scale":
        low, high = 0.0, full_scale
    elif options.normalize == "percentile":
        low, high = np.percentile(raw, [options.low_percentile, options.high_percentile])
    else:
        raise ValueError(f"normalize must be 'full scale' or 'percentile', got {options.normalize!r}")
    image = np.clip((raw - low) / (high - low), 0.0, 1.0)
    if options.crop is not None:
        image = image[options.crop]
    return gaussian_blur(image, options.blur)


def gaussian_upsample(image: NDArray[np.float64], factor: int, width: float) -> NDArray[np.float64]:
    """Gaussian kernel average on a grid ``factor`` times finer.

    Fine pixel ``(i, j)`` covers ``[i, i + 1] / factor x [j, j + 1] / factor``
    of the original grid; its value is the mean of the original pixels weighted
    by a Gaussian of standard deviation ``width`` (original pixels) in the
    distance between the centers. The kernel is separable, so the weights are
    normalized per axis; near the border only the pixels inside count.
    Factor 1 returns the image unchanged.
    """
    if factor == 1:
        return image

    def weights(length):
        fine = (np.arange(length * factor) + 0.5) / factor
        distance = fine[:, None] - (np.arange(length) + 0.5)[None, :]
        kernel = np.exp(-0.5 * (distance / width) ** 2)
        return kernel / kernel.sum(axis=1, keepdims=True)

    return weights(image.shape[0]) @ image @ weights(image.shape[1]).T


def downsample_labels(labels: NDArray[np.int_], factor: int) -> NDArray[np.int_]:
    """Labels of the original pixels: the most frequent label of each block (the first of a tie)."""
    if factor == 1:
        return labels
    rows, columns = labels.shape[0] // factor, labels.shape[1] // factor
    blocks = labels[:rows * factor, :columns * factor].reshape(rows, factor, columns, factor)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(rows, columns, factor * factor)
    counts = (blocks[..., :, None] == blocks[..., None, :]).sum(axis=-1)
    return np.take_along_axis(blocks, counts.argmax(axis=-1)[..., None], axis=-1)[..., 0]


def camera(image: NDArray[np.float64], levels: int = 255) -> NDArray[np.float64]:
    """Clip to [0, 1] and quantize to ``levels`` steps, like an 8-bit camera."""
    return np.round(np.clip(image, 0.0, 1.0) * levels) / levels


@dataclass(frozen=True)
class CameraScene:
    seed: int
    droplets: list[Droplet]
    image: NDArray[np.float64]
    truth: NDArray[np.int_]

    @property
    def name(self) -> str:
        return f"camera scene, seed {self.seed}"


def camera_scene(seed: int, shape: tuple[int, int] = (128, 128), n_droplets: int = 180,
                 radius: tuple[float, float] = (0.8, 2.2), peak: tuple[float, float] = (0.1, 1.3),
                 blur: float = 0.5, background: float = 0.01, noise: float = 0.01,
                 background_variation: float = 0.02, layout: str = "uniform") -> CameraScene:
    """Small dense droplets rendered like an 8-bit camera image.

    The defaults match the Golgi image of ``data_lucy``: the quartiles of the
    areas above 0.1 (about 7, 11 and 20 px), the background noise, and peaks
    above 1 that saturate, as about half of the real maxima do. The truth is
    the droplets' footprints, as in ``simulate``.
    """
    rng = np.random.default_rng([seed, 2])
    droplets = sample_droplets(rng, shape, layout, n_droplets=n_droplets, radius=radius, peak=peak,
                               elongated_fraction=0.5, max_elongation=2.0)
    image = droplet_image(droplets, shape, blur=blur, background=background, noise=noise, seed=seed,
                          background_variation=background_variation, background_scale=20.0)
    return CameraScene(seed, droplets, camera(image), true_labels(droplets, shape))


def segment_table(labels: NDArray[np.int_], image: NDArray[np.float64], factor: int = 1):
    """Area, equivalent radius, centroid and intensity of every segment, as a DataFrame.

    ``labels`` and ``image`` may be upsampled by ``factor``; areas, radii and
    positions are in original pixels (the centroid as a pixel index).
    """
    import pandas as pd

    ids = np.unique(labels[labels > 0])
    areas = ndimage.sum_labels(np.ones_like(image), labels, ids) / factor**2
    centroids = (np.array(ndimage.center_of_mass(np.ones_like(image), labels, ids)).reshape(-1, 2) + 0.5) / factor - 0.5
    return pd.DataFrame({
        "label": ids, "area": areas, "radius": np.sqrt(areas / np.pi),
        "row": centroids[:, 0], "column": centroids[:, 1],
        "mean": ndimage.mean(image, labels, ids), "max": ndimage.maximum(image, labels, ids),
    }).set_index("label")


def fine_truth(droplets: list[Droplet], shape: tuple[int, int], factor: int) -> NDArray[np.int_]:
    """True labels on a grid ``factor`` times finer: the footprints drawn at the fine pixel centers."""
    scaled = [Droplet(droplet.row * factor, droplet.column * factor, droplet.radius * factor, droplet.peak,
                      droplet.elongation, droplet.angle) for droplet in droplets]
    return true_labels(scaled, (shape[0] * factor, shape[1] * factor))
