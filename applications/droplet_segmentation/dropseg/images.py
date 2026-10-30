"""Images for the segmentation: loading, preprocessing and upsampling.

The segmentation works on intensities in [0, 1]: its thresholds are
intensities in these units. ``load_image`` reads an image file as floats;
``preprocess`` scales it to [0, 1] and optionally crops and blurs it.

Upsampling gives small droplets more pixels, so that the outlines are smooth
curves rather than staircases of a few pixels. ``gaussian_upsample`` is a
kernel average on the finer grid: every fine pixel is the Gaussian-weighted
mean of the original pixels around it. Its level sets are smooth curves
through the original samples' neighborhoods, so a waist between two droplets
or the roundness of a droplet can show below the pixel size. The kernel
width trades smoothness of the outlines against blurring neighboring
droplets together. ``downsample_labels`` maps labels back to the original
pixels.
"""

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def image_files(folder) -> list[Path]:
    """Image files in ``folder``, sorted by name."""
    return sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def load_image(path, channel: int | None = None) -> tuple[NDArray[np.float64], float]:
    """Read a 2D image as floats, with the largest value its type can hold.

    Color images need ``channel``; a gray image saved as RGB has equal channels
    and is read as gray. Returns ``(raw, full_scale)``: for an 8-bit image
    ``full_scale`` is 255, for a 16-bit image 65535, for a float image its
    maximum.
    """
    raw = np.asarray(Image.open(path))
    if raw.ndim == 3:
        if channel is None:
            if not all((raw[..., 0] == raw[..., c]).all() for c in range(1, min(raw.shape[-1], 3))):
                raise ValueError(f"{path} has {raw.shape[-1]} channels; pass channel")
            channel = 0
        raw = raw[..., channel]
    if raw.ndim != 2:
        raise ValueError(f"{path} is not a 2D image (shape {raw.shape})")
    full_scale = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else float(raw.max())
    return raw.astype(float), full_scale


@dataclass(frozen=True)
class Preprocessing:
    """How a raw image becomes the input of the segmentation.

    Attributes
    ----------
    normalize : {"full scale", "percentile"}
        Divide by the largest value of the image type (255 for 8 bit), or map
        the ``low_percentile`` and ``high_percentile`` of the image to 0 and 1,
        for images that use only part of their range (e.g. 12-bit data saved
        as 16 bit). Values outside [0, 1] are clipped.
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

    def record(self) -> dict:
        """The options as a plain dict, for saving."""
        record = asdict(self)
        record["crop"] = None if self.crop is None else [[s.start, s.stop] for s in self.crop]
        return record


def preprocess(raw: NDArray[np.float64], full_scale: float,
               options: Preprocessing = Preprocessing()) -> NDArray[np.float64]:
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


def gaussian_blur(image: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Separable Gaussian blur, truncated at four standard deviations, with edge padding."""
    if sigma == 0:
        return image
    radius = int(np.ceil(4 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    for axis in (0, 1):
        image = np.apply_along_axis(
            lambda line: np.convolve(np.pad(line, radius, mode="edge"), kernel, "valid"), axis, image)
    return image


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
