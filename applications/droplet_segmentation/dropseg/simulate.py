"""Simulated fluorescence images of droplets with known labels, for examples and tests.

A droplet in focus is a small sphere, so its brightness follows its thickness:
a dome ``peak * sqrt(1 - d**2)`` in the normalized distance ``d`` from its
center. Overlapping droplets add up, and the microscope blurs the sum with
its point spread function, modelled as a Gaussian. The image lies on a dim
background with Gaussian noise; the background level can vary smoothly
across the image, as uneven illumination does in a microscope.

The true label of a pixel is the droplet whose own dome is brightest there,
and 0 where no dome reaches: the droplets' footprints, independent of the
blur, the noise and of how the segmentation works.

``camera_scene`` draws small, dense droplets rendered like an 8-bit camera
image (quantized and saturating): the droplets are only a few pixels across,
where upsampling helps. ``sample_droplets`` and ``droplet_image`` draw larger
scenes with any droplet sizes.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .images import gaussian_blur


@dataclass(frozen=True)
class Droplet:
    """A dome-shaped droplet.

    Attributes
    ----------
    row, column : float
        Center in pixel coordinates.
    radius : float
        Radius along the rows; along the columns it is ``radius * elongation``.
    peak : float
        Brightness at the center above the background.
    elongation : float
        Ratio of the second radius to ``radius``; 1 for a round droplet.
    angle : float
        Rotation in degrees of the second axis away from the columns.
    """

    row: float
    column: float
    radius: float
    peak: float
    elongation: float = 1.0
    angle: float = 0.0

    @property
    def extent(self) -> float:
        """Largest radius."""
        return self.radius * max(self.elongation, 1.0)

    def signal(self, shape: tuple[int, int]) -> NDArray[np.float64]:
        """Brightness of this droplet alone at the pixel centers."""
        rows, columns = np.mgrid[0:shape[0], 0:shape[1]] + 0.5
        cos, sin = np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))
        along_rows = cos * (rows - self.row) - sin * (columns - self.column)
        along_columns = sin * (rows - self.row) + cos * (columns - self.column)
        distance = np.hypot(along_rows / self.radius, along_columns / (self.radius * self.elongation))
        return self.peak * np.sqrt(np.clip(1 - distance**2, 0, None))


def _signals(droplets, shape):
    # Each droplet is computed on its bounding box only, so that large scenes stay fast.
    for droplet in droplets:
        reach = int(np.ceil(droplet.extent)) + 1
        low = np.maximum([int(droplet.row) - reach, int(droplet.column) - reach], 0)
        high = np.minimum([int(droplet.row) + reach + 1, int(droplet.column) + reach + 1], shape)
        local = Droplet(droplet.row - low[0], droplet.column - low[1], droplet.radius, droplet.peak,
                        droplet.elongation, droplet.angle)
        yield (slice(low[0], high[0]), slice(low[1], high[1])), local.signal(tuple(high - low))


def droplet_image(droplets, shape, blur=1.0, background=0.15, noise=0.02, seed=7, background_variation=0.0,
                  background_scale=40.0) -> NDArray[np.float64]:
    """Blurred, summed droplet domes on a background with Gaussian noise.

    Parameters
    ----------
    droplets : sequence of Droplet
    shape : tuple of int
    blur : float
        Standard deviation of the Gaussian point spread function in pixels;
        0 disables it.
    background : float
        Mean background brightness.
    noise : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Seed of the noise generator.
    background_variation : float
        Largest deviation of the background from ``background``: a Gaussian
        random field of correlation length ``background_scale`` pixels,
        scaled to this amplitude. 0 keeps the background constant.
    background_scale : float
    """
    rng = np.random.default_rng(seed)
    signal = np.zeros(shape)
    for window, local in _signals(droplets, shape):
        signal[window] += local
    image = gaussian_blur(signal, blur) + background + noise * rng.standard_normal(shape)
    if background_variation:
        field = gaussian_blur(rng.standard_normal(shape), background_scale)
        image = image + background_variation * field / np.abs(field).max()
    return image


def true_labels(droplets, shape) -> NDArray[np.int_]:
    """Label each pixel by the droplet contributing most to it; 0 is background.

    Label ``k`` belongs to ``droplets[k - 1]``.
    """
    best = np.zeros(shape)
    labels = np.zeros(shape, dtype=int)
    for label, (window, local) in enumerate(_signals(droplets, shape), start=1):
        brighter = local > best[window]
        best[window][brighter] = local[brighter]
        labels[window][brighter] = label
    return labels


def sample_droplets(rng: np.random.Generator, shape: tuple[int, int] = (192, 192), n_droplets: int = 12,
                    radius: tuple[float, float] = (5.0, 14.0), peak: tuple[float, float] = (0.4, 1.0),
                    elongated_fraction: float = 0.3, max_elongation: float = 1.8) -> list[Droplet]:
    """Draw droplets uniformly over the image.

    Radii, peaks and orientations are uniform in their ranges; a fraction of
    the droplets is elongated by a uniform factor up to ``max_elongation``.
    Every droplet lies fully inside the image, two pixels from the border.
    Overlaps are not prevented.
    """
    droplets = []
    while len(droplets) < n_droplets:
        size = rng.uniform(*radius)
        elongation = rng.uniform(1.0, max_elongation) if rng.random() < elongated_fraction else 1.0
        droplet = Droplet(0.0, 0.0, size, rng.uniform(*peak), elongation, rng.uniform(0.0, 180.0))
        margin = droplet.extent + 2
        row, column = rng.uniform(margin, np.array(shape) - margin)
        droplets.append(Droplet(float(row), float(column), droplet.radius, droplet.peak, droplet.elongation,
                                droplet.angle))
    return droplets


def camera(image: NDArray[np.float64], levels: int = 255) -> NDArray[np.float64]:
    """Clip to [0, 1] and quantize to ``levels`` steps, like an 8-bit camera."""
    return np.round(np.clip(image, 0.0, 1.0) * levels) / levels


@dataclass(frozen=True)
class Scene:
    """A simulated image with its droplets and true labels."""

    droplets: list[Droplet]
    image: NDArray[np.float64]
    truth: NDArray[np.int_]


def camera_scene(seed: int = 0, shape: tuple[int, int] = (128, 128), n_droplets: int = 180,
                 radius: tuple[float, float] = (0.8, 2.2), peak: tuple[float, float] = (0.1, 1.3),
                 blur: float = 0.5, background: float = 0.01, noise: float = 0.01,
                 background_variation: float = 0.02) -> Scene:
    """Small dense droplets rendered like an 8-bit camera image.

    Droplets of radius 0.8 to 2.2 px, half of them elongated up to 2:1, with
    peaks of 0.1 to 1.3 above a dark background; peaks above 1 saturate.
    """
    rng = np.random.default_rng([seed, 2])
    droplets = sample_droplets(rng, shape, n_droplets=n_droplets, radius=radius, peak=peak,
                               elongated_fraction=0.5, max_elongation=2.0)
    image = droplet_image(droplets, shape, blur=blur, background=background, noise=noise, seed=seed,
                          background_variation=background_variation, background_scale=20.0)
    return Scene(droplets, camera(image), true_labels(droplets, shape))


def count_found(labels: NDArray[np.int_], truth: NDArray[np.int_], min_iou: float = 0.5) -> int:
    """True droplets matched by a segment with an intersection over union above ``min_iou``.

    Above 0.5, a droplet and a segment match at most one of each other.
    """
    both = (truth > 0) & (labels > 0)
    pairs, intersection = np.unique(np.stack([truth[both], labels[both]]), axis=1, return_counts=True)
    true_sizes = np.bincount(truth.ravel())
    segment_sizes = np.bincount(labels.ravel())
    union = true_sizes[pairs[0]] + segment_sizes[pairs[1]] - intersection
    return int((intersection / union > min_iou).sum())
