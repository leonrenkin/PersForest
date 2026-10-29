"""Simulated fluorescence images of lipid droplets with known labels.

A droplet in focus is a small sphere, so its brightness follows its thickness:
a dome ``peak * sqrt(1 - d**2)`` in the normalized distance ``d`` from its
center. Overlapping droplets add up (``overlap="sum"``), or the brighter one
covers the other (``overlap="max"``), and the microscope blurs the result with
its point spread function, modelled as a Gaussian. With the sum, the steep
edges of two overlapping domes add up to a ridge, which can create a maximum
in the overlap. The image lies on a dim background with Gaussian noise; the
background level can vary smoothly across the image, as uneven illumination
does in a microscope.

The true label of a pixel is the droplet whose own dome is brightest there,
and 0 where no dome reaches: the droplets' footprints, independent of the
blur, the noise and of how the segmentation works.

``DROPLETS`` is the hand-made scene of the notebook walkthrough;
``sample_droplets`` draws random scenes for the benchmark.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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


SHAPE = (215, 220)

DROPLETS = (
    # isolated droplets of different sizes
    Droplet(35, 40, 16, 1.0), Droplet(40, 170, 8, 0.7), Droplet(125, 35, 6, 0.55), Droplet(125, 185, 12, 0.85),
    # overlapping pair with a valley between the droplets
    Droplet(80, 95, 13, 0.9), Droplet(84, 115, 10, 0.8),
    # cluster of three
    Droplet(35, 100, 7, 0.65), Droplet(31, 112, 7, 0.75), Droplet(41, 118, 6, 0.6),
    # small droplet on the flank of a large one
    Droplet(125, 110, 18, 0.95), Droplet(113, 126, 6, 0.6),
    # pairs without a valley: the sum has a single maximum, the outline a waist
    Droplet(185, 30, 10, 0.7), Droplet(185, 44, 10, 0.7),
    Droplet(185, 95, 12, 0.85), Droplet(185, 113, 8, 0.45),
    # an elongated droplet: convex, so it must not be split
    Droplet(185, 175, 8, 0.8, elongation=2.2),
)
CASES = ("isolated",) * 4 + ("pair",) * 2 + ("cluster",) * 3 + ("on flank",) * 2 + ("no valley",) * 4 + ("elongated",)


def droplet_image(droplets=DROPLETS, shape=SHAPE, blur=1.0, background=0.15, noise=0.02,
                  seed=7, overlap="sum", background_variation=0.0,
                  background_scale=40.0) -> NDArray[np.float64]:
    """Blurred droplet domes on a background with Gaussian noise.

    Parameters
    ----------
    droplets : sequence of Droplet
    shape : tuple of int
    overlap : {"sum", "max"}
        Whether overlapping domes add up or the brighter one covers the other.
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
    if overlap not in ("sum", "max"):
        raise ValueError(f"overlap must be 'sum' or 'max', got {overlap!r}")
    rng = np.random.default_rng(seed)
    signals = np.stack([droplet.signal(shape) for droplet in droplets])
    image = gaussian_blur(signals.sum(axis=0) if overlap == "sum" else signals.max(axis=0), blur)
    image = image + background + noise * rng.standard_normal(shape)
    if background_variation:
        field = gaussian_blur(rng.standard_normal(shape), background_scale)
        image = image + background_variation * field / np.abs(field).max()
    return image


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


def true_labels(droplets=DROPLETS, shape=SHAPE) -> NDArray[np.int_]:
    """Label each pixel by the droplet contributing most to it; 0 is background.

    Label ``k`` belongs to ``droplets[k - 1]``.
    """
    signals = np.stack([droplet.signal(shape) for droplet in droplets])
    return np.where(signals.max(axis=0) > 0, signals.argmax(axis=0) + 1, 0)


LAYOUTS = ("uniform", "clustered")


def sample_droplets(rng: np.random.Generator, shape: tuple[int, int] = (192, 192), layout: str = "uniform",
                    n_droplets: int = 12, radius: tuple[float, float] = (5.0, 14.0),
                    peak: tuple[float, float] = (0.4, 1.0), elongated_fraction: float = 0.3,
                    max_elongation: float = 1.8, n_clusters: int = 4,
                    cluster_spread: float = 9.0) -> list[Droplet]:
    """Draw a random scene of droplets.

    Radii, peaks and orientations are uniform in their ranges; a fraction of
    the droplets is elongated by a uniform factor up to ``max_elongation``.
    Every droplet lies fully inside the image, two pixels from the border:
    beyond the grid counts as brighter than everything, so droplets cut by the
    border are out of scope. Overlaps are not prevented.

    Parameters
    ----------
    rng : numpy.random.Generator
    shape : tuple of int
    layout : {"uniform", "clustered"}
        ``"uniform"`` places centers uniformly; ``"clustered"`` places them
        around ``n_clusters`` uniform cluster centers with Gaussian offsets of
        standard deviation ``cluster_spread``, so that many droplets overlap.
    n_droplets : int
    radius, peak : tuple of float
        Ranges of the radius (pixels) and of the peak brightness.
    elongated_fraction, max_elongation : float
    n_clusters : int
    cluster_spread : float
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    centers = rng.uniform((0, 0), shape, size=(n_clusters, 2))
    droplets = []
    while len(droplets) < n_droplets:
        size = rng.uniform(*radius)
        elongation = rng.uniform(1.0, max_elongation) if rng.random() < elongated_fraction else 1.0
        droplet = Droplet(0.0, 0.0, size, rng.uniform(*peak), elongation, rng.uniform(0.0, 180.0))
        margin = droplet.extent + 2
        if layout == "uniform":
            row, column = rng.uniform(margin, np.array(shape) - margin)
        else:
            row, column = centers[rng.integers(n_clusters)] + rng.normal(0.0, cluster_spread, 2)
            if not (margin <= row <= shape[0] - margin and margin <= column <= shape[1] - margin):
                continue
        droplets.append(Droplet(float(row), float(column), droplet.radius, droplet.peak, droplet.elongation,
                                droplet.angle))
    return droplets
