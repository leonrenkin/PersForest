"""A small synthetic z-stack of envelope-stained nuclei, for trying the pipeline and for tests.

Each nucleus is an ellipsoid with a thin bright envelope, a dim inside and a
cloud of textured perinuclear ER around it; the stack is blurred more along z
than in x and y, as in a confocal stack, and gets noise. One nucleus is cut by
the right image edge, and two touch each other. It is meant to exercise the
code, not to stand in for real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
import tifffile

from .io import Stack


@dataclass(frozen=True)
class SyntheticNucleus:
    """Centre ``(z, y, x)`` and semi-axes ``(z, y, x)`` in µm."""

    centre: tuple[float, float, float]
    semi_axes: tuple[float, float, float]


#: Voxel size and shape of the default example: 8 x 41.6 x 52 µm.
UM_PER_PIXEL = 0.065
Z_STEP_UM = 0.4
SHAPE = (20, 640, 800)

#: Nuclei of the default example. The last one is cut by the right image
#: edge; the third and fourth touch.
NUCLEI = (
    SyntheticNucleus((4.0, 11.0, 12.0), (2.2, 5.5, 7.0)),
    SyntheticNucleus((3.8, 12.0, 30.0), (2.0, 6.0, 5.0)),
    SyntheticNucleus((4.2, 30.0, 14.0), (2.2, 5.0, 5.5)),
    SyntheticNucleus((4.0, 30.5, 27.2), (2.0, 5.0, 5.0)),
    SyntheticNucleus((3.8, 26.0, 50.0), (2.0, 6.0, 5.0)),
)


def synthetic_volume(shape: tuple[int, int, int] = SHAPE, um_per_pixel: float = UM_PER_PIXEL,
                     z_step_um: float = Z_STEP_UM, nuclei: tuple[SyntheticNucleus, ...] = NUCLEI,
                     seed: int = 0) -> NDArray[np.uint16]:
    """The synthetic stack as 16-bit intensities ``(slices, rows, cols)``.

    Parameters
    ----------
    shape : tuple of int
    um_per_pixel, z_step_um : float
        Voxel size.
    nuclei : tuple of SyntheticNucleus
    seed : int
        Seed of the texture and the noise.
    """
    rng = np.random.default_rng(seed)
    z, y, x = (np.arange(n, dtype=np.float32) * step
               for n, step in zip(shape, (z_step_um, um_per_pixel, um_per_pixel)))
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    # Distance outside the nearest envelope in µm (negative inside), approximated
    # by the normalized ellipsoid radius times the mean semi-axis.
    distance = np.full(shape, np.inf, dtype=np.float32)
    for nucleus in nuclei:
        (cz, cy, cx), (az, ay, ax) = nucleus.centre, nucleus.semi_axes
        radius = np.sqrt(((zz - cz) / az) ** 2 + ((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2)
        distance = np.minimum(distance, (radius - 1) * (ay + ax) / 2)
    texture = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32),
                                      (0.4 / z_step_um, 0.3 / um_per_pixel, 0.3 / um_per_pixel))
    texture = np.clip(0.6 + texture / (3 * texture.std()), 0.25, None)
    envelope = np.exp(-0.5 * (distance / 0.2) ** 2)
    perinuclear = np.where(distance > 0, np.exp(-distance / 3.0), 0) * texture
    inside = np.where(distance < 0, 0.08, 0)
    volume = np.maximum(envelope, 0.6 * perinuclear) + inside + 0.04
    volume = ndimage.gaussian_filter(volume, (0.3 / z_step_um, 0.12 / um_per_pixel, 0.12 / um_per_pixel))
    counts = rng.poisson(volume * 3000.0) + 100
    return np.clip(counts, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def synthetic_stack(seed: int = 0, **kwargs) -> Stack:
    """The default synthetic example as a ``Stack`` (see ``synthetic_volume`` for the options)."""
    um_per_pixel = kwargs.get("um_per_pixel", UM_PER_PIXEL)
    z_step_um = kwargs.get("z_step_um", Z_STEP_UM)
    return Stack(synthetic_volume(seed=seed, **kwargs), um_per_pixel, z_step_um, name="synthetic")


def write_synthetic_stack(path: str | Path, seed: int = 0) -> Path:
    """Write the default synthetic example as an ImageJ TIFF with its voxel size and return the path."""
    stack = synthetic_volume(seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, stack, imagej=True, resolution=(1 / UM_PER_PIXEL, 1 / UM_PER_PIXEL),
                     metadata={"unit": "um", "spacing": Z_STEP_UM, "axes": "ZYX"}, compression="zlib")
    return path
