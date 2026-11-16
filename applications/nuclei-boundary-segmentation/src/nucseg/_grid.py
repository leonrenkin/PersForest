"""Binning, normalization and shape measures of binary masks, shared by the 2D and 3D steps."""

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import Delaunay


def bin_xy(array: NDArray, factor: int) -> NDArray[np.float64]:
    """Average blocks of ``factor`` x ``factor`` pixels over the last two axes.

    Rows and columns beyond the last full block are dropped.
    """
    if factor < 1:
        raise ValueError("bin_factor must be at least 1")
    *lead, h, w = array.shape
    h, w = h - h % factor, w - w % factor
    blocks = array[..., :h, :w].reshape(*lead, h // factor, factor, w // factor, factor)
    return blocks.mean(axis=(-3, -1), dtype=np.float64)


def unbin_xy(labels: NDArray, factor: int, shape: tuple[int, ...]) -> NDArray:
    """Repeat every binned pixel as a ``factor`` x ``factor`` block in an array of ``shape``.

    Pixels beyond the last full block are 0.
    """
    up = np.repeat(np.repeat(labels, factor, axis=-2), factor, axis=-1)
    out = np.zeros(shape, dtype=labels.dtype)
    out[..., : up.shape[-2], : up.shape[-1]] = up
    return out


def normalize(image: NDArray[np.float64], percentile: float) -> NDArray[np.float64]:
    """Scale an image so that its ``percentile`` is 1."""
    scale = np.percentile(image, percentile)
    if not scale > 0:
        raise ValueError(f"the {percentile} percentile of the image is {scale}; the image is (almost) empty")
    return image / scale


def touches_edge(mask: NDArray[np.bool_]) -> bool:
    """Whether a 2D region reaches the outermost pixels of the image."""
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def width(mask: NDArray[np.bool_]) -> float:
    """Twice the largest distance from a pixel of the region to its outside, in pixels.

    The image edge counts as outside.
    """
    return 2.0 * float(ndimage.distance_transform_edt(np.pad(mask, 1)).max())


def convex_hull_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Pixels whose centers lie in the convex hull of the region's pixel squares."""
    points = np.argwhere(mask & ~ndimage.binary_erosion(mask))
    corners = (points[:, None, :] + np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) - 0.5).reshape(-1, 2)
    inside = Delaunay(corners).find_simplex(np.argwhere(np.ones(mask.shape, dtype=bool))) >= 0
    return inside.reshape(mask.shape)


def solidity(mask: NDArray[np.bool_]) -> float:
    """Area of a region over the area of its convex hull (pixel squares included)."""
    rows, cols = np.nonzero(mask)
    sub = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return float(sub.sum() / convex_hull_mask(np.pad(sub, 1)).sum())


def bounding_box(pixels: NDArray[np.intp], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """``(row_start, row_stop, col_start, col_stop)`` of flat pixel indices (stops exclusive)."""
    rows, cols = np.unravel_index(pixels, shape)
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def disk(radius: int) -> NDArray[np.bool_]:
    """Binary disk of ``radius`` pixels."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return xx**2 + yy**2 <= radius**2
