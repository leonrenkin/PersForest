"""Results of a segmentation as files: label images, a table of segments, a boundary overlay, the parameters.

``save_results`` writes, into a folder, for an image named ``stem``:

- ``stem_labels.tif``: the labels on the original pixels, 0 for the
  background (16 bit, or 32 bit beyond 65535 segments); opens in Fiji or
  napari as a label image. Upsampled labels are reduced to the original
  pixels by majority.
- ``stem_labels_<f>x.tif``: when upsampled by ``f``, the labels on the fine grid.
- ``stem_segments.csv``: area, equivalent radius, centroid and intensity per
  segment, in original pixels.
- ``stem_boundaries.png``: the image with the segment boundaries, every
  original pixel drawn as ``scale`` x ``scale`` pixels (``boundary_overlay``).
- ``stem_parameters.json``: the parameters and preprocessing, which
  ``python -m dropseg --parameters`` reads back.
"""

from pathlib import Path
import json

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

from .images import Preprocessing
from .images import downsample_labels
from .parameters import Parameters
from .whole_image import Segmentation


def segment_table(labels: NDArray[np.int_], image: NDArray[np.float64], factor: int = 1):
    """Area, equivalent radius, centroid and intensity of every segment, as a pandas DataFrame.

    ``labels`` and ``image`` may be upsampled by ``factor``; areas, radii and
    positions are in original pixels (the centroid as a pixel index: row 0 is
    the center of the first row). ``mean`` and ``max`` are intensities in [0, 1].
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


def boundary_overlay(image, labels, factor=1, scale=16, vmax=None, groups=None, edge_color=(255, 225, 0),
                     contact_color=(255, 40, 40), lobe_color=(0, 200, 255), width=3, rim_color=(20, 20, 20)):
    """RGB array of ``image`` with the segment boundaries drawn on it, for saving at high resolution.

    Every pixel of ``image`` becomes a ``scale`` x ``scale`` block, so the
    original pixels stay visible; ``labels`` lies on a grid ``factor`` times
    finer than ``image`` (``scale`` must be a multiple of ``factor``). Edges
    towards the background are drawn in ``edge_color`` (yellow), edges
    between two segments in ``contact_color`` (red), ``width`` output pixels
    wide and centered on the pixel edge. With ``groups``
    (``Segmentation.groups``, on the grid of ``labels``), edges between two
    segments of one outline, cut apart by the lobe splitting, are drawn in
    ``lobe_color`` (blue) instead; edges between segments of different
    outlines are cuts of the forest. Every edge has a rim of one output pixel
    in ``rim_color`` on both sides, so that it also shows on saturated
    pixels; ``rim_color=None`` omits it. Brightness above ``vmax`` is white.
    """
    if scale % factor:
        raise ValueError(f"scale {scale} must be a multiple of factor {factor}")
    step = scale // factor
    gray = (np.clip(image / (vmax or image.max() or 1.0), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)[..., None].repeat(3, axis=2)
    span = np.arange(step + 1)
    # Rims first, under all edges, then the edges themselves.
    passes = [(width + 2, True), (width, False)] if rim_color is not None else [(width, False)]
    for line_width, rim in passes:
        offsets = np.arange(line_width) - line_width // 2
        _draw_edges(rgb, labels, groups, step, span, offsets, rim_color if rim else None,
                    edge_color, contact_color, lobe_color)
    return rgb


def _draw_edges(rgb, labels, groups, step, span, offsets, rim_color, edge_color, contact_color, lobe_color):
    """Draw the boundary edges of ``labels`` into ``rgb``, all in ``rim_color`` if given; see ``boundary_overlay``."""
    width = len(offsets)
    for axis in (0, 1):
        first = labels[:-1, :] if axis == 0 else labels[:, :-1]
        second = labels[1:, :] if axis == 0 else labels[:, 1:]
        differs = (first != second) & ((first > 0) | (second > 0))
        contact = differs & (first > 0) & (second > 0)
        lobe = np.zeros_like(contact)
        if groups is not None:
            first_group = groups[:-1, :] if axis == 0 else groups[:, :-1]
            second_group = groups[1:, :] if axis == 0 else groups[:, 1:]
            lobe = contact & (first_group == second_group)
        for mask, color in ((differs & ~contact, edge_color), (contact & ~lobe, contact_color), (lobe, lobe_color)):
            color = rim_color or color
            rows, columns = np.nonzero(mask)
            # The edge lies after pixel (row, column) along the axis and spans the pixel across it.
            across = ((rows if axis == 0 else columns) + 1) * step
            along = (columns if axis == 0 else rows) * step
            lines = np.clip(across[:, None, None] + offsets[None, :, None], 0, None)
            extent = np.broadcast_to(along[:, None, None] + span[None, None, :], (len(along), width, step + 1))
            lines = np.broadcast_to(lines, extent.shape)
            keep = (lines < rgb.shape[axis]) & (extent < rgb.shape[1 - axis])
            if axis == 0:
                rgb[lines[keep], extent[keep]] = color
            else:
                rgb[extent[keep], lines[keep]] = color


def save_labels(path, labels: NDArray[np.int_]):
    """Write a label image as a TIFF: 16 bit if the labels fit, 32 bit otherwise."""
    dtype = np.uint16 if labels.max() <= np.iinfo(np.uint16).max else np.int32
    Image.fromarray(labels.astype(dtype)).save(path)


def save_results(folder, stem: str, image: NDArray[np.float64], segmentation: Segmentation, parameters: Parameters,
                 preprocessing: Preprocessing | None = None, scale: int = 8, vmax: float | None = None,
                 extra: dict | None = None) -> list[Path]:
    """Write the results of ``segmentation`` of ``image`` (original pixels) into ``folder``; see the module.

    ``scale`` is the number of output pixels per original pixel in the
    boundary overlay, rounded up to a multiple of the upsampling factor.
    ``extra`` is added to the JSON record. Returns the files written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    factor = segmentation.factor
    scale = -(-scale // factor) * factor
    paths = {name: folder / f"{stem}_{name}" for name in ("labels.tif", "segments.csv", "boundaries.png",
                                                          "parameters.json")}
    save_labels(paths["labels.tif"], segmentation.original_labels)
    if factor > 1:
        paths["fine"] = folder / f"{stem}_labels_{factor}x.tif"
        save_labels(paths["fine"], segmentation.labels)
    segment_table(segmentation.labels, segmentation.image, factor).to_csv(paths["segments.csv"])
    overlay = boundary_overlay(image, segmentation.labels, factor, scale, vmax, segmentation.groups)
    Image.fromarray(overlay).save(paths["boundaries.png"])
    record = {"image": stem, "shape": list(image.shape), "segments": segmentation.n_segments,
              "parameters": parameters.record(),
              "preprocessing": None if preprocessing is None else preprocessing.record(), **(extra or {})}
    paths["parameters.json"].write_text(json.dumps(record, indent=1))
    return list(paths.values())
