"""Reading z-stacks and writing results.

Stacks are read with ``tifffile``: ImageJ TIFFs (also hyperstacks), OME-TIFFs
and plain (multi-page) TIFFs. The voxel size comes from the ImageJ or OME
metadata unless it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING
import csv
import json
import re

import numpy as np
from numpy.typing import NDArray
import tifffile

from .parameters import Parameters
from .parameters import Parameters3D

if TYPE_CHECKING:
    from .segment2d import Segmentation
    from .segment3d import Segmentation3D

TIFF_SUFFIXES = (".tif", ".tiff")

_UM_PER_UNIT = {"micron": 1.0, "microns": 1.0, "um": 1.0, "µm": 1.0, "μm": 1.0, "\\u00b5m": 1.0,
                "nm": 1e-3, "nanometer": 1e-3, "mm": 1e3, "millimeter": 1e3}


@dataclass
class Stack:
    """A z-stack ``(slices, rows, cols)`` with its voxel size.

    A 2D image is a stack of one slice.

    Attributes
    ----------
    data : ndarray of float32, shape (slices, rows, cols)
    um_per_pixel : float
        Pixel size along rows and columns.
    z_step_um : float or None
        Distance between slices, if known; needed for the 3D step.
    name : str
        Used for the names of the result files.
    """

    data: NDArray[np.float32]
    um_per_pixel: float
    z_step_um: float | None = None
    name: str = "stack"
    source: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float32)
        if data.ndim == 2:
            data = data[None]
        if data.ndim != 3:
            raise ValueError(f"a stack has shape (slices, rows, cols), not {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("the stack contains non-finite values")
        if not self.um_per_pixel > 0:
            raise ValueError(f"um_per_pixel must be positive, not {self.um_per_pixel}")
        if self.z_step_um is not None and not self.z_step_um > 0:
            raise ValueError(f"z_step_um must be positive, not {self.z_step_um}")
        self.data = data
        self.um_per_pixel = float(self.um_per_pixel)
        self.z_step_um = None if self.z_step_um is None else float(self.z_step_um)


def read_stack(path: str | Path, *, channel: int | None = None, um_per_pixel: float | None = None,
               z_step_um: float | None = None) -> Stack:
    """Read a TIFF z-stack and its voxel size.

    Parameters
    ----------
    path : str or Path
    channel : int, optional
        Channel of a multichannel file (0-based). Required when the file has
        more than one channel, so that channels are never mixed into z.
    um_per_pixel, z_step_um : float, optional
        Override the voxel size of the metadata. Without ``um_per_pixel`` the
        file must carry a pixel size in a known length unit.

    Raises
    ------
    ValueError
        For several channels without ``channel``, several time points, axes
        other than channel, z, y and x, or a missing pixel size.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes
        found_um, found_z = _voxel_size(tif)
    data = _to_zyx(data, axes, channel, path)
    if um_per_pixel is None:
        if found_um is None:
            raise ValueError(f"{path}: no pixel size in the ImageJ or OME metadata; pass um_per_pixel")
        um_per_pixel = found_um
    return Stack(data, um_per_pixel, z_step_um if z_step_um is not None else found_z, path.stem, str(path))


def _to_zyx(data: NDArray, axes: str, channel: int | None, path: Path) -> NDArray:
    # tifffile names a plain page sequence I (or Q); treat it as z.
    axes = axes.replace("I", "Z").replace("Q", "Z")
    for name in ("C", "S"):
        if name not in axes:
            continue
        index = axes.index(name)
        n = data.shape[index]
        if channel is None and n > 1:
            raise ValueError(f"{path}: {n} channels (axes {axes}); choose one with channel=")
        chosen = 0 if channel is None else channel
        if not 0 <= chosen < n:
            raise ValueError(f"{path}: channel {chosen} outside 0..{n - 1}")
        data = np.take(data, chosen, axis=index)
        axes = axes.replace(name, "")
        channel = None
    if channel not in (None, 0):
        raise ValueError(f"{path}: has a single channel, cannot use channel={channel}")
    for index in reversed(range(len(axes))):
        name = axes[index]
        if name in "ZYX":
            continue
        if data.shape[index] > 1:
            kind = "time points" if name == "T" else f"entries along axis {name}"
            raise ValueError(f"{path}: {data.shape[index]} {kind} (axes {axes}); save one z-stack per file")
        data = data.squeeze(axis=index)
        axes = axes[:index] + axes[index + 1:]
    if "Z" not in axes:
        data, axes = data[None], "Z" + axes
    if sorted(axes) != ["X", "Y", "Z"]:
        raise ValueError(f"{path}: cannot read axes {axes} as a z-stack")
    return np.transpose(data, [axes.index(a) for a in "ZYX"])


def _voxel_size(tif: tifffile.TiffFile) -> tuple[float | None, float | None]:
    """Pixel size and z step in µm from ImageJ or OME metadata, ``None`` where absent."""
    if tif.is_ome and tif.ome_metadata:
        xml = tif.ome_metadata
        pixel = _ome_length(xml, "PhysicalSizeX")
        if pixel is not None:
            return pixel, _ome_length(xml, "PhysicalSizeZ")
    if tif.is_imagej:
        info = tif.imagej_metadata or {}
        scale = _UM_PER_UNIT.get(str(info.get("unit", "")).strip().lower())
        tag = tif.pages[0].tags.get("XResolution")
        if scale is not None and tag is not None:
            numerator, denominator = tag.value
            per_unit = numerator / denominator
            if per_unit > 0:
                spacing = info.get("spacing")
                return scale / per_unit, None if spacing is None else scale * float(spacing)
    return None, None


def _ome_length(xml: str, name: str) -> float | None:
    value = re.search(rf'\b{name}="([^"]+)"', xml)
    if value is None:
        return None
    unit = re.search(rf'\b{name}Unit="([^"]+)"', xml)
    scale = _UM_PER_UNIT.get(unit.group(1).lower() if unit else "µm")
    return None if scale is None else scale * float(value.group(1))


def find_stacks(inputs: list[str | Path]) -> list[Path]:
    """TIFF files among ``inputs``, folders expanded (not recursively), sorted per folder."""
    paths: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            paths += sorted(p for p in item.iterdir() if p.suffix.lower() in TIFF_SUFFIXES)
        elif item.exists():
            paths.append(item)
        else:
            raise FileNotFoundError(item)
    return paths


# ----- writing -----

def write_labels(path: str | Path, labels: NDArray, um_per_pixel: float, z_step_um: float | None = None) -> None:
    """Write a 2D or 3D label image as a compressed 16-bit ImageJ TIFF with its voxel size."""
    if labels.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError("more labels than a 16-bit image holds")
    info: dict = {"unit": "um", "axes": "ZYX" if labels.ndim == 3 else "YX"}
    if labels.ndim == 3 and z_step_um is not None:
        info["spacing"] = z_step_um
    tifffile.imwrite(path, labels.astype(np.uint16), imagej=True, resolution=(1 / um_per_pixel, 1 / um_per_pixel),
                     metadata=info, compression="zlib")


def write_table(path: str | Path, rows: list[dict]) -> None:
    """Write table rows (dicts with the same keys) as CSV."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["label"])
        writer.writeheader()
        writer.writerows(rows)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def run_record(stack: Stack, parameters: Parameters, parameters_3d: Parameters3D | None = None,
               z_step_um: float | None = None) -> dict:
    """Everything needed to repeat a run: versions, input, voxel size and parameters."""
    return {
        "nucseg_version": _version("nucseg"),
        "persforest_version": _version("persforest"),
        "source": stack.source,
        "um_per_pixel": stack.um_per_pixel,
        "z_step_um": z_step_um if z_step_um is not None else stack.z_step_um,
        "parameters": parameters.to_dict(),
        "parameters_3d": None if parameters_3d is None else parameters_3d.to_dict(),
    }


def read_parameters(path: str | Path) -> tuple[Parameters, Parameters3D | None]:
    """Parameters from a ``_parameters.json`` of an earlier run, or from a JSON dict of 2D parameters.

    A run file holds ``"parameters"`` and ``"parameters_3d"`` (the latter
    ``null`` for a 2D-only run); other keys are ignored.
    """
    values = json.loads(Path(path).read_text())
    if "parameters" in values:
        third = values.get("parameters_3d")
        return Parameters.from_dict(values["parameters"]), None if third is None else Parameters3D.from_dict(third)
    return Parameters.from_dict(values), None


def save_results(out_dir: str | Path, stack: Stack, segmentation: Segmentation,
                 segmentation_3d: Segmentation3D | None = None, *, figures: bool = True) -> list[Path]:
    """Write the results of one stack to ``out_dir`` and return the written paths.

    Files, named after ``stack.name``:

    - ``_labels.tif``: 2D labels at full resolution (0 = background, 1, 2, ...
      as in the table), with the pixel size for Fiji/ImageJ.
    - ``_nuclei.csv``: one row per nucleus.
    - ``_parameters.json``: versions, input, voxel size and all parameters;
      ``--parameters`` of the command line reads it back.
    - ``_overview.png``, ``_gallery.png``: outlines on the image, for checking
      (with ``figures``).
    - With ``segmentation_3d``: ``_labels_3d.tif`` (full resolution in x and
      y, same labels), ``_nuclei_3d.csv``, and ``_sections_3d.png``.
    """
    from . import plots

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = stack.name
    written = []

    def path(suffix: str) -> Path:
        written.append(out_dir / f"{name}{suffix}")
        return written[-1]

    write_labels(path("_labels.tif"), segmentation.labels(), stack.um_per_pixel)
    write_table(path("_nuclei.csv"), segmentation.table())
    z_step = None if segmentation_3d is None else segmentation_3d.z_step_um
    record = run_record(stack, segmentation.parameters,
                        None if segmentation_3d is None else segmentation_3d.parameters, z_step)
    path("_parameters.json").write_text(json.dumps(record, indent=2) + "\n")
    if segmentation_3d is not None:
        write_labels(path("_labels_3d.tif"), segmentation_3d.labels(full_resolution=True), stack.um_per_pixel,
                     segmentation_3d.z_step_um)
        write_table(path("_nuclei_3d.csv"), segmentation_3d.table())
    if figures:
        import matplotlib.pyplot as plt

        made = [("_overview.png", plots.figure_overview(segmentation, stack, title=f"{name}: "
                                                         f"{len(segmentation.nuclei)} nuclei")),
                ("_gallery.png", plots.figure_gallery(segmentation, stack))]
        if segmentation_3d is not None:
            made.append(("_sections_3d.png", plots.plot_sections(segmentation_3d, title=name)))
        for suffix, fig in made:
            if fig is not None:
                fig.savefig(path(suffix), dpi=110)
                plt.close(fig)
    return written
