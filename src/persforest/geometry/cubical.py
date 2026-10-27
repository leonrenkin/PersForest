"""Physical geometry of cubical grids."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CubicalGeometry:
    """Physical placement of a cubical grid.

    Along axis ``a``, doubled coordinate ``c`` lies at
    ``origin[a] + (c / 2) * spacing[a]``. With ``filtration_type="top_cells"``
    the origin is the lower corner of the first pixel or voxel; with
    ``"vertices"`` it is the first vertex.

    Attributes
    ----------
    shape : tuple of int
        Shape of the value array.
    spacing : tuple of float
        Positive cell size per array axis.
    origin : tuple of float
        Position of doubled coordinate 0 per array axis.
    filtration_type : {"top_cells", "vertices"}
        Convention of the value array.
    """

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    filtration_type: str

    @property
    def dim(self) -> int:
        return len(self.shape)

    def coordinates(self, cell: tuple[int, ...]) -> NDArray[np.float64]:
        """Return the physical center of ``cell``."""
        return np.asarray(self.origin) + np.asarray(cell, dtype=float) / 2 * np.asarray(self.spacing)

    def cell_measure(self, cell: tuple[int, ...]) -> float:
        """Return the length, area or volume of ``cell`` (1 for a vertex)."""
        return math.prod(s for c, s in zip(cell, self.spacing) if c & 1)

    # Simplicial measurements index or convert their second argument as a point
    # array; fail with an actionable message instead of an unrelated error.
    def _reject_point_cloud_use(self, *args, **kwargs):
        raise TypeError(
            "This measurement expects a point cloud, but the forest was built from a "
            "cubical grid. Use persforest.measurements.cubical (e.g. "
            "cubical_chain_volume, cubical_chain_interior_volume) or a callable "
            "accepting (chain, CubicalGeometry).")

    __array__ = _reject_point_cloud_use
    __getitem__ = _reject_point_cloud_use


def make_cubical_geometry(shape: tuple[int, ...], spacing, origin, filtration_type: str) -> CubicalGeometry:
    """Validate ``spacing`` and ``origin`` (scalars or one value per axis)."""
    d = len(shape)

    def per_axis(value, default, name):
        if value is None:
            return (default,) * d
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
        if array.ndim == 0:
            array = np.full(d, float(array))
        if array.shape != (d,):
            raise ValueError(f"{name} must be a scalar or have one value per axis ({d})")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return tuple(float(v) for v in array)

    spacing = per_axis(spacing, 1.0, "spacing")
    if min(spacing) <= 0:
        raise ValueError("spacing must be positive")
    return CubicalGeometry(tuple(shape), spacing, per_axis(origin, 0.0, "origin"), filtration_type)
