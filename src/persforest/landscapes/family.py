"""Rank explicit measurement-profile kernels on a shared sample grid."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._validation import positive_integer, sample_grid, validate_landscape_request, validate_tolerance
from .functions import PiecewiseLinearFunction, _freeze_metadata, _readonly_vector
from .kernels import landscape_kernel
from .profiles import MeasurementProfiles, ProfileBar


@dataclass(frozen=True, eq=False)
class MeasurementLandscapeFamily:
    """Sampled landscape levels, with exact kernels and captured bar identities.

    Landscape rows are samples, not an exact piecewise-linear representation of
    their rank crossings. ``evaluate_on_grid`` retrieves stored coordinates
    only. Recompute from profiles to obtain ranks at new coordinates.
    """
    forest_id: str
    label: str
    rescaling: str
    x_grid: NDArray[np.float64]
    bar_kernels: Mapping[int, PiecewiseLinearFunction]
    landscapes: Mapping[int, PiecewiseLinearFunction]
    bars: tuple[ProfileBar, ...] = ()
    extra_meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grid = validate_landscape_request(1, 2, self.rescaling, self.x_grid)
        grid = _readonly_vector(grid, "x_grid")
        kernels, landscapes = dict(self.bar_kernels), dict(self.landscapes)
        bars = tuple(self.bars)
        if len({bar.bar_id for bar in bars}) != len(bars) or set(kernels) != {bar.bar_id for bar in bars}:
            raise ValueError("Captured bars and kernel IDs must align without duplicates")
        for level, landscape in landscapes.items():
            positive_integer(level, "landscape level")
            if not isinstance(landscape, PiecewiseLinearFunction) or not np.array_equal(landscape.xs, grid):
                raise ValueError("Every landscape level must use the stored x_grid")
        if not landscapes or set(landscapes) != set(range(1, max(landscapes) + 1)):
            raise ValueError("Landscape levels must be consecutive starting at 1")
        if any(not isinstance(kernel, PiecewiseLinearFunction) for kernel in kernels.values()):
            raise TypeError("bar_kernels must contain PiecewiseLinearFunction values")
        object.__setattr__(self, "x_grid", grid)
        object.__setattr__(self, "bars", bars)
        object.__setattr__(self, "bar_kernels", MappingProxyType(kernels))
        object.__setattr__(self, "landscapes", MappingProxyType(landscapes))
        object.__setattr__(self, "extra_meta", _freeze_metadata(self.extra_meta))

    @property
    def bar_ids(self) -> tuple[int, ...]:
        """Return source-profile IDs in the requested construction order."""
        return tuple(bar.bar_id for bar in self.bars)

    def evaluate_on_grid(
        self, grid: NDArray[np.float64] | None = None,
        levels: int | Sequence[int] | None = None,
    ) -> NDArray[np.float64]:
        """Retrieve stored samples as a (levels, coordinates) array.

        Parameters
        ----------
        grid : array-like, optional
            Nonempty, finite, nondecreasing 1-D subset of stored coordinates.
            Coordinates must match exactly. Defaults to the complete grid.
        levels : int or sequence of int, optional
            An integer m requests levels 1 through m. A sequence requests levels
            in its given order, including repetitions. Defaults to all levels.
            Noninteger, boolean, or uncomputed levels raise ValueError.
        """
        query = sample_grid(self.x_grid if grid is None else grid)
        indices = np.searchsorted(self.x_grid, query)
        if (np.any(indices >= self.x_grid.size)
                or not np.array_equal(self.x_grid[indices], query)):
            raise ValueError(
                "Evaluation grid must be a subset of the stored landscape x_grid "
                "(exact coordinate matches required). Recompute landscapes with "
                "x_grid set to the requested grid or a larger grid containing all requested points."
            )
        if levels is None:
            ks = sorted(self.landscapes)
        elif isinstance(levels, (int, np.integer, bool, np.bool_)):
            ks = list(range(1, positive_integer(levels, "levels") + 1))
        else:
            try:
                ks = [positive_integer(k, "landscape level") for k in levels]
            except TypeError as error:
                raise ValueError("levels must be an integer or sequence of integers") from error
        missing = sorted(set(ks) - self.landscapes.keys())
        if missing:
            raise ValueError(
                f"Landscape levels {missing} were not computed. Recompute landscapes "
                f"with max_k >= {max(missing)} or request only computed levels."
            )
        return np.array([self.landscapes[k].ys[indices] for k in ks], dtype=float).reshape(len(ks), query.size)


def landscapes_from_profiles(
    profiles: MeasurementProfiles, *, bar_ids: Sequence[Any] | None = None,
    label: str | None = None, max_k: int = 5, num_grid_points: int = 512,
    mode: Literal["raw", "pyramid"] = "pyramid",
    x_grid: NDArray[np.float64] | None = None, tol: float = 1e-12,
) -> MeasurementLandscapeFamily:
    """Build sampled ranked landscapes from an explicit measurement result.

    Parameters
    ----------
    profiles : MeasurementProfiles
        Captured endpoints and scalar profiles. No forest, callable, or registry
        lookup is needed, even if the original source has since changed.
    bar_ids : sequence, optional
        Result-local IDs, captured bars, or original source bars. Defaults to all
        profiles. Order is retained; missing or duplicate selections raise errors.
    label : str, optional
        Result label; defaults to the profile label. Never a cache lookup key.
    max_k : int, default 5
        Positive rank count. Ranks beyond the selected bar count are zero.
    num_grid_points : int, default 512
        Inferred-grid sample count, at least two. Ignored with explicit x_grid.
    mode : {'raw', 'pyramid'}, default 'pyramid'
        Kernel normalization. Pyramid uses lambda(x) = 0.5 * g(2x).
    x_grid : array-like, optional
        Shared finite strictly increasing construction grid, copied on input.
    tol : float, default 1e-12
        Convolution event tolerance, independent of source filtration_tol.

    Returns
    -------
    MeasurementLandscapeFamily
        Read-only stored samples, exact per-bar kernels, selected bar snapshots,
        and source/measurement settings. Empty selections give zero landscapes
        on x_grid, or on the established default domain [0, 1].
    """
    grid = validate_landscape_request(max_k, num_grid_points, mode, x_grid)
    tol = validate_tolerance(tol)
    if not isinstance(profiles, MeasurementProfiles):
        raise TypeError("profiles must be MeasurementProfiles")
    if label is not None and not isinstance(label, str):
        raise TypeError("label must be a string")
    indices = profiles.indices(bar_ids)
    if len(set(indices)) != len(indices):
        raise ValueError("bar_ids must not contain duplicate bars")
    if profiles.baseline != 0.0:
        raise ValueError("Landscape profiles must have baseline=0")
    # Resolve the complete selection first, then compute each kernel once.
    kernels = {i: landscape_kernel(profiles, i, mode=mode, tol=tol) for i in indices}
    if grid is None:
        if kernels:
            left = min(kernel.domain[0] for kernel in kernels.values())
            right = max(kernel.domain[1] for kernel in kernels.values())
            if right <= left:
                raise ValueError("Cannot infer a positive-width grid; provide an explicit x_grid")
        else:
            left, right = 0.0, 1.0
        grid = np.linspace(left, right, num_grid_points)
    values = np.array([kernel(grid) for kernel in kernels.values()], dtype=float).reshape(len(kernels), grid.size)
    values.sort(axis=0)
    levels = {}
    for k in range(1, max_k + 1):
        ys = values[-k] if k <= len(kernels) else np.zeros(grid.size)
        levels[k] = PiecewiseLinearFunction(
            grid, ys, (float(grid[0]), float(grid[-1])),
            metadata={"k": k, "mode": mode},
        )
    return MeasurementLandscapeFamily(
        forest_id=profiles.forest_id, label=profiles.label if label is None else label,
        rescaling=mode, x_grid=grid, bar_kernels=kernels, landscapes=levels,
        bars=tuple(profiles.bars[i] for i in indices),
        extra_meta={**profiles.extra_meta, "profile_label": profiles.label,
                    "measurement_name": profiles.measurement_name,
                    "measurement_params": profiles.measurement_params,
                    "signed": profiles.signed, "baseline": profiles.baseline,
                    "bar_ids": indices, "n_bars": len(indices),
                    "empty_barcode": not indices, "num_grid_points": grid.size,
                    "mode": mode, "tol": tol},
    )
