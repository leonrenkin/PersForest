"""Measurement profiles captured from cycle progressions.

Representatives and coordinates are borrowed during measurement. Results own
scalar data and endpoint snapshots; numerical reuse never reads live bars.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._validation import require_computed_barcode, sample_grid
from .functions import StepFunctionData, _freeze_metadata

CycleValueFunc = Callable[[Any, np.ndarray], float]


@dataclass(frozen=True, eq=False)
class ProfileBar:
    """Captured bar endpoints and a result-local integer identifier.

    IDs remain attached to their profile when a subset is requested in another
    order. They do not identify bars across independently computed results.
    """
    bar_id: int
    birth: float
    death: float
    root_id: int | None = None

    def __post_init__(self) -> None:
        if (isinstance(self.bar_id, (bool, np.bool_))
                or not isinstance(self.bar_id, (int, np.integer)) or self.bar_id < 0):
            raise ValueError("bar_id must be a nonnegative integer")
        if np.isnan((self.birth, self.death)).any() or self.death < self.birth:
            raise ValueError("Bar endpoints must be ordered and not NaN")

    def lifespan(self) -> float:
        """Return the captured bar length."""
        return self.death - self.birth


@dataclass(frozen=True, eq=False)
class MeasurementProfiles:
    """Owned measurement data with immutable bar ordering and lookup maps.

    ``bars`` contains endpoint snapshots. Lookups accept a result-local integer
    ID, a captured bar, or an original source bar. Source bars are borrowed only
    for identity lookup, never for numerical computation. ``forest_id`` is a
    descriptive process-local label, not a persistent provenance identifier.
    """
    forest_id: str
    label: str
    baseline: float
    bars: tuple[ProfileBar, ...]
    step_functions: Mapping[int, StepFunctionData]
    signed: bool = False
    measurement_name: str | None = None
    measurement_params: Mapping[str, Any] = field(default_factory=dict)
    extra_meta: Mapping[str, Any] = field(default_factory=dict)
    _source_bars: tuple[Any, ...] = field(default=(), repr=False)
    _source_identity: int | None = field(default=None, repr=False)
    _bar_to_index: Mapping[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bars = tuple(self.bars)
        steps = dict(self.step_functions)
        if any(not isinstance(bar, ProfileBar) for bar in bars):
            raise TypeError("bars must contain ProfileBar endpoint snapshots")
        if tuple(bar.bar_id for bar in bars) != tuple(range(len(bars))):
            raise ValueError("Profile bar IDs must be consecutive in stored order")
        if (any(isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)) for key in steps)
                or set(steps) != set(range(len(bars)))):
            raise ValueError("Every stored bar must have exactly one measurement profile")
        if not isinstance(self.signed, (bool, np.bool_)):
            raise TypeError("signed must be a boolean")
        if not isinstance(self.label, str) or not isinstance(self.forest_id, str):
            raise TypeError("label and forest_id must be strings")
        if self.measurement_name is not None and not isinstance(self.measurement_name, str):
            raise TypeError("measurement_name must be a string")
        if not isinstance(self.measurement_params, Mapping) or not isinstance(self.extra_meta, Mapping):
            raise TypeError("measurement_params and extra_meta must be mappings")
        baseline = float(self.baseline)
        for bar in bars:
            step = steps[bar.bar_id]
            if not isinstance(step, StepFunctionData):
                raise TypeError("step_functions must contain StepFunctionData")
            if step.baseline != baseline and not (np.isnan(step.baseline) and np.isnan(baseline)):
                raise ValueError("Profile baselines must match the result baseline")
            if np.any(step.starts < bar.birth) or np.any(step.ends > bar.death):
                raise ValueError(f"Profile {bar.bar_id} extends outside its captured bar")
        source_bars = tuple(self._source_bars)
        if source_bars and len(source_bars) != len(bars):
            raise ValueError("Source bar lookup must align with captured bars")
        lookup = {id(bar): i for i, bar in enumerate(bars)}
        if len({id(bar) for bar in source_bars}) != len(source_bars):
            raise ValueError("Source bars must not be duplicated")
        lookup.update({id(bar): i for i, bar in enumerate(source_bars)})
        object.__setattr__(self, "bars", bars)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "step_functions", MappingProxyType(steps))
        object.__setattr__(self, "_source_bars", source_bars)
        object.__setattr__(self, "_bar_to_index", MappingProxyType(lookup))
        object.__setattr__(self, "measurement_params", _freeze_metadata(self.measurement_params))
        object.__setattr__(self, "extra_meta", _freeze_metadata(self.extra_meta))

    @property
    def bar_ids(self) -> tuple[int, ...]:
        """Return IDs in the stored profile order."""
        return tuple(bar.bar_id for bar in self.bars)

    def _index(self, bar_or_index: Any) -> int:
        if isinstance(bar_or_index, (bool, np.bool_)):
            raise ValueError("Bar IDs must be integers, not booleans")
        if isinstance(bar_or_index, (int, np.integer)):
            index = int(bar_or_index)
        else:
            index = self._bar_to_index.get(id(bar_or_index), -1)
        if index not in self.step_functions:
            raise ValueError(
                f"Requested bar {bar_or_index!r} has no measurement profile. "
                f"Available bar IDs: {self.bar_ids}. Recompute profiles including the requested bar."
            )
        return index

    def indices(self, bars: Sequence[Any] | None = None) -> tuple[int, ...]:
        """Resolve a complete ordered selection before numerical work."""
        return self.bar_ids if bars is None else tuple(self._index(bar) for bar in bars)

    def get(self, bar_or_index: Any) -> StepFunctionData:
        """Return the profile for a stored ID, captured bar, or source bar."""
        return self.step_functions[self._index(bar_or_index)]

    def __getitem__(self, bar_or_index: Any) -> StepFunctionData:
        return self.get(bar_or_index)

    def evaluate_on_grid(self, grid: NDArray[np.float64], bars: Sequence[Any] | None = None) -> NDArray[np.float64]:
        """Evaluate selected profiles on a finite, nondecreasing 1-D grid.

        Returns an array with one row per requested bar, preserving selection
        order and repetitions. An empty selection has shape ``(0, len(grid))``.
        """
        grid = sample_grid(grid)
        indices = self.indices(bars)
        return np.array([self.step_functions[i].eval_on_grid(grid) for i in indices], dtype=float).reshape(len(indices), grid.size)

    def require_source(self, forest: object, bar: Any) -> None:
        """Check association before pairing a profile with live forest rendering."""
        require_computed_barcode(forest)
        index = self._index(bar)
        captured = self.bars[index]
        if (self._source_identity != id(forest) or bar not in forest.barcode
                or not self._source_bars or self._source_bars[index] is not bar
                or (bar.birth, bar.death) != (captured.birth, captured.death)):
            raise ValueError("Measurement profiles do not match this forest and bar; recompute profiles")


def _measurement_geometry(forest):
    """Return the second measurement argument: ``forest.geometry`` or, for generic
    forest-like inputs without it, ``forest.point_cloud``."""
    geometry = getattr(forest, "geometry", None)
    return forest.point_cloud if geometry is None else geometry


def _build_step_function_data(
    forest: object, bar: Any, cycle_func: CycleValueFunc, baseline: float = 0.0,
) -> StepFunctionData:
    """Measure descending representatives once and store ascending [start,end) data.

    Callbacks borrow representatives and coordinates and must not mutate them.
    Only scalar records are retained, including when representatives are reconstructed from diffs.
    """
    require_computed_barcode(forest)
    if bar not in forest.barcode:
        raise ValueError("Bar is not in barcode of forest")
    records = []
    for cycle in forest.iter_bar_cycle_reps(bar):
        value = cycle_func(cycle, _measurement_geometry(forest))
        if not np.isscalar(value) or np.iscomplexobj(value):
            raise ValueError("Measurements must return a real scalar")
        records.append((cycle.active_start, cycle.active_end, float(value)))
    records.reverse()
    starts, ends, vals = np.asarray(records, dtype=float).reshape(-1, 3).T
    return StepFunctionData(
        starts, ends, vals, baseline, (float(bar.birth), float(bar.death)),
        metadata={"bar_birth": bar.birth, "bar_death": bar.death, "root_id": getattr(bar, "root_id", None)},
    )


def compute_measurement_profiles(
    forest: object, cycle_func: CycleValueFunc, label: str, *,
    signed: bool = False, min_bar_length: float = 0.0, baseline: float = 0.0,
    cache: bool = True, measurement_name: str | None = None,
    measurement_params: Mapping[str, Any] | None = None,
) -> MeasurementProfiles:
    """Capture per-bar measurement profiles from a computed forest.

    Parameters
    ----------
    forest : object
        Forest-like source with barcode, point_cloud and iter_bar_cycle_reps.
        An explicitly false ``_barcode_computed`` is rejected before work.
    cycle_func : callable
        Read-only ``(chain, point_cloud) -> real scalar`` measurement.
    label : str
        Result label and optional forest registry key; overwrites on success.
    signed : bool, default False
        Cancel opposing simplices before measurement, matching the public
        forest method. True measures representatives as stored.
    min_bar_length : float, default 0
        Inclusive finite, nonnegative length threshold.
    baseline : float, default 0
        Value outside recorded half-open intervals. Landscapes require zero.
    cache : bool, default True
        Publish the result to ``forest.measurement_profiles`` only on success.
    measurement_name : str, optional
        Descriptive name, never used as a callable identity. Defaults to the
        ``measurement_name`` attribute of ``cycle_func`` if it has one (as
        the ``on_*_loops`` measurements do), else to label.
    measurement_params : mapping, optional
        Descriptive parameters. Defaults to the ``measurement_params``
        attribute of ``cycle_func`` if it has one, else empty; never guessed
        from other properties of the callable.

    Returns
    -------
    MeasurementProfiles
        Endpoint snapshots, scalar data, preprocessing and source metadata.
        Negative/nonfinite scalar values and infinite bar deaths remain usable
        for standalone profiles; landscape construction validates its domain.
    """
    require_computed_barcode(forest)
    if not callable(cycle_func):
        raise TypeError("cycle_func must be callable")
    if not isinstance(signed, (bool, np.bool_)):
        raise TypeError("signed must be a boolean")
    if not isinstance(cache, (bool, np.bool_)):
        raise TypeError("cache must be a boolean")
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    # Measurements may describe themselves through these two attributes;
    # attributes of other types (e.g. on mocks) are ignored.
    if measurement_name is None and isinstance(getattr(cycle_func, "measurement_name", None), str):
        measurement_name = cycle_func.measurement_name
    if measurement_params is None and isinstance(getattr(cycle_func, "measurement_params", None), Mapping):
        measurement_params = cycle_func.measurement_params
    if measurement_name is not None and not isinstance(measurement_name, str):
        raise TypeError("measurement_name must be a string")
    if measurement_params is not None and not isinstance(measurement_params, Mapping):
        raise TypeError("measurement_params must be a mapping")
    # Check descriptive inputs before invoking possibly expensive measurements.
    measurement_params = _freeze_metadata({} if measurement_params is None else measurement_params)
    if not np.isscalar(baseline) or np.iscomplexobj(baseline):
        raise ValueError("baseline must be a real scalar")
    baseline = float(baseline)
    if not np.isscalar(min_bar_length) or isinstance(min_bar_length, (bool, np.bool_)):
        raise ValueError("min_bar_length must be finite and nonnegative")
    min_bar_length = float(min_bar_length)
    if not np.isfinite(min_bar_length) or min_bar_length < 0:
        raise ValueError("min_bar_length must be finite and nonnegative")
    # PFBar sets have no stable iteration order. Sort only this extraction's
    # records, without changing forest construction, ties, or representative order.
    source_bars = list(forest.barcode)
    for i, bar in enumerate(source_bars):
        ProfileBar(i, float(bar.birth), float(bar.death))
    if isinstance(forest.barcode, (set, frozenset)):
        source_bars.sort(key=lambda bar: (bar.birth, bar.death, tuple(getattr(bar, "_node_progression", ()))))
    source_manifest = tuple((float(bar.birth), float(bar.death), tuple(getattr(bar, "_node_progression", ()))) for bar in source_bars)
    selected = [bar for bar in source_bars if bar.death - bar.birth >= min_bar_length]
    bars = tuple(ProfileBar(i, float(bar.birth), float(bar.death), getattr(bar, "root_id", None)) for i, bar in enumerate(selected))
    if signed:
        measurement = cycle_func
    else:
        def measurement(chain, points):
            return cycle_func(chain.unsigned(), points)
    steps = {i: _build_step_function_data(forest, bar, measurement, baseline) for i, bar in enumerate(selected)}
    result = MeasurementProfiles(
        forest_id=str(getattr(forest, "id", id(forest))), label=label, baseline=baseline,
        bars=bars, step_functions=steps, signed=bool(signed),
        measurement_name=label if measurement_name is None else measurement_name,
        measurement_params=measurement_params,
        extra_meta={"min_bar_length": min_bar_length, "source_barcode": source_manifest,
                    "filtration_tol": getattr(forest, "filtration_tol", None),
                    "reduced": getattr(forest, "reduced", None)},
        _source_bars=tuple(selected), _source_identity=id(forest),
    )
    if cache:
        if not hasattr(forest, "measurement_profiles"):
            forest.measurement_profiles = {}
        forest.measurement_profiles[label] = result
    return result
