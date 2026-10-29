"""The whole segmentation as one function of the image and its settings.

The notebook walks through the same steps one by one; ``segment`` runs them
for the benchmark, where settings and images vary.
"""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Callable
import time

import numpy as np
from numpy.typing import NDArray

from persforest import PersistenceForest
from persforest.barcode import PFBar
from persforest.measurements.loops import LoopMeasurement
from persforest.measurements.loops import loops_convex_hull_perimeter_deficit

from candidates import Branches
from candidates import branch_structure
from candidates import long_bars
from segmentation import fill_in
from segmentation import label_regions
from segmentation import open_labels
from segmentation import outline_mask
from segmentation import paint
from selection import Selection
from selection import Stable
from selection import Threshold
from selection import select_cycle
from splitting import Split
from splitting import split_selection


@dataclass(frozen=True)
class Settings:
    """All parameters of the segmentation; see the notebook for their meaning."""

    # stage 1; "diffs" reconstructs representatives instead of storing them, for large images
    storage: str = "cycles"
    min_bar_length: float = 0.1
    # stage 2
    shape_functional: Callable = loops_convex_hull_perimeter_deficit
    curve: str = "midpoint"
    scale: float | None = None
    selection: str = "stable"
    stability_bandwidth: float = 0.002
    stability_tol: float = 0.002
    stability_max_defect: float | None = 0.08
    max_defect: float = 0.05
    max_value: float = 0.3
    min_value: float | None = 0.2
    # stage 3
    split_defect: float = 0.02
    min_lobe_radius: float = 2.5
    min_lobe_lifespan: float = 0.5
    lobe_curve: str = "raw"
    lobe_scale: float | None = None
    lobe_max_defect: float = 0.2
    # segments
    seed_late_peaks: bool = False
    fill_in: str | None = "flood"
    boundary_radius: float | None = 1.118  # half the diagonal of a 1 x 2 pixel rectangle

    @property
    def measure(self) -> LoopMeasurement:
        return LoopMeasurement(self.shape_functional, curve=self.curve,
                               scale=self.scale if self.curve == "smoothed" else None)

    @property
    def lobe_measure(self) -> LoopMeasurement:
        return LoopMeasurement(self.shape_functional, curve=self.lobe_curve,
                               scale=self.lobe_scale if self.lobe_curve == "smoothed" else None)

    @property
    def rule(self) -> Threshold | Stable:
        if self.selection == "threshold":
            return Threshold(self.max_defect)
        if self.selection == "stable":
            return Stable(self.stability_bandwidth, self.stability_tol, self.stability_max_defect)
        raise ValueError(f"selection must be 'stable' or 'threshold', got {self.selection!r}")


@dataclass
class Result:
    """Every intermediate product of ``segment``."""

    forest: PersistenceForest
    bars: list[PFBar]
    branches: Branches
    selections: dict[PFBar, Selection]
    late: list[PFBar]
    splits: dict[PFBar, Split]
    regions: list[NDArray[np.bool_]]
    split_outlines: list[NDArray[np.bool_]]
    crowded: list[NDArray[np.bool_]]
    stage2_labels: NDArray[np.int_]
    region_labels: NDArray[np.int_]
    labels: NDArray[np.int_]
    timings: dict[str, float] = field(default_factory=dict)


def peak(bar: PFBar, forest: PersistenceForest | None = None) -> tuple[int, int]:
    """Pixel of a cubical bar's peak: the last pixel to enter lies in its last representative.

    ``forest`` is needed when the representatives are not stored (``storage="diffs"``).
    """
    stored = bar.cycle_reps[-1] if bar.cycle_reps else None
    last = stored if stored is not None else next(iter(forest.iter_bar_cycle_reps(bar)))
    (row, column), _ = next(iter(last.interior))
    return row // 2, column // 2


def search_start(branches: Branches, bar: PFBar, settings: Settings) -> float:
    """Start of a bar's outline search: after its splits below ``max_value``, and not below ``min_value``."""
    start = branches.search_start(bar, until=settings.max_value)
    return start if settings.min_value is None else max(start, settings.min_value)


def segment(image: NDArray[np.float64], settings: Settings = Settings()) -> Result:
    """Segment the bright droplets of ``image``."""
    timings = {}
    clock = time.perf_counter()
    forest = PersistenceForest.from_cubical_grid(image, storage=settings.storage)
    bars = long_bars(forest, settings.min_bar_length)
    branches = branch_structure(forest, bars)
    outlined = [bar for bar in bars if bar.birth < settings.max_value]
    late = [bar for bar in bars if bar.birth >= settings.max_value]
    timings["forest"] = time.perf_counter() - clock

    clock = time.perf_counter()
    selections = {bar: select_cycle(forest, bar, settings.measure, settings.rule,
                                    search_start(branches, bar, settings), max_value=settings.max_value)
                  for bar in outlined}
    timings["outlines"] = time.perf_counter() - clock

    clock = time.perf_counter()
    splits = {bar: split_selection(selection, forest.geometry, settings.min_lobe_radius,
                                   settings.min_lobe_lifespan, settings.lobe_measure, settings.lobe_max_defect)
              for bar, selection in selections.items() if selection.defect > settings.split_defect}
    timings["splitting"] = time.perf_counter() - clock

    empty = np.zeros(image.shape, dtype=int)
    result = Result(forest, bars, branches, selections, late, splits, [], [], [], empty, empty, empty, timings)
    return relabel(result, image, settings)


def relabel(result: Result, image: NDArray[np.float64], settings: Settings) -> Result:
    """Turn the outlines and lobes of ``result`` into segments with the segment options of ``settings``.

    Only ``seed_late_peaks``, ``fill_in`` and ``boundary_radius`` are read, so
    segment options can be compared without recomputing the outlines.
    """
    clock = time.perf_counter()
    shape = image.shape
    forest, branches = result.forest, result.branches
    regions, split_outlines = [], []
    for bar, selection in result.selections.items():
        outline = outline_mask(selection.cycle, forest.geometry, shape)
        split = result.splits.get(bar)
        if split is not None and split.lobes:
            regions += [outline_mask(lobe.cycle, split.points, shape) for lobe in split.lobes]
            split_outlines.append(outline)
        else:
            regions.append(outline)
    peaks, cores = {}, {}
    if settings.seed_late_peaks:
        peaks = {bar: peak(bar, forest) for bar in result.bars}
        cores = {bar: outline_mask(forest.cycle_for_bar_at(bar, branches.search_start(bar)), forest.geometry, shape)
                 for bar in result.bars}
    region_labels, _, crowded = label_regions(regions, peaks, cores, shape)
    labels = region_labels
    if settings.fill_in:
        for container in crowded + split_outlines:
            labels = fill_in(labels, container, image, settings.fill_in)
    if settings.boundary_radius:
        labels = open_labels(labels, settings.boundary_radius)
    stage2_labels = paint([outline_mask(selection.cycle, forest.geometry, shape)
                           for selection in result.selections.values()], shape)
    timings = {**result.timings, "segments": time.perf_counter() - clock}
    return replace(result, regions=regions, split_outlines=split_outlines, crowded=crowded,
                   stage2_labels=stage2_labels, region_labels=region_labels, labels=labels, timings=timings)
