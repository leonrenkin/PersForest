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
from persforest.measurements.loops import loops_convex_hull_perimeter_deficit
from persforest.measurements.loops import on_midpoint_loops
from persforest.measurements.loops import on_raw_loops
from persforest.measurements.loops import on_smoothed_loops

from candidates import Branches
from candidates import branch_structure
from candidates import long_bars
from segmentation import drop_segments
from segmentation import fill_in
from segmentation import grow
from segmentation import label_regions
from segmentation import open_labels
from segmentation import outline_groups
from segmentation import outline_mask
from segmentation import paint
from selection import Selection
from selection import Stable
from selection import Threshold
from selection import select_cycle
from splitting import Split
from splitting import split_selection


def _loop_measure(functional: Callable, curve: str, scale: float | None) -> Callable:
    """``functional`` on the loops named by ``curve``; ``scale`` is the smoothing sigma."""
    if curve == "smoothed":
        return on_smoothed_loops(functional, sigma=scale)
    return {"raw": on_raw_loops, "midpoint": on_midpoint_loops}[curve](functional)


@dataclass(frozen=True)
class Settings:
    """All parameters of the segmentation, lengths in pixels of the grid it runs on.

    ``droplets.Parameters`` is the reduced public set, which derives these;
    ``droplet_segmentation.ipynb`` explains every one of them.
    """

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
    min_fraction: float | None = None  # dim bars: search from this fraction of the way from birth to peak
    birth_offset: float = 0.0  # search at least this far above the bar's birth, past its first wiggly outlines
    # stage 3
    split_defect: float = 0.02
    min_lobe_radius: float = 2.5
    min_lobe_lifespan: float = 0.5
    lobe_curve: str = "raw"
    lobe_scale: float | None = None
    lobe_max_defect: float = 0.2
    lobe_waist_ratio: float | None = None  # lobes by waist / lobe width instead of the two minima
    lobe_min_rise: float | None = None  # every lobe's brightest pixel lies this far above the bar's birth
    # segments
    seed_late_peaks: bool = False
    divide_below: float | None = None  # seed only peaks that split off below this value; None for all
    fill_in: str | None = "flood"
    boundary_radius: float | None = 1.118  # half the diagonal of a 1 x 2 pixel rectangle
    grow_floor: float | None = None  # grow segments down to this intensity
    # final filter
    min_area: float = 0.0  # drop smaller segments, pixels

    @property
    def measure(self) -> Callable:
        return _loop_measure(self.shape_functional, self.curve, self.scale)

    @property
    def lobe_measure(self) -> Callable:
        return _loop_measure(self.shape_functional, self.lobe_curve, self.lobe_scale)

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
    grown_labels: NDArray[np.int_] | None = None  # after the fill-in and growing
    smoothed_labels: NDArray[np.int_] | None = None  # after the opening, before the final filter
    groups: NDArray[np.int_] | None = None  # the outline each segment of ``labels`` comes from (``outline_groups``)
    dimmed: NDArray[np.bool_] | None = None  # territory of dim lobes, removed after the flooding
    broken: list[int] = field(default_factory=list)  # segments the smoothing cut into several pieces >= min_area
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
    """Start of a bar's outline search: after its splits below ``max_value``, and not below a minimal level.

    The minimal level is ``min_value``, or, with ``min_fraction``, the level
    ``min_fraction`` of the way from the bar's birth to its peak if that is
    lower: a fixed level cuts only the tip of a dim droplet. A droplet whose bar
    dies at or below the level is searched from its last split. The search
    never starts closer than ``birth_offset`` to the bar's birth.
    """
    start = branches.search_start(bar, until=settings.max_value)
    if settings.min_value is not None:
        level = settings.min_value
        if settings.min_fraction is not None:
            level = min(level, bar.birth + settings.min_fraction * (bar.death - bar.birth))
        start = start if level >= bar.death else max(start, level)
    return max(start, bar.birth + settings.birth_offset) if settings.birth_offset < bar.lifespan() else start


# The steps of ``segment``, each a function of the products of the previous ones.


def find_peaks(forest: PersistenceForest, settings: Settings) -> list[PFBar]:
    """Step 1: the bars at least ``min_bar_length`` long, one per candidate droplet."""
    return long_bars(forest, settings.min_bar_length)


def split_forest(forest: PersistenceForest, bars: list[PFBar],
                 settings: Settings) -> tuple[Branches, list[PFBar], list[PFBar]]:
    """Step 2: which bars split off which, and which are outlined.

    Bars born below ``max_value`` are separate droplets and get an outline;
    *late* bars, born above it, lie inside their elder's outline. A bar born
    above ``max_value`` that splits off no selected bar has no elder outline
    to lie in, and is outlined itself: e.g. a bright droplet whose surroundings
    reach the border of the grid, beyond which counts as brighter than
    everything, so that it separates from the border only high up.
    """
    branches = branch_structure(forest, bars)
    outlined = [bar for bar in bars if bar.birth < settings.max_value or branches.parent[bar] is None]
    late = [bar for bar in bars if bar.birth >= settings.max_value and branches.parent[bar] is not None]
    return branches, outlined, late


def find_outlines(forest: PersistenceForest, branches: Branches, outlined: list[PFBar],
                  settings: Settings) -> dict[PFBar, Selection]:
    """Step 3: the first reliable representative of every outlined bar, from its search start on."""
    selections = {}
    for bar in outlined:
        start = search_start(branches, bar, settings)
        # The birth offset can lift the start of a bar born just below max_value above it; it is searched freely.
        max_value = settings.max_value if start < settings.max_value else None
        selections[bar] = select_cycle(forest, bar, settings.measure, settings.rule, start, max_value=max_value)
    return selections


def find_lobes(forest: PersistenceForest, selections: dict[PFBar, Selection], image: NDArray[np.float64],
               settings: Settings) -> dict[PFBar, Split]:
    """Step 4: the lobes of 8-shaped outlines, for outlines whose defect exceeds ``split_defect``.

    With ``lobe_min_rise``, a lobe is a droplet only if its brightest pixel
    lies at least that far above the birth of the outline's bar: the
    bar-length check of step 1, applied to the lobes, which have no bar of
    their own. Lobes failing it are listed in ``dim_lobes`` (and stay in
    ``lobes``): they take part in the flooding, so that the other lobes do
    not flood into them, and are removed after it (``relabel``). A dim lobe is
    cut off like a droplet but, like a dim peak in step 1, is not one.
    """
    splits = {}
    for bar, selection in selections.items():
        if selection.defect <= settings.split_defect:
            continue
        split = split_selection(selection, forest.geometry, settings.min_lobe_radius, settings.min_lobe_lifespan,
                                settings.lobe_measure, settings.lobe_max_defect, waist_ratio=settings.lobe_waist_ratio)
        if split.lobes and settings.lobe_min_rise is not None:
            level = bar.birth + settings.lobe_min_rise - 1e-9  # 8-bit levels are exact; this absorbs rounding
            masks = [outline_mask(lobe.cycle, split.points, image.shape) for lobe in split.lobes]
            dim = [lobe for lobe, mask in zip(split.lobes, masks) if not (mask.any() and image[mask].max() >= level)]
            split = replace(split, dim_lobes=dim)
        splits[bar] = split
    return splits


def segment(image: NDArray[np.float64], settings: Settings = Settings()) -> Result:
    """Segment the bright droplets of ``image``."""
    timings = {}
    clock = time.perf_counter()
    forest = PersistenceForest.from_cubical_grid(image, storage=settings.storage)
    bars = find_peaks(forest, settings)
    branches, outlined, late = split_forest(forest, bars, settings)
    timings["forest"] = time.perf_counter() - clock

    clock = time.perf_counter()
    selections = find_outlines(forest, branches, outlined, settings)
    timings["outlines"] = time.perf_counter() - clock

    clock = time.perf_counter()
    splits = find_lobes(forest, selections, image, settings)
    timings["splitting"] = time.perf_counter() - clock
    return relabel(outlines_result(forest, bars, branches, selections, late, splits, image.shape, timings),
                   image, settings)


def outlines_result(forest: PersistenceForest, bars: list[PFBar], branches: Branches,
                    selections: dict[PFBar, Selection], late: list[PFBar], splits: dict[PFBar, Split],
                    shape: tuple[int, int], timings: dict[str, float] | None = None) -> Result:
    """A ``Result`` holding the products of steps 1 to 4, to be completed by ``relabel``."""
    empty = np.zeros(shape, dtype=int)
    return Result(forest, bars, branches, selections, late, splits, regions=[], split_outlines=[], crowded=[],
                  stage2_labels=empty, region_labels=empty, labels=empty, timings=dict(timings or {}))


def relabel(result: Result, image: NDArray[np.float64], settings: Settings) -> Result:
    """Steps 5 to 7: turn the outlines and lobes of ``result`` into segments.

    Only ``seed_late_peaks``, ``divide_below``, ``fill_in``, ``grow_floor``,
    ``boundary_radius`` and ``min_area`` are read, so these
    options can be compared without recomputing the outlines.

    With ``divide_below``, a region is divided only at the peaks that split off
    (are born) below it: two maxima that separate only high up, near the top
    of one blob, stay one segment. At or below ``max_value`` no region is divided.
    """
    clock = time.perf_counter()
    shape = image.shape
    forest, branches = result.forest, result.branches
    regions, split_outlines, dim_regions = [], [], []
    for bar, selection in result.selections.items():
        outline = outline_mask(selection.cycle, forest.geometry, shape)
        split = result.splits.get(bar)
        if split is not None and split.lobes:
            for lobe in split.lobes:
                regions.append(outline_mask(lobe.cycle, split.points, shape))
                if any(lobe is dim for dim in split.dim_lobes):
                    dim_regions.append(regions[-1])
            split_outlines.append(outline)
        else:
            regions.append(outline)
    peaks, cores = {}, {}
    if settings.seed_late_peaks:
        seeded = [bar for bar in result.bars if settings.divide_below is None or bar.birth < settings.divide_below]
        peaks = {bar: peak(bar, forest) for bar in seeded}
        cores = {bar: outline_mask(forest.cycle_for_bar_at(bar, branches.search_start(bar)), forest.geometry, shape)
                 for bar in seeded}
    region_labels, _, crowded = label_regions(regions, peaks, cores, shape)
    # Step 5: flooding, inside the outlines holding several segments and then down to the floor.
    labels = region_labels
    if settings.fill_in:
        for container in crowded + split_outlines:
            labels = fill_in(labels, container, image, settings.fill_in)
    if settings.grow_floor is not None:
        labels = grow(labels, image, settings.grow_floor)
    # Dim lobes have kept the others out of their territory; now it becomes background.
    dim_ids = {np.bincount(region_labels[region & (region_labels > 0)]).argmax()
               for region in dim_regions if (region & (region_labels > 0)).any()}
    dimmed = np.isin(labels, list(dim_ids)) if dim_ids else np.zeros(shape, dtype=bool)
    labels = np.where(dimmed, 0, labels)
    grown_labels = labels
    # Step 6: smoothing the boundaries.
    if settings.boundary_radius:
        labels = open_labels(labels, settings.boundary_radius)
    smoothed_labels = labels
    # Step 7: the final filter.
    labels, broken = drop_segments(labels, settings.min_area)
    stage2_labels = paint([outline_mask(selection.cycle, forest.geometry, shape)
                           for selection in result.selections.values()], shape)
    groups = outline_groups(labels, region_labels, stage2_labels)
    timings = {**result.timings, "segments": time.perf_counter() - clock}
    return replace(result, regions=regions, split_outlines=split_outlines, crowded=crowded,
                   stage2_labels=stage2_labels, region_labels=region_labels, labels=labels,
                   grown_labels=grown_labels, smoothed_labels=smoothed_labels, groups=groups, dimmed=dimmed, broken=broken,
                   timings=timings)
