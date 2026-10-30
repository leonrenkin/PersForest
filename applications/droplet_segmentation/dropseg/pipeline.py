"""The steps of the segmentation as functions, and ``segment``, which runs them on one grid.

``Settings`` holds the parameters in units of the grid the steps run on
(fine pixels when upsampled); ``Parameters.settings`` derives them. Every
step is a function of the products of the previous ones, so that
``stages.Stages`` can cache them one by one.
"""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import time

import numpy as np
from numpy.typing import NDArray

from persforest import PersistenceForest
from persforest.barcode import PFBar
from persforest.measurements.loops import loops_convex_hull_perimeter_deficit
from persforest.measurements.loops import on_midpoint_loops

from .lobes import Split
from .lobes import split_selection
from .outlines import Selection
from .outlines import select_cycle
from .peaks import Branches
from .peaks import branch_structure
from .peaks import long_bars
from .segments import drop_segments
from .segments import fill_in
from .segments import grow
from .segments import open_labels
from .segments import outline_groups
from .segments import outline_mask
from .segments import paint

# Convex-hull perimeter deficit of the midpoint loops, which cut the staircase corners of pixel outlines.
OUTLINE_MEASURE = on_midpoint_loops(loops_convex_hull_perimeter_deficit)
# Outlines with at most this deficit are convex enough to hold no waist and are not examined for lobes.
SPLIT_DEFECT = 0.02


@dataclass(frozen=True)
class Settings:
    """The parameters of the steps, lengths in pixels of the grid they run on.

    Attributes
    ----------
    storage : {"cycles", "diffs"}
        Storage of the forest; "diffs" reconstructs representatives instead
        of storing them, which needs less memory for large grids. The
        segments are the same.
    min_bar_length : float
        Step 1, and the minimal rise of a lobe in step 4.
    split_level : float
        Step 2.
    min_level : float
        Step 3, and the floor of the flooding in step 5.
    birth_offset : float
        Step 3: outlines lie at least this far (intensity) above the bar's birth.
    max_defect : float
        Step 3: the outline is the first with at most this convex-hull
        perimeter deficit; inf takes the first.
    split_lobes : bool
        Step 4.
    lobe_waist : float
        Step 4: the waist ratio s / rho of a lobe.
    smooth_radius : float or None
        Step 6, grid pixels.
    min_area : float
        Step 7, grid pixels.
    """

    storage: str = "cycles"
    min_bar_length: float = 0.2
    split_level: float = 0.3
    min_level: float = 0.1
    birth_offset: float = 0.02
    max_defect: float = float("inf")
    split_lobes: bool = True
    lobe_waist: float = 0.75
    smooth_radius: float | None = None
    min_area: float = 2.0


@dataclass
class Result:
    """Every intermediate product of ``segment``.

    Attributes
    ----------
    forest : PersistenceForest
        Cubical forest of the image.
    bars : list of PFBar
        Step 1: the candidate peaks.
    branches : Branches
        Step 2: which candidate splits off which.
    selections : dict
        Step 3: the outline of every outlined bar.
    late : list of PFBar
        Step 2: bars born above ``split_level``, merged into their elder's outline.
    splits : dict
        Step 4: the splits of the outlines examined for lobes.
    regions : list of ndarray of bool
        Outlines and lobes as masks.
    split_outlines : list of ndarray of bool
        Masks of the outlines that were split into lobes.
    outline_labels : ndarray of int
        The step-3 outlines as a label image.
    region_labels : ndarray of int
        The regions as a label image, before flooding.
    labels : ndarray of int
        The final segments; 0 is background.
    grown_labels : ndarray of int
        After the flooding of step 5 (dim lobes removed).
    smoothed_labels : ndarray of int
        After the smoothing of step 6, before the filter.
    groups : ndarray of int
        The outline each segment of ``labels`` comes from (``segments.outline_groups``).
    dimmed : ndarray of bool
        Territory of dim lobes, removed after the flooding.
    broken : list of int
        Segments the smoothing cut into several pieces of at least ``min_area``.
    timings : dict
        Seconds per stage.
    """

    forest: PersistenceForest
    bars: list[PFBar]
    branches: Branches
    selections: dict[PFBar, Selection]
    late: list[PFBar]
    splits: dict[PFBar, Split]
    regions: list[NDArray[np.bool_]]
    split_outlines: list[NDArray[np.bool_]]
    outline_labels: NDArray[np.int_]
    region_labels: NDArray[np.int_]
    labels: NDArray[np.int_]
    grown_labels: NDArray[np.int_] | None = None
    smoothed_labels: NDArray[np.int_] | None = None
    groups: NDArray[np.int_] | None = None
    dimmed: NDArray[np.bool_] | None = None
    broken: list[int] = field(default_factory=list)
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
    """Start of a bar's outline search: after its splits below ``split_level``, and not below ``min_level``.

    A droplet whose bar dies at or below ``min_level`` is searched from its
    last split. The search never starts closer than ``birth_offset`` to the
    bar's birth.
    """
    start = branches.search_start(bar, until=settings.split_level)
    start = start if settings.min_level >= bar.death else max(start, settings.min_level)
    return max(start, bar.birth + settings.birth_offset) if settings.birth_offset < bar.lifespan() else start


def find_peaks(forest: PersistenceForest, settings: Settings) -> list[PFBar]:
    """Step 1: the bars at least ``min_bar_length`` long, one per candidate droplet."""
    return long_bars(forest, settings.min_bar_length)


def split_forest(forest: PersistenceForest, bars: list[PFBar],
                 settings: Settings) -> tuple[Branches, list[PFBar], list[PFBar]]:
    """Step 2: which bars split off which, and which are outlined.

    Bars born below ``split_level`` are separate droplets and get an outline;
    *late* bars, born above it, lie inside their elder's outline. A bar born
    above ``split_level`` that splits off no selected bar has no elder outline
    to lie in, and is outlined itself: e.g. a bright droplet whose surroundings
    reach the border of the grid, beyond which counts as brighter than
    everything, so that it separates from the border only high up.
    """
    branches = branch_structure(forest, bars)
    outlined = [bar for bar in bars if bar.birth < settings.split_level or branches.parent[bar] is None]
    late = [bar for bar in bars if bar.birth >= settings.split_level and branches.parent[bar] is not None]
    return branches, outlined, late


def find_outlines(forest: PersistenceForest, branches: Branches, outlined: list[PFBar],
                  settings: Settings) -> dict[PFBar, Selection]:
    """Step 3: the outline of every outlined bar, from its search start on."""
    selections = {}
    for bar in outlined:
        start = search_start(branches, bar, settings)
        # The birth offset can lift the start of a bar born just below split_level above it; it is searched freely.
        max_value = settings.split_level if start < settings.split_level else None
        selections[bar] = select_cycle(forest, bar, OUTLINE_MEASURE, settings.max_defect, start, max_value=max_value)
    return selections


def find_lobes(forest: PersistenceForest, selections: dict[PFBar, Selection], image: NDArray[np.float64],
               settings: Settings) -> dict[PFBar, Split]:
    """Step 4: the lobes of 8-shaped outlines, for outlines whose defect exceeds ``SPLIT_DEFECT``.

    A lobe is a droplet only if its brightest pixel lies at least
    ``min_bar_length`` above the birth of the outline's bar: the bar-length
    check of step 1, applied to the lobes, which have no bar of their own.
    Lobes failing it are listed in ``dim_lobes`` (and stay in ``lobes``):
    they take part in the flooding, so that the other lobes do not flood
    into them, and are removed after it (``relabel``).
    """
    if not settings.split_lobes:
        return {}
    splits = {}
    for bar, selection in selections.items():
        if selection.defect <= SPLIT_DEFECT:
            continue
        split = split_selection(selection, forest.geometry, settings.lobe_waist)
        if split.lobes:
            level = bar.birth + settings.min_bar_length - 1e-9  # 8-bit levels are exact; this absorbs rounding
            masks = [outline_mask(lobe.cycle, split.points, image.shape) for lobe in split.lobes]
            dim = [lobe for lobe, mask in zip(split.lobes, masks) if not (mask.any() and image[mask].max() >= level)]
            split = replace(split, dim_lobes=dim)
        splits[bar] = split
    return splits


def segment(image: NDArray[np.float64], settings: Settings = Settings()) -> Result:
    """Segment the bright droplets of ``image``, all steps on its own grid."""
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
    timings["lobes"] = time.perf_counter() - clock
    return relabel(outlines_result(forest, bars, branches, selections, late, splits, image.shape, timings),
                   image, settings)


def outlines_result(forest: PersistenceForest, bars: list[PFBar], branches: Branches,
                    selections: dict[PFBar, Selection], late: list[PFBar], splits: dict[PFBar, Split],
                    shape: tuple[int, int], timings: dict[str, float] | None = None) -> Result:
    """A ``Result`` holding the products of steps 1 to 4, to be completed by ``relabel``."""
    empty = np.zeros(shape, dtype=int)
    return Result(forest, bars, branches, selections, late, splits, regions=[], split_outlines=[],
                  outline_labels=empty, region_labels=empty, labels=empty, timings=dict(timings or {}))


def relabel(result: Result, image: NDArray[np.float64], settings: Settings) -> Result:
    """Steps 5 to 7: turn the outlines and lobes of ``result`` into segments.

    Only ``min_level``, ``smooth_radius`` and ``min_area`` are read, so these
    can be changed without recomputing the outlines.
    """
    clock = time.perf_counter()
    shape = image.shape
    forest = result.forest
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
    region_labels = paint(regions, shape)
    # Step 5: flooding, inside the split outlines and then down to min_level.
    labels = region_labels
    for container in split_outlines:
        labels = fill_in(labels, container, image)
    labels = grow(labels, image, settings.min_level)
    # Dim lobes have kept the others out of their territory; now it becomes background.
    dim_ids = {np.bincount(region_labels[region & (region_labels > 0)]).argmax()
               for region in dim_regions if (region & (region_labels > 0)).any()}
    dimmed = np.isin(labels, list(dim_ids)) if dim_ids else np.zeros(shape, dtype=bool)
    labels = np.where(dimmed, 0, labels)
    grown_labels = labels
    # Step 6: smoothing the boundaries.
    if settings.smooth_radius:
        labels = open_labels(labels, settings.smooth_radius)
    smoothed_labels = labels
    # Step 7: the final filter.
    labels, broken = drop_segments(labels, settings.min_area)
    outline_labels = paint([outline_mask(selection.cycle, forest.geometry, shape)
                            for selection in result.selections.values()], shape)
    groups = outline_groups(labels, region_labels, outline_labels)
    timings = {**result.timings, "segments": time.perf_counter() - clock}
    return replace(result, regions=regions, split_outlines=split_outlines, outline_labels=outline_labels,
                   region_labels=region_labels, labels=labels, grown_labels=grown_labels,
                   smoothed_labels=smoothed_labels, groups=groups, dimmed=dimmed, broken=broken, timings=timings)
