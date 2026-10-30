"""Step 3: the outline of a bar, the first representative after a search start that is convex enough.

Along a bar, the shape measure (the convex-hull perimeter deficit) is a
piecewise constant *profile*: one value per representative, held on the
representative's half-open activity interval. The outline is the first
representative with a defect of at most ``max_defect``; with
``max_defect = inf`` it is simply the first representative after the start.

The search can be bounded by a maximal filtration value: if no representative
qualifies before it, the one active at that value is selected. Otherwise, if
none qualifies, the one with the lowest defect is selected.
"""

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np
from numpy.typing import NDArray

from persforest import PersistenceForest
from persforest.barcode import PFBar
from persforest.chains import SignedChain


@dataclass(frozen=True)
class Profile:
    """The shape measure of a bar's representatives on a search interval.

    ``starts[i]``, ``ends[i]`` delimit the part of the search interval on which
    ``cycles[i]`` is active; ``values[i]`` is its measure (NaN if the measure
    failed).
    """

    bar: PFBar
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    values: NDArray[np.float64]
    cycles: list[SignedChain]


@dataclass(frozen=True)
class Selection:
    """The representative chosen for a bar.

    Attributes
    ----------
    bar : PFBar
    value : float
        Filtration value from which the representative is active (within the
        search interval).
    cycle : SignedChain
    defect : float
    reason : {"rule", "max value", "lowest defect"}
        Why this representative was chosen.
    profile : Profile
        The representatives measured; it ends at the chosen one unless the
        search failed.
    """

    bar: PFBar
    value: float
    cycle: SignedChain
    defect: float
    reason: str
    profile: Profile

    @property
    def fallback(self) -> bool:
        """True if no representative was convex enough."""
        return self.reason != "rule"


def representatives(forest: PersistenceForest, bar: PFBar, start: float):
    """Representatives of ``bar`` active after ``start``, by increasing filtration value.

    With ``storage="cycles"`` these are the stored representatives. With
    ``storage="diffs"`` they are reconstructed from the bar's death downward,
    which stops at ``start``, and returned in increasing order.
    """
    if forest.storage == "cycles":
        return bar.cycle_reps
    window = []
    for cycle in forest.iter_bar_cycle_reps(bar):
        if cycle.active_end <= start:
            break
        window.append(cycle)
    return window[::-1]


def defect_profile(forest: PersistenceForest, bar: PFBar, measure: Callable, start: float,
                   end: float | None = None, max_defect: float | None = None) -> Profile:
    """Measure the representatives of ``bar`` active on ``[start, end)``.

    ``measure`` is ``(chain, geometry) -> float``. With ``max_defect``,
    measuring stops at the first representative with a defect of at most it.
    """
    end = bar.death if end is None else end
    starts, ends, values, cycles = [], [], [], []
    for cycle in representatives(forest, bar, start):
        low, high = max(cycle.active_start, start), min(cycle.active_end, end)
        if low >= high:
            continue
        try:
            value = float(measure(cycle, forest.geometry))
        except ValueError:
            value = math.nan
        starts.append(low)
        ends.append(high)
        values.append(value)
        cycles.append(cycle)
        if max_defect is not None and value <= max_defect:
            break
    return Profile(bar, np.array(starts), np.array(ends), np.array(values), list(cycles))


def select_cycle(forest: PersistenceForest, bar: PFBar, measure: Callable, max_defect: float,
                 start: float, max_value: float | None = None) -> Selection:
    """Select the first representative of ``bar`` after ``start`` with a defect of at most ``max_defect``.

    With ``max_value``, the search ends there, and if no representative
    qualifies, the one active at ``max_value`` is chosen (if the bar is still
    alive).
    """
    end = bar.death
    if max_value is not None and max_value < end:
        if start >= max_value:
            raise ValueError(f"Search start {start} lies above max_value {max_value}")
        end = max_value
    else:
        max_value = None
    profile = defect_profile(forest, bar, measure, start, end, max_defect)
    if len(profile.values) == 0:
        raise ValueError(f"No representative of bar [{bar.birth}, {bar.death}) is active after {start}")
    if profile.values[-1] <= max_defect:
        return Selection(bar, float(profile.starts[-1]), profile.cycles[-1], float(profile.values[-1]), "rule",
                         profile)
    if max_value is not None:
        cycle = forest.cycle_for_bar_at(bar, max_value)
        return Selection(bar, max_value, cycle, float(measure(cycle, forest.geometry)), "max value", profile)
    index = int(np.nanargmin(profile.values))
    return Selection(bar, float(profile.starts[index]), profile.cycles[index], float(profile.values[index]),
                     "lowest defect", profile)
