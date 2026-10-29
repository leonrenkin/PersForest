"""Stage 2: the first reliable representative of a bar.

Along a bar, the shape measure (e.g. a convex-hull deficit) is a piecewise
constant *profile*: one value per representative, held on the
representative's half-open activity interval. A selection rule picks the
first representative after a search start that is reliable:

``Threshold(max_defect)``
    the first value with ``defect <= max_defect``.
``Stable(bandwidth, tol, max_defect=None)``
    the first representative on a plateau of the profile: the *local
    deviation*, the mean of ``|defect(s) - defect(t)|`` over the neighborhood
    of the representative, weighted by a Gaussian of standard deviation
    ``bandwidth`` (filtration units) and the time each representative is
    active, is at most ``tol``. The neighborhood is cut off at four
    bandwidths and at the ends of the search interval. Comparing with the
    value at ``t``, not with the local average, matters: a falling noisy
    profile crosses its own average many times before it settles. With
    ``max_defect``, the defect must also not exceed it.

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

_CUTOFF = 4.0  # neighborhoods of the stable rule end at this many bandwidths


@dataclass(frozen=True)
class Threshold:
    max_defect: float


@dataclass(frozen=True)
class Stable:
    bandwidth: float
    tol: float
    max_defect: float | None = None


@dataclass(frozen=True)
class Profile:
    """The shape measure of a bar's representatives on a search interval.

    ``starts[i]``, ``ends[i]`` delimit the part of the search interval on which
    ``cycles[i]`` is active; ``values[i]`` is its measure (NaN if the measure
    failed, e.g. when smoothing collapses a tiny loop).
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
        The profile the choice was made on; it ends shortly after the chosen
        representative unless the search failed.
    """

    bar: PFBar
    value: float
    cycle: SignedChain
    defect: float
    reason: str
    profile: Profile

    @property
    def fallback(self) -> bool:
        """True if no representative satisfied the rule."""
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
                   end: float | None = None, until: Callable[[Profile], bool] | None = None) -> Profile:
    """Measure the representatives of ``bar`` active on ``[start, end)``.

    Parameters
    ----------
    forest : PersistenceForest
    bar : PFBar
    measure : callable
        ``(chain, geometry) -> float``, e.g. a ``LoopMeasurement``.
    start, end : float
        Search interval; ``end`` defaults to the death of the bar.
    until : callable, optional
        Called with the profile measured so far after every representative;
        measuring stops when it returns True.
    """
    end = bar.death if end is None else end
    starts, ends, values, cycles = [], [], [], []

    def profile():
        return Profile(bar, np.array(starts), np.array(ends), np.array(values), list(cycles))

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
        if until is not None and until(profile()):
            break
    return profile()


def deviation_at(profile: Profile, index: int, bandwidth: float) -> float:
    """Local deviation of the profile around representative ``index``."""
    center = (profile.starts[index] + profile.ends[index]) / 2
    low, high = center - _CUTOFF * bandwidth, center + _CUTOFF * bandwidth
    near = (profile.ends > low) & (profile.starts < high) & ~np.isnan(profile.values)
    lower = (np.maximum(profile.starts[near], low) - center) / (bandwidth * math.sqrt(2))
    upper = (np.minimum(profile.ends[near], high) - center) / (bandwidth * math.sqrt(2))
    weights = np.array([math.erf(b) - math.erf(a) for a, b in zip(lower, upper)])
    return float(weights @ np.abs(profile.values[near] - profile.values[index]) / weights.sum())


def local_deviation(profile: Profile, bandwidth: float) -> NDArray[np.float64]:
    """Local deviation around every representative of the profile."""
    return np.array([deviation_at(profile, index, bandwidth) for index in range(len(profile.values))])


def passes(profile: Profile, index: int, rule: Threshold | Stable) -> bool:
    """Whether representative ``index`` satisfies ``rule``."""
    value = profile.values[index]
    if math.isnan(value):
        return False
    if isinstance(rule, Threshold):
        return value <= rule.max_defect
    if rule.max_defect is not None and value > rule.max_defect:
        return False
    return deviation_at(profile, index, rule.bandwidth) <= rule.tol


def select_index(profile: Profile, rule: Threshold | Stable) -> int | None:
    """Index of the first representative satisfying ``rule``, or None."""
    return next((index for index in range(len(profile.values)) if passes(profile, index, rule)), None)


def select_cycle(forest: PersistenceForest, bar: PFBar, measure: Callable, rule: Threshold | Stable,
                 start: float, end: float | None = None, max_value: float | None = None) -> Selection:
    """Select the first reliable representative of ``bar`` on ``[start, end)``.

    With ``max_value``, the search ends there, and if no representative
    qualifies, the one active at ``max_value`` is chosen (if the bar is still
    alive). Representatives are measured only until the choice is certain:
    for ``Threshold`` up to the first passing one, for ``Stable`` until the
    neighborhood of a stable one has been measured.
    """
    end = bar.death if end is None else end
    if max_value is not None and max_value < end:
        if start >= max_value:
            raise ValueError(f"Search start {start} lies above max_value {max_value}")
        end = max_value
    else:
        max_value = None
    reach = 0.0 if isinstance(rule, Threshold) else _CUTOFF * rule.bandwidth
    checked = 0

    def decided(profile):
        # A representative is decided once every representative within reach
        # of its center has been measured.
        nonlocal checked
        while checked < len(profile.values):
            center = (profile.starts[checked] + profile.ends[checked]) / 2
            if reach and center + reach > profile.starts[-1]:
                return False
            if passes(profile, checked, rule):
                return True
            checked += 1
        return False

    profile = defect_profile(forest, bar, measure, start, end, until=decided)
    if len(profile.values) == 0:
        raise ValueError(f"No representative of bar [{bar.birth}, {bar.death}) is active after {start}")
    index = select_index(profile, rule)
    if index is not None:
        return Selection(bar, float(profile.starts[index]), profile.cycles[index],
                         float(profile.values[index]), "rule", profile)
    if max_value is not None:
        cycle = forest.cycle_for_bar_at(bar, max_value)
        return Selection(bar, max_value, cycle, float(measure(cycle, forest.geometry)), "max value", profile)
    index = int(np.nanargmin(profile.values))
    return Selection(bar, float(profile.starts[index]), profile.cycles[index], float(profile.values[index]),
                     "lowest defect", profile)
