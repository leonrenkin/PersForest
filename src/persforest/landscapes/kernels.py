"""landscapes.kernels: mechanically extracted reference implementation."""

from bisect import bisect_right
from persforest.landscapes._validation import validate_landscape_kernel_input
from persforest.landscapes._validation import validate_tolerance
from persforest.landscapes.functions import PiecewiseLinearFunction
from persforest.landscapes.functions import StepFunctionData
from persforest.landscapes.profiles import MeasurementProfiles
from typing import Callable
from typing import List
from typing import Literal
from typing import Tuple
import math
import numpy as np

def _build_convolution_with_indicator(
            starts: List[float],
            ends: List[float],
            vals: List[float],
            a: float,
            b: float,
            *,
            tol: float = 1e-12
    ) -> Tuple[Callable[[float], float], List[float], List[float]]:
        """
        Compute ``h(x) = (f * 1_[a,b])(x)``.

        ``f`` is piecewise constant with values ``vals[i]`` on
        ``[starts[i], ends[i]]`` and zero elsewhere. ``1_[a,b]`` is the
        indicator of ``[a, b]``.

        Parameters
        ----------
        starts, ends, vals : list of float
            Interval boundaries and values describing the step function f.
        a, b : float
            Bounds of the indicator kernel; must satisfy a <= b.
        tol : float, default 1e-12
            Quantisation tolerance for merging nearly coincident event points.
            Intervals narrower than tol are ignored. Measurement amplitudes
            are never thresholded by this coordinate tolerance.

        Returns
        -------
        tuple
            ``(h, xs, ys)`` where ``h`` evaluates the piecewise-linear
            convolution, ``xs`` are knot locations and ``ys`` are values at
            those knots.
        """
        tol = validate_tolerance(tol)
        if not np.isfinite((a, b)).all():
            raise ValueError("Convolution interval endpoints must be finite")
        if not all(np.isfinite(values).all() for values in (starts, ends, vals)):
            raise ValueError("Convolution profile endpoints and values must be finite")
        if b < a:
            raise ValueError("Require a <= b for the indicator [a,b].")
        if not (len(starts) == len(ends) == len(vals)):
            raise ValueError("starts, ends, vals must have equal length.")
        n = len(starts)
        if n == 0:
            # No support: convolution is identically zero
            def h_zero(_: float) -> float: return 0.0
            return h_zero, [], []

        # Helper to merge nearly identical event keys to improve numerical stability
        def _quantize(x: float) -> float:
            if tol <= 0:
                return x
            # snap to nearest multiple of tol
            return round(x / tol) * tol

        # Build slope-change "events"
        events = {}  # x -> delta_slope
        def add_event(x: float, delta: float):
            qx = _quantize(x)
            events[qx] = events.get(qx, 0.0) + float(delta)

        for s, e, v in zip(starts, ends, vals):
            if e < s:
                raise ValueError(f"Encountered interval with end < start: [{s}, {e}].")
            if v == 0 or abs(e - s) < tol:
                # tol has coordinate units; it must not discard small values.
                continue
            add_event(a + s, +v)
            add_event(a + e, -v)
            add_event(b + s, -v)
            add_event(b + e, +v)

        if not events:
            def h_zero(_: float) -> float: return 0.0
            return h_zero, [], []

        xs = sorted(events.keys())
        # We know h(x)=0 for x < xs[0] (the window [x-b, x-a] is fully left of f’s support).
        ys: List[float] = [0.0]

        # Sweep: on (xs[i], xs[i+1]) the slope is constant; update slope at the left endpoint.
        slope = 0.0
        slopes_per_interval: List[float] = []
        for i in range(len(xs) - 1):
            x_i, x_next = xs[i], xs[i + 1]
            slope += events[x_i]              # slope just to the right of x_i
            slopes_per_interval.append(slope) # slope on (x_i, x_{i+1})
            y_next = ys[-1] + slope * (x_next - x_i)
            ys.append(y_next)

        # Consume the last event to bring slope back (should return to ~0)
        slope += events[xs[-1]]
        # Optional check (tolerant to rounding)
        if not math.isclose(slope, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            # Not fatal; tiny residue can appear from floating noise
            pass

        # Build a fast evaluator by linear interpolation on the piecewise-linear segments
        def h(x: float) -> float:
            if not xs:
                return 0.0
            if x <= xs[0] or x >= xs[-1]:
                return 0.0
            i = bisect_right(xs, x) - 1
            # segment i is (xs[i], xs[i+1]) with slope slopes_per_interval[i]
            return ys[i] + slopes_per_interval[i] * (x - xs[i])

        return h, xs, ys

def convolve_profile(
    profile: StepFunctionData, birth: float, death: float, *, tol: float = 1e-12,
) -> PiecewiseLinearFunction:
    """Convolve a zero-baseline profile with the indicator of [birth, death].

    Parameters
    ----------
    profile : StepFunctionData
        Ordered interval data. Negative values are supported by raw convolution.
    birth, death : float
        Finite, ordered indicator endpoints.
    tol : float, default 1e-12
        Finite nonnegative coordinate tolerance for event quantization and
        ignoring intervals narrower than tol. Does not threshold values.

    Returns
    -------
    PiecewiseLinearFunction
        Raw convolution g(x), zero outside its captured knot support.

    Notes
    -----
    Nonzero baselines are rejected rather than silently discarded. Infinite
    endpoints remain supported by standalone profiles, not finite convolution.
    """
    if not isinstance(profile, StepFunctionData):
        raise TypeError("profile must be StepFunctionData")
    if profile.baseline != 0.0:
        raise ValueError("Convolution requires baseline=0; nonzero baselines are not convolved")
    _, xs, ys = _build_convolution_with_indicator(
        profile.starts.tolist(), profile.ends.tolist(), profile.vals.tolist(),
        birth, death, tol=tol,
    )
    domain = (float(xs[0]), float(xs[-1])) if len(xs) > 1 else (float(birth), float(death))
    return PiecewiseLinearFunction(
        np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), domain,
        metadata={**profile.metadata, "kernel_type": "raw_convolution", "tol": float(tol)},
    )


def landscape_kernel(
    profiles: MeasurementProfiles, bar_id: object, *,
    mode: Literal["raw", "pyramid"] = "pyramid", tol: float = 1e-12,
) -> PiecewiseLinearFunction:
    """Build a kernel from an explicitly selected captured measurement profile.

    Parameters
    ----------
    profiles : MeasurementProfiles
        Self-contained measurement result; no live forest is consulted.
    bar_id : int or bar
        Result-local ID, captured bar, or original source bar.
    mode : {'raw', 'pyramid'}, default 'pyramid'
        Raw convolution g(x), or lambda(x) = 0.5 * g(2x).
    tol : float, default 1e-12
        Coordinate tolerance for event quantization and ignoring intervals
        narrower than tol, distinct from the source forest's filtration
        reduction tolerance. Does not threshold measurement amplitudes.

    Returns
    -------
    PiecewiseLinearFunction
        Exact kernel between its numerically constructed knots. Landscape
        inputs must have finite endpoints, nonnegative finite measurements,
        and zero baseline. Invalid modes and missing profiles raise errors.
    """
    if mode not in ("raw", "pyramid"):
        raise ValueError("mode must be 'raw' or 'pyramid'")
    tol = validate_tolerance(tol)
    if not isinstance(profiles, MeasurementProfiles):
        raise TypeError("profiles must be MeasurementProfiles")
    index = profiles._index(bar_id)
    bar = profiles.bars[index]
    profile = profiles[index]
    validate_landscape_kernel_input(bar, profile)
    raw = convolve_profile(profile, bar.birth, bar.death, tol=tol)
    if mode == "raw":
        return raw
    xs, ys = raw.xs / 2.0, raw.ys / 2.0
    domain = (float(xs[0]), float(xs[-1])) if xs.size > 1 else raw.domain
    return PiecewiseLinearFunction(
        xs, ys, domain,
        metadata={**raw.metadata, "kernel_type": "pyramid_rescaled",
                  "rescaling_formula": "lambda(x) = 0.5 * g(2x)"},
    )
