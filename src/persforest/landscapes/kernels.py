"""landscapes.kernels: mechanically extracted reference implementation."""

from bisect import bisect_right
from persforest.landscapes._validation import validate_landscape_kernel_input
from persforest.landscapes.functions import PiecewiseLinearFunction
from persforest.landscapes.functions import StepFunctionData
from persforest.landscapes.profiles import CycleValueFunc
from persforest.landscapes.profiles import _build_step_function_data
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
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

        Returns
        -------
        tuple
            ``(h, xs, ys)`` where ``h`` evaluates the piecewise-linear
            convolution, ``xs`` are knot locations and ``ys`` are values at
            those knots.
        """
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
            if abs(v) < tol or abs(e - s) < tol:
                # Zero value or degenerate interval contributes nothing
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

def compute_convolution_kernel_for_bar(
        forest,
        bar,
        cycle_func: CycleValueFunc,
        *,
        tol: float = 1e-12,
        sf: Optional[StepFunctionData] = None,
    ) -> PiecewiseLinearFunction:
        """
        Compute the raw convolution kernel for one bar.

        The kernel is ``g(x) = (f * 1_[birth, death])(x)``, where ``f`` is the
        step function induced by ``cycle_func`` on the bar's cycle
        representatives.

        Parameters
        ----------
        forest :
            Forest-like object with `barcode` and `point_cloud`.
        bar :
            Bar instance from `forest.barcode`.
        cycle_func : CycleValueFunc
            Callable that assigns a scalar value to each cycle representative.
        tol : float, default 1e-12
            Numerical tolerance used when merging breakpoint events.
        sf : StepFunctionData, optional
            If given, reuse this step function instead of rebuilding it.

        Returns
        -------
        PiecewiseLinearFunction
            Piecewise-linear convolution kernel defined on the bar interval.
        """
        if sf is None:
            sf = _build_step_function_data(forest=forest,
                                           bar=bar, 
                                           cycle_func=cycle_func, 
                                           baseline=0.0)

        a, b = bar.birth, bar.death
        h, xs, ys = _build_convolution_with_indicator(
            sf.starts.tolist(),
            sf.ends.tolist(),
            sf.vals.tolist(),
            a,
            b,
            tol=tol,
        )

        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)

        if xs_arr.size > 1:
            domain = (float(xs_arr[0]), float(xs_arr[-1]))
        else:
            domain = (float(a), float(b))

        return PiecewiseLinearFunction(
            xs=xs_arr,
            ys=ys_arr,
            domain=domain,
            metadata={
                **sf.metadata,
                "kernel_type": "raw_convolution",
            },
        )

def compute_measurement_interval_landscape(
        forest,
        cycle_func: CycleValueFunc,
        bar,
        sf: Optional[StepFunctionData] = None,
    ):
        """
        Helper that computes the (raw) convolution kernel g
        for a single bar and returns (h, xs, ys) as before, but implemented via
        the new PiecewiseLinearFunction.

        Note: this uses the *raw* convolution (no pyramid rescaling).
        For landscapes, prefer compute_measurement_landscape_family.

        Parameters
        ----------
        forest :
            Forest-like object with `barcode`.
        cycle_func : CycleValueFunc
            Scalar-valued functional on cycle representatives.
        bar :
            Bar from the forest barcode; if None, `forest.max_bar()` is used.
        sf : StepFunctionData, optional
            If given, reuse this step function instead of rebuilding it.

        Returns
        -------
        tuple
            `(h, xs, ys)` where `h` is a callable evaluator and `xs`, `ys`
            describe the breakpoints of the piecewise-linear kernel.
        """
        if bar is None:
            bar = forest.max_bar()

        if bar not in forest.barcode:
            raise ValueError("Bar is not in barcode of forest")

        kernel = compute_convolution_kernel_for_bar(
            forest=forest,
            bar=bar,
            cycle_func=cycle_func,
            sf=sf,
        )

        xs = kernel.xs.tolist()
        ys = kernel.ys.tolist()

        def h(x: float) -> float:
            return float(kernel(x))

        return h, xs, ys

def compute_landscape_kernel_for_bar(
        forest,
        bar,
        cycle_func: CycleValueFunc,
        *,
        mode: Literal["raw", "pyramid"] = "pyramid",
        tol: float = 1e-12,
        sf: Optional[StepFunctionData] = None,
    ) -> PiecewiseLinearFunction:
        """
        Build the kernel used for landscapes for a single bar.

        cycle_func values must be finite and nonnegative.

        - mode="raw":      return g(x) = (f * 1_[birth,death])(x)
        - mode="pyramid":  return λ(x) = 1/2 * g(2x)

        The "pyramid" mode should reproduce the usual persistence landscape
        shape when f ≡ 1.

        Parameters
        ----------
        forest :
            Forest-like object with `barcode`.
        bar :
            Bar from the forest barcode.
        cycle_func : CycleValueFunc
            Functional used to evaluate each cycle representative.
        mode : {"raw", "pyramid"}, default "pyramid"
            Whether to return the raw convolution or the pyramid-rescaled kernel.
        tol : float, default 1e-12
            Tolerance forwarded to the convolution helper.
        sf : StepFunctionData, optional
            If given, reuse this step function instead of rebuilding it.

        Returns
        -------
        PiecewiseLinearFunction
            Kernel ready to be sampled on a grid.
        """
        if sf is None:
            sf = _build_step_function_data(
                forest=forest, bar=bar, cycle_func=cycle_func, baseline=0.0
            )
        validate_landscape_kernel_input(bar, sf)

        raw_kernel = compute_convolution_kernel_for_bar(
            forest = forest,
            bar=bar,
            cycle_func=cycle_func,
            tol=tol,
            sf=sf,
        )

        if mode == "raw":
            # Just return a shallow copy with explicit metadata
            return PiecewiseLinearFunction(
                xs=raw_kernel.xs.copy(),
                ys=raw_kernel.ys.copy(),
                domain=raw_kernel.domain,
                metadata={
                    **raw_kernel.metadata,
                    "kernel_type": "raw_convolution",
                },
            )

        # mode == "pyramid"
        xs = raw_kernel.xs
        ys = raw_kernel.ys

        # λ(x) = 1/2 * g(2x)
        xs_scaled = xs / 2.0
        ys_scaled = ys / 2.0

        domain = (float(xs_scaled[0]), float(xs_scaled[-1])) if xs_scaled.size > 1 else raw_kernel.domain

        return PiecewiseLinearFunction(
            xs=xs_scaled,
            ys=ys_scaled,
            domain=domain,
            metadata={
                **raw_kernel.metadata,
                "kernel_type": "pyramid_rescaled",
                "rescaling_formula": "lambda(x) = 0.5 * g(2x)",
            },
        )
