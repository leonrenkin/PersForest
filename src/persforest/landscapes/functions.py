"""landscapes.functions: mechanically extracted reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from numpy.typing import NDArray
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING
from collections.abc import Mapping, Sequence
from types import MappingProxyType
import numpy as np
from persforest.landscapes._validation import sample_grid

if TYPE_CHECKING:
    import matplotlib.axes

def _readonly_vector(values, name: str) -> NDArray[np.float64]:
    """Own a one-dimensional numeric buffer with no writable backing array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.frombuffer(array.tobytes(), dtype=float)


def _freeze_metadata(value: Any) -> Any:
    """Snapshot descriptive metadata without retaining mutable caller containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return value
    raise TypeError("Metadata must contain scalar values, sequences, or mappings")


def _domain(domain: Sequence[float]) -> tuple[float, float]:
    if len(domain) != 2:
        raise ValueError("domain must contain two endpoints")
    start, end = map(float, domain)
    if np.isnan((start, end)).any() or end < start:
        raise ValueError("domain endpoints must be ordered and not NaN")
    return start, end


@dataclass(frozen=True, eq=False)
class StepFunctionData:
    """
    Piecewise-constant function data.

    f(t) = vals[i]  on [starts[i], ends[i])
         = baseline elsewhere.

    All arrays are 1D and aligned by index. `domain` is the global range
    where the function is potentially non-zero (for convenience).

    Intervals must be ordered and nonoverlapping. Gaps use the baseline;
    zero-width records are allowed. Negative/nonfinite values and infinite
    endpoints can be stored here; landscapes impose stricter requirements.
    """
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    vals: NDArray[np.float64]
    baseline: float
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate interval structure and capture owned, read-only arrays."""
        for name in ("starts", "ends", "vals"):
            object.__setattr__(self, name, _readonly_vector(getattr(self, name), name))
        if not (self.starts.size == self.ends.size == self.vals.size):
            raise ValueError("starts, ends, vals must have the same length")
        if (np.isnan(self.starts).any() or np.isnan(self.ends).any()
                or np.any(self.ends < self.starts)
                or np.any(self.starts[1:] < self.ends[:-1])):
            raise ValueError("Profile intervals must be ordered, nonoverlapping, and have end >= start")
        domain = _domain(self.domain)
        if np.any(self.starts < domain[0]) or np.any(self.ends > domain[1]):
            raise ValueError("Profile intervals must lie inside the domain")
        if not np.isscalar(self.baseline):
            raise ValueError("baseline must be a scalar")
        object.__setattr__(self, "baseline", float(self.baseline))
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        *,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        show_baseline: bool = False,
        title: Optional[str] = None,
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        **line_kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Plot the step function represented by this object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, a new figure and axis are created.
        x_range : (float, float), optional
            Restrict the x-axis to the given range; defaults to the stored domain.
        y_range : (float, float), optional
            Restrict the y-axis to the given range; auto-scales when omitted.
        show_baseline : bool, default False
            Whether to draw a dashed line at `baseline`.
        title : str, optional
            Title for the plot.
        baseline_kwargs : dict, optional
            Extra keyword arguments forwarded to the baseline line plot.
        **line_kwargs :
            Keyword arguments forwarded to `ax.plot` for the step curve.

        Returns
        -------
        matplotlib.axes.Axes
            The axis with the rendered step function.
        """
        from persforest.plotting.landscapes import _plot_step_function
        return _plot_step_function(self, ax, x_range=x_range, y_range=y_range, show_baseline=show_baseline, title=title, baseline_kwargs=baseline_kwargs, **line_kwargs)
    
    def eval_on_grid(self, grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the step function on a given grid of points.

        Parameters
        ----------
        grid : numpy.ndarray
            1D array of points at which to evaluate the function.
            must be non-empty and sorted in non-decreasing order.

        Returns
        -------
        numpy.ndarray
            Array of function values at the provided grid points.
        """
        x_grid = sample_grid(grid)
        y_vals = np.full_like(x_grid, fill_value=self.baseline, dtype=float)

        if self.starts.size == 0:
            return np.full_like(x_grid, fill_value=float(self.baseline), dtype=float)

        i = 0
        for j,x in enumerate(x_grid):
            if x < self.starts[0]:
                continue
            if x >= self.ends[-1]:
                break

            while i < self.starts.size and x >= self.ends[i]:
                i += 1
            if i < self.starts.size and self.starts[i] <= x < self.ends[i]:
                y_vals[j] = self.vals[i]

        return y_vals

@dataclass(frozen=True, eq=False)
class PiecewiseLinearFunction:
    """
    Piecewise-linear function specified by breakpoints (xs, ys).
    Between xs[i] and xs[i+1] the function is linear. Outside [xs[0], xs[-1]]
    the function evaluates to 0.0 by default.
    """
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate finite knots and capture owned, read-only arrays."""
        xs = _readonly_vector(self.xs, "xs")
        ys = _readonly_vector(self.ys, "ys")
        if xs.size != ys.size or not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError("xs and ys must be aligned finite arrays")
        if np.any(np.diff(xs) <= 0):
            raise ValueError("xs must be strictly increasing")
        domain = _domain(self.domain)
        if not np.isfinite(domain).all():
            raise ValueError("Piecewise-linear domain endpoints must be finite")
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def __call__(self, x: Union[NDArray[np.float64], float]):
        """
        Evaluate the piecewise-linear function at one or many points.

        Parameters
        ----------
        x : float or array-like of float
            Locations at which to sample the function.

        Returns
        -------
        float or numpy.ndarray
            Scalar if `x` is scalar, otherwise an array of the same shape as `x`.
        """
        if self.xs.size == 0:
            if np.isscalar(x):
                return 0.0
            x_arr = np.asarray(x, dtype=float)
            return np.zeros_like(x_arr)

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.interp(x_arr, self.xs, self.ys, left=0.0, right=0.0)

        if np.isscalar(x):
            # np.interp returns a scalar np.ndarray for scalar input
            return float(y_arr)
        return y_arr
