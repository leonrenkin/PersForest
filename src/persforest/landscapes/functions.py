"""landscapes.functions: mechanically extracted reference implementation."""

from dataclasses import dataclass
from dataclasses import field
from numpy.typing import NDArray
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
import matplotlib
import matplotlib.axes
import numpy as np

@dataclass
class StepFunctionData:
    """
    Piecewise-constant function data.

    f(t) = vals[i]  on [starts[i], ends[i])
         = baseline elsewhere.

    All arrays are 1D and aligned by index. `domain` is the global range
    where the function is potentially non-zero (for convenience).

    We assume that ends[i] = starts[i+1] for all i, i.e., no gaps or overlaps between intervals.
    """
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    vals: NDArray[np.float64]
    baseline: float
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

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
        if grid.size == 0:
            raise ValueError("x_grid must be non-empty")
        if not np.all(np.isfinite(grid)):
            raise ValueError("x_grid contains non-finite values")
        if np.any(np.diff(grid) < 0):
            raise ValueError("x_grid must be sorted in non-decreasing order")

        x_grid = np.asarray(grid, dtype=float).ravel()
        y_vals = np.full_like(grid, fill_value=self.baseline, dtype=float)

        if self.starts.size == 0:
            return np.full_like(x_grid, fill_value=float(self.baseline), dtype=float)
        if not (self.starts.size == self.ends.size == self.vals.size):
            raise ValueError("starts, ends, vals must have the same length")

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

@dataclass
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
