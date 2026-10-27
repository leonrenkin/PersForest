"""landscapes.profiles: mechanically extracted reference implementation."""

from dataclasses import dataclass
from dataclasses import field
from numpy.typing import NDArray
from persforest.landscapes.functions import StepFunctionData
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import numpy as np

CycleValueFunc = Callable[[Any, np.ndarray], float]

@dataclass
class BarcodeFunctionals:
    """
    Container for per-bar step functions (StepFunctionData) for a fixed cycle_func.
    Aligned with the bar ordering used in compute_measurement_landscape_family.
    """
    forest_id: str
    label: str
    baseline: float
    bars: List[Any]  # PFBar or bar-like objects
    step_functions: Dict[int, StepFunctionData]  # index -> StepFunctionData
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    # built once for convenience
    _bar_to_index: Dict[Any, int] = field(init=False, repr=False)

    def __post_init__(self):
        """Build the lookup from bar object to stored row index."""
        self._bar_to_index = {bar: i for i, bar in enumerate(self.bars)}

    def get(self, bar_or_index: Union[int, Any]) -> StepFunctionData:
        """
        Return the step function for a bar object or stored integer index.
        """
        if isinstance(bar_or_index, int):
            return self.step_functions[bar_or_index]
        return self.step_functions[self._bar_to_index[bar_or_index]]

    def __getitem__(self, bar_or_index: Union[int, Any]) -> StepFunctionData:
        """Alias for ``get`` to support bracket access."""
        return self.get(bar_or_index)
    
    def evaluate_on_grid(
        self,
        grid: NDArray[np.float64],
        bars: Optional[Sequence[Union[int, Any]]] = None,
    ) -> NDArray[np.float64]:
        """Evaluate stored per-bar step functions on a common grid.

        Parameters
        ----------
        grid:
            One-dimensional, monotonically non-decreasing sample grid.
        bars:
            Optional subset of bars (bar objects) or indices to evaluate.
            If omitted, evaluates for all bars in `self.bars` order.

        Returns
        -------
        numpy.ndarray
            Array of shape (B, G) where G=len(grid) and B is the number of
            requested bars.
        """
        grid_arr = np.asarray(grid, dtype=float).ravel()
        if grid_arr.size == 0:
            raise ValueError("grid must be non-empty")
        if not np.all(np.isfinite(grid_arr)):
            raise ValueError("grid contains non-finite values")
        if np.any(np.diff(grid_arr) < 0):
            raise ValueError("grid must be sorted in non-decreasing order")

        if bars is None:
            indices = list(range(len(self.bars)))
        else:
            indices = []
            for b in bars:
                if isinstance(b, int):
                    indices.append(b)
                else:
                    indices.append(self._bar_to_index[b])

        out = np.empty((len(indices), grid_arr.size), dtype=float)
        for row, idx in enumerate(indices):
            if idx not in self.step_functions:
                raise KeyError(f"No StepFunctionData stored for bar index {idx}")
            out[row, :] = self.step_functions[idx].eval_on_grid(grid_arr)

        return out

def _build_step_function_data(
        forest,
        bar,
        cycle_func: CycleValueFunc,
        baseline: float = 0.0,
    ) -> StepFunctionData:
        """
        Build the piecewise-constant measurement function for one bar.

        Each cycle representative contributes an interval
        `[cycle.active_start, cycle.active_end]` on which the function takes the
        value `cycle_func(cycle, forest.point_cloud)`. Outside these intervals,
        the function equals `baseline`.

        Parameters
        ----------
        forest :
            Forest-like object providing ``point_cloud`` and ``barcode``.
            Requires ``iter_bar_cycle_reps(bar)`` in descending filtration order.
        bar :
            Bar instance taken from ``forest.barcode``.
        cycle_func : CycleValueFunc
            Callable that assigns a scalar to each cycle representative.
        baseline : float, default 0.0
            Value of the function outside the active intervals.

        Returns
        -------
        StepFunctionData
            Structured representation of the piecewise-constant function.
        """
        if bar not in forest.barcode:
            raise ValueError("Bar is not in barcode of forest")

        records = [
            (cycle.active_start, cycle.active_end,
             float(cycle_func(cycle, forest.point_cloud)))
            for cycle in forest.iter_bar_cycle_reps(bar)
        ]
        # The iterator runs from death toward birth. Reverse only scalar
        # records so reconstructed chains need not be retained.
        records.reverse()
        data = np.asarray(records, dtype=float).reshape(-1, 3)
        starts, ends, vals = data.T

        if starts.size == 0:
            # Degenerate: no representatives
            domain = (float(bar.birth), float(bar.death))
        else:
            domain = (float(starts.min()), float(ends.max()))

        return StepFunctionData(
            starts=starts,
            ends=ends,
            vals=vals,
            baseline=baseline,
            domain=domain,
            metadata={
                "bar_birth": bar.birth,
                "bar_death": bar.death,
                "root_id": getattr(bar, "root_id", None),
            },
        )

def compute_barcode_functionals(
    forest,
    cycle_func: CycleValueFunc,
    label: str,
    *,
    min_bar_length: float = 0.0,
    baseline: float = 0.0,
    cache: bool = True,
) -> BarcodeFunctionals:
    """
    Compute per-bar step functions for one cycle functional.

    Parameters
    ----------
    forest :
        Forest-like object with ``barcode``, ``point_cloud`` and
        ``barcode_functionals`` when caching is enabled.
    cycle_func : CycleValueFunc
        Callable ``(cycle_rep, point_cloud) -> float``.
    label : str
        Cache key and label for the resulting container.
    min_bar_length : float
        Ignore bars with lifespan below this threshold.
    baseline : float
        Value outside each bar's representative intervals.
    cache : bool
        If True, store the result in ``forest.barcode_functionals[label]``.

    Returns
    -------
    BarcodeFunctionals
        Step-function data for the selected bars.
    """

    bars = [bar for bar in forest.barcode if (bar.death - bar.birth) >= min_bar_length]
    step_functions: Dict[int, StepFunctionData] = {}

    for i, bar in enumerate(bars):
        sf = _build_step_function_data(forest=forest, bar=bar, cycle_func=cycle_func, baseline=baseline)
        step_functions[i] = sf

    bf = BarcodeFunctionals(
        forest_id=getattr(forest, "id", str(id(forest))),
        label=label,
        baseline=baseline,
        bars=bars,
        step_functions=step_functions,
        extra_meta={"min_bar_length": min_bar_length},
    )

    if cache:
        if not hasattr(forest, "barcode_functionals"):
            forest.barcode_functionals = {}
        key = label
        forest.barcode_functionals[key] = bf

    return bf
