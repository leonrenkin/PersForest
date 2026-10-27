"""landscapes.family: mechanically extracted reference implementation."""

from dataclasses import dataclass
from dataclasses import field
from numpy.typing import NDArray
from persforest.landscapes._validation import validate_landscape_request
from persforest.landscapes.functions import PiecewiseLinearFunction
from persforest.landscapes.kernels import compute_landscape_kernel_for_bar
from persforest.landscapes.profiles import CycleValueFunc
from persforest.landscapes.profiles import compute_barcode_functionals
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union
import numpy as np

@dataclass
class MeasurementLandscapeFamily:
    """
    Container for a family of measurement landscapes for one forest and cycle
    functional.

    - bar_kernels: per-bar kernels (typically already rescaled if mode="pyramid")
    - landscapes: k -> sampled λ_k (k-th landscape) as PiecewiseLinearFunction

    Landscape values are grid approximations.
    """
    forest_id: str 
    label: str
    rescaling: str  # "raw" or "pyramid", where "pyramid" is default
    x_grid: NDArray[np.float64]
    bar_kernels: Dict[int, PiecewiseLinearFunction]
    landscapes: Dict[int, PiecewiseLinearFunction]
    extra_meta: Dict[str, object] = field(default_factory=dict)
    
    def evaluate_on_grid(
        self,
        grid: NDArray[np.float64] | None = None,
        levels: Union[int, Sequence[int]] | None = None,
    ) -> NDArray[np.float64]:
        """Retrieve selected measurement landscape levels on a stored-grid subset.

        Parameters
        ----------
        grid:
            One-dimensional, monotonically non-decreasing sample grid. Every
            coordinate must exactly match a coordinate in ``self.x_grid``;
            if None, uses the full stored grid.
        levels:
            If an int m is given, evaluates levels k=1..m. If a sequence is
            given, evaluates those k-values in the provided order. If None,
            evaluates levels 1 through the highest computed level (``max_k``).

        Returns
        -------
        numpy.ndarray
            Array of shape (L, G) where G=len(grid) and L is the number of
            requested levels.
        """
        if grid is None:
            grid = self.x_grid

        grid_arr = np.asarray(grid, dtype=float).ravel()
        if grid_arr.size == 0:
            raise ValueError("grid must be non-empty")
        if not np.all(np.isfinite(grid_arr)):
            raise ValueError("grid contains non-finite values")
        if np.any(np.diff(grid_arr) < 0):
            raise ValueError("grid must be sorted in non-decreasing order")

        # Match grid coordinates exactly: snapping nearby points could conceal
        # an unsampled peak or crossing on a finely spaced construction grid.
        indices = np.searchsorted(self.x_grid, grid_arr)
        if (np.any(indices >= self.x_grid.size)
                or not np.array_equal(self.x_grid[indices], grid_arr)):
            raise ValueError(
                "Evaluation grid must be a subset of the stored landscape x_grid "
                "(exact coordinate matches required). Recompute landscapes with "
                "x_grid set to the requested grid or a larger grid containing all "
                "requested points."
            )

        if levels is None:
            ks = list(range(1, max(self.landscapes, default=0) + 1))
        elif isinstance(levels, int):
            if levels < 1:
                raise ValueError("levels must be >= 1")
            ks = list(range(1, levels + 1))
        else:
            ks = [int(k) for k in levels]
            if any(k < 1 for k in ks):
                raise ValueError("all requested landscape levels must be >= 1")

        missing = sorted({k for k in ks if k not in self.landscapes})
        if missing:
            raise ValueError(
                f"Landscape levels {missing} were not computed. "
                f"Recompute landscapes with max_k >= {max(missing)} "
                "or request only computed levels."
            )

        out = np.empty((len(ks), grid_arr.size), dtype=float)
        for i, k in enumerate(ks):
            f = self.landscapes[k]
            out[i, :] = f.ys[indices]

        return out

def compute_measurement_landscape_family(
        forest,
        cycle_func: CycleValueFunc,
        label: str,
        *,
        max_k: int = 5,
        num_grid_points: int = 512,
        mode: Literal["raw", "pyramid"] = "pyramid",
        min_bar_length: float = 0.0,
        x_grid: Optional[NDArray[np.float64]] = None,
        cache: bool = True,
        cache_functionals: bool = False,
        functionals_label: Optional[str] = None,
        compute_functionals: bool = True,
    ) -> MeasurementLandscapeFamily:
        """
        Compute a measurement landscape family for a forest.

        If x_grid is provided, all landscapes are evaluated on that grid
        (and num_grid_points is ignored). This is crucial for consistent
        vectorisations across multiple forest objects.

        Parameters
        ----------
        forest:
            Forest-like object with ``barcode``, ``point_cloud``,
            ``landscape_families`` and ``barcode_functionals``.
        cycle_func:
            Function ``f(cycle_rep, point_cloud) -> scalar``. 
            cycle_func values must be finite and nonnegative.
        label:
            Key used to store the family in ``forest.landscape_families``.
        max_k:
            Number of landscapes λ_1, ..., λ_max_k to compute.
        num_grid_points:
            Number of grid points used when x_grid is None.
        mode:
            "raw" or "pyramid" (see compute_landscape_kernel_for_bar).
        min_bar_length:
            Ignore bars with (death - birth) < min_bar_length.
        x_grid:
            Optional fixed grid on which to evaluate the landscapes.
        cache:
            If True, save the result to ``forest.landscape_families[label]``.
        cache_functionals:
            If True, cache computed barcode functionals on the forest.
        functionals_label:
            Label used to cache/retrieve barcode functionals.
            If None, uses `label`.
        compute_functionals:
            If True, compute barcode functionals. If False, reuse cached
            functionals with ``functionals_label``.

        Returns
        -------
        MeasurementLandscapeFamily
            Collection of per-bar kernels and λ_k landscapes on the shared grid.
        """
        if not hasattr(forest, "barcode"):
            raise AttributeError("Forest object has no 'barcode' attribute. Did you compute it?")
        x_grid = validate_landscape_request(max_k, num_grid_points, mode, x_grid)

        # 1. Filter bars by length
        bars = [
            bar for bar in forest.barcode
            if (bar.death - bar.birth) >= min_bar_length
        ]

        if functionals_label is None:
            functionals_label = label
        if cache_functionals is None:
            cache_functionals = cache


        if not bars:
            # --- Empty barcode (after min_bar_length filtering): return 0-landscapes instead of error ---
            # Common grid (still needed for a well-formed family / vectorization)
            if x_grid is None:
                # No bars => no natural domain; choose a stable default
                x_grid = np.linspace(0.0, 1.0, num_grid_points)
            # Build zero landscapes on the chosen grid
            xmin, xmax = float(x_grid[0]), float(x_grid[-1])

            def _zero_plf(k: int) -> PiecewiseLinearFunction:
                return PiecewiseLinearFunction(
                    xs=x_grid,
                    ys=np.zeros_like(x_grid),
                    domain=(xmin, xmax),
                    metadata={"k": k, "mode": mode, "empty_barcode": True},
                )

            landscapes = {k: _zero_plf(k) for k in range(1, max_k + 1)}

            fam = MeasurementLandscapeFamily(
                forest_id=getattr(forest, "id", str(id(forest))),
                label=label,
                rescaling=mode,
                x_grid=x_grid,
                bar_kernels={},  # no bars => no kernels
                landscapes=landscapes,
                extra_meta={
                    "min_bar_length": min_bar_length,
                    "n_bars": 0,
                    "empty_barcode": True,
                    "functionals_label": functionals_label,
                },
            )

            if cache:
                if not hasattr(forest, "landscape_families"):
                    forest.landscape_families = {}
                forest.landscape_families[label] = fam

            return fam

        if not compute_functionals:
            bf =forest.barcode_functionals.get(functionals_label, None)
            if bf is None:
                raise ValueError(
                    f"Requested to reuse cached barcode functionals "
                    f"with label '{functionals_label}', but none found."
                )
        else:
            bf = compute_barcode_functionals(
                forest=forest,
                cycle_func=cycle_func,
                label=functionals_label,
                min_bar_length=min_bar_length,
                cache=False,
            )

        bar_kernels: Dict[int, PiecewiseLinearFunction] = {}
        global_min_x = float("inf")
        global_max_x = float("-inf")

        for i, bar in enumerate(bars):
            kernel = compute_landscape_kernel_for_bar(
                forest,
                bar,
                cycle_func,
                mode=mode,
                sf=bf[bar] #access precomputed step function for this bar
            )
            bar_kernels[i] = kernel

            global_min_x = min(global_min_x, kernel.domain[0])
            global_max_x = max(global_max_x, kernel.domain[1])

        if not np.isfinite(global_min_x) or not np.isfinite(global_max_x):
            raise RuntimeError("Failed to infer a finite global domain for the kernels.")

        if global_max_x <= global_min_x:
            raise RuntimeError(
                f"Non-positive domain width: [{global_min_x}, {global_max_x}]"
            )

        # 3. Common grid
        if x_grid is None:
            x_grid = np.linspace(global_min_x, global_max_x, num_grid_points)
        num_grid_points = x_grid.size  # ensure consistency

        # 4. Evaluate all kernels on the grid
        n_bars = len(bars)
        values = np.zeros((n_bars, num_grid_points), dtype=float)
        for i, kernel in bar_kernels.items():
            values[i, :] = kernel(x_grid)

        # 5. Compute order statistics along the bar axis
        # sorted_vals[k, j] = (k+1)-th largest value at x_grid[j]
        sorted_vals = np.sort(values, axis=0)[::-1, :]  # descending along axis 0
        max_possible_k = sorted_vals.shape[0]

        landscapes: Dict[int, PiecewiseLinearFunction] = {}
        for k in range(1, max_k + 1):
            if k <= max_possible_k:
                y_k = sorted_vals[k - 1, :]
            else:
                # If we ask for more landscapes than bars, pad with zeros.
                y_k = np.zeros(num_grid_points, dtype=float)

            landscapes[k] = PiecewiseLinearFunction(
                xs=x_grid.copy(),
                ys=y_k.copy(),
                domain=(float(x_grid[0]), float(x_grid[-1])),
                metadata={
                    "k": k,
                    "mode": mode,
                    "min_bar_length": min_bar_length,
                },
            )

        # 6. Assemble family object
        forest_id = getattr(forest, "name", f"Forest@{id(forest)}")

        family = MeasurementLandscapeFamily(
            forest_id=forest_id,
            label=label,
            rescaling=mode,
            x_grid=x_grid,
            bar_kernels=bar_kernels,
            landscapes=landscapes,
            extra_meta={
                "num_grid_points": num_grid_points,
                "min_bar_length": min_bar_length,
                "global_min_x": global_min_x,
                "global_max_x": global_max_x,
            },
        )

        # Publish caches only after the complete family succeeds.
        if cache_functionals and compute_functionals:
            if not hasattr(forest, "barcode_functionals"):
                forest.barcode_functionals = {}
            forest.barcode_functionals[functionals_label] = bf
        if cache:
            if not hasattr(forest, "landscape_families"):
                forest.landscape_families = {}

            key = label
            forest.landscape_families[key] = family

        return family
