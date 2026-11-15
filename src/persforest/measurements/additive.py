"""Additive measurements: sums of fixed per-cell weights over a representative.

An additive measurement assigns every cell a weight that does not depend on
the representative, and measures a chain as the sum of the weights of its
boundary cells (counted once per stored orientation) or of its interior cells.
Lengths, areas, volumes and cell counts are additive; shape functionals such as
circularity, curvature or connected components are not.

Measurement profiles and landscapes evaluate additive measurements
incrementally from the forest's stored simplex diffs instead of re-measuring
every representative, which reduces their cost from the total size of all
representatives to the total size of the diffs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from persforest.chains import SignedChain

AdditiveSupport = Literal["boundary", "interior"]
CellWeights = Callable[[Sequence[tuple], Any], Any]


def _require_chain_interior(signed_chain: SignedChain) -> None:
    """Raise an actionable error when a chain has no interior data."""
    if not signed_chain.interior_available or signed_chain.interior is None:
        raise ValueError(
            "Interior is unavailable. Obtain the chain through "
            "forest.iter_bar_cycle_reps(bar) or forest.cycle_for_bar_at(bar, value)."
        )


class AdditiveMeasurement:
    """Measurement given by a sum of per-cell weights.

    Instances are ordinary ``(chain, geometry) -> float`` measurements and can
    be used wherever a measurement callable is accepted. Profile and landscape
    computations recognize them and evaluate them incrementally.

    Parameters
    ----------
    cell_weights : callable
        ``cell_weights(cells, geometry)`` returns one finite weight per cell,
        as an array-like of shape ``(len(cells),)``. ``cells`` is a list of
        cell IDs (increasing vertex tuples for simplicial forests, doubled
        coordinates for cubical grids) and ``geometry`` is ``forest.geometry``.
        Weights must not depend on orientation or on the representative.
        It should be vectorized; it is called once per profile computation
        for every cell of the support dimension.
    support : {"boundary", "interior"}
        ``"boundary"`` sums over the codimension-one cells of the chain,
        once per stored orientation; after ``SignedChain.unsigned()`` (the
        default in profiles), cells present in both orientations cancel.
        ``"interior"`` sums over the top-dimensional interior cells and
        requires interior data on the chain.
    name : str, optional
        Value of ``__name__``. Animations use it as the default profile label.
    doc : str, optional
        Value of ``__doc__``.
    geometry_check : callable, optional
        ``geometry_check(geometry)`` raises for unsupported geometry. It runs
        before any other validation, also for empty chains.
    measurement_name : str, optional
        Descriptive name recorded by profiles. If omitted, profiles fall back
        to their label.
    measurement_params : mapping, optional
        Descriptive parameters recorded by profiles.

    Notes
    -----
    Direct evaluation uses ``math.fsum`` (exactly rounded). Incremental
    evaluation uses compensated running sums and agrees with direct evaluation
    up to floating-point rounding; it returns exactly zero for empty chains.
    """

    def __init__(
        self,
        cell_weights: CellWeights,
        support: AdditiveSupport,
        *,
        name: str | None = None,
        doc: str | None = None,
        geometry_check: Callable[[Any], None] | None = None,
        measurement_name: str | None = None,
        measurement_params: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(cell_weights):
            raise TypeError("cell_weights must be callable")
        if support not in ("boundary", "interior"):
            raise ValueError(f'support must be "boundary" or "interior", not {support!r}')
        if geometry_check is not None and not callable(geometry_check):
            raise TypeError("geometry_check must be callable")
        if measurement_name is not None and not isinstance(measurement_name, str):
            raise TypeError("measurement_name must be a string")
        if measurement_params is not None and not isinstance(measurement_params, Mapping):
            raise TypeError("measurement_params must be a mapping")
        self.cell_weights = cell_weights
        self.support = support
        self.geometry_check = geometry_check
        self.measurement_name = measurement_name
        self.measurement_params = MappingProxyType(dict(measurement_params or {}))
        self.__name__ = name if name is not None else f"additive_{support}_measurement"
        self.__qualname__ = self.__name__
        self.__doc__ = doc

    def __repr__(self) -> str:
        return f"AdditiveMeasurement({self.__name__}, support={self.support!r})"

    def check_geometry(self, geometry: Any) -> None:
        """Run ``geometry_check`` if one was given."""
        if self.geometry_check is not None:
            self.geometry_check(geometry)

    def weights(self, cells: Sequence[tuple], geometry: Any) -> NDArray[np.float64]:
        """Return validated weights of ``cells``.

        Raises
        ------
        ValueError
            If the result does not have one finite real weight per cell.
        """
        cells = list(cells)
        if not cells:
            return np.zeros(0, dtype=float)
        weights = np.asarray(self.cell_weights(cells, geometry))
        if np.iscomplexobj(weights):
            raise ValueError("cell_weights must return real weights")
        weights = weights.astype(float, copy=False)
        if weights.shape != (len(cells),):
            raise ValueError(
                f"cell_weights must return shape ({len(cells)},), got {weights.shape}")
        if not np.all(np.isfinite(weights)):
            raise ValueError("cell_weights must return finite weights")
        return weights

    def __call__(self, signed_chain: SignedChain, geometry: Any) -> float:
        """Return the sum of the weights of the chain's support cells."""
        self.check_geometry(geometry)
        if self.support == "interior":
            _require_chain_interior(signed_chain)
            entries = signed_chain.interior
        else:
            entries = signed_chain.signed_simplices
        if not entries:
            return 0.0
        return float(math.fsum(self.weights([cell for cell, _ in entries], geometry)))
