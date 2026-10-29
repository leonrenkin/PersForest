"""Stage 3: split 8-shaped outlines with the alpha filtration of their points.

Two droplets without a valley between them share one bar: their summed
brightness has a single maximum. Their outline is still 8-shaped, and the
waist shows in the alpha filtration of the outline's grid vertices, taken as
plain points. The interior hole of the alpha complex is born when the
outline closes, splits in two when the alpha radius reaches half the width of
the waist, and each part dies at about the radius of its lobe. A lobe is
therefore a bar with

- ``death >= min_lobe_radius``: the lobe is droplet-sized, and
- ``lifespan >= min_lobe_lifespan``: the waist is narrower than the lobe.

The concave pockets *outside* the outline also become holes of the alpha
complex; only bars whose region lies inside the outline count. Every lobe is
then searched like a stage-2 bar, from its last long split on. Its vertices
are grid vertices, so it maps back to pixels exactly.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib.path import Path
from numpy.typing import NDArray

from persforest import PersistenceForest
from persforest.geometry.loops import cycle_loops
from persforest.geometry.planar import polygon_area

from candidates import Branches
from candidates import branch_structure
from candidates import long_bars
from selection import Selection
from selection import Threshold
from selection import select_cycle


def outer_loop(cycle, geometry) -> NDArray[np.float64]:
    """Vertices of the loop of ``cycle`` that encloses the largest area.

    For a cubical cycle these are the grid vertices of its outer staircase,
    which ignores holes and outlying pieces of the region.
    """
    loops = cycle_loops(cycle, geometry)
    return max(loops, key=lambda loop: abs(polygon_area(loop, signed=True)))


@dataclass(frozen=True)
class Split:
    """Result of splitting one stage-2 selection.

    Attributes
    ----------
    selection : Selection
        The stage-2 selection that was split.
    points : ndarray, shape (n, 2)
        Grid vertices of its outer loop, the input of the alpha forest.
    forest : PersistenceForest
        Alpha forest of ``points``.
    lobe_bars : list of PFBar
        Long alpha bars whose region lies inside the outline.
    branches : Branches
        Branch structure of ``lobe_bars``.
    lobes : list of Selection
        One selection per lobe bar; empty if fewer than two lobes were found,
        in which case the outline is kept.
    """

    selection: Selection
    points: NDArray[np.float64]
    forest: PersistenceForest
    lobe_bars: list
    branches: Branches
    lobes: list[Selection]


def split_selection(selection: Selection, geometry, min_lobe_radius: float, min_lobe_lifespan: float,
                    measure: Callable, max_defect: float) -> Split:
    """Look for lobes of a stage-2 outline in the alpha forest of its vertices.

    Parameters
    ----------
    selection : Selection
        Stage-2 selection of a cubical forest.
    geometry : CubicalGeometry
        Geometry of that forest.
    min_lobe_radius, min_lobe_lifespan : float
        Minimal death and lifespan of a lobe bar, in alpha radius units.
    measure : callable
        Shape measure of the alpha representatives, ``(chain, points) -> float``.
    max_defect : float
        Threshold of the lobe search.
    """
    outline = outer_loop(selection.cycle, geometry)
    points = np.unique(outline, axis=0)
    forest = PersistenceForest.from_points(points, storage="cycles")
    inside = Path(outline)
    lobe_bars = [bar for bar in long_bars(forest, min_lobe_lifespan, min_lobe_radius)
                 if inside.contains_point(outer_loop(bar.cycle_reps[-1], points).mean(axis=0))]
    branches = branch_structure(forest, lobe_bars)
    lobes = []
    if len(lobe_bars) > 1:
        lobes = [select_cycle(forest, bar, measure, Threshold(max_defect), branches.search_start(bar))
                 for bar in lobe_bars]
    return Split(selection, points, forest, lobe_bars, branches, lobes)
