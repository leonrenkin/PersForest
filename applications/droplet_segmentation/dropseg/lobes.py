"""Step 4: split 8-shaped outlines with the alpha filtration of their points.

Two droplets without a valley between them share one bar: their summed
brightness has a single maximum. Their outline is still 8-shaped, and the
waist shows in the alpha filtration of the outline's grid vertices, taken as
plain points. The interior hole of the alpha complex is born when the
outline closes, splits in two when the alpha radius reaches half the width of
the waist, and each part dies at about the radius of its lobe. The bar of
the smaller lobe is therefore born at the split radius s, about half the
waist width, and dies at its lobe radius rho. A lobe is a bar with
``s / rho <= waist_ratio``: the waist is at most that fraction of the lobe's
width. The ratio does not depend on the size of the droplet or the
resolution of the grid; only lobes of at least ``STAIRCASE_LOBE_RADIUS``
grid pixels count, since smaller ones are the corners of the pixel staircase.

The concave pockets *outside* the outline also become holes of the alpha
complex; only bars whose region lies inside the outline count. Every lobe is
then given the outline of its alpha bar, from its last split on. Its vertices
are grid vertices, so it maps back to pixels exactly.
"""

from dataclasses import dataclass
from dataclasses import field
import math

import numpy as np
from matplotlib.path import Path
from numpy.typing import NDArray

from persforest import PersistenceForest
from persforest.geometry.loops import raw_loops
from persforest.geometry.planar import polygon_area
from persforest.measurements.loops import loops_convex_hull_perimeter_deficit
from persforest.measurements.loops import on_raw_loops

from .outlines import Selection
from .outlines import select_cycle
from .peaks import Branches
from .peaks import branch_structure
from .peaks import long_bars

# Alpha lobes of a staircase outline up to a domino (circumradius sqrt(5) / 2) are its corners, not droplets.
STAIRCASE_LOBE_RADIUS = 1.2
# The outline of a lobe is the first alpha representative of its bar at most this far from convex.
LOBE_MAX_DEFECT = 0.2
LOBE_MEASURE = on_raw_loops(loops_convex_hull_perimeter_deficit)


def outer_loop(cycle, geometry) -> NDArray[np.float64]:
    """Vertices of the loop of ``cycle`` that encloses the largest area.

    For a cubical cycle these are the grid vertices of its outer staircase,
    which ignores holes and outlying pieces of the region.
    """
    loops = raw_loops(cycle, geometry)
    return max(loops, key=lambda loop: abs(polygon_area(loop, signed=True)))


@dataclass(frozen=True)
class Split:
    """Result of splitting one outline.

    Attributes
    ----------
    selection : Selection
        The outline that was split.
    points : ndarray, shape (n, 2)
        Grid vertices of its outer loop, the input of the alpha forest.
    forest : PersistenceForest
        Alpha forest of ``points``.
    lobe_bars : list of PFBar
        Alpha bars with a narrow enough waist whose region lies inside the outline.
    branches : Branches
        Branch structure of ``lobe_bars``.
    lobes : list of Selection
        One selection per lobe bar; empty if fewer than two lobes were found,
        in which case the outline is kept.
    radius : float
        Equivalent radius of the outline, ``sqrt(area / pi)``, in pixels.
    dim_lobes : list of Selection
        The lobes too dim to be droplets (set by ``pipeline.find_lobes``);
        they are also in ``lobes``, take part in the flooding and are removed
        after it.
    """

    selection: Selection
    points: NDArray[np.float64]
    forest: PersistenceForest
    lobe_bars: list
    branches: Branches
    lobes: list[Selection]
    radius: float
    dim_lobes: list[Selection] = field(default_factory=list)


def split_selection(selection: Selection, geometry, waist_ratio: float) -> Split:
    """Look for lobes of an outline in the alpha forest of its vertices.

    Parameters
    ----------
    selection : Selection
        Outline in a cubical forest.
    geometry : CubicalGeometry
        Geometry of that forest.
    waist_ratio : float
        Lobes are the alpha bars with ``birth / death <= waist_ratio`` and a
        death of at least ``STAIRCASE_LOBE_RADIUS``.
    """
    outline = outer_loop(selection.cycle, geometry)
    radius = math.sqrt(abs(polygon_area(outline, signed=True)) / math.pi)
    points = np.unique(outline, axis=0)
    forest = PersistenceForest.from_points(points, storage="cycles")
    inside = Path(outline)
    candidates = [bar for bar in long_bars(forest) if bar.death >= STAIRCASE_LOBE_RADIUS
                  and bar.birth <= waist_ratio * bar.death]
    lobe_bars = [bar for bar in candidates
                 if inside.contains_point(outer_loop(bar.cycle_reps[-1], points).mean(axis=0))]
    branches = branch_structure(forest, lobe_bars)
    lobes = []
    if len(lobe_bars) > 1:
        lobes = [select_cycle(forest, bar, LOBE_MEASURE, LOBE_MAX_DEFECT, branches.search_start(bar))
                 for bar in lobe_bars]
    return Split(selection, points, forest, lobe_bars, branches, lobes, radius)
