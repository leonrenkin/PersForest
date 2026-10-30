"""Steps 1 and 2: candidate bars and the branches that split off them.

A bar's representatives run from its birth, where it splits off an elder bar
(or where its tree starts), to its death. A long bar ``child`` *splits off*
``parent`` if the forest node at which ``child`` is born lies on the
progression of ``parent``. Before that split, ``parent``'s cycle still
encloses ``child``'s region; from the split on, the two cycles bound disjoint
regions. Searching ``parent`` for an outline only from its last long split on
therefore keeps outlines from containing each other. Splits above a given
value can be ignored, leaving those droplets inside their elder's outline.

The functions work on any persistence forest, cubical or simplicial.
"""

from dataclasses import dataclass
import math

from persforest import PersistenceForest
from persforest.barcode import PFBar


def long_bars(forest: PersistenceForest, min_lifespan: float = 0.0) -> list[PFBar]:
    """Bars with ``lifespan >= min_lifespan``, longest first.

    ``forest.barcode`` is a set; ties, common in 8-bit images with saturated
    peaks, are broken by the forest's node ids, so that the order and hence
    the labels are the same in every run.
    """
    bars = [bar for bar in forest.barcode if bar.lifespan() >= min_lifespan]
    return sorted(bars, key=lambda bar: (-bar.lifespan(), bar.birth, bar.death, bar._node_progression[0]))


@dataclass(frozen=True)
class Branches:
    """Which of the selected bars split off which.

    Attributes
    ----------
    bars : list of PFBar
        The selected bars.
    parent : dict
        The selected bar each bar splits off, or None if it splits off no
        selected bar (it starts its own tree, or splits off a bar that was
        not selected).
    children : dict
        The selected bars splitting off each bar, by increasing birth.
    """

    bars: list[PFBar]
    parent: dict[PFBar, PFBar | None]
    children: dict[PFBar, list[PFBar]]

    def search_start(self, bar: PFBar, until: float = math.inf) -> float:
        """Birth of ``bar``, or the birth of its last child born below ``until``, if later.

        A child born at ``until`` counts as late, like the bars born at the
        maximal search value; ties are common in 8-bit images.
        """
        return max([bar.birth] + [child.birth for child in self.children[bar] if child.birth < until])


def branch_structure(forest: PersistenceForest, bars: list[PFBar]) -> Branches:
    """Find which of ``bars`` split off which.

    A bar is born at the parent of the first node of its progression. The bar
    whose progression contains that node is the one it splits off; if that
    bar is not among ``bars``, the search continues with the bar it split off
    in turn.
    """
    owner = {node_id: bar for bar in forest.barcode for node_id in bar._node_progression}

    def split_off(bar):
        first = forest.nodes[bar._node_progression[0]]
        return owner.get(first.parent)

    selected = set(bars)
    parent = {}
    for bar in bars:
        elder = split_off(bar)
        while elder is not None and elder not in selected:
            elder = split_off(elder)
        parent[bar] = elder
    children = {bar: sorted((child for child in bars if parent[child] is bar), key=lambda child: child.birth)
                for bar in bars}
    return Branches(list(bars), parent, children)
