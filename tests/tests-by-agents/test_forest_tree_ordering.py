from types import SimpleNamespace
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from persforest.PersistenceForest import PFNode
from persforest.forest_plotting import _plot_persistence_forest_generic


def _forest_with_distinct_tree_endpoints():
    # (birth, death, lifespan): root 0 = (0, 10, 10),
    # root 2 = (2, 7, 5), root 4 = (1, 4, 3).
    specs = [(0, 0.0, None), (1, 10.0, 0),
             (2, 2.0, None), (3, 7.0, 2),
             (4, 1.0, None), (5, 4.0, 4)]
    nodes = {i: PFNode(i, value, None, set(), parent)
             for i, value, parent in specs}
    for node in nodes.values():
        if node.parent is not None:
            nodes[node.parent].children.add(node.id)
    return SimpleNamespace(nodes=nodes, barcode=[])


class ForestTreeOrderingTests(unittest.TestCase):
    def test_tree_display_order(self):
        cases = [
        ("lifespan", True, (0, 2, 4)),
        ("lifespan", False, (4, 2, 0)),
        ("birth", True, (2, 4, 0)),
        ("birth", False, (0, 4, 2)),
        ("death", True, (0, 2, 4)),
        ("death", False, (4, 2, 0)),
        ]
        for sort, descending, expected in cases:
            with self.subTest(sort=sort, descending=descending):
                _, layout = _plot_persistence_forest_generic(
                    _forest_with_distinct_tree_endpoints(), sort=sort,
                    descending=descending, show=False, return_layout=True,
                )
                self.assertEqual(layout["roots"], expected)
                plt.close("all")

    def test_max_trees_selects_longest_before_display_sort(self):
        _, layout = _plot_persistence_forest_generic(
            _forest_with_distinct_tree_endpoints(), max_trees=2, sort="birth",
            descending=True, show=False, return_layout=True,
        )
        self.assertEqual(layout["roots"], (2, 0))
        plt.close("all")

    def test_tree_order_ties_use_root_id_in_both_directions(self):
        forest = _forest_with_distinct_tree_endpoints()
        forest.nodes[2].filt_val = forest.nodes[0].filt_val
        for descending in (False, True):
            _, layout = _plot_persistence_forest_generic(
                forest, sort="birth", descending=descending, show=False,
                return_layout=True,
            )
            tied_roots = [root for root in layout["roots"] if root in (0, 2)]
            self.assertEqual(tied_roots, [0, 2])
        plt.close("all")

    def test_invalid_tree_order_options(self):
        for kwargs in ({"sort": "unknown"}, {"descending": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                _plot_persistence_forest_generic(
                    _forest_with_distinct_tree_endpoints(), show=False, **kwargs,
                )
