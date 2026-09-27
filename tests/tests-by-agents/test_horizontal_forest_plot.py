"""Regression checks for horizontal persistence-forest plots."""
import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from persforest.PersistenceForest import PFBar, PFNode
from persforest.forest_plotting import _plot_persistence_forest_generic


class HorizontalForestPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_horizontal_filtration_geometry_and_bar_colors(self):
        nodes = {
            0: PFNode(0, 0, None, {1, 2}, None),
            1: PFNode(1, 2, None, {3}, 0),
            2: PFNode(2, 3, None, set(), 0),
            3: PFNode(3, 5, None, set(), 1),
        }
        bars = [PFBar(0, 5, (1, 3), [], True, 0),
                PFBar(0, 3, (2,), [], False, 0)]
        palette = {bars[0]: "red", bars[1]: "blue"}
        forest = SimpleNamespace(
            nodes=nodes, barcode=bars, _get_color_map=lambda _: palette
        )
        ax, layout = _plot_persistence_forest_generic(
            forest, orientation="horizontal", coloring="bars", nodes="all",
            return_layout=True, show=False,
        )
        ax.figure.canvas.draw()

        self.assertEqual(layout["orientation"], "horizontal")
        self.assertEqual({i: position[0] for i, position in layout["positions"].items()},
                         {i: node.filt_val for i, node in nodes.items()})
        self.assertEqual(ax.get_xlabel(), "Filtration value")
        self.assertEqual(len(ax.get_yticks()), 0)
        self.assertEqual(set(map(tuple, ax.collections[0].get_edgecolors())),
                         {to_rgba("red"), to_rgba("blue")})
        np.testing.assert_allclose(
            ax.collections[1].get_offsets(),
            [layout["positions"][i] for i in layout["marker_ids"]],
        )
        for path in ax.collections[0].get_paths():
            self.assertLessEqual(path.vertices[0, 0], path.vertices[-1, 0])

        ax.figure.set_size_inches(5, 8)
        ax.figure.canvas.draw()
        np.testing.assert_allclose(
            ax.collections[1].get_offsets(),
            [layout["positions"][i] for i in layout["marker_ids"]],
        )

        with self.assertRaisesRegex(ValueError, "orientation"):
            _plot_persistence_forest_generic(forest, orientation="diagonal", show=False)


if __name__ == "__main__":
    unittest.main()
