import unittest
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from persforest.barcode import PFBar
from persforest.plotting.barcode import _plot_barcode_generic as plot


def forest_with(bars):
    return SimpleNamespace(dim=2, barcode=bars,
                           color_map_bars=dict(zip(bars, ['red', 'blue', 'green'])))


class BarcodeOrientationTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_vertical_is_transpose_with_same_sort_selection_and_colors(self):
        bars = [PFBar(1, 4, (), []), PFBar(2, 2.1, (), []), PFBar(0, 5, (), [])]
        forest = forest_with(bars)
        fig, axes = plt.subplots(1, 2)
        for ax, orientation in zip(axes, ['horizontal', 'vertical']):
            plot(forest, ax=ax, orientation=orientation, coloring='bars',
                 min_bar_length=.2, max_bars=2, sort='length', descending=True, tight_layout=False)
        for horizontal, vertical in zip(axes[0].collections, axes[1].collections):
            np.testing.assert_allclose(horizontal.get_segments()[0][:, ::-1], vertical.get_segments()[0])
            np.testing.assert_equal(horizontal.get_colors(), vertical.get_colors())
        np.testing.assert_allclose(axes[1].collections[0].get_segments()[0], [[0, 0], [0, 5]])
        self.assertEqual(tuple(axes[1].collections[0].get_colors()[0]), to_rgba('green'))
        self.assertEqual(axes[0].get_xlabel(), 'filtration value')
        self.assertEqual(axes[1].get_ylabel(), 'filtration value')
        self.assertEqual(len(axes[1].get_xticks()), 0)

    def test_vertical_preserves_shared_filtration_ticks(self):
        fig, axes = plt.subplots(1, 2, sharey=True)
        axes[0].set_yticks([0, 1, 2, 3])
        plot(forest_with([PFBar(1, 3, (), [])]), ax=axes[1], orientation='vertical',
             coloring='bars', ylabel='Scale', tight_layout=False)
        np.testing.assert_equal(axes[0].get_yticks(), [0, 1, 2, 3])
        np.testing.assert_equal(axes[0].get_ylim(), axes[1].get_ylim())
        self.assertEqual(axes[1].get_ylabel(), 'Scale')

    def test_infinite_bars_have_arrows_in_both_orientations(self):
        forest = forest_with([PFBar(0, np.inf, (), []), PFBar(2, np.inf, (), [])])
        fig, axes = plt.subplots(1, 2)
        for ax, orientation in zip(axes, ['horizontal', 'vertical']):
            plot(forest, ax=ax, orientation=orientation, coloring='bars', tight_layout=False)
            axis = 1 if orientation == 'vertical' else 0
            limits = ax.get_ylim() if axis else ax.get_xlim()
            self.assertTrue(np.isfinite(limits).all())
            self.assertGreater(limits[1], 2)
            self.assertEqual(len(ax.texts), 2)
            for arrow in ax.texts:
                self.assertGreater(arrow.xy[axis], arrow.get_position()[axis])
                self.assertEqual(arrow.xy[1-axis], arrow.get_position()[1-axis])

    def test_negative_birth_and_infinite_bar_after_finite_deaths(self):
        forest = forest_with([PFBar(-2, -1, (), []), PFBar(4, np.inf, (), [])])
        fig, ax = plt.subplots()
        plot(forest, ax=ax, orientation='vertical', coloring='bars', tight_layout=False)
        self.assertLessEqual(ax.get_ylim()[0], -2)
        self.assertGreater(ax.get_ylim()[1], 4)
        self.assertTrue(all(np.isfinite(line.get_segments()[0]).all() for line in ax.collections))

    def test_default_shows_at_most_100_longest_bars_and_says_so(self):
        bars = [PFBar(0, 1 + i, ()) for i in range(130)]
        forest = SimpleNamespace(dim=2, barcode=bars, color_map_bars={b: 'red' for b in bars})
        ax = plot(forest, coloring='bars', tight_layout=False)
        [collection] = ax.collections
        self.assertEqual(len(collection.get_segments()), 100)
        self.assertEqual(min(s[1][0] for s in collection.get_segments()), 31)  # the 100 longest
        self.assertEqual(ax.get_title(loc='right'), '100 longest of 130 bars')
        self.assertEqual(ax.get_title(loc='left'), '$H_{1}$ barcode')
        all_bars = plot(forest, coloring='bars', max_bars=0, tight_layout=False)
        self.assertEqual(len(all_bars.collections[0].get_segments()), 130)
        self.assertEqual(all_bars.get_title(loc='right'), '130 bars')
        # Bars get thinner with more rows, within 0.6 to 6 points.
        widths = [ax.collections[0].get_linewidths()[0], all_bars.collections[0].get_linewidths()[0]]
        self.assertGreater(widths[0], widths[1])
        self.assertTrue(all(0.6 <= w <= 6 for w in widths))

    def test_tree_sort_groups_trees_with_gaps(self):
        # Tree 1 holds the longest bar; tree 0 comes second; a gap row separates them.
        bars = [PFBar(0, 2, (), tree_id=0), PFBar(1, 9, (), tree_id=1), PFBar(0.5, 1, (), tree_id=0),
                PFBar(3, 4, (), tree_id=1)]
        forest = SimpleNamespace(dim=2, barcode=bars, color_map_forest={b: 'red' for b in bars})
        ax = plot(forest, sort='tree', tight_layout=False)
        rows = {(s[0][0], s[1][0]): s[0][1] for s in ax.collections[0].get_segments()}
        self.assertEqual(rows, {(1, 9): 0, (3, 4): 1, (0, 2): 3, (0.5, 1): 4})

    def test_invalid_orientation(self):
        with self.assertRaisesRegex(ValueError, 'orientation'):
            plot(forest_with([PFBar(0, 1, (), [])]), orientation='diagonal')


if __name__ == '__main__':
    unittest.main()
