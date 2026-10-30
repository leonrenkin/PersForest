import unittest
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from persforest.PersistenceForest import PFBar
from persforest.forest_plotting import _plot_barcode_generic as plot


def forest_with(bars):
    return SimpleNamespace(barcode=bars, color_map_bars=dict(zip(bars, ['red', 'blue', 'green'])))


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

    def test_invalid_orientation(self):
        with self.assertRaisesRegex(ValueError, 'orientation'):
            plot(forest_with([PFBar(0, 1, (), [])]), orientation='diagonal')


if __name__ == '__main__':
    unittest.main()
