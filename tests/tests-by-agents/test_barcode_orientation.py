import unittest
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from persforest.PersistenceForest import PFBar
from persforest.forest_plotting import _plot_barcode_generic as plot


def forest_with(bars):
    return SimpleNamespace(barcode=bars, color_map_bars=dict(zip(bars, ['red', 'blue', 'green'])))


class BarcodeOrientationTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_invalid_orientation(self):
        with self.assertRaisesRegex(ValueError, 'orientation'):
            plot(forest_with([PFBar(0, 1, (), [])]), orientation='diagonal')


if __name__ == '__main__':
    unittest.main()
