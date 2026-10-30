"""2D cubical bars are the superlevel H0 pairs of the image framed by a bright border (Alexander duality).

The loops of the sublevel filtration of a 2D image enclose regions brighter than
their surroundings: a bar is born when the loop around a bright spot closes, at
the level where the spot separates from a brighter neighbor or the frame, and
dies at the spot's maximum. Beyond the grid counts as brighter than everything.
"""
import unittest

import gudhi
import numpy as np

from persforest import PersistenceForest


def superlevel_pairs(values):
    """Finite superlevel H0 pairs (birth, death) = (saddle, peak) of ``values`` from GUDHI."""
    pairs = [pair for dimension, pair in gudhi.CubicalComplex(vertices=-values).persistence() if dimension == 0]
    return {(-death, -birth) for birth, death in pairs if np.isfinite(death)}


def bright_frame(values):
    return np.pad(values, 1, constant_values=values.max() + 1)


class SuperlevelDualityTests(unittest.TestCase):
    def assert_dual(self, image):
        bars = {(bar.birth, bar.death) for bar in PersistenceForest.from_cubical_grid(image).barcode}
        self.assertEqual(bars, superlevel_pairs(bright_frame(image)))

    def test_random_image(self):
        self.assert_dual(np.random.default_rng(3).random((17, 23)))

    def test_blurred_spots(self):
        rows, columns = np.mgrid[0:40, 0:48] + 0.5
        image = np.zeros((40, 48))
        for row, column, radius, peak in ((10, 12, 5, 1.0), (14, 21, 4, 0.7), (30, 34, 6, 0.8), (31, 8, 3, 0.4)):
            image += peak * np.exp(-((rows - row) ** 2 + (columns - column) ** 2) / (2 * radius**2))
        image += 0.01 * np.random.default_rng(5).standard_normal(image.shape)
        self.assert_dual(image)


if __name__ == "__main__":
    unittest.main()
