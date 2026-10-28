"""Cubical forests: measurements, landscapes, supported plots and explicit gating."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persforest import PersistenceForest
from persforest.measurements.basic import (
    constant_one_functional, signed_chain_edge_length, signed_chain_interior_volume,
)
from persforest.measurements.cubical import cubical_chain_interior_volume, cubical_chain_volume
from fixtures_small_filtrations import reps


def centered(shape):
    """Grid of ones with one high center cell: a single bar [1, 3)."""
    values = np.ones(shape)
    values[tuple(n // 2 for n in shape)] = 3
    return values


class CubicalMeasurementTests(unittest.TestCase):
    def test_boundary_and_interior_measures_with_anisotropic_spacing(self):
        for shape, spacing, boundary, interior in (((3, 3), (2.0, 3.0), 2 * 2 + 2 * 3, 6.0),
                                                   ((3, 3, 3), (1.0, 2.0, 3.0), 2 * 6 + 2 * 3 + 2 * 2, 6.0)):
            for storage in ("diffs", "cycles"):
                with self.subTest(shape=shape, storage=storage):
                    forest = PersistenceForest.from_cubical_grid(centered(shape), spacing=spacing, storage=storage)
                    [bar] = forest.barcode
                    [rep] = reps(forest, bar)
                    self.assertAlmostEqual(cubical_chain_volume(rep, forest.geometry), boundary)
                    self.assertAlmostEqual(cubical_chain_interior_volume(rep, forest.geometry), interior)

    def test_opposite_orientations_count_separately_unless_cancelled(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)))
        [rep] = reps(forest, next(iter(forest.barcode)))
        cell, sign = next(iter(rep.signed_simplices))
        doubled = type(rep)(set(rep.signed_simplices) | {(cell, -sign)})
        self.assertEqual(cubical_chain_volume(doubled, forest.geometry), 5.0)
        self.assertEqual(cubical_chain_volume(doubled.unsigned(), forest.geometry), 3.0)

    def test_measurements_reject_the_wrong_geometry(self):
        cubical = PersistenceForest.from_cubical_grid(centered((3, 3)))
        rep = reps(cubical, next(iter(cubical.barcode)))[0]
        for function in (signed_chain_edge_length, signed_chain_interior_volume):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(TypeError, "cubical grid"):
                function(rep, cubical.geometry)
        points = PersistenceForest.from_points(np.random.default_rng(0).random((20, 2)))
        with self.assertRaisesRegex(TypeError, "CubicalGeometry"):
            cubical_chain_volume(reps(points, points.max_bar())[0], points.geometry)
        self.assertIs(points.geometry, points.point_cloud)

    def test_profiles_and_landscapes_receive_the_grid_geometry(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)), spacing=0.5)
        profiles = forest.compute_measurement_profiles(cubical_chain_interior_volume, "area")
        np.testing.assert_allclose(profiles.evaluate_on_grid(np.array([1.0, 2.0, 2.9])), [[0.25, 0.25, 0.25]])
        family = forest.compute_measurement_landscapes(constant_one_functional, "one", x_grid=np.linspace(0, 4, 9))
        self.assertTrue(family.bar_kernels)


class CubicalPlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_barcode_forest_and_measurement_plots_work(self):
        forest = PersistenceForest.from_cubical_grid(np.random.default_rng(3).random((8, 8)))
        self.assertIn("H_{1}", forest.plot_barcode().get_title())
        forest.plot_persistence_forest(show=False)
        forest.compute_measurement_landscapes(cubical_chain_volume, "length", x_grid=np.linspace(0, 1, 11))
        forest.plot_measurement_landscapes(label="length")
        forest.plot_barcode_measurement(cubical_chain_volume, show=False)

    def test_the_delaunay_dual_plot_raises_explicitly(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)))
        with self.assertRaisesRegex(NotImplementedError, "plot_at_filtration_with_dual"):
            forest.plot_at_filtration_with_dual(2)


if __name__ == "__main__":
    unittest.main()
