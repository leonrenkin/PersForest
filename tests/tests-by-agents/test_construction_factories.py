"""Construction factories: equivalence with the shorthand, options and Delaunay–Čech."""
import unittest
import warnings

import gudhi
import numpy as np

from persforest import PersistenceForest
from engine_golden import snapshot
from fixtures_small_filtrations import reps, triangle
from oracle_chains import boundary, coefficients


def gudhi_bars(simplex_tree, dimension, min_length=1e-9):
    simplex_tree.compute_persistence()
    return sorted((float(b), float(d)) for b, d in simplex_tree.persistence_intervals_in_dimension(dimension)
                  if np.isfinite(d) and d - b > min_length)


def forest_bars(forest, min_length=1e-9):
    return sorted((b.birth, b.death) for b in forest.barcode if b.death - b.birth > min_length)


def custom_entries():
    fixture = triangle()
    entries = sorted(fixture.values.items(), key=lambda item: (item[1], len(item[0]), item[0]))
    return fixture.points, entries


class FactoryEquivalenceTests(unittest.TestCase):
    def test_from_points_matches_the_shorthand(self):
        points = np.random.default_rng(3).random((40, 2))
        for reduce in (False, True):
            with self.subTest(reduce=reduce):
                shorthand = PersistenceForest(points, reduce=reduce)
                factory = PersistenceForest.from_points(points, reduce=reduce)
                self.assertEqual(snapshot(factory), snapshot(shorthand))
                self.assertEqual((factory.filtration_type, factory.filtration_units), ("alpha", "radius"))
                self.assertEqual(shorthand.filtration_type, "alpha")
                self.assertFalse(hasattr(factory, "simplex_tree"))

    def test_from_simplicial_filtration_matches_the_shorthand_and_warns_at_the_caller(self):
        points, entries = custom_entries()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            factory = PersistenceForest.from_simplicial_filtration(points, entries)
            shorthand = PersistenceForest(points, filtration=entries)
        self.assertEqual(snapshot(factory), snapshot(shorthand))
        self.assertEqual(len(caught), 2)
        self.assertTrue(all(w.filename == __file__ for w in caught))
        self.assertIn("meaningless", str(caught[0].message))
        self.assertEqual((factory.filtration_type, factory.filtration_units),
                         ("custom_simplicial", "custom"))
        self.assertFalse(hasattr(factory, "simplex_tree"))

    def test_options_are_keyword_only_and_validated_before_input(self):
        points = np.random.default_rng(0).random((10, 2))
        with self.assertRaises(TypeError):
            PersistenceForest.from_points(points, "alpha")
        with self.assertRaisesRegex(ValueError, "filtration_type must be one of"):
            PersistenceForest.from_points(points, filtration_type="rips")
        for filtration_type in ("alpha", "delaunay_cech"):
            with self.assertRaisesRegex(ValueError, "1D point clouds are not supported"):
                PersistenceForest.from_points(np.arange(5.0)[:, None], filtration_type=filtration_type)
        with self.assertRaises(TypeError):
            PersistenceForest.from_points(points, storage="diffs")
        with self.assertRaisesRegex(ValueError, "filtration_tol"):
            PersistenceForest.from_simplicial_filtration(points, [], filtration_tol=-1)

    def test_delayed_barcode_through_a_factory(self):
        points = np.random.default_rng(5).random((30, 2))
        delayed = PersistenceForest.from_points(points, filtration_type="delaunay_cech", compute_barcode=False)
        self.assertEqual(delayed.barcode, set())
        delayed.compute_barcode()
        self.assertEqual(snapshot(delayed),
                         snapshot(PersistenceForest.from_points(points, filtration_type="delaunay_cech")))


class DelaunayCechTests(unittest.TestCase):
    def test_bars_match_gudhi_persistence_for_both_point_filtrations(self):
        # Both come from persforest's builders in 2D and 3D, compared with GUDHI's
        # exact mode. GUDHI's exact Delaunay–Čech radii are themselves inexact (up
        # to 1e-10 relative seen); test_delaunay_builder checks ours in rational arithmetic.
        reference = {"alpha": (lambda points: gudhi.AlphaComplex(points=points, precision="exact"), 1e-13),
                     "delaunay_cech": (lambda points: gudhi.DelaunayCechComplex(points=points, precision="exact"),
                                       1e-9)}
        for dim, n in ((2, 60), (3, 40)):
            for seed in (0, 1):
                points = np.random.default_rng(seed).random((n, dim))
                for filtration_type, (complex_for, rtol) in reference.items():
                    tree = complex_for(points).create_simplex_tree(output_squared_values=False)
                    expected = gudhi_bars(tree, dim - 1)
                    with self.subTest(dim=dim, seed=seed, filtration_type=filtration_type):
                        forest = PersistenceForest.from_points(points, filtration_type=filtration_type)
                        self.assertTrue(expected)
                        np.testing.assert_allclose(forest_bars(forest), expected, rtol=rtol, atol=0)
                        self.assertEqual(forest.filtration_units, "radius")

    def test_representatives_are_bounded_cycles(self):
        for dim, n in ((2, 50), (3, 30)):
            points = np.random.default_rng(11).random((n, dim))
            forest = PersistenceForest.from_points(points, filtration_type="delaunay_cech")
            for bar in forest.barcode:
                for rep in reps(forest, bar):
                    support = coefficients(rep.signed_simplices)
                    self.assertEqual(boundary(support), {})
                    self.assertEqual(boundary(coefficients(rep.interior)), support)

    def test_delaunay_cech_values_never_exceed_alpha_values(self):
        # Sanity check on the input values: the minimal enclosing ball is no
        # larger than the smallest empty circumscribing ball.
        points = np.random.default_rng(2).random((50, 2))
        alpha = dict((tuple(s), v) for s, v in PersistenceForest.from_points(points).filtration)
        cech = dict((tuple(s), v) for s, v in
                    PersistenceForest.from_points(points, filtration_type="delaunay_cech").filtration)
        self.assertEqual(alpha.keys(), cech.keys())
        self.assertTrue(all(cech[s] <= alpha[s] + 1e-12 for s in alpha))
        self.assertTrue(any(cech[s] < alpha[s] - 1e-9 for s in alpha))


if __name__ == "__main__":
    unittest.main()
