"""Construction factories: equivalence with the shorthand, options and Delaunay–Čech."""
import unittest
import warnings

import gudhi
import numpy as np

from persforest import PersistenceForest
from engine_golden import snapshot
from fixtures_small_filtrations import reps, triangle
from oracle_chains import boundary, coefficients

STORAGES = ("diffs", "cycles")


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
        for storage in STORAGES:
            for reduce in (False, True):
                with self.subTest(storage=storage, reduce=reduce):
                    shorthand = PersistenceForest(points, storage=storage, reduce=reduce)
                    factory = PersistenceForest.from_points(points, storage=storage, reduce=reduce)
                    self.assertEqual(snapshot(factory), snapshot(shorthand))
                    self.assertEqual((factory.filtration_type, factory.filtration_units), ("alpha", "radius"))
                    self.assertEqual(shorthand.filtration_type, "alpha")
                    self.assertIsNotNone(factory.simplex_tree)

    def test_from_simplicial_filtration_matches_the_shorthand_and_warns_at_the_caller(self):
        points, entries = custom_entries()
        for storage in STORAGES:
            with self.subTest(storage=storage):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    factory = PersistenceForest.from_simplicial_filtration(points, entries, storage=storage)
                    shorthand = PersistenceForest(points, filtration=entries, storage=storage)
                self.assertEqual(snapshot(factory), snapshot(shorthand))
                self.assertEqual(len(caught), 2)
                self.assertTrue(all(w.filename == __file__ for w in caught))
                self.assertIn("meaningless", str(caught[0].message))
                self.assertEqual((factory.filtration_type, factory.filtration_units),
                                 ("custom_simplicial", "custom"))
                self.assertIsNone(factory.simplex_tree)

    def test_options_are_keyword_only_and_validated_before_input(self):
        points = np.random.default_rng(0).random((10, 2))
        with self.assertRaises(TypeError):
            PersistenceForest.from_points(points, "alpha")
        with self.assertRaisesRegex(ValueError, "filtration_type must be one of"):
            PersistenceForest.from_points(points, filtration_type="rips")
        with self.assertRaisesRegex(ValueError, "storage must be"):
            PersistenceForest.from_points(np.full((3, 2), np.nan), storage="stored")
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
        for dim, n in ((2, 60), (3, 40)):
            for seed in (0, 1):
                points = np.random.default_rng(seed).random((n, dim))
                for filtration_type, complex_type in (("alpha", gudhi.AlphaComplex),
                                                      ("delaunay_cech", gudhi.DelaunayCechComplex)):
                    tree = complex_type(points=points).create_simplex_tree(output_squared_values=False)
                    expected = gudhi_bars(tree, dim - 1)
                    for storage in STORAGES:
                        with self.subTest(dim=dim, seed=seed, filtration_type=filtration_type, storage=storage):
                            forest = PersistenceForest.from_points(
                                points, filtration_type=filtration_type, storage=storage)
                            self.assertTrue(expected)
                            self.assertEqual(forest_bars(forest), expected)
                            self.assertEqual(forest.filtration_units, "radius")

    def test_representatives_are_bounded_cycles_and_storages_agree(self):
        for dim, n in ((2, 50), (3, 30)):
            points = np.random.default_rng(11).random((n, dim))
            records = {}
            for storage in STORAGES:
                forest = PersistenceForest.from_points(points, filtration_type="delaunay_cech", storage=storage)
                for bar in forest.barcode:
                    for rep in reps(forest, bar):
                        support = coefficients(rep.signed_simplices)
                        self.assertEqual(boundary(support), {})
                        self.assertEqual(boundary(coefficients(rep.interior)), support)
                records[storage] = snapshot(forest)
            self.assertEqual(records["diffs"], records["cycles"])

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
