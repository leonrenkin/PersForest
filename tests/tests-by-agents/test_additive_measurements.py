"""Incremental evaluation of additive measurements against direct measurement."""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from persforest import PersistenceForest
from persforest.chains import SignedChain
from persforest.geometry.cubical import CubicalGeometry
from persforest.measurements.additive import AdditiveMeasurement
from persforest.measurements.basic import signed_chain_interior_volume, signed_chain_volume
from persforest.measurements.cubical import cubical_chain_interior_volume, cubical_chain_volume
from fixtures_small_filtrations import annulus, build, disconnected, fan, mode_options, split
from oracle_chains import simplex_volume

# Incremental values are compensated running sums; direct values use fsum.
RTOL, ATOL = 1e-12, 1e-14


def _direct(measurement):
    """Plain callable with the same values, which profiles never evaluate incrementally."""
    return lambda chain, geometry: measurement(chain, geometry)


def _forests():
    rng = np.random.default_rng(11)
    yield "alpha2", PersistenceForest.from_points(rng.uniform(0, 1, (400, 2)))
    yield "alpha3", PersistenceForest.from_points(rng.uniform(0, 1, (150, 3)))
    yield "cech2", PersistenceForest.from_points(rng.uniform(0, 1, (200, 2)),
                                                 filtration_type="delaunay_cech")
    yield "alpha2-cycles", PersistenceForest.from_points(rng.uniform(0, 1, (200, 2)), storage="cycles")
    yield "cubical2", PersistenceForest.from_cubical_grid(rng.uniform(0, 1, (20, 20)), spacing=(0.5, 1.5))
    yield "cubical2-vertices", PersistenceForest.from_cubical_grid(
        rng.uniform(0, 1, (15, 15)), filtration_type="vertices")
    yield "cubical3", PersistenceForest.from_cubical_grid(rng.uniform(0, 1, (6, 6, 6)))
    yield "cubical3-cycles", PersistenceForest.from_cubical_grid(rng.uniform(0, 1, (5, 5, 5)), storage="cycles")
    for fixture in (split, fan, annulus, disconnected):
        for name, options in mode_options():
            yield f"{fixture.__name__}-{name}", build(fixture(), **options)


def _measurements(forest):
    if isinstance(forest.geometry, CubicalGeometry):
        return (cubical_chain_volume, cubical_chain_interior_volume)
    return (signed_chain_volume, signed_chain_interior_volume)


class IncrementalEqualsDirectTests(unittest.TestCase):
    def assertProfilesClose(self, incremental, direct):
        self.assertEqual(incremental.bar_ids, direct.bar_ids)
        for bar_id in direct.bar_ids:
            got, expected = incremental[bar_id], direct[bar_id]
            np.testing.assert_array_equal(got.starts, expected.starts)
            np.testing.assert_array_equal(got.ends, expected.ends)
            np.testing.assert_allclose(got.vals, expected.vals, rtol=RTOL, atol=ATOL)
            # Exact zeros where the representative measures zero; never negative.
            np.testing.assert_array_equal(got.vals[expected.vals == 0], 0.0)
            self.assertTrue(np.all(got.vals >= 0))

    def test_profiles_match_for_all_builtins_conventions_and_storage(self):
        for name, forest in _forests():
            for measurement in _measurements(forest):
                for signed in (False, True):
                    with self.subTest(forest=name, measurement=measurement.__name__, signed=signed):
                        incremental = forest.compute_measurement_profiles(
                            measurement, "m", signed=signed, cache=False)
                        direct = forest.compute_measurement_profiles(
                            _direct(measurement), "m", signed=signed, cache=False)
                        self.assertProfilesClose(incremental, direct)

    def test_long_progressions_stay_within_tolerance(self):
        # One long-lived cycle whose boundary grows and shrinks over many steps.
        rng = np.random.default_rng(0)
        angles = rng.uniform(0, 2 * np.pi, 3000)
        points = np.c_[np.cos(angles), np.sin(angles)] + rng.normal(0, 0.05, (3000, 2))
        forest = PersistenceForest.from_points(points)
        for measurement in (signed_chain_volume, signed_chain_interior_volume):
            with self.subTest(measurement=measurement.__name__):
                self.assertProfilesClose(
                    forest.compute_measurement_profiles(measurement, "m", cache=False),
                    forest.compute_measurement_profiles(_direct(measurement), "m", cache=False))

    def test_landscapes_match(self):
        forest = PersistenceForest.from_points(np.random.default_rng(2).uniform(0, 1, (500, 2)))
        grid = np.linspace(0, 0.1, 64)
        incremental = forest.compute_measurement_landscapes(signed_chain_volume, "a", x_grid=grid, cache=False)
        direct = forest.compute_measurement_landscapes(_direct(signed_chain_volume), "b", x_grid=grid, cache=False)
        np.testing.assert_allclose(incremental.evaluate_on_grid(grid), direct.evaluate_on_grid(grid),
                                   rtol=RTOL, atol=ATOL)

    def test_custom_measurement_counts_unsigned_cells(self):
        edge_count = AdditiveMeasurement(lambda cells, geometry: np.ones(len(cells)), "boundary")
        forest = PersistenceForest.from_points(np.random.default_rng(6).uniform(0, 1, (200, 2)))
        profiles = forest.compute_measurement_profiles(edge_count, "count", cache=False)
        for profile_bar, bar in zip(profiles.bars, profiles._source_bars):
            counts = [len(rep.unsigned().signed_simplices) for rep in forest.iter_bar_cycle_reps(bar)]
            np.testing.assert_array_equal(profiles[profile_bar.bar_id].vals, counts[::-1])

    def test_forest_like_input_without_diffs_uses_direct_evaluation(self):
        forest = PersistenceForest.from_points(np.random.default_rng(8).uniform(0, 1, (150, 2)))
        generic = SimpleNamespace(barcode=forest.barcode, geometry=forest.geometry,
                                  iter_bar_cycle_reps=forest.iter_bar_cycle_reps)
        from persforest.landscapes.profiles import compute_measurement_profiles
        self.assertProfilesClose(
            compute_measurement_profiles(forest, signed_chain_volume, "m"),
            compute_measurement_profiles(generic, signed_chain_volume, "m"))


class AdditiveMeasurementCallTests(unittest.TestCase):
    def test_volume_call_matches_gram_oracle(self):
        rng = np.random.default_rng(9)
        for dim in (2, 3):
            points = rng.normal(size=(12, dim))
            facets = [tuple(sorted(rng.choice(12, dim, replace=False))) for _ in range(20)]
            terms = {(facet, int(sign)) for facet, sign in zip(facets, rng.choice([-1, 1], 20))}
            chain = SignedChain(terms)
            expected = math.fsum(simplex_volume(points, facet) for facet, _ in terms)
            self.assertAlmostEqual(signed_chain_volume(chain, points), expected, places=12)
            self.assertEqual(signed_chain_volume(SignedChain(set()), points), 0.0)

    def test_interior_volume_call(self):
        points = np.array([(0., 0.), (2., 0.), (0., 3.), (2., 3.)])
        chain = SignedChain(set(), interior_available=True, interior={((0, 1, 2), 1), ((1, 2, 3), -1)})
        self.assertAlmostEqual(signed_chain_interior_volume(chain, points), 6.0, places=12)
        with self.assertRaises(ValueError):
            signed_chain_interior_volume(SignedChain(set()), points)
        bad = SignedChain(set(), interior_available=True, interior={((0, 1), 1)})
        with self.assertRaises(ValueError):
            signed_chain_interior_volume(bad, points)

    def test_cubical_measures_use_spacing(self):
        forest = PersistenceForest.from_cubical_grid(np.zeros((3, 3)), spacing=(0.5, 2.0))
        chain = SignedChain({((1, 0), 1), ((0, 1), -1), ((1, 1), 1)})
        self.assertAlmostEqual(cubical_chain_volume(chain, forest.geometry), 0.5 + 2.0 + 1.0)
        interior = SignedChain(set(), interior_available=True, interior={((1, 1), 1)})
        self.assertAlmostEqual(cubical_chain_interior_volume(interior, forest.geometry), 1.0)

    def test_geometry_mismatch_raises(self):
        cubical = PersistenceForest.from_cubical_grid(np.random.default_rng(1).uniform(0, 1, (6, 6)))
        points = PersistenceForest.from_points(np.random.default_rng(1).uniform(0, 1, (30, 2)))
        with self.assertRaises(TypeError):
            signed_chain_volume(SignedChain(set()), cubical.geometry)
        with self.assertRaises(TypeError):
            cubical_chain_volume(SignedChain(set()), points.geometry)
        with self.assertRaises(TypeError):
            cubical.compute_measurement_profiles(signed_chain_volume, "m", cache=False)
        with self.assertRaises(TypeError):
            points.compute_measurement_profiles(cubical_chain_volume, "m", cache=False)

    def test_validation(self):
        with self.assertRaises(ValueError):
            AdditiveMeasurement(lambda cells, geometry: np.ones(len(cells)), "edges")
        with self.assertRaises(TypeError):
            AdditiveMeasurement("not callable", "boundary")
        chain = SignedChain({((0, 1), 1), ((1, 2), 1)})
        points = np.zeros((3, 2))
        wrong_shape = AdditiveMeasurement(lambda cells, geometry: np.ones(len(cells) + 1), "boundary")
        with self.assertRaises(ValueError):
            wrong_shape(chain, points)
        not_finite = AdditiveMeasurement(lambda cells, geometry: np.full(len(cells), np.nan), "boundary")
        with self.assertRaises(ValueError):
            not_finite(chain, points)

    def test_builtins_keep_names_and_profile_metadata(self):
        self.assertEqual(signed_chain_volume.__name__, "signed_chain_volume")
        self.assertEqual(cubical_chain_interior_volume.__name__, "cubical_chain_interior_volume")
        self.assertIn("simplex volume", signed_chain_volume.__doc__)
        forest = PersistenceForest.from_points(np.random.default_rng(1).uniform(0, 1, (40, 2)))
        profiles = forest.compute_measurement_profiles(signed_chain_volume, "length", cache=False)
        self.assertEqual((profiles.measurement_name, dict(profiles.measurement_params)), ("length", {}))


if __name__ == "__main__":
    unittest.main()
