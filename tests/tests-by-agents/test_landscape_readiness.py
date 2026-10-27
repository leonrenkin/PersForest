"""Readiness belongs at measurement boundaries; captured results stand alone."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from persforest import PersistenceForest
from persforest.landscapes.family import landscapes_from_profiles
from persforest.landscapes.profiles import compute_measurement_profiles, _build_step_function_data
from persforest.landscapes.kernels import landscape_kernel
from fixtures_small_filtrations import build, triangle, profile_forest, measured


class LandscapeReadinessTests(unittest.TestCase):
    def test_uncomputed_boundaries_reject_before_work_or_cache_changes(self):
        forest = build(triangle(), compute_barcode=False)
        forbidden = Mock(side_effect=AssertionError('Measurement must not run'))
        sentinel = object()
        forest.measurement_profiles['existing'] = sentinel
        forest.landscape_families['existing'] = sentinel
        calls = [lambda: compute_measurement_profiles(forest, forbidden, 'existing'),
                 lambda: forest.compute_measurement_profiles(forbidden, 'existing'),
                 lambda: forest.compute_barcode_functionals(forbidden, 'existing'),
                 lambda: _build_step_function_data(forest, object(), forbidden)]
        for cache in (False, True):
            calls.append(lambda cache=cache: forest.compute_measurement_landscapes(
                forbidden, 'existing', cache=cache, cache_profiles=cache))
        for call in calls:
            with self.assertRaisesRegex(RuntimeError, 'call compute_barcode'):
                call()
            self.assertEqual(forest.measurement_profiles, {'existing': sentinel})
            self.assertEqual(forest.landscape_families, {'existing': sentinel})
        forbidden.assert_not_called()
        self.assertFalse(forest._barcode_computed)

    def test_numerical_result_is_independent_of_subsequent_source_state(self):
        forest = profile_forest([(0, 4, [(0, 4, 1)])])
        profiles = compute_measurement_profiles(forest, measured, 'one', cache=False)
        forest._barcode_computed = False
        forest.barcode[0].death = 100
        with patch.object(forest, 'iter_bar_cycle_reps', side_effect=AssertionError('Must not reconstruct')):
            self.assertEqual(landscape_kernel(profiles, 0)(2), 2)
            family = landscapes_from_profiles(profiles, x_grid=[0, 2, 4], max_k=1)
        np.testing.assert_array_equal(family.evaluate_on_grid(), [[0, 2, 0]])
        with self.assertRaisesRegex(RuntimeError, 'call compute_barcode'):
            profiles.require_source(forest, forest.barcode[0])

    def test_delayed_computation_enables_measurement_entry_points(self):
        forest = build(triangle(), compute_barcode=False)
        forest.compute_barcode()
        profiles = forest.compute_measurement_profiles(lambda *_: 1, 'one')
        self.assertTrue(profiles.bars)
        family = forest.compute_measurement_landscapes(lambda *_: 1, 'one')
        self.assertTrue(np.any(family.evaluate_on_grid() > 0))

    def test_computed_empty_and_generic_empty_remain_valid(self):
        for forest in (PersistenceForest(np.empty((0, 2))), profile_forest([])):
            forbidden = Mock(side_effect=AssertionError('Empty barcode needs no measurements'))
            profiles = compute_measurement_profiles(forest, forbidden, 'empty')
            family = landscapes_from_profiles(profiles, x_grid=[0, 1], max_k=2)
            self.assertEqual(profiles.bars, ())
            np.testing.assert_array_equal(family.evaluate_on_grid(), np.zeros((2, 2)))
            forbidden.assert_not_called()
