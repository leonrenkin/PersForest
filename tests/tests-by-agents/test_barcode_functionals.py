"""P1–P5/W1–W4: half-open profiles, row identity, caching and signedness."""
from copy import deepcopy
from unittest.mock import Mock
import unittest
import numpy as np
from persforest.forest_landscapes import StepFunctionData, compute_barcode_functionals, _build_step_function_data
from persforest.cycle_rep_vectorisations import signed_chain_area, signed_chain_edge_length
from fixtures_small_filtrations import split, fan, build, profile_forest, measured, reps


class BarcodeFunctionalTests(unittest.TestCase):
    def test_f2_exact_step_arrays_and_endpoint_samples(self):
        grid = np.arange(9, dtype=float)
        forest = build(split())
        profiles = compute_barcode_functionals(forest, signed_chain_area, 'area')
        for index, bar in enumerate(profiles.bars):
            step = profiles[bar]
            self.assertIs(step, profiles[index])
            self.assertIs(step, profiles.get(bar))
            if bar.birth == 1:
                expected = ([1, 3], [3, 7], [6, 2], [0, 6, 6, 2, 2, 2, 2, 0, 0])
            else:
                expected = ([3], [5], [4], [0, 0, 0, 4, 4, 0, 0, 0, 0])
            for actual, value in zip((step.starts, step.ends, step.vals, step.eval_on_grid(grid)), expected):
                np.testing.assert_array_equal(actual, value)
            self.assertEqual(step.domain, (bar.birth, bar.death))
            self.assertEqual(step.metadata['bar_birth'], bar.birth)
            self.assertEqual(step.metadata['bar_death'], bar.death)
            np.testing.assert_array_equal(profiles.evaluate_on_grid(grid, bars=[bar, index]), [expected[-1]] * 2)

    def test_birth_transition_death_and_duplicate_grid_coordinates(self):
        forest = profile_forest([(1, 7, [(1, 3, 6), (3, 7, 2)])])
        step = _build_step_function_data(forest, forest.barcode[0], measured)
        grid = np.array([np.nextafter(1, -np.inf), 1, np.nextafter(1, np.inf),
                         np.nextafter(3, -np.inf), 3, 3, np.nextafter(3, np.inf),
                         np.nextafter(7, -np.inf), 7, np.nextafter(7, np.inf)])
        np.testing.assert_array_equal(step.eval_on_grid(grid), [0, 6, 6, 6, 2, 2, 2, 2, 0, 0])
        np.testing.assert_array_equal(step.eval_on_grid(np.array([3.])), [2])

    def test_baseline_empty_representatives_and_empty_barcode_shapes(self):
        forest = profile_forest([(1, 2, []), (3, 5, [(3, 5, 4)])])
        profiles = compute_barcode_functionals(forest, measured, 'custom', baseline=9)
        np.testing.assert_array_equal(profiles.evaluate_on_grid(np.arange(7.)),
                                      [[9] * 7, [9, 9, 9, 4, 4, 9, 9]])
        self.assertEqual(profiles[0].domain, (1, 2))
        self.assertEqual(profiles[0].starts.size, 0)
        empty = compute_barcode_functionals(profile_forest([]), measured, 'empty')
        self.assertEqual(empty.evaluate_on_grid(np.array([0., 1.])).shape, (0, 2))

    def test_spy_is_called_once_per_representative_without_mutation(self):
        forest = build(fan())
        snapshots = [deepcopy(bar.cycle_reps) for bar in forest.barcode]
        spy = Mock(side_effect=signed_chain_edge_length)
        profiles = compute_barcode_functionals(forest, spy, 'length')
        self.assertEqual(spy.call_count, sum(len(b.cycle_reps) for b in forest.barcode))
        self.assertTrue(all(call.args[1] is forest.point_cloud for call in spy.call_args_list))
        self.assertEqual(snapshots, [b.cycle_reps for b in forest.barcode])
        for bar in profiles.bars:
            np.testing.assert_allclose(profiles[bar].vals,
                [signed_chain_edge_length(r, forest.point_cloud) for r in bar.cycle_reps])

    def test_zero_and_numpy_scalar_measurements(self):
        forest = profile_forest([(1, 3, [(1, 2, 4), (2, 3, 7)])])
        for func, expected in [(lambda *_: 0, [0, 0]), (lambda r, _: np.float64(r.value), [4, 7])]:
            profile = compute_barcode_functionals(forest, func, 'test')[0]
            np.testing.assert_array_equal(profile.vals, expected)
            self.assertEqual(profile.vals.dtype, np.dtype(float))

    def test_filters_are_inclusive_cache_is_explicit_and_label_overwrites(self):
        forest = build(split())
        uncached = compute_barcode_functionals(forest, signed_chain_area, 'area', cache=False)
        self.assertEqual(forest.barcode_functionals, {})
        saved = compute_barcode_functionals(forest, signed_chain_area, 'area', min_bar_length=2)
        self.assertIs(forest.barcode_functionals['area'], saved)
        self.assertEqual(len(saved.bars), 2)
        only_long = compute_barcode_functionals(forest, signed_chain_area, 'area', min_bar_length=np.nextafter(2, np.inf))
        self.assertIs(forest.barcode_functionals['area'], only_long)
        self.assertEqual([(b.birth, b.death) for b in only_long.bars], [(1, 7)])
        self.assertEqual(len(uncached.bars), 2)

    def test_missing_bars_indices_and_invalid_grids(self):
        forest = profile_forest([(1, 2, [(1, 2, 1)])])
        profiles = compute_barcode_functionals(forest, measured, 'test')
        with self.assertRaises(ValueError):
            _build_step_function_data(forest, object(), measured)
        for reference in (5, object()):
            with self.assertRaises(KeyError):
                profiles[reference]
        for grid in (np.array([]), np.array([2., 1.]), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                profiles.evaluate_on_grid(grid)
            with self.assertRaises(ValueError):
                profiles[0].eval_on_grid(grid)
        malformed = StepFunctionData(np.array([0., 1.]), np.array([1.]), np.array([1., 2.]), 0, (0, 2))
        with self.assertRaises(ValueError):
            malformed.eval_on_grid(np.array([.5]))

    def test_diff_only_profile_order_matches_literal_intervals(self):
        forest = build(fan(), diff_only_mode=True, keep_simplex_diff=True)
        profiles = compute_barcode_functionals(forest, signed_chain_area, 'area')
        bar = next(b for b in forest.barcode if b.birth == 1)
        np.testing.assert_array_equal(profiles[bar].starts, [1, 2, 3, 4])
        np.testing.assert_array_equal(profiles[bar].ends, [2, 3, 4, 7])
        np.testing.assert_array_equal(profiles[bar].vals, [8, 8, 6, 2])
        self.assertEqual([r.active_start for r in forest.iter_bar_cycle_reps(bar)], [4, 3, 2, 1])

    def test_infinite_bar_profile_at_finite_times(self):
        forest = profile_forest([(1, np.inf, [(1, np.inf, 2)])])
        profiles = compute_barcode_functionals(forest, measured, 'infinite')
        np.testing.assert_array_equal(profiles.evaluate_on_grid([0, 1, 100]), [[0, 2, 2]])
