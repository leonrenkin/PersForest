"""Explicit measurement-result composition, ownership and failure contracts."""
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import runpy
import unittest
from unittest.mock import Mock, patch

import numpy as np
from persforest import PersistenceForest
from persforest.landscapes.family import MeasurementLandscapeFamily, landscapes_from_profiles
from persforest.landscapes.functions import StepFunctionData, PiecewiseLinearFunction
from persforest.landscapes.kernels import convolve_profile, landscape_kernel
from persforest.landscapes.profiles import compute_measurement_profiles, MeasurementProfiles, ProfileBar
from persforest.measurements.basic import constant_one_functional, signed_chain_edge_length
from fixtures_small_filtrations import build, split, fan, disconnected, profile_forest, measured
from oracle_landscapes import landscape


class ProfileCompositionTests(unittest.TestCase):
    def test_landscape_levels_are_samples_and_cannot_interpolate(self):
        profiles = MeasurementProfiles('synthetic', 'one', 0, (ProfileBar(0, 0, 4),),
            {0: StepFunctionData([0], [4], [1], 0, (0, 4))})
        family = landscapes_from_profiles(profiles, x_grid=[0, 4], max_k=1)
        samples = family.landscapes[1]
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, family.x_grid.shape)
        np.testing.assert_array_equal(samples, [0, 0])
        self.assertFalse(callable(samples))
        with self.assertRaises(TypeError):
            samples(2)
        with self.assertRaisesRegex(ValueError, 'Recompute landscapes'):
            family.evaluate_on_grid([2])
        self.assertEqual(family.bar_kernels[0](2), 2)
        refined = landscapes_from_profiles(profiles, x_grid=[0, 2, 4], max_k=1)
        np.testing.assert_array_equal(refined.evaluate_on_grid([2]), [[2]])

    def test_sample_constructor_validates_alignment_and_owns_values(self):
        values = np.array([1., 2., 3.])
        family = MeasurementLandscapeFamily('synthetic', 'samples', 'pyramid',
            [0, 1, 2], {}, {1: values})
        values[1] = 99
        np.testing.assert_array_equal(family.landscapes[1], [1, 2, 3])
        with self.assertRaises(ValueError):
            family.landscapes[1][1] = 99
        for invalid in ([0, 1], [[0, 1, 2]], [0, np.nan, 2], [0, np.inf, 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MeasurementLandscapeFamily('synthetic', 'bad', 'pyramid',
                    [0, 1, 2], {}, {1: invalid})

    def test_signed_false_defaults_match_across_measurement_entry_points(self):
        forest = build(fan())
        standalone = compute_measurement_profiles(forest, signed_chain_edge_length, 'standalone', cache=False)
        unsigned = forest.compute_measurement_profiles(signed_chain_edge_length, 'unsigned', signed=False, cache=False)
        default = forest.compute_measurement_profiles(signed_chain_edge_length, 'default', cache=False)
        legacy = forest.compute_barcode_functionals(signed_chain_edge_length, 'legacy', cache=False)
        signed = compute_measurement_profiles(forest, signed_chain_edge_length, 'signed', signed=True, cache=False)
        for result in (standalone, unsigned, default, legacy):
            self.assertFalse(result.signed)
            for bar in forest.barcode:
                np.testing.assert_array_equal(result[bar].vals, unsigned[bar].vals)
        long = next(bar for bar in forest.barcode if bar.birth == 1)
        self.assertTrue(signed.signed)
        np.testing.assert_allclose(signed[long].vals - unsigned[long].vals,
                                   [0, 2 * np.sqrt(2), 0, 0], atol=1e-12)
        grid = np.linspace(0, 8, 65)
        final = forest.compute_measurement_landscapes(signed_chain_edge_length, 'final', x_grid=grid, cache=False)
        np.testing.assert_array_equal(final.evaluate_on_grid(),
            landscapes_from_profiles(standalone, x_grid=grid).evaluate_on_grid())
        self.assertFalse(final.extra_meta['signed'])

    def test_arrays_are_owned_and_read_only_at_every_result_boundary(self):
        starts, ends, vals = np.array([0.]), np.array([4.]), np.array([1.])
        profile = StepFunctionData(starts, ends, vals, 0, (0, 4))
        starts[0], ends[0], vals[0] = 1, 3, 9
        np.testing.assert_array_equal(profile.eval_on_grid([0, 1, 3, 4]), [1, 1, 1, 0])
        profiles = MeasurementProfiles('synthetic', 'one', 0, (ProfileBar(0, 0, 4),), {0: profile})
        grid = np.arange(5.)
        family = landscapes_from_profiles(profiles, x_grid=grid, max_k=1)
        grid[1] = 1.5
        np.testing.assert_array_equal(family.evaluate_on_grid([1]), [[1]])
        with self.assertRaisesRegex(ValueError, 'subset'):
            family.evaluate_on_grid([1.5])
        for array in (profile.starts, profile.ends, profile.vals, family.x_grid,
                      family.landscapes[1],
                      family.bar_kernels[0].xs, family.bar_kernels[0].ys):
            with self.subTest(array=array), self.assertRaises(ValueError):
                array[0] = 99
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        knots, values = np.array([0., 1.]), np.array([0., 2.])
        linear = PiecewiseLinearFunction(knots, values, (0, 1))
        values[1] = 20
        self.assertEqual(linear(.5), 1)

    def test_containers_and_endpoint_snapshots_cannot_desynchronize(self):
        forest = profile_forest([(0, 4, [(0, 4, 1)]), (1, 5, [(1, 5, 3)])])
        source = forest.barcode[0]
        profiles = compute_measurement_profiles(forest, measured, 'values', signed=True)
        with self.assertRaises(FrozenInstanceError):
            profiles.bars = tuple(reversed(profiles.bars))
        with self.assertRaises(FrozenInstanceError):
            profiles.bars[0].death = 100
        with self.assertRaises(TypeError):
            profiles.step_functions[0] = profiles[1]
        source.death = 100
        source.cycle_reps[0].value = 100
        forest.barcode.reverse()
        self.assertEqual(profiles.bars[0].death, 4)
        self.assertEqual(profiles[source].vals.tolist(), [1])
        self.assertEqual(landscape_kernel(profiles, source)(2), 2)
        with self.assertRaisesRegex(ValueError, 'do not match'):
            profiles.require_source(forest, source)

    def test_selection_order_identity_missing_and_duplicate_ids(self):
        forest = profile_forest([(0, 4, [(0, 4, 1)]), (0, 4, [(0, 4, 3)])])
        profiles = compute_measurement_profiles(forest, measured, 'values', signed=True, cache=False)
        selected = landscapes_from_profiles(profiles, bar_ids=[1, 0], x_grid=[0, 2, 4], max_k=3)
        self.assertEqual(selected.bar_ids, (1, 0))
        self.assertEqual(tuple(selected.bar_kernels), (1, 0))
        np.testing.assert_array_equal(selected.evaluate_on_grid(), [[0, 6, 0], [0, 2, 0], [0, 0, 0]])
        foreign = compute_measurement_profiles(forest, measured, 'other', signed=True, cache=False).bars[0]
        for selection in ([2], [-1], [True], [0.5], [foreign], [object()]):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                landscapes_from_profiles(profiles, bar_ids=selection)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            landscapes_from_profiles(profiles, bar_ids=[0, profiles.bars[0]])
        np.testing.assert_array_equal(profiles.evaluate_on_grid([2], bars=[1, 0, 1]), [[3], [1], [3]])

    def test_equal_endpoint_forest_bars_keep_rank_multiplicity(self):
        forest = build(disconnected())
        profiles = forest.compute_measurement_profiles(constant_one_functional, 'one')
        family = landscapes_from_profiles(profiles, x_grid=[2, 4, 6], max_k=3)
        self.assertEqual(len(profiles.bars), 2)
        np.testing.assert_array_equal(family.evaluate_on_grid(), [[0, 2, 0], [0, 2, 0], [0, 0, 0]])

    def test_invalid_requests_do_not_measure_or_replace_cache_entries(self):
        forest = build(split())
        sentinel = object()
        forest.measurement_profiles['one'] = sentinel
        forest.landscape_families['one'] = sentinel
        forbidden = Mock(side_effect=AssertionError('Invalid requests must not measure'))
        for options in ({'mode': 'typo'}, {'max_k': True}, {'max_k': 1.5},
                        {'x_grid': [[0, 1]]}, {'tol': -1}, {'tol': np.nan},
                        {'tol': np.inf}, {'cache_profiles': None},
                        {'signed': 'false'}, {'measurement_params': []}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                forest.compute_measurement_landscapes(forbidden, 'one', **options)
        forbidden.assert_not_called()
        self.assertIs(forest.measurement_profiles['one'], sentinel)
        self.assertIs(forest.landscape_families['one'], sentinel)

    def test_direct_invalid_modes_tolerances_and_levels_raise(self):
        forest = profile_forest([(0, 4, [(0, 4, 1)])])
        profiles = compute_measurement_profiles(forest, measured, 'one', signed=True)
        for mode in ('typo', None):
            with self.assertRaisesRegex(ValueError, 'mode'):
                landscape_kernel(profiles, 0, mode=mode)
        for tol in (-1, np.nan, np.inf, True):
            with self.assertRaisesRegex(ValueError, 'tol'):
                landscape_kernel(profiles, 0, tol=tol)
        family = landscapes_from_profiles(profiles, x_grid=[0, 2, 4])
        for levels in (True, np.bool_(False), 1.5, [1.5], ['1'], [True]):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                family.evaluate_on_grid(levels=levels)
        np.testing.assert_array_equal(family.evaluate_on_grid(levels=np.int64(1)), [[0, 2, 0]])

    def test_invalid_profile_structure_and_source_endpoints_raise(self):
        for starts, ends, vals in (([0, 1], [2, 3], [1, 2]),
                                   ([2, 0], [3, 1], [1, 2]),
                                   ([1], [0], [1]), ([0], [1, 2], [1]),
                                   ([[0]], [1], [1]), ([np.nan], [1], [1]),
                                   ([-1], [1], [1])):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                StepFunctionData(starts, ends, vals, 0, (0, 3))
        forbidden = Mock(side_effect=AssertionError('Invalid source must not measure'))
        for endpoints in ((np.nan, 1), (2, 1)):
            forest = profile_forest([(*endpoints, [])])
            with self.assertRaises(ValueError):
                compute_measurement_profiles(forest, forbidden, 'bad', signed=True, min_bar_length=100)
        forbidden.assert_not_called()

    def test_gaps_and_zero_width_records_preserve_half_open_values(self):
        profile = StepFunctionData([0, 1, 3], [1, 1, 4], [2, 99, 5], -1, (0, 4))
        np.testing.assert_array_equal(profile.eval_on_grid([-1, 0, 1, 2, 3, 4]), [-1, 2, -1, -1, 5, -1])
        zero = StepFunctionData([0, 1, 3], [1, 1, 4], [2, 99, 5], 0, (0, 4))
        specs = [(0, 4, [(0, 1, 2), (1, 1, 99), (3, 4, 5)])]
        profiles = MeasurementProfiles('synthetic', 'gaps', 0, (ProfileBar(0, 0, 4),), {0: zero})
        grid = np.linspace(-1, 5, 49)
        np.testing.assert_allclose(landscapes_from_profiles(profiles, x_grid=grid, max_k=1).evaluate_on_grid(), landscape(grid, specs, 1))

    def test_nonzero_baseline_is_never_silently_discarded(self):
        for specs in ([], [(0, 4, [(0, 4, 1)])]):
            forest = profile_forest(specs)
            profiles = compute_measurement_profiles(forest, measured, 'baseline', signed=True, baseline=9)
            with self.assertRaisesRegex(ValueError, 'baseline=0'):
                landscapes_from_profiles(profiles)
            if specs:
                self.assertEqual(profiles[0].eval_on_grid([-1])[0], 9)
                with self.assertRaisesRegex(ValueError, 'baseline=0'):
                    landscape_kernel(profiles, 0)
                with self.assertRaisesRegex(ValueError, 'baseline=0'):
                    convolve_profile(profiles[0], 0, 4)

    def test_metadata_is_captured_and_propagated_without_inference(self):
        forest = build(split())
        params = {'scale': 2, 'options': [1, {'normalize': False}]}
        profiles = forest.compute_measurement_profiles(signed_chain_edge_length, 'length',
            signed=True, measurement_name='edge length', measurement_params=params)
        params['options'][1]['normalize'] = True
        family = landscapes_from_profiles(profiles, bar_ids=[0], max_k=1, mode='raw', tol=0)
        self.assertFalse(profiles.measurement_params['options'][1]['normalize'])
        self.assertTrue(family.extra_meta['signed'])
        self.assertEqual(family.extra_meta['measurement_name'], 'edge length')
        self.assertEqual(family.extra_meta['tol'], 0)
        self.assertEqual(family.forest_id, profiles.forest_id)
        self.assertEqual(len(profiles.extra_meta['source_barcode']), len(forest.barcode))
        with self.assertRaises(TypeError):
            profiles.measurement_params['scale'] = 3

    def test_legacy_entry_point_is_only_a_small_forest_delegate(self):
        forest = build(split())
        with patch.object(forest, 'compute_measurement_profiles', return_value='result') as canonical:
            self.assertEqual(forest.compute_barcode_functionals(constant_one_functional, 'one', signed=True), 'result')
        canonical.assert_called_once_with(constant_one_functional, 'one', signed=True)
        import persforest.landscapes.profiles as module
        self.assertFalse(hasattr(module, 'BarcodeFunctionals'))
        self.assertFalse(hasattr(module, 'compute_barcode_functionals'))
        self.assertFalse(hasattr(forest, 'barcode_functionals'))

    def test_successful_empty_results_publish_both_requested_caches(self):
        for forest, threshold in ((PersistenceForest(np.empty((0, 2))), 0), (build(split()), 100)):
            forbidden = Mock(side_effect=AssertionError('No bars selected'))
            family = forest.compute_measurement_landscapes(forbidden, 'empty',
                min_bar_length=threshold, cache_profiles=True, max_k=2, x_grid=[0, 1])
            self.assertIs(forest.landscape_families['empty'], family)
            self.assertEqual(forest.measurement_profiles['empty'].bars, ())
            np.testing.assert_array_equal(family.evaluate_on_grid(), [[0, 0], [0, 0]])
            forbidden.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('pandas'), 'Regression example needs the examples extra')
    def test_regression_two_grids_match_independent_rank_oracle(self):
        path = Path(__file__).resolve().parents[2] / 'research/tendril_regression/featurize_point_clouds.py'
        function = runpy.run_path(str(path))['featurize_point_cloud']
        theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        points = np.column_stack((np.cos(theta), np.sin(theta)))
        short, first = np.linspace(0, .4, 11), np.linspace(0, 1, 51)
        actual = function(points, constant_one_functional, False, 3, short, first)
        forest = PersistenceForest(points)
        specs = [(bar.birth, bar.death, [(bar.birth, bar.death, 1)]) for bar in forest.barcode]
        expected = np.concatenate((landscape(first, specs, 1).ravel(), landscape(short, specs, 3)[1:].ravel()))
        self.assertEqual(actual.shape, (73,))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        with patch.dict(function.__globals__, {'PersistenceForest': Mock(side_effect=AssertionError('Must validate grids first'))}):
            with self.assertRaises(ValueError):
                function(points, constant_one_functional, False, 3, short[::-1], first)


class ProfileRenderingTests(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_explicit_family_plot_and_comparison_fail_before_artists(self):
        import matplotlib.pyplot as plt
        from persforest.plotting.landscapes import plot_landscape_family, plot_landscape_comparison
        profiles = compute_measurement_profiles(profile_forest([(0, 4, [(0, 4, 1)])]), measured, 'one', signed=True)
        family = landscapes_from_profiles(profiles, max_k=1, x_grid=[0, 2, 4])
        ax = plot_landscape_family(family, show=False)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 2, 0])
        plt.close('all')
        with self.assertRaises(ValueError):
            plot_landscape_comparison([family], k=2)
        with self.assertRaises(ValueError):
            plot_landscape_comparison([family], labels=[])
        self.assertEqual(plt.get_fignums(), [])

    def test_animation_reuses_profiles_and_checks_source(self):
        import matplotlib.pyplot as plt
        from persforest.plotting.animation.measurements import animate_barcode_measurement_generic
        forest = build(split())
        bar = forest.max_bar()
        profiles = forest.compute_measurement_profiles(constant_one_functional, 'one')
        with patch.object(forest, 'iter_bar_cycle_reps', side_effect=AssertionError('Must reuse profiles')):
            anim, fig = animate_barcode_measurement_generic(forest, bar=bar, profiles=profiles, frames=2, t_max=7)
            fig.canvas.draw()
        plt.close(fig)
        for options in ({'cycle_func': constant_one_functional, 'profiles': profiles},
                        {'profiles': profiles, 'frames': 0}, {'profiles': profiles, 't_max': np.inf}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                animate_barcode_measurement_generic(forest, bar=bar, **options)
        foreign = build(split())
        with self.assertRaises(ValueError):
            animate_barcode_measurement_generic(foreign, profiles=profiles, frames=2, t_max=7)
        self.assertEqual(plt.get_fignums(), [])
