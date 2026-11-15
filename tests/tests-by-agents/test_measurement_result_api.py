"""Signedness defaults, forest-owned caches, zero kernels and explicit family plotting."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from persforest.landscapes.family import landscapes_from_profiles
from persforest.landscapes.kernels import landscape_kernel
from persforest.landscapes.functions import StepFunctionData
from persforest.landscapes.profiles import MeasurementProfiles, ProfileBar, compute_measurement_profiles
from persforest.measurements.basic import constant_one_functional, signed_chain_edge_length
from fixtures_small_filtrations import build, fan, split, profile_forest, measured


def _profiles(*specs):
    """Synthetic profiles from ``(birth, death, value)`` with one interval per bar."""
    bars = tuple(ProfileBar(i, b, d) for i, (b, d, _) in enumerate(specs))
    steps = {i: StepFunctionData([b], [d], [v], 0.0, (b, d)) for i, (b, d, v) in enumerate(specs)}
    return MeasurementProfiles('synthetic', 'values', 0.0, bars, steps)


class ZeroKernelDomainTests(unittest.TestCase):
    def test_zero_kernel_domain_matches_nonzero_kernel_support(self):
        profiles = _profiles((1.0, 2.0, 0.0), (1.0, 2.0, 1.0))
        self.assertEqual(landscape_kernel(profiles, 0, mode='raw').domain, (2.0, 4.0))
        self.assertEqual(landscape_kernel(profiles, 1, mode='raw').domain, (2.0, 4.0))
        for bar_id in (0, 1):
            self.assertEqual(landscape_kernel(profiles, bar_id, mode='pyramid').domain, (1.0, 2.0))
        self.assertEqual(landscape_kernel(profiles, 0, mode='raw')(3.0), 0.0)

    def test_raw_inferred_grid_is_not_widened_by_zero_profiles(self):
        zero_only = landscapes_from_profiles(_profiles((1.0, 2.0, 0.0)), mode='raw', num_grid_points=3)
        np.testing.assert_array_equal(zero_only.x_grid, [2.0, 3.0, 4.0])
        mixed = landscapes_from_profiles(_profiles((0.1, 0.2, 0.0), (0.5, 1.0, 1.0)),
                                         mode='raw', num_grid_points=5)
        self.assertEqual(mixed.x_grid[0], 0.2)
        self.assertEqual(mixed.x_grid[-1], 2.0)


class SignednessTests(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_animation_defaults_to_unsigned_and_records_requested_convention(self):
        import persforest.plotting.animation.measurements as module
        forest = build(fan())
        results = []

        def capture(*args, **kwargs):
            results.append(compute_measurement_profiles(*args, **kwargs))
            return results[-1]
        with patch.object(module, 'compute_measurement_profiles', side_effect=capture):
            for kwargs, expected in (({}, False), ({'signed': False}, False), ({'signed': True}, True)):
                with self.subTest(kwargs=kwargs):
                    anim, fig = forest.animate_barcode_measurement(signed_chain_edge_length, frames=2, **kwargs)
                    fig.canvas.draw()
                    self.assertIs(results[-1].signed, expected)
        long = next(bar for bar in forest.barcode if bar.birth == 1)
        reference = {signed: forest.compute_measurement_profiles(
            signed_chain_edge_length, str(signed), signed=signed, cache=False)[long].vals for signed in (False, True)}
        self.assertFalse(np.array_equal(reference[False], reference[True]))
        for result, signed in zip(results, (False, False, True)):
            np.testing.assert_array_equal(result[long].vals, reference[signed])

    def test_forest_animation_reuses_explicit_profiles(self):
        forest = build(split())
        profiles = forest.compute_measurement_profiles(constant_one_functional, 'one')
        with patch.object(forest, 'iter_bar_cycle_reps', side_effect=AssertionError('Must reuse profiles')):
            anim, fig = forest.animate_barcode_measurement(profiles=profiles, frames=2, t_max=7)
            fig.canvas.draw()
        with self.assertRaises(ValueError):
            forest.animate_barcode_measurement(constant_one_functional, profiles=profiles, frames=2)

    def test_single_bar_plot_uses_the_profile_convention(self):
        forest = build(fan())
        long = next(bar for bar in forest.barcode if bar.birth == 1)
        for signed in (False, True):
            with self.subTest(signed=signed):
                expected = forest.compute_measurement_profiles(
                    signed_chain_edge_length, 'x', signed=signed, cache=False)[long]
                kwargs = {} if not signed else {'signed': True}
                _, step = forest.plot_barcode_measurement(signed_chain_edge_length, bar=long, **kwargs)
                np.testing.assert_array_equal(step.vals, expected.vals)
        with self.assertRaises(TypeError):
            forest.plot_barcode_measurement(signed_chain_edge_length, signed='no', bar=long)


class ForestOwnedCacheTests(unittest.TestCase):
    def test_numerical_profile_function_never_touches_a_registry(self):
        self.assertNotIn('cache', inspect.signature(compute_measurement_profiles).parameters)
        forest = profile_forest([(0, 4, [(0, 4, 1)])])
        del forest.measurement_profiles
        profiles = compute_measurement_profiles(forest, measured, 'one', signed=True)
        self.assertFalse(hasattr(forest, 'measurement_profiles'))
        self.assertEqual(profiles[0].vals.tolist(), [1])

    def test_forest_method_publishes_only_on_request_and_success(self):
        forest = build(split())
        with self.assertRaisesRegex(TypeError, 'cache'):
            forest.compute_measurement_profiles(constant_one_functional, 'one', cache=None)
        self.assertEqual(forest.measurement_profiles, {})
        with self.assertRaises(ValueError):
            forest.compute_measurement_profiles(lambda *_: 1j, 'one')
        self.assertEqual(forest.measurement_profiles, {})
        profiles = forest.compute_measurement_profiles(constant_one_functional, 'one')
        self.assertIs(forest.measurement_profiles['one'], profiles)

    def test_family_does_not_copy_the_source_barcode_list(self):
        forest = build(split())
        profiles = forest.compute_measurement_profiles(constant_one_functional, 'one')
        family = landscapes_from_profiles(profiles, max_k=1)
        self.assertIn('source_barcode', profiles.extra_meta)
        self.assertNotIn('source_barcode', family.extra_meta)
        for key in ('min_bar_length', 'filtration_tol', 'reduced', 'signed', 'measurement_name'):
            self.assertIn(key, family.extra_meta)


class ExplicitFamilyPlottingTests(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_forest_plots_accept_family_results_and_labels(self):
        forest = build(split())
        grid = np.linspace(0, 8, 17)
        cached = forest.compute_measurement_landscapes(constant_one_functional, 'cached', x_grid=grid, max_k=2)
        loose = forest.compute_measurement_landscapes(
            signed_chain_edge_length, 'loose', x_grid=grid, max_k=2, cache=False)
        self.assertNotIn('loose', forest.landscape_families)
        ax = forest.plot_measurement_landscapes(loose, show=False)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), loose.landscapes[1])
        ax = forest.plot_landscape_comparison_between_measurements(['cached', loose])
        self.assertEqual(len(ax.lines), 2)
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), loose.landscapes[1])
        self.assertEqual([line.get_label() for line in ax.lines], ['cached', 'loose'])
        with self.assertRaisesRegex(KeyError, 'available labels'):
            forest.plot_measurement_landscapes('missing', show=False)


class PublicExportTests(unittest.TestCase):
    def test_explicit_composition_is_importable_from_the_landscapes_package(self):
        import persforest.landscapes as landscapes
        self.assertIs(landscapes.landscapes_from_profiles, landscapes_from_profiles)
        self.assertIs(landscapes.landscape_kernel, landscape_kernel)
        self.assertIs(landscapes.MeasurementProfiles, MeasurementProfiles)
        for name in landscapes.__all__:
            self.assertTrue(hasattr(landscapes, name), name)


if __name__ == '__main__':
    unittest.main()
