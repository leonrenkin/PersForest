"""K1–K7: independent interval-overlap integration and exact knot values."""
import unittest
import numpy as np
from persforest.forest_landscapes import (
    _build_convolution_with_indicator, compute_convolution_kernel_for_bar,
    compute_landscape_kernel_for_bar, compute_measurement_interval_landscape,
    compute_barcode_functionals,
)
from fixtures_small_filtrations import profile_forest, measured
from oracle_landscapes import kernel, event_grid


class LandscapeKernelTests(unittest.TestCase):
    def test_constant_profiles_reproduce_tents_and_raw_scaling(self):
        for birth, death in [(2, 6), (-3, 1), (0, 2)]:
            for value in (0, 1, 6):
                forest = profile_forest([(birth, death, [(birth, death, value)])])
                bar = forest.barcode[0]
                pyramid = compute_landscape_kernel_for_bar(forest, bar, measured)
                raw = compute_landscape_kernel_for_bar(forest, bar, measured, mode='raw')
                grid = np.linspace(birth - 1, death + 1, 49)
                expected = value * np.maximum(0, np.minimum(grid - birth, death - grid))
                np.testing.assert_allclose(pyramid(grid), expected, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(raw(2 * grid), 2 * expected, atol=1e-12, rtol=1e-12)
                self.assertEqual(pyramid((birth + death) / 2), value * (death - birth) / 2)

    def test_piecewise_area_profile_has_hand_derived_knots_slopes_and_integral(self):
        forest = profile_forest([(1, 7, [(1, 3, 6), (3, 7, 2)]), (3, 5, [(3, 5, 4)])])
        for index, xs, ys, integral in [(0, [1, 2, 4, 5, 7], [0, 6, 10, 4, 0], 30),
                                        (1, [3, 4, 5], [0, 4, 0], 4)]:
            actual = compute_landscape_kernel_for_bar(forest, forest.barcode[index], measured)
            np.testing.assert_allclose(actual(xs), ys, atol=1e-12, rtol=1e-12)
            self.assertAlmostEqual(np.trapezoid(actual.ys, actual.xs), integral)
            if index == 0:
                np.testing.assert_allclose(np.diff(actual(xs)) / np.diff(xs), [6, 2, -6, -2])

    def test_overlap_oracle_at_all_events_midpoints_and_outside(self):
        specs = [(1, 7, [(1, 3, 6), (3, 7, 2)]),
                 (0, 4, [(0, 1, 2), (1, 2, 0), (2, 4, 4)]),
                 (-2, 3, [(-2, -.5, 3), (-.5, 2, 1), (2, 3, 5)])]
        for birth, death, records in specs:
            forest = profile_forest([(birth, death, records)])
            for mode in ('raw', 'pyramid'):
                grid = event_grid(birth, death, records)
                if mode == 'raw':
                    grid *= 2
                actual = compute_landscape_kernel_for_bar(forest, forest.barcode[0], measured, mode=mode)
                expected = [kernel(t, birth, death, records, raw=mode == 'raw') for t in grid]
                np.testing.assert_allclose(actual(grid), expected, atol=1e-12, rtol=1e-12)
                self.assertTrue(np.all(actual(grid) >= -1e-12))
                for knot in actual.xs:
                    self.assertAlmostEqual(actual(np.nextafter(knot, -np.inf)), actual(np.nextafter(knot, np.inf)))
                scale = 1 if mode == 'raw' else .25
                expected_integral = scale * (death - birth) * sum(v * (e - s) for s, e, v in records)
                self.assertAlmostEqual(np.trapezoid(actual.ys, actual.xs), expected_integral)

    def test_splitting_constant_intervals_and_zero_width_records_do_not_change_kernel(self):
        records = [[(0, 4, 2)], [(0, 1, 2), (1, 2, 2), (2, 4, 2)],
                   [(0, 2, 2), (2, 2, 999), (2, 4, 2)]]
        grid = np.linspace(-1, 5, 65)
        for pieces in records:
            forest = profile_forest([(0, 4, pieces)])
            actual = compute_landscape_kernel_for_bar(forest, forest.barcode[0], measured)
            np.testing.assert_allclose(actual(grid), 2 * np.maximum(0, np.minimum(grid, 4 - grid)))

    def test_empty_zero_and_zero_width_support_are_zero(self):
        for records, birth, death in [([], 1, 2), ([(1, 2, 0)], 1, 2), ([(1, 1, 5)], 1, 1)]:
            forest = profile_forest([(birth, death, records)])
            actual = compute_landscape_kernel_for_bar(forest, forest.barcode[0], measured)
            self.assertTrue(np.isfinite(actual(np.arange(4.))).all())
            np.testing.assert_array_equal(actual(np.arange(4.)), 0)
        h, xs, ys = _build_convolution_with_indicator([], [], [], 1, 2)
        self.assertEqual((h(-5), h(0), h(5), xs, ys), (0, 0, 0, [], []))

    def test_invalid_convolution_inputs(self):
        for args in [([0], [1], [1], 2, 1), ([0, 1], [1], [1], 0, 2),
                     ([2], [1], [1], 0, 2)]:
            with self.assertRaises(ValueError):
                _build_convolution_with_indicator(*args)

    def test_event_quantization_has_separate_bounded_error(self):
        tau = 2. ** -12
        start, end = tau / 8, 1 + tau / 8
        records = [(start, end, 2)]
        grid = np.linspace(-1, 3, 257)
        for tol in (0, tau, 1e-12):
            h, xs, ys = _build_convolution_with_indicator([start], [end], [2], 0, 1, tol=tol)
            error = max(abs(h(t) - kernel(t, 0, 1, records, raw=True)) for t in grid)
            self.assertLessEqual(error, 8 * tol + 1e-12)
            self.assertTrue(np.isfinite(ys).all())
        # Explicitly characterize selected tolerance's deletion of narrow support.
        h, _, _ = _build_convolution_with_indicator([0], [tau / 2], [1], 0, 1, tol=tau)
        self.assertEqual(h(.5), 0)
        h, _, _ = _build_convolution_with_indicator([0], [2 * tau], [1], 0, 1, tol=tau)
        self.assertAlmostEqual(h(.5), 2 * tau)

    def test_legacy_wrapper_is_raw_and_reuses_precomputed_profile(self):
        forest = profile_forest([(2, 6, [(2, 6, 1)])])
        bar = forest.barcode[0]
        profile = compute_barcode_functionals(forest, measured, 'one')[bar]
        def forbidden(*_):
            raise AssertionError('Precomputed profile should not call the functional')
        raw = compute_convolution_kernel_for_bar(forest, bar, forbidden, sf=profile)
        h, xs, ys = compute_measurement_interval_landscape(forest, forbidden, None, sf=profile)
        np.testing.assert_array_equal(xs, [4, 8, 12])
        np.testing.assert_array_equal(ys, [0, 4, 0])
        self.assertEqual(h(8), 4)
        np.testing.assert_allclose(raw(xs), ys)
        with self.assertRaises(ValueError):
            compute_measurement_interval_landscape(forest, measured, object())

    def test_per_bar_linearity_and_filtration_translation(self):
        first, second = [(0, 1, 1), (1, 4, 3)], [(0, 1, 4), (1, 4, 2)]
        together = [(s, e, a + b) for (s, e, a), (_, _, b) in zip(first, second)]
        forest = profile_forest([(0, 4, records) for records in (first, second, together)])
        functions = [compute_landscape_kernel_for_bar(forest, b, measured) for b in forest.barcode]
        grid = np.linspace(-1, 5, 49)
        np.testing.assert_allclose(functions[2](grid), functions[0](grid) + functions[1](grid))
        translated = profile_forest([(7, 11, [(s + 7, e + 7, v) for s, e, v in first])])
        shifted = compute_landscape_kernel_for_bar(translated, translated.barcode[0], measured)
        np.testing.assert_allclose(shifted(grid + 7), functions[0](grid))
