"""L1–L7/W1–W4: sampled rank landscapes and public wrappers."""
from unittest.mock import Mock
import unittest
import numpy as np
from persforest.forest_landscapes import (
    compute_measurement_landscape_family as family, compute_barcode_functionals,
    compute_landscape_kernel_for_bar, compute_convolution_kernel_for_bar,
    PiecewiseLinearFunction,
)
from persforest.cycle_rep_vectorisations import signed_chain_area, signed_chain_edge_length, constant_one_functional
from fixtures_small_filtrations import build, split, fan, disconnected, profile_forest, measured
from oracle_landscapes import landscape


class LandscapeFamilyTests(unittest.TestCase):
    def test_f2_end_to_end_area_and_constant_one(self):
        forest = build(split())
        grid = np.arange(9.)
        for func, expected in [(signed_chain_area, [[0, 0, 6, 8, 10, 4, 2, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]),
                               (constant_one_functional, [[0, 0, 1, 2, 3, 2, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]])]:
            actual = forest.compute_measurement_landscapes(func, 'test', x_grid=grid, max_k=3)
            np.testing.assert_allclose(actual.evaluate_on_grid(grid, levels=3), expected + [[0] * 9])

    def test_rank_crossing_exact_when_grid_includes_crossing(self):
        specs = [(0, 4, [(0, 4, 1)]), (1, 5, [(1, 5, 1)])]
        grid = np.array([0., 1, 2, 2.5, 3, 4, 5])
        actual = family(profile_forest(specs), measured, 'one', x_grid=grid, max_k=3)
        np.testing.assert_allclose(actual.evaluate_on_grid([2, 2.5, 3], levels=2), [[2, 1.5, 2], [1, 1.5, 1]])
        dense = np.linspace(-1, 6, 101)
        np.testing.assert_allclose(actual.evaluate_on_grid(dense, levels=3), landscape(dense, specs, 3))

    def test_coarse_grid_has_documented_interpolation_error_not_exact_envelope(self):
        specs = [(0, 4, [(0, 4, 1)]), (1, 5, [(1, 5, 1)])]
        grid = np.arange(6.)
        actual = family(profile_forest(specs), measured, 'one', x_grid=grid, max_k=2)
        np.testing.assert_allclose(actual.evaluate_on_grid(grid, levels=2), landscape(grid, specs, 2))
        # D7 compatibility: known approximation, not a mathematical oracle.
        np.testing.assert_allclose(actual.evaluate_on_grid([2.5], levels=2), [[2], [1]])
        error = abs(actual.evaluate_on_grid([2.5], levels=2) - landscape([2.5], specs, 2))
        np.testing.assert_allclose(error, .5)  # M*h/2, with M=h=1.

    def test_duplicate_bars_preserve_multiplicity_and_pad_ranks(self):
        forest = build(disconnected())
        grid = np.linspace(1, 7, 25)
        actual = family(forest, constant_one_functional, 'one', x_grid=grid, max_k=4)
        tent = np.maximum(0, np.minimum(grid - 2, 6 - grid))
        np.testing.assert_allclose(actual.evaluate_on_grid(grid, levels=4), [tent, tent, np.zeros(25), np.zeros(25)])
        self.assertEqual(len(actual.bar_kernels), 2)

    def test_rank_oracle_order_sum_and_positive_scaling(self):
        specs = [(0, 4, [(0, 1, 2), (1, 4, 1)]), (1, 5, [(1, 5, 3)]),
                 (-1, 2, [(-1, 0, 4), (0, 2, 2)])]
        grid = np.linspace(-2, 6, 65)
        for mode in ('raw', 'pyramid'):
            actual = family(profile_forest(specs), measured, 'test', x_grid=grid, max_k=5, mode=mode)
            values = actual.evaluate_on_grid(grid, levels=5)
            np.testing.assert_allclose(values, landscape(grid, specs, 5, raw=mode == 'raw'))
            self.assertTrue(np.all(np.diff(values, axis=0) <= 1e-12))
            self.assertTrue(np.all(values >= -1e-12))
            np.testing.assert_allclose(values.sum(axis=0), sum(k(grid) for k in actual.bar_kernels.values()))
            scaled = family(profile_forest(specs), lambda r, p: 3 * measured(r, p), 'scaled', x_grid=grid, max_k=5, mode=mode)
            np.testing.assert_allclose(scaled.evaluate_on_grid(grid, levels=5), 3 * values)

    def test_empty_filtered_and_zero_measurement_families(self):
        for forest, options in [(profile_forest([]), {}),
                                (profile_forest([(1, 3, [(1, 3, 4)])]), {'min_bar_length': 3}),
                                (profile_forest([(1, 3, [(1, 3, 0)])]), {})]:
            actual = family(forest, measured, 'zero', num_grid_points=7, max_k=3, **options)
            self.assertEqual(len(actual.x_grid), 7)
            np.testing.assert_array_equal(actual.evaluate_on_grid(np.linspace(-2, 5, 11), levels=3), np.zeros((3, 11)))
        empty = family(profile_forest([]), measured, 'empty', num_grid_points=7)
        np.testing.assert_array_equal(empty.x_grid, np.linspace(0, 1, 7))

    def test_level_selection_fill_values_and_invalid_evaluation(self):
        actual = family(profile_forest([(0, 4, [(0, 4, 1)])]), measured, 'one', x_grid=np.arange(5.), max_k=2)
        np.testing.assert_allclose(actual.evaluate_on_grid([2], levels=[2, 1, 2, 5], fill_value=9), [[0], [2], [0], [9]])
        self.assertEqual(actual.evaluate_on_grid([2], levels=[]).shape, (0, 1))
        for levels in (0, -1, [0, 1]):
            with self.assertRaises(ValueError):
                actual.evaluate_on_grid([2], levels=levels)
        for grid in ([], [1, 0], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                actual.evaluate_on_grid(grid)
        np.testing.assert_array_equal(actual.evaluate_on_grid([-1, 5], levels=2), np.zeros((2, 2)))

    def test_piecewise_linear_scalar_array_empty_and_outside(self):
        f = PiecewiseLinearFunction(np.array([0., 1, 2]), np.array([0., 2, 0]), (0, 2))
        self.assertIsInstance(f(.5), float)
        self.assertEqual(f(.5), 1)
        np.testing.assert_array_equal(f(np.array([[-1., 0], [1, 3]])), [[0, 0], [2, 0]])
        empty = PiecewiseLinearFunction(np.array([]), np.array([]), (0, 1))
        self.assertEqual(empty(.5), 0)
        np.testing.assert_array_equal(empty([[0, 1]]), [[0, 0]])

    def test_explicit_grid_wins_and_shape_checks(self):
        forest = profile_forest([(0, 4, [(0, 4, 1)])])
        grid = np.array([0., 2, 4])
        actual = family(forest, measured, 'one', x_grid=grid, num_grid_points=999)
        np.testing.assert_array_equal(actual.x_grid, grid)
        self.assertEqual(actual.rescaling, 'pyramid')
        for invalid in (np.array([1.]), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                family(forest, measured, 'bad', x_grid=invalid)


class CacheAndWrapperTests(unittest.TestCase):
    def test_cache_flags_overwrite_and_reuse_without_functional_calls(self):
        forest = profile_forest([(1, 7, [(1, 3, 6), (3, 7, 2)]), (3, 5, [(3, 5, 4)])])
        grid = np.arange(9.)
        first = family(forest, measured, 'area', x_grid=grid, cache=False)
        self.assertEqual(forest.landscape_families, {})
        self.assertEqual(forest.barcode_functionals, {})
        saved = family(forest, measured, 'area', x_grid=grid, cache_functionals=True, functionals_label='profiles')
        self.assertIs(saved, forest.landscape_families['area'])
        forbidden = Mock(side_effect=AssertionError('cached functional must not be called'))
        reused = family(forest, forbidden, 'reuse', x_grid=grid, compute_functionals=False, functionals_label='profiles')
        forbidden.assert_not_called()
        np.testing.assert_allclose(reused.evaluate_on_grid(grid, levels=5), first.evaluate_on_grid(grid, levels=5))
        overwritten = family(forest, lambda *_: 0, 'area', x_grid=grid)
        self.assertIs(forest.landscape_families['area'], overwritten)
        np.testing.assert_array_equal(overwritten.evaluate_on_grid(grid, levels=5), np.zeros((5, 9)))
        with self.assertRaisesRegex(ValueError, 'none found'):
            family(forest, measured, 'missing', compute_functionals=False)

    def test_reordered_and_filtered_bars_reuse_profiles_by_object_identity(self):
        specs = [(1, 7, [(1, 3, 6), (3, 7, 2)]), (3, 5, [(3, 5, 4)])]
        forest = profile_forest(specs)
        compute_barcode_functionals(forest, measured, 'profiles')
        forest.barcode.reverse()
        grid = np.arange(9.)
        actual = family(forest, measured, 'cached', x_grid=grid, compute_functionals=False,
                        functionals_label='profiles', max_k=2)
        np.testing.assert_allclose(actual.evaluate_on_grid(grid, levels=2), landscape(grid, specs, 2))
        filtered = family(forest, measured, 'long', x_grid=grid, min_bar_length=6,
                          compute_functionals=False, functionals_label='profiles', max_k=1)
        np.testing.assert_allclose(filtered.evaluate_on_grid(grid), landscape(grid, specs[:1], 1))
        compute_barcode_functionals(forest, measured, 'partial', min_bar_length=6)
        with self.assertRaises(KeyError):
            family(forest, measured, 'incompatible', x_grid=grid, compute_functionals=False, functionals_label='partial')

    def test_public_signedness_profiles_kernel_difference_and_legacy_alias(self):
        forest = build(fan())
        grid = np.linspace(0, 8, 65)
        families = []
        for signed in (False, True):
            actual = forest.compute_measurement_landscapes(signed_chain_edge_length, str(signed),
                signed=signed, x_grid=grid, max_k=4, cache_functionals=True)
            profiles = forest.barcode_functionals[str(signed)]
            a = 4 + np.sqrt(2) + np.sqrt(10)
            literal = {
                1: [(1, 2, 8 + 4 * np.sqrt(2)),
                    (2, 3, 8 + (6 if signed else 4) * np.sqrt(2)),
                    (3, 4, 4 + 5 * np.sqrt(2) + np.sqrt(10)), (4, 7, a)],
                3: [(3, 5, a)], 4: [(4, 6, 4 * np.sqrt(2) + 2 * np.sqrt(10))],
            }
            specs = [(b.birth, b.death, literal[b.birth]) for b in profiles.bars]
            for b in profiles.bars:
                np.testing.assert_allclose(profiles[b].vals, [v for _, _, v in literal[b.birth]],
                                           rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.evaluate_on_grid(grid, levels=4), landscape(grid, specs, 4),
                                       rtol=1e-12, atol=1e-12)
            alias = forest.compute_generalized_landscape_family(signed_chain_edge_length, 'alias',
                signed=signed, x_grid=grid, max_k=4, cache=False, min_bar_length=0)
            np.testing.assert_allclose(alias.evaluate_on_grid(grid, levels=4), actual.evaluate_on_grid(grid, levels=4))
            families.append(actual)
        long = next(b for b in forest.barcode if b.birth == 1)
        unsigned = forest.barcode_functionals['False'][long]
        signed = forest.barcode_functionals['True'][long]
        np.testing.assert_allclose(signed.vals - unsigned.vals, [0, 2 * np.sqrt(2), 0, 0])
        a = compute_landscape_kernel_for_bar(forest, long, signed_chain_edge_length, sf=unsigned)
        b = compute_landscape_kernel_for_bar(forest, long, signed_chain_edge_length, sf=signed)
        expected = np.sqrt(2) * np.maximum(0, np.minimum(3, 2 * grid - 1) - np.maximum(2, 2 * grid - 7))
        np.testing.assert_allclose(b(grid) - a(grid), expected, atol=1e-12)


class InvalidLandscapeContractTests(unittest.TestCase):
    """D8: landscape validation and remaining expected-failure targets."""

    def test_d8_infinite_kernel_has_actionable_error(self):
        forest = profile_forest([(1, np.inf, [(1, np.inf, 1)])])
        with self.assertRaisesRegex(ValueError, 'finite'):
            compute_landscape_kernel_for_bar(forest, forest.barcode[0], measured)


    def test_d8_infinite_family_has_actionable_error(self):
        forest = profile_forest([(1, np.inf, [(1, np.inf, 1)])])
        with self.assertRaisesRegex(ValueError, 'finite'):
            family(forest, measured, 'infinite', x_grid=[0, 1, 2])

    def test_d8_negative_scientific_measurement_rejects(self):
        forest = profile_forest([(0, 2, [(0, 2, -1)])])
        for mode in ('raw', 'pyramid'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                    family(forest, measured, 'negative', mode=mode, cache_functionals=True)
        self.assertEqual(forest.barcode_functionals, {})
        self.assertEqual(forest.landscape_families, {})

    def test_d8_nonfinite_measurement_rejects(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                forest = profile_forest([(0, 2, [(0, 2, value)])])
                with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                    family(forest, measured, 'nonfinite', cache_functionals=True)
                self.assertEqual(forest.barcode_functionals, {})

    def test_d8_direct_kernel_and_cached_profile_values_are_validated(self):
        forest = profile_forest([(0, 2, [(0, 2, -1)])])
        bar = forest.barcode[0]
        with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
            compute_landscape_kernel_for_bar(forest, bar, measured)

        profiles = compute_barcode_functionals(forest, measured, 'signed')
        self.assertEqual(profiles[bar].vals.tolist(), [-1])
        self.assertEqual(compute_convolution_kernel_for_bar(forest, bar, measured, sf=profiles[bar])(2), -2)
        for mode in ('raw', 'pyramid'):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                    compute_landscape_kernel_for_bar(forest, bar, measured, sf=profiles[bar], mode=mode)
                with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                    family(forest, measured, 'cached', mode=mode,
                           compute_functionals=False, functionals_label='signed')

    def test_d8_public_wrapper_rejects_negative_and_preserves_existing_cache(self):
        forest = build(split())
        existing = forest.compute_measurement_landscapes(
            constant_one_functional, 'measure', cache_functionals=True
        )
        profiles = forest.barcode_functionals['measure']
        with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
            forest.compute_measurement_landscapes(
                lambda *_: -1, 'measure', cache_functionals=True
            )
        self.assertIs(forest.landscape_families['measure'], existing)
        self.assertIs(forest.barcode_functionals['measure'], profiles)


def _invalid_family_case(options):
    def test(self):
        forest = profile_forest([(0, 2, [(0, 2, 1)])])
        with self.assertRaises(ValueError):
            family(forest, measured, 'invalid', **options)
    return test


for _name, _options in [('descending_grid', {'x_grid': [2., 1, 0]}),
                        ('nan_grid', {'x_grid': [0., np.nan, 2]}),
                        ('infinite_grid', {'x_grid': [0., np.inf]}),
                        ('zero_levels', {'max_k': 0}), ('one_grid_point', {'num_grid_points': 1})]:
    setattr(InvalidLandscapeContractTests, 'test_d8_' + _name, _invalid_family_case(_options))
