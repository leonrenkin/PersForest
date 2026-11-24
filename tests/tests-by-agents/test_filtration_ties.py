"""T1–T10: exact ties, elder choices, and explicit reduction tolerances."""
from itertools import permutations
import unittest
import numpy as np
from persforest.measurements.basic import signed_chain_area, constant_one_functional
from persforest.landscapes.kernels import landscape_kernel
from persforest.landscapes.profiles import compute_measurement_profiles
from fixtures_small_filtrations import (
    triangle, split, fan, disconnected, annulus, build, chain, active_reps, positive_bars,
    mode_options, reps,
)
from oracle_chains import sample_times


def variants():
    f = triangle()
    yield 'T1', f.changed({s: 2 for s in f.values if len(s) > 1}), []
    yield 'T2', f.changed({(1, 2): 6}), []
    f = split()
    yield 'T3', f.changed({(0, 1): 1}), [(1, 5), (1, 7)]
    yield 'T4', f.changed({(0, 1, 2): 3}), [(1, 7)]
    yield 'T5', f.changed({(0, 1, 2): 7}), [(1, 7), (3, 7)]
    yield 'T6', f.changed({(0, 1): 7, (0, 1, 2): 7}), [(1, 7)]
    f = fan()
    yield 'T7', f.changed({(1, 3): 2}), [(1, 7), (2, 5), (4, 6)]
    yield 'T8', f.changed({(0, 3): 3, (2, 3): 3}), [(1, 7), (3, 5), (3, 6)]
    yield 'T9', disconnected(), [(2, 6), (2, 6)]
    yield 'T10', annulus(), [(2, 5)] + [(3, 5)] * 7


def active_signature(forest, time):
    return sorted(tuple(sorted(rep.signed_simplices)) for rep in active_reps(forest, time))


class FiltrationTieTests(unittest.TestCase):
    def assert_separated_beyond_tolerance(self, forest):
        """In a reduced forest, a bar's birth and critical values differ by more than the tolerance."""
        for bar in forest.barcode:
            values = (bar.birth,) + bar.critical_values
            self.assertTrue(all(b > a + forest.filtration_tol for a, b in zip(values, values[1:])), values)

    def test_t1_to_t10_barcode_and_observable_reduction_equivalence(self):
        for name, fixture, expected in variants():
            reference = build(fixture, reduce=False, filtration_tol=0)
            for mode, options in mode_options():
                for tol in (0, 1e-12):
                    with self.subTest(case=name, mode=mode, tol=tol):
                        forest = build(fixture, filtration_tol=tol, **options)
                        self.assertEqual(positive_bars(forest), expected)
                        for time in sample_times(fixture.values):
                            self.assertEqual(active_signature(forest, time), active_signature(reference, time))
                        if options['reduce']:
                            self.assertTrue(all(b.death > b.birth for b in forest.barcode))
                            self.assert_separated_beyond_tolerance(forest)

    def test_tied_split_has_no_observable_transient_chain(self):
        cases = dict((name, fixture) for name, fixture, _ in variants())
        for name, time, expected in [
            ('T3', 1, [chain('+01 -02 +12'), chain('-01 +03 -13')]),
            ('T4', 3, [chain('-01 +03 -13')]),
            ('T6', 6, [chain('-02 +12 +03 -13')]),
            ('T7', 2, [chain('+01 -03 +13'), chain('-02 +03 +12 -13')]),
            ('T8', 3, [chain('+01 -03 +13'), chain('-02 +03 -23'), chain('+12 -13 +23')]),
        ]:
            with self.subTest(case=name):
                self.assertEqual(active_signature(build(cases[name]), time),
                                 sorted(tuple(sorted(s)) for s in expected))

    def test_legal_equal_value_orders_preserve_active_basis(self):
        for name, fixture, expected in variants():
            entries = sorted(fixture.values.items(), key=lambda item: (item[1], len(item[0]), item[0]))
            groups = {}
            for s, t in entries:
                groups.setdefault((t, len(s)), []).append((s, t))
            # Explicitly change one same-dimension block at a time; faces stay first.
            orders = [entries]
            for group in groups.values():
                if 1 < len(group) <= 4:
                    indices = [entries.index(item) for item in group]
                    for perm in permutations(group):
                        reordered = entries.copy()
                        for index, item in zip(indices, perm):
                            reordered[index] = item
                        orders.append(reordered)
                elif len(group) > 4:
                    reordered = entries.copy()
                    for index, item in zip([entries.index(item) for item in group], reversed(group)):
                        reordered[index] = item
                    orders.append(reordered)
            reference = build(fixture, order=entries)
            for order in orders:
                with self.subTest(case=name, order=order):
                    forest = build(fixture, order=order)
                    self.assertEqual(positive_bars(forest), expected)
                    for time in sample_times(fixture.values):
                        self.assertEqual(active_signature(forest, time), active_signature(reference, time))
                    if name not in ('T5', 'T9', 'T10'):
                        signature = lambda f: sorted((b.birth, b.death,
                            tuple((r.active_start, r.active_end, tuple(sorted(r.signed_simplices)))
                                  for r in reps(f, b))) for b in f.barcode)
                        self.assertEqual(signature(forest), signature(reference))

    def test_fixed_tied_order_is_repeatable_across_modes(self):
        fixture = split().changed({(0, 1, 2): 7})
        for _, options in mode_options():
            forests = [build(fixture, **options) for _ in range(2)]
            signatures = []
            for forest in forests:
                bar = next(b for b in forest.barcode if b.birth == 1)
                rep = next(r for r in reps(forest, bar) if r.active_start == 3)
                self.assertEqual(rep.signed_simplices, chain('-01 +03 -13'))
                signatures.append(active_signature(forest, 4))
            self.assertEqual(*signatures)

    def test_equal_deaths_allow_two_measurement_landscapes(self):
        fixture = split().changed({(0, 1, 2): 7})
        entries = sorted(fixture.values.items(), key=lambda item: (item[1], len(item[0]), item[0]))
        results, ones = [], []
        for swap in (False, True):
            order = entries.copy()
            if swap:
                order[-2:] = reversed(order[-2:])
            forest = build(fixture, order=order)
            bar = next(b for b in forest.barcode if b.birth == 1)
            results.append(landscape_kernel(compute_measurement_profiles(forest, signed_chain_area, "test", signed=True), bar)(3))
            ones.append(landscape_kernel(compute_measurement_profiles(forest, constant_one_functional, "test", signed=True), bar)(3))
        self.assertEqual(sorted(results), [8, 10])
        self.assertEqual(ones, [2, 2])

    def test_near_tied_deaths_swap_elder_and_do_not_imply_general_stability(self):
        for exponent in (8, 12, 16):
            epsilon = 2. ** -exponent
            values = []
            for sign in (-1, 1):
                fixture = split().changed({(0, 1, 2): 7, (0, 1, 3): 7 + sign * epsilon})
                forest = build(fixture, filtration_tol=0)
                bar = next(b for b in forest.barcode if b.birth == 1)
                rep = next(iter(forest.iter_bar_cycle_reps(bar)))
                self.assertEqual(rep.signed_simplices,
                                 chain('-01 +03 -13') if sign == 1 else chain('+01 -02 +12'))
                values.append(landscape_kernel(compute_measurement_profiles(forest, signed_chain_area, "test", signed=True), bar)(3))
            np.testing.assert_allclose(values, [10, 8], rtol=0, atol=1e-12)


class ReductionToleranceTests(unittest.TestCase):
    def test_small_bar_removed_only_inside_tolerance(self):
        tau = 2. ** -20
        for tol in (0, tau):
            for gap in (0, tau / 2, tau, 2 * tau):
                with self.subTest(tol=tol, gap=gap):
                    fixture = triangle().changed({(0, 1, 2): 2 + gap})
                    forest = build(fixture, filtration_tol=tol)
                    self.assertEqual(positive_bars(forest), [] if gap <= tol else [(2, 2 + gap)])

    def test_reduced_tie_forests_are_separated_beyond_tolerance(self):
        for name, fixture, _ in variants():
            with self.subTest(case=name):
                FiltrationTieTests.assert_separated_beyond_tolerance(self, build(fixture))

    def test_invalid_tolerance_rejected(self):
        for tol in (-1, float('nan'), float('inf')):
            with self.subTest(tol=tol), self.assertRaisesRegex(ValueError, 'filtration_tol'):
                build(triangle(), filtration_tol=tol)

    def test_d11_collapsed_near_tie_has_no_gap_in_stored_representative_activity(self):
        """Reduction changes node heights but leaves stored interval metadata stale."""
        tau = 2. ** -20
        forest = build(fan().changed({(1, 3): 2 + tau / 2}), filtration_tol=tau)
        for bar in forest.barcode:
            actual = reps(forest, bar)
            self.assertEqual(actual[0].active_start, bar.birth)
            self.assertEqual(actual[-1].active_end, bar.death)
            self.assertEqual([r.active_end for r in actual[:-1]],
                             [r.active_start for r in actual[1:]])

    def test_near_tie_diff_reconstruction_covers_whole_reduced_bar(self):
        tau = 2. ** -20
        for gap in (tau / 2, tau, 2 * tau):
            forest = build(fan().changed({(1, 3): 2 + gap}), filtration_tol=tau)
            for bar in forest.barcode:
                actual = reps(forest, bar)
                self.assertEqual(actual[0].active_start, bar.birth)
                self.assertEqual(actual[-1].active_end, bar.death)
                self.assertEqual([r.active_end for r in actual[:-1]],
                                 [r.active_start for r in actual[1:]])
