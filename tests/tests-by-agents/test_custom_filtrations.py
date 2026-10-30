"""F1–F6: literal representatives, barcode multiplicity and reconstruction."""
import copy
import math
import unittest
import numpy as np
from persforest import cycle_rep_vectorisations as measurements
from fixtures_small_filtrations import (
    triangle, split, fan, disconnected, tetrahedra, annulus, build, chain,
    mode_options, positive_bars, reps, active_reps, simplex_tree,
)


class CustomFiltrationTests(unittest.TestCase):
    def test_f1_to_f5_literal_chains_interiors_and_intervals_in_every_mode(self):
        for factory in (triangle, split, fan, disconnected, tetrahedra):
            fixture = factory()
            for mode, options in mode_options():
                with self.subTest(fixture=factory.__name__, mode=mode):
                    forest = build(fixture, **options)
                    self.assertEqual(positive_bars(forest), sorted((b, d) for b, d, _ in fixture.expected))
                    unmatched = list(fixture.expected)
                    for bar in forest.barcode:
                        if bar.death <= bar.birth:
                            continue
                        actual = [r for r in reps(forest, bar) if r.active_end > r.active_start]
                        signature = [(r.active_start, r.active_end, frozenset(r.signed_simplices)) for r in actual]
                        match = next((e for e in unmatched if (bar.birth, bar.death) == e[:2]
                                      and signature == [(r.start, r.end, r.simplices) for r in e[2]]), None)
                        self.assertIsNotNone(match, signature)
                        unmatched.remove(match)
                        if options.get('compute_interior') or options.get('diff_only_mode'):
                            for observed, expected in zip(actual, match[2]):
                                self.assertTrue(observed.interior_available)
                                self.assertEqual(observed.interior, expected.interior)
                    self.assertFalse(unmatched)

    def test_f1_measurements_and_irrelevant_edge(self):
        fixture = triangle()
        forest = build(fixture, keep_simplex_diff=True, compute_interior=True)
        rep = next(iter(forest.barcode)).cycle_reps[0]
        for func, expected in [(measurements.signed_chain_edge_length, 12),
                               (measurements.signed_chain_area, 6),
                               (measurements.signed_chain_interior_volume, 6),
                               (measurements.signed_chain_circularity, math.pi / 6),
                               (measurements.signed_chain_excess_curvature_normalized, 0)]:
            with self.subTest(function=func.__name__):
                self.assertAlmostEqual(func(rep, fixture.points), expected, places=12)
        fixture.points = np.vstack((fixture.points, [[10, 0], [11, 0]]))
        fixture.values |= {(3,): 0, (4,): 0, (3, 4): 3}
        self.assertEqual(positive_bars(build(fixture)), [(2, 6)])

    def test_f2_interior_activity_preserves_bar_ownership(self):
        forest = build(split(), keep_simplex_diff=True, compute_interior=True)
        actual = {s: sorted((b.birth, b.death, start, end) for b, start, end in rows)
                  for s, rows in forest.interior_simplex_activity().items()}
        self.assertEqual(actual, {(0, 1, 2): [(1, 7, 1, 3), (3, 5, 3, 5)],
                                  (0, 1, 3): [(1, 7, 1, 7)]})

    def test_f3_doubled_edge_changes_length_but_not_projection_or_area(self):
        forest = build(fan())
        bar = next(b for b in forest.barcode if b.birth == 1)
        first, doubled, remaining, last = bar.cycle_reps
        self.assertEqual(doubled.unsigned().signed_simplices, first.signed_simplices)
        self.assertEqual(doubled.only_double_simplices().signed_simplices, chain('+03 -03'))
        expected_lengths = [8 + 4 * math.sqrt(2), 8 + 6 * math.sqrt(2),
                            4 + 5 * math.sqrt(2) + math.sqrt(10), 4 + math.sqrt(2) + math.sqrt(10)]
        np.testing.assert_allclose([measurements.signed_chain_edge_length(r, forest.point_cloud)
                                    for r in bar.cycle_reps], expected_lengths, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose([measurements.signed_chain_area(r, forest.point_cloud)
                                    for r in bar.cycle_reps], [8, 8, 6, 2])
        self.assertEqual(len(first.signed_simplices), 3)
        self.assertEqual(len(doubled.signed_simplices), 5)

    def test_f5_exact_interior_volumes(self):
        forest = build(tetrahedra(), keep_simplex_diff=True, compute_interior=True)
        for bar in forest.barcode:
            expected = [.5, 1 / 3] if bar.birth == 1 else [1 / 6]
            np.testing.assert_allclose([measurements.signed_chain_interior_volume(r, forest.point_cloud)
                                        for r in bar.cycle_reps], expected, rtol=1e-12, atol=1e-12)

    def test_f6_annular_interior_and_simultaneous_eight_way_split(self):
        for mode, options in mode_options():
            if not (options.get('compute_interior') or options.get('diff_only_mode')):
                continue
            with self.subTest(mode=mode):
                forest = build(annulus(), **options)
                self.assertEqual(positive_bars(forest), [(2, 5)] + [(3, 5)] * 7)
                [rep] = active_reps(forest, 2)
                self.assertEqual(rep.signed_simplices, chain('+01 +12 +23 -03 -45 -56 -67 +47'))
                self.assertEqual({s for s, _ in rep.interior},
                                 {(0, 1, 5), (0, 4, 5), (1, 2, 6), (1, 5, 6),
                                  (2, 3, 7), (2, 6, 7), (0, 3, 4), (3, 4, 7)})
                for func, expected in [(measurements.signed_chain_area, 12),
                                       (measurements.signed_chain_interior_volume, 12),
                                       (measurements.signed_chain_edge_length, 24),
                                       (measurements.signed_chain_connected_components, 2),
                                       (measurements.signed_chain_circularity, math.pi / 12),
                                       (measurements.signed_chain_convex_hull_area_deficit, .25),
                                       (measurements.signed_chain_convex_hull_perimeter_deficit, 1 / 3)]:
                    self.assertAlmostEqual(func(rep, forest.point_cloud), expected, places=12)
                self.assertEqual(len(active_reps(forest, 3)), 8)
                self.assertEqual(active_reps(forest, 5), [])

    def test_fixture_validation_rejects_silent_face_repair_and_bad_order(self):
        fixture = triangle()
        del fixture.values[(0, 1)]
        with self.assertRaisesRegex(ValueError, 'face'):
            simplex_tree(fixture)
        fixture = triangle().changed({(0, 1): 8})
        with self.assertRaisesRegex(ValueError, 'face'):
            simplex_tree(fixture)
        fixture = triangle()
        tree = simplex_tree(fixture)  # GUDHI's iterator requires its owner alive.
        with self.assertRaises(ValueError):
            build(fixture, order=list(reversed(list(tree.get_filtration()))))

    def test_diff_iterator_sets_are_independent_and_repeatable(self):
        forest = build(fan(), keep_simplex_diff=True, diff_only_mode=True)
        bar = next(b for b in forest.barcode if b.birth == 1)
        before = copy.deepcopy(bar.cycle_reps)
        first = list(forest.iter_bar_cycle_reps(bar))
        second = list(forest.iter_bar_cycle_reps(bar))
        self.assertEqual(first, second)
        self.assertEqual(before, bar.cycle_reps)
        self.assertEqual(len({id(r.interior) for r in first}), len(first))
        self.assertEqual(len({id(r.signed_simplices) for r in first}), len(first))
        first[0].interior.clear()
        first[0].signed_simplices.clear()
        self.assertEqual(second, list(forest.iter_bar_cycle_reps(bar)))
        self.assertTrue(first[1].interior)
        self.assertTrue(first[1].signed_simplices)

    def test_normal_iterator_identity_and_missing_diff_errors(self):
        forest = build(fan())
        for bar in forest.barcode:
            self.assertTrue(all(a is b for a, b in zip(forest.iter_bar_cycle_reps(bar), reversed(bar.cycle_reps))))
        with self.assertRaisesRegex(ValueError, 'not in barcode'):
            next(forest.iter_bar_cycle_reps(object()))
        forest = build(fan(), keep_simplex_diff=True, diff_only_mode=True)
        bar = next(iter(forest.barcode))
        forest.nodes[bar._node_progression[-1]]._simplex_diff_available = False
        with self.assertRaisesRegex(ValueError, 'simplex_diff'):
            next(forest.iter_bar_cycle_reps(bar))
