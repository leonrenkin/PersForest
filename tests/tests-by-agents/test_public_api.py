"""Q1–Q7/M1–M2: public queries, boundaries, delayed computation and errors."""
from copy import deepcopy
import unittest
import numpy as np
from persforest import PersistenceForest
from persforest.barcode import PFBar
from persforest.chains import SignedChain
from persforest.measurements.basic import constant_one_functional
from fixtures_small_filtrations import triangle, split, fan, build, chain, positive_bars


class PublicAPITests(unittest.TestCase):
    def test_public_entry_point_is_callable_class(self):
        self.assertIsInstance(PersistenceForest, type)
        self.assertIsInstance(PersistenceForest([[0, 0], [1, 0], [0, 1]]), PersistenceForest)

    def test_half_open_bar_and_cycle_queries_at_all_endpoint_sides(self):
        forest = build(split())
        for event in (1., 3., 5., 7.):
            for time in (np.nextafter(event, -np.inf), event, np.nextafter(event, np.inf)):
                with self.subTest(time=time):
                    expected = []
                    if 1 <= time < 3:
                        expected = [chain('-02 +12 +03 -13')]
                    elif 3 <= time < 5:
                        expected = [chain('+01 -02 +12'), chain('-01 +03 -13')]
                    elif 5 <= time < 7:
                        expected = [chain('-01 +03 -13')]
                    actual = [frozenset(r.signed_simplices) for r in forest.cycle_reps_at(time)]
                    self.assertCountEqual(actual, expected)
                    self.assertEqual(len(forest.active_bars_at(time)), len(expected))
                    for bar in forest.barcode:
                        if bar.birth <= time < bar.death:
                            rep = forest.cycle_for_bar_at(bar, time)
                            self.assertLessEqual(rep.active_start, time)
                            self.assertGreater(rep.active_end, time)
                        else:
                            with self.assertRaises(ValueError):
                                forest.cycle_for_bar_at(bar, time)

    def test_d2_active_cycles_include_birth_and_exclude_death(self):
        forest = build(split())
        self.assertEqual([len(forest.active_cycles_at(t)) for t in (1, 3, 5, 7)], [1, 2, 1, 0])

    def test_length_selection_and_relative_position(self):
        forest = build(split())
        self.assertEqual(forest.max_bar().lifespan(), 6)
        self.assertEqual([b.lifespan() for b in forest.longest_bars(2)], [6, 2])
        self.assertEqual(forest.longest_bars(0), [])
        self.assertEqual(len(forest.cycle_reps_at(4, min_bar_length=2)), 2)
        self.assertEqual(len(forest.cycle_reps_at(4, min_bar_length=np.nextafter(2, np.inf))), 1)
        birth = forest.barcode_cycle_reps(relative_position=0)
        self.assertEqual(birth[0].signed_simplices, chain('-02 +12 +03 -13'))
        middle = forest.barcode_cycle_reps(relative_position=.5)
        self.assertEqual(middle[0].signed_simplices, chain('-01 +03 -13'))
        for invalid in (-.1, 1.1):
            with self.assertRaises(ValueError):
                forest.barcode_cycle_reps(relative_position=invalid)
        # D9: 1 currently rejects because it requests excluded death; no new endpoint policy.
        with self.assertRaises(ValueError):
            forest.barcode_cycle_reps(relative_position=1)

    def test_empty_selection_and_max_bar_error(self):
        forest = build(triangle().changed({(1, 2): 6}))
        self.assertEqual(forest.active_bars_at(6), [])
        self.assertEqual(forest.cycle_reps_at(6), [])
        self.assertEqual(forest.barcode_cycle_reps(), [])
        self.assertEqual(forest.longest_bars(2), [])
        with self.assertRaises(ValueError):
            forest.max_bar()

    def test_bar_level_forest_structure(self):
        forest = build(fan())
        bars = {(bar.birth, bar.death): bar for bar in forest.barcode}
        trunk, first, second = bars[1, 7], bars[3, 5], bars[4, 6]
        self.assertIsNone(forest.parent_bar(trunk))
        self.assertIs(forest.parent_bar(first), trunk)
        self.assertIs(forest.parent_bar(second), trunk)
        self.assertEqual(forest.child_bars(trunk), [first, second])
        self.assertEqual(forest.child_bars(first), [])
        self.assertEqual({bar.tree_id for bar in forest.barcode}, {trunk.tree_id})
        # The trunk's representative changes where branches split off it, and at its death.
        self.assertEqual(trunk.critical_values, (2.0, 3.0, 4.0, 7.0))
        self.assertEqual(first.critical_values, (5.0,))
        self.assertEqual(sorted(bar.index for bar in forest.barcode), [0, 1, 2])
        for bar in forest.barcode:
            reps = list(reversed(list(forest.iter_bar_cycle_reps(bar))))
            self.assertEqual(tuple(rep.active_end for rep in reps), bar.critical_values)
            self.assertEqual(reps[0].active_start, bar.birth)

    def test_bar_queries_reject_foreign_bars(self):
        forest = build(fan())
        foreign = next(iter(build(fan()).barcode))
        for query in (forest.parent_bar, forest.child_bars, lambda bar: forest.cycle_for_bar_at(bar, bar.birth)):
            with self.assertRaisesRegex(ValueError, 'not in barcode'):
                query(foreign)

    def test_delayed_barcode_matches_eager_and_repeats_are_rejected(self):
        forest = build(fan(), compute_barcode=False)
        self.assertEqual(forest.barcode, set())
        forest.compute_barcode()
        self.assertEqual(positive_bars(forest), [(1, 7), (3, 5), (4, 6)])
        self.assertEqual(positive_bars(forest), positive_bars(build(fan())))
        with self.assertRaisesRegex(RuntimeError, 'already been called'):
            forest.compute_barcode()

    def test_d9_empty_computed_barcode_still_rejects_repeat(self):
        forest = PersistenceForest(np.empty((0, 2)))
        self.assertEqual(forest.barcode, set())
        with self.assertRaisesRegex(RuntimeError, 'already been called'):
            forest.compute_barcode()

    def test_delayed_barcode_rejects_measurement_until_computed(self):
        forest = build(fan(), compute_barcode=False)
        with self.assertRaisesRegex(RuntimeError, 'compute_barcode'):
            forest.compute_measurement_landscapes(constant_one_functional, 'one')
        with self.assertRaisesRegex(RuntimeError, 'compute_barcode'):
            forest.compute_measurement_profiles(constant_one_functional, 'one')
        self.assertEqual(forest.measurement_profiles, {})
        self.assertEqual(forest.landscape_families, {})
        forest.compute_barcode()
        self.assertTrue(forest.compute_measurement_landscapes(
            constant_one_functional, 'one').bar_kernels)

        empty = PersistenceForest(np.empty((0, 2)))
        self.assertEqual(empty.compute_measurement_landscapes(
            constant_one_functional, 'one').bar_kernels, {})

    def test_removed_options_and_representative_interiors(self):
        forest = build(fan())
        bar = forest.max_bar()
        self.assertIsInstance(forest.cycle_for_bar_at(bar, bar.birth), SignedChain)
        self.assertTrue(all(rep.interior_available for rep in forest.iter_bar_cycle_reps(bar)))
        for removed in ('storage', 'keep_simplex_diff', 'compute_interior', 'diff_only_mode'):
            with self.subTest(removed=removed), self.assertRaises(TypeError):
                build(triangle(), **{removed: True})

    def test_hand_made_pfbar(self):
        bar = PFBar(1, np.inf, (2.0, np.inf), tree_id=3)
        self.assertEqual(bar.lifespan(), np.inf)
        self.assertEqual(bar.critical_values, (2.0, np.inf))
        self.assertEqual((bar.tree_id, bar.index), (3, None))
