"""Q1–Q7/M1–M2: public queries, boundaries, delayed computation and errors."""
from copy import deepcopy
import unittest
import numpy as np
from persforest import PersistenceForest
from persforest.PersistenceForest import PFBar, SignedChain
from persforest.cycle_rep_vectorisations import constant_one_functional
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
                            rep = bar.cycle_at_filtration_value(time)
                            self.assertLessEqual(rep.active_start, time)
                            self.assertGreater(rep.active_end, time)
                        else:
                            with self.assertRaises(ValueError):
                                bar.cycle_at_filtration_value(time)


    def test_d2_active_nodes_include_birth_and_exclude_death(self):
        forest = build(split())
        self.assertEqual([len(forest.active_nodes_at(t)) for t in (1, 3, 5, 7)], [1, 2, 1, 0])


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

    def test_leaf_paths_and_roots_are_reciprocal(self):
        forest = build(fan())
        self.assertEqual(len(forest.roots), 1)
        root = forest.nodes[next(iter(forest.roots))]
        leaves = forest.leaves_below_node(root)
        self.assertEqual(len(leaves), 3)
        for leaf_id in leaves:
            leaf = forest.nodes[leaf_id]
            self.assertIs(forest.get_root(leaf), root)
            path = forest.leaf_to_node_path(leaf, root)
            self.assertEqual(path[0], leaf_id)
            self.assertEqual(path[-1], root.id)
            self.assertEqual(forest.node_to_leaf_path(leaf, root), path[::-1])
            self.assertEqual(len(path), len(set(path)))
            for child, parent in zip(path, path[1:]):
                self.assertEqual(forest.nodes[child].parent, parent)
                self.assertIn(child, forest.nodes[parent].children)
        a, b = [forest.nodes[i] for i in sorted(leaves)[:2]]
        with self.assertRaises(ValueError):
            forest.leaf_to_node_path(a, b)

    def test_delayed_barcode_matches_eager_and_repeats_are_rejected(self):
        for diffs in (False, True):
            forest = build(fan(), keep_simplex_diff=diffs, compute_barcode=False)
            self.assertEqual(forest.barcode, set())
            forest.compute_barcode()
            self.assertEqual(positive_bars(forest), [(1, 7), (3, 5), (4, 6)])
            self.assertEqual(positive_bars(forest), positive_bars(build(fan(), keep_simplex_diff=diffs)))
            with self.assertRaisesRegex(RuntimeError, 'already been called'):
                forest.compute_barcode()
            direct_method = forest.compute_barcode_diff if diffs else forest.compute_barcode_cycles
            with self.assertRaisesRegex(RuntimeError, 'already been called'):
                direct_method()

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
            forest.compute_generalized_landscape_family(constant_one_functional, 'one')
        self.assertEqual(forest.barcode_functionals, {})
        self.assertEqual(forest.landscape_families, {})
        forest.compute_barcode()
        self.assertTrue(forest.compute_measurement_landscapes(
            constant_one_functional, 'one').bar_kernels)

        empty = PersistenceForest(np.empty((0, 2)))
        self.assertEqual(empty.compute_measurement_landscapes(
            constant_one_functional, 'one').bar_kernels, {})

    def test_missing_interior_dependency_and_diff_only_bar_access_error(self):
        forest_with_interior = build(triangle(), compute_interior=True)
        self.assertTrue(forest_with_interior.keep_simplex_diff)
        self.assertTrue(forest_with_interior.compute_interior)
        forest = build(fan(), diff_only_mode=True, keep_simplex_diff=True)
        bar = forest.max_bar()
        with self.assertRaisesRegex(ValueError, 'diff_only_mode'):
            bar.cycle_at_filtration_value(bar.birth)
        self.assertIsInstance(forest.cycle_for_bar_at(bar, bar.birth), SignedChain)
        with self.assertRaisesRegex(ValueError, 'diff_only_mode'):
            build(triangle(), compute_interior=True, diff_only_mode=True, keep_simplex_diff=True)

    def test_d9_missing_interior_activity_error_is_actionable(self):
        forest = build(triangle())
        with self.assertRaisesRegex(ValueError, 'keep_simplex_diff=True'):
            forest.interior_simplex_activity()

    def test_infinite_pfbar_finite_time_queries(self):
        rep = SignedChain(set(chain('+01 -02 +12')), 1, np.inf)
        bar = PFBar(1, np.inf, (1,), [rep])
        self.assertEqual(bar.lifespan(), np.inf)
        self.assertIs(bar.cycle_at_filtration_value(1), rep)
        self.assertIs(bar.cycle_at_filtration_value(100), rep)
        with self.assertRaises(ValueError):
            bar.cycle_at_filtration_value(np.inf)
