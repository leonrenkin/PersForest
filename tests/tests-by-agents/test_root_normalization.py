"""Roots represent independent births after zero-duration edges collapse."""
import unittest
from unittest.mock import patch

from fixtures_small_filtrations import (
    MODULE, active_reps, build, chain, fan, mode_options, positive_bars, reps,
    split, triangle,
)


def build_split(fixture, **options):
    """Insert the shared edge last among tied edges to create a root merge."""
    order = sorted(fixture.values.items(),
                   key=lambda item: (item[1], len(item[0]), item[0] == (0, 1), item[0]))
    return build(fixture, order=order, **options)


def progression_signature(forest):
    """Compare signed active chains and interiors without root identifiers."""
    return sorted(
        (bar.birth, bar.death, bar.is_max_tree_bar,
         tuple((rep.active_start, rep.active_end,
                tuple(sorted(rep.signed_simplices)),
                tuple(sorted(rep.interior)) if rep.interior_available else None)
               for rep in reps(forest, bar)))
        for bar in forest.barcode
    )


class RootNormalizationTests(unittest.TestCase):
    def assert_normalized(self, forest):
        self.assertEqual(forest.roots, {n.id for n in forest.nodes.values()
                                       if n.parent is None})
        self.assertEqual(list(forest.nodes), sorted(forest.nodes))
        for root_id in forest.roots:
            root = forest.nodes[root_id]
            self.assertEqual(len(root.children), 1)
            child = forest.nodes[next(iter(root.children))]
            self.assertEqual(child.parent, root_id)
            self.assertIs(root.cycle, child.cycle)
            self.assertEqual(root._simplex_diff_available, forest.keep_simplex_diff)
            self.assertIsNone(root._interior_diff)
            self.assertIsNone(root._codim1_simplex_diff)
            self.assertIsNone(root._barcode_interior_diff)
            self.assertIsNone(root._barcode_codim1_simplex_diff)

    def test_two_and_three_simultaneous_births_in_all_storage_modes(self):
        fixtures = [
            (split().changed({(0, 1): 1}), [(1, 5), (1, 7)], build_split),
            (fan().changed({(0, 3): 1, (1, 3): 1, (2, 3): 1}),
             [(1, 5), (1, 6), (1, 7)], build),
        ]
        for fixture, expected, builder in fixtures:
            for mode, options in mode_options():
                if not options['reduce']:
                    continue
                with self.subTest(branches=len(expected), mode=mode):
                    # Compare against the same reduced forest before normalization.
                    with patch.object(MODULE.PersistenceForest, '_normalize_roots'):
                        before = builder(fixture, **options)
                    forest = builder(fixture, **options)
                    self.assert_normalized(forest)
                    self.assertEqual(len(before.roots), 1)
                    self.assertEqual(len(forest.roots), len(expected))
                    self.assertEqual(positive_bars(forest), expected)
                    self.assertEqual(progression_signature(forest), progression_signature(before))
                    self.assertEqual({b.root_id for b in forest.barcode}, forest.roots)
                    self.assertTrue(all(b.is_max_tree_bar for b in forest.barcode))
                    self.assertEqual(len(active_reps(forest, 1)), len(expected))
                    self.assertEqual(len(active_reps(forest, 5)), len(expected) - 1)
                    self.assertEqual(active_reps(forest, 7), [])
                    original_root_id = next(iter(before.roots))
                    original_children = before.nodes[original_root_id].children
                    self.assertEqual(forest.nodes[original_root_id].children,
                                     {min(original_children)})
                    self.assertEqual(
                        [nid for nid in forest.nodes if nid not in forest.roots],
                        [nid for nid in before.nodes if nid not in before.roots],
                    )

    def test_literal_signed_cycles_and_interiors_at_birth(self):
        forest = build_split(split().changed({(0, 1): 1}),
                       keep_simplex_diff=True, compute_interior=True)
        expected = {5: (chain('+01 -02 +12'), chain('+012')),
                    7: (chain('-01 +03 -13'), chain('-013'))}
        for bar in forest.barcode:
            root = forest.nodes[bar.root_id]
            support, interior = expected[bar.death]
            self.assertEqual(root.cycle.signed_simplices, support)
            self.assertEqual(root.cycle.interior, interior)
            self.assertEqual((root.cycle.active_start, root.cycle.active_end),
                             (1, bar.death))

    def test_unary_root_gets_surviving_child_cycle(self):
        fixture = split().changed({(0, 1): 1, (0, 1, 2): 1})
        for mode, options in mode_options():
            if not options['reduce']:
                continue
            with self.subTest(mode=mode):
                forest = build_split(fixture, **options)
                self.assert_normalized(forest)
                self.assertEqual(positive_bars(forest), [(1, 7)])
                self.assertEqual(active_reps(forest, 1)[0].signed_simplices,
                                 chain('-01 +03 -13'))

    def test_tolerance_controls_whether_split_is_at_birth(self):
        tolerance = 1e-6
        for delta, root_count in [(0.5 * tolerance, 2), (2 * tolerance, 1)]:
            with self.subTest(delta=delta):
                forest = build(split().changed({(0, 1): 1 + delta}),
                               filtration_tol=tolerance)
                self.assert_normalized(forest)
                self.assertEqual(len(forest.roots), root_count)
                expected_birth = 1 if root_count == 2 else 1 + delta
                self.assertEqual(positive_bars(forest), [(1, 7), (expected_birth, 5)]
                                 if root_count == 1 else [(1, 5), (1, 7)])

    def test_normalization_is_idempotent_and_deterministic(self):
        fixture = fan().changed({(0, 3): 1, (1, 3): 1, (2, 3): 1})
        forest = build(fixture, keep_simplex_diff=True, compute_barcode=False)
        def topology(forest):
            return [(n.id, n.parent, tuple(sorted(n.children)), n.filt_val)
                    for n in forest.nodes.values()]
        before = topology(forest)
        forest._normalize_roots()
        self.assertEqual(topology(forest), before)
        self.assertEqual(topology(build(fixture, compute_barcode=False)), before)
        self.assertEqual(next(forest._node_id), max(forest.nodes) + 1)

    def test_unreduced_forest_does_not_normalize(self):
        with patch.object(MODULE.PersistenceForest, '_normalize_roots') as normalize:
            forest = build_split(split().changed({(0, 1): 1}), reduce=False)
        normalize.assert_not_called()
        self.assertEqual(len(forest.roots), 1)

    def test_empty_reduced_forest_stays_empty(self):
        forest = build(triangle().changed({(0, 1, 2): 2}))
        self.assertEqual(forest.nodes, {})
        self.assertEqual(forest.roots, set())
        self.assertEqual(forest.barcode, set())
        forest._normalize_roots()
        self.assertEqual(forest.nodes, {})
