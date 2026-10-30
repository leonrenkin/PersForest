"""Regression checks for deferred, independently owned barcode differences."""

import unittest

from persforest.PersistenceForest import PersistenceForest, PFNode
from fixtures_small_filtrations import build, disconnected, fan, split, tetrahedra


def nested_forest() -> PersistenceForest:
    """Build a bookkeeping tree with unary paths and a three-child merge."""
    forest = object.__new__(PersistenceForest)
    forest.keep_simplex_diff = True
    forest._barcode_computed = False
    forest.barcode = set()
    parents = {1: 8, 2: 7, 3: 5, 4: 6, 5: 7, 6: 7, 7: 8, 8: 9, 9: None}
    forest.nodes = {
        nid: PFNode(
            id=nid, filt_val=float(11 - nid), cycle=None,
            children={child for child, parent in parents.items() if parent == nid},
            parent=parent,
            _interior_diff={((nid, nid + 1, nid + 2), 1)},
            _codim1_simplex_diff={((nid, nid + 1), 1), ((nid, nid + 1), -1)},
        )
        for nid, parent in parents.items()
    }
    return forest


class BarcodeDiffAccumulationTests(unittest.TestCase):
    def test_nested_merges_preserve_sources_and_own_their_accumulators(self):
        forest = nested_forest()
        fields = ('_interior_diff', '_codim1_simplex_diff')
        sources = {(nid, field): getattr(node, field).copy()
                   for nid, node in forest.nodes.items() for field in fields}
        forest.compute_barcode_diff()

        self.assertEqual(
            sorted((bar._node_progression, bar.birth, bar.death,
                    bar.root_id, bar.is_max_tree_bar) for bar in forest.barcode),
            [((5, 3), 4., 8., 9, False), ((6, 4), 4., 7., 9, False),
             ((7, 2), 3., 9., 9, False), ((8, 1), 2., 10., 9, True)],
        )
        for field in fields:
            cache_field = '_barcode' + field
            lower = getattr(forest.nodes[7], cache_field)
            upper = getattr(forest.nodes[8], cache_field)
            self.assertEqual(lower, set().union(*(sources[nid, field] for nid in (3, 4, 5, 6))))
            self.assertEqual(upper, set().union(*(sources[nid, field] for nid in (2, 3, 4, 5, 6, 7))))
            for nid, node in forest.nodes.items():
                self.assertEqual(getattr(node, field), sources[nid, field])
                self.assertIsNot(lower, getattr(node, field))
                self.assertIsNot(upper, getattr(node, field))
            self.assertIsNot(lower, upper)
            upper.clear()
            self.assertEqual(lower, set().union(*(sources[nid, field] for nid in (3, 4, 5, 6))))

    def test_root_reaching_sources_are_never_materialized(self):
        class UnreadableDiff:
            def __iter__(self):
                raise AssertionError('Root-reaching differences must not be aggregated')

        forest = nested_forest()
        # Make the root itself branching: both leaves 1 and 2 reach it.
        forest.nodes[1].parent = 9
        forest.nodes[7].parent = 9
        forest.nodes[9].children = {1, 7}
        del forest.nodes[8]
        for nid in (1, 2, 7):
            forest.nodes[nid]._interior_diff = UnreadableDiff()
            forest.nodes[nid]._codim1_simplex_diff = UnreadableDiff()
        forest.compute_barcode_diff()
        self.assertEqual(sum(bar.is_max_tree_bar for bar in forest.barcode), 2)
        self.assertIsNone(forest.nodes[9]._barcode_interior_diff)
        self.assertIsNone(forest.nodes[9]._barcode_codim1_simplex_diff)

    def test_none_and_empty_differences_stay_distinct(self):
        forest = nested_forest()
        for node in forest.nodes.values():
            node._interior_diff = None
            node._codim1_simplex_diff = None
        forest.nodes[4]._interior_diff = set()
        forest.compute_barcode_diff()
        for nid in (7, 8):
            self.assertEqual(forest.nodes[nid]._barcode_interior_diff, set())
            self.assertIsNone(forest.nodes[nid]._barcode_codim1_simplex_diff)

    def test_literal_representatives_and_interiors_across_modes(self):
        for fixture in (split(), fan(), disconnected(shift_time=1), tetrahedra()):
            for reduce in (False, True):
                for options in ({'diff_only_mode': True}, {}, {'compute_interior': True}):
                    with self.subTest(fixture=fixture.points.tolist(), reduce=reduce, options=options):
                        forest = build(fixture, reduce=reduce, keep_simplex_diff=True, **options)
                        actual = sorted(forest.barcode, key=lambda bar: (bar.birth, bar.death))
                        expected = sorted(fixture.expected, key=lambda bar: (bar[0], bar[1]))
                        self.assertEqual(len(actual), len(expected))
                        for bar, (birth, death, expected_reps) in zip(actual, expected):
                            self.assertEqual((bar.birth, bar.death), (birth, death))
                            reps = list(reversed(list(forest.iter_bar_cycle_reps(bar))))
                            self.assertEqual(len(reps), len(expected_reps))
                            for rep, expected_rep in zip(reps, expected_reps):
                                self.assertEqual((rep.active_start, rep.active_end),
                                                 (expected_rep.start, expected_rep.end))
                                self.assertEqual(rep.signed_simplices, expected_rep.simplices)
                                if options:
                                    self.assertEqual(rep.interior, expected_rep.interior)


if __name__ == '__main__':
    unittest.main()
