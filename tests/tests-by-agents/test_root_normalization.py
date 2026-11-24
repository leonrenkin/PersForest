"""Roots represent independent births after zero-duration edges collapse."""
import unittest

from fixtures_small_filtrations import (
    active_reps, build, chain, fan, positive_bars, split, triangle,
)


def build_split(fixture, **options):
    """Insert the shared edge last among tied edges to create a root merge."""
    order = sorted(fixture.values.items(),
                   key=lambda item: (item[1], len(item[0]), item[0] == (0, 1), item[0]))
    return build(fixture, order=order, **options)


def structure(forest):
    """Bars with their forest structure, in the forest's deterministic order."""
    bars = sorted(forest.barcode, key=lambda bar: bar.index)
    return [(bar.birth, bar.death, bar.critical_values, bar.tree_id,
             None if forest.parent_bar(bar) is None else forest.parent_bar(bar).index)
            for bar in bars]


class RootNormalizationTests(unittest.TestCase):
    def test_simultaneous_births_become_separate_trees(self):
        fixtures = [
            (split().changed({(0, 1): 1}), [(1, 5), (1, 7)], build_split),
            (fan().changed({(0, 3): 1, (1, 3): 1, (2, 3): 1}), [(1, 5), (1, 6), (1, 7)], build),
        ]
        for fixture, expected, builder in fixtures:
            with self.subTest(branches=len(expected)):
                forest = builder(fixture)
                self.assertEqual(positive_bars(forest), expected)
                self.assertTrue(all(forest.parent_bar(bar) is None for bar in forest.barcode))
                self.assertEqual(len({bar.tree_id for bar in forest.barcode}), len(expected))
                self.assertEqual(len(active_reps(forest, 1)), len(expected))
                self.assertEqual(len(active_reps(forest, 5)), len(expected) - 1)
                self.assertEqual(active_reps(forest, 7), [])

    def test_literal_signed_cycles_and_interiors_at_birth(self):
        forest = build_split(split().changed({(0, 1): 1}))
        expected = {5: (chain('+01 -02 +12'), chain('+012')),
                    7: (chain('-01 +03 -13'), chain('-013'))}
        for bar in forest.barcode:
            rep = forest.cycle_for_bar_at(bar, bar.birth)
            support, interior = expected[bar.death]
            self.assertEqual(rep.signed_simplices, support)
            self.assertEqual(rep.interior, interior)
            self.assertEqual((rep.active_start, rep.active_end), (1, bar.death))

    def test_unary_root_keeps_the_surviving_branch(self):
        forest = build_split(split().changed({(0, 1): 1, (0, 1, 2): 1}))
        self.assertEqual(positive_bars(forest), [(1, 7)])
        self.assertEqual(active_reps(forest, 1)[0].signed_simplices, chain('-01 +03 -13'))

    def test_tolerance_controls_whether_split_is_at_birth(self):
        tolerance = 1e-6
        for delta, tree_count in [(0.5 * tolerance, 2), (2 * tolerance, 1)]:
            with self.subTest(delta=delta):
                forest = build(split().changed({(0, 1): 1 + delta}), filtration_tol=tolerance)
                self.assertEqual(len({bar.tree_id for bar in forest.barcode}), tree_count)
                expected_birth = 1 if tree_count == 2 else 1 + delta
                self.assertEqual(positive_bars(forest), [(1, 7), (expected_birth, 5)]
                                 if tree_count == 1 else [(1, 5), (1, 7)])

    def test_construction_is_deterministic(self):
        fixture = fan().changed({(0, 3): 1, (1, 3): 1, (2, 3): 1})
        self.assertEqual(structure(build(fixture)), structure(build(fixture)))

    def test_unreduced_forest_does_not_split_roots(self):
        forest = build_split(split().changed({(0, 1): 1}), reduce=False)
        self.assertEqual(len({bar.tree_id for bar in forest.barcode}), 1)

    def test_empty_reduced_forest_stays_empty(self):
        forest = build(triangle().changed({(0, 1, 2): 2}))
        self.assertEqual(forest.barcode, set())


if __name__ == '__main__':
    unittest.main()
