"""Interior activity from diffs, with explicit merge and tie expectations."""

from collections import defaultdict
from itertools import combinations
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from persforest import PersistenceForest


def split_forest(dim=2, shared_value=3.0, deaths=(5.0, 7.0), **options):
    """Build two embedded simplices sharing a codimension-one face."""
    points = ([[0, 0], [4, 0], [0, 2], [0, -1]] if dim == 2 else
              [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -2]])
    shared = tuple(range(dim))
    simplices = (shared + (dim,), shared + (dim + 1,))
    values = {}
    for simplex, death in zip(simplices, deaths):
        for size in range(1, dim + 1):
            for face in combinations(simplex, size):
                values[face] = 0.0 if size < dim else 1.0
        values[simplex] = death
    values[shared] = shared_value
    filtration = sorted(values.items(), key=lambda item: (item[1], len(item[0]), item[0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return PersistenceForest(points, filtration=filtration, **options)


def normalized(activity):
    """Compare ownership across forests without relying on object identity."""
    return {simplex: sorted((bar._node_progression, bar.birth, bar.death, start, end)
                            for bar, start, end in intervals)
            for simplex, intervals in activity.items()}


def stored_activity(forest):
    """Reference the previous cumulative-interior implementation."""
    activity = defaultdict(list)
    for bar in forest.barcode:
        first_rep = bar.cycle_reps[0]
        for simplex, _orientation in first_rep.interior:
            simplex_key = tuple(sorted(simplex))
            active_end = max(rep.active_end for rep in bar.cycle_reps
                             if any(tuple(sorted(s)) == simplex_key for s, _ in rep.interior))
            activity[simplex_key].append((bar, first_rep.active_start, active_end))
    return dict(activity)


class InteriorSimplexActivityTests(unittest.TestCase):
    def test_seeded_alpha_forests_match_stored_interiors(self):
        for dim in (2, 3):
            for seed in (0, 7):
                points = np.random.default_rng(seed).normal(size=(24, dim))
                for reduce in (False, True):
                    with self.subTest(dim=dim, seed=seed, reduce=reduce):
                        stored = PersistenceForest(points, reduce=reduce,
                                                   keep_simplex_diff=True, compute_interior=True)
                        expected = normalized(stored_activity(stored))
                        self.assertEqual(normalized(stored.interior_simplex_activity()), expected)
                        for mode in ({}, {"diff_only_mode": True}):
                            forest = PersistenceForest(points, reduce=reduce,
                                                       keep_simplex_diff=True, **mode)
                            self.assertEqual(normalized(forest.interior_simplex_activity()), expected)

    def test_explicit_merge_ownership_in_all_diff_modes(self):
        for dim in (2, 3):
            for options in ({"keep_simplex_diff": True},
                            {"keep_simplex_diff": True, "compute_interior": True},
                            {"keep_simplex_diff": True, "diff_only_mode": True}):
                with self.subTest(dim=dim, options=options):
                    forest = split_forest(dim, **options)
                    actual = {simplex: sorted((bar.birth, bar.death, start, end)
                                              for bar, start, end in intervals)
                              for simplex, intervals in forest.interior_simplex_activity().items()}
                    shared = tuple(range(dim))
                    self.assertEqual(actual, {
                        shared + (dim,): [(1, 7, 1, 3), (3, 5, 3, 5)],
                        shared + (dim + 1,): [(1, 7, 1, 7)],
                    })

    def test_matches_stored_interiors_at_exact_ties(self):
        for dim in (2, 3):
            for reduce in (False, True):
                for shared, deaths in ((3, (5, 7)), (1, (5, 7)), (3, (3, 7)),
                                       (3, (7, 7)), (7, (7, 7)), (1, (1, 1))):
                    with self.subTest(dim=dim, reduce=reduce, shared=shared, deaths=deaths):
                        options = dict(dim=dim, shared_value=shared, deaths=deaths,
                                       reduce=reduce, keep_simplex_diff=True)
                        stored = split_forest(**options, compute_interior=True)
                        expected = normalized(stored_activity(stored))
                        for mode in ({}, {"diff_only_mode": True}):
                            forest = split_forest(**options, **mode)
                            self.assertEqual(normalized(forest.interior_simplex_activity()), expected)

    def test_near_tie_uses_reduced_node_endpoints(self):
        tolerance = 2.0 ** -20
        for gap in (tolerance / 2, tolerance, 2 * tolerance):
            for mode in ({}, {"compute_interior": True}, {"diff_only_mode": True}):
                with self.subTest(gap=gap, mode=mode):
                    forest = split_forest(shared_value=3, deaths=(3 + gap, 7),
                                          filtration_tol=tolerance, keep_simplex_diff=True, **mode)
                    intervals = sorted((bar.birth, bar.death, start, end)
                                       for bar, start, end in forest.interior_simplex_activity()[(0, 1, 2)])
                    expected = [(1, 7, 1, 3)]
                    if gap > tolerance:
                        expected.append((3, 3 + gap, 3, 3 + gap))
                    self.assertEqual(intervals, expected)

    def test_intervals_match_half_open_representative_membership(self):
        forest = split_forest(keep_simplex_diff=True, diff_only_mode=True)
        activity = forest.interior_simplex_activity()
        for bar in forest.barcode:
            reps = list(forest.iter_bar_cycle_reps(bar))
            for time in (1, 2, 3, 4, 5, 6, 7):
                expected = {tuple(sorted(simplex)) for rep in reps
                            if rep.active_start <= time < rep.active_end
                            for simplex, _ in rep.interior}
                actual = {simplex for simplex, intervals in activity.items()
                          if any(owner is bar and start <= time < end
                                 for owner, start, end in intervals)}
                self.assertEqual(actual, expected)

    def test_delayed_and_empty_barcodes(self):
        forest = split_forest(keep_simplex_diff=True, compute_barcode=False)
        with self.assertRaisesRegex(RuntimeError, "compute_barcode"):
            forest.interior_simplex_activity()
        forest.compute_barcode()
        self.assertEqual(normalized(forest.interior_simplex_activity()),
                         normalized(split_forest(keep_simplex_diff=True).interior_simplex_activity()))
        empty = PersistenceForest(np.empty((0, 2)), keep_simplex_diff=True)
        self.assertEqual(empty.interior_simplex_activity(), {})
        self.assertEqual(split_forest(shared_value=1, deaths=(1, 1),
                                      keep_simplex_diff=True).interior_simplex_activity(), {})

    def test_missing_diffs_and_wrong_barcode_builder(self):
        with self.assertRaisesRegex(ValueError, "keep_simplex_diff=True"):
            split_forest().interior_simplex_activity()
        forest = split_forest(keep_simplex_diff=True, compute_barcode=False)
        forest.compute_barcode_cycles()
        with self.assertRaisesRegex(RuntimeError, "computed from simplex diffs"):
            forest.interior_simplex_activity()
        forest = split_forest(keep_simplex_diff=True)
        bar = next(iter(forest.barcode))
        forest.nodes[bar._node_progression[-1]]._simplex_diff_available = False
        with self.assertRaisesRegex(ValueError, "diffs are unavailable"):
            forest.interior_simplex_activity()

    def test_query_is_repeatable_without_representative_reconstruction(self):
        forest = split_forest(keep_simplex_diff=True, diff_only_mode=True)
        before = {node_id: (None if node._interior_diff is None else node._interior_diff.copy(),
                            None if node._barcode_interior_diff is None else node._barcode_interior_diff.copy())
                  for node_id, node in forest.nodes.items()}
        with patch.object(forest, "iter_bar_cycle_reps", side_effect=AssertionError("Reconstructed cycles")):
            first = forest.interior_simplex_activity()
            self.assertEqual(first, forest.interior_simplex_activity())
        for node_id, node in forest.nodes.items():
            self.assertEqual((node._interior_diff, node._barcode_interior_diff), before[node_id])
        self.assertTrue(all(rep is None for bar in forest.barcode for rep in bar.cycle_reps))


if __name__ == "__main__":
    unittest.main()
