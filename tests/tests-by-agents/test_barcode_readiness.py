"""Barcode capabilities and internal builder dispatch after the readiness fix."""

import unittest
from unittest.mock import patch

import numpy as np

from persforest import PersistenceForest
from test_interior_simplex_activity import split_forest


class BarcodeReadinessTests(unittest.TestCase):
    def test_dispatch_marks_only_completed_diff_barcodes_ready(self):
        for options in ({}, {"keep_simplex_diff": True}, {"diff_only_mode": True},
                        {"keep_simplex_diff": True, "compute_interior": True}):
            with self.subTest(options=options):
                forest = split_forest(**options)
                self.assertTrue(forest._barcode_computed)
                self.assertEqual(forest._barcode_diffs_ready, forest.keep_simplex_diff)
                before = forest._barcode_diffs_ready
                with self.assertRaisesRegex(RuntimeError, "already been called"):
                    forest.compute_barcode()
                self.assertEqual(forest._barcode_diffs_ready, before)

    def test_empty_and_delayed_results_have_explicit_readiness(self):
        forest = PersistenceForest(np.empty((0, 2)), keep_simplex_diff=True,
                                   compute_barcode=False)
        self.assertFalse(forest._barcode_diffs_ready)
        for query in (forest.interior_simplex_activity, forest._add_interior_to_barcode):
            with self.assertRaisesRegex(RuntimeError, "call compute_barcode"):
                query()
        forest.compute_barcode()
        self.assertTrue(forest._barcode_diffs_ready)
        self.assertEqual(forest.interior_simplex_activity(), {})
        forest._add_interior_to_barcode()

        eager = PersistenceForest(np.empty((0, 2)), keep_simplex_diff=True,
                                  compute_barcode=False)
        eager._compute_barcode_cycles()
        with self.assertRaisesRegex(RuntimeError, "computed from simplex diffs"):
            eager.interior_simplex_activity()

    def test_eager_builder_rejects_interior_consumers_before_mutation(self):
        forest = split_forest(keep_simplex_diff=True, compute_barcode=False)
        forest._compute_barcode_cycles()
        bar = next(iter(forest.barcode))
        reps = list(forest.iter_bar_cycle_reps(bar))
        self.assertTrue(reps)  # Stored boundary representatives remain usable.
        before = [(rep.interior_available, rep.interior) for rep in reps]
        for query in (forest.interior_simplex_activity, forest._add_interior_to_barcode,
                      lambda: forest._add_interior_to_bar(bar)):
            with self.assertRaisesRegex(RuntimeError, "computed from simplex diffs"):
                query()
        self.assertEqual([(rep.interior_available, rep.interior) for rep in reps], before)

    def test_diff_only_reconstruction_rejects_eager_builder(self):
        forest = split_forest(diff_only_mode=True, compute_barcode=False)
        forest._compute_barcode_cycles()
        bar = next(iter(forest.barcode))
        with self.assertRaisesRegex(RuntimeError, "computed from simplex diffs"):
            list(forest.iter_bar_cycle_reps(bar))
        with self.assertRaisesRegex(RuntimeError, "computed from simplex diffs"):
            forest.cycle_for_bar_at(bar, bar.birth)

    def test_failed_builder_does_not_publish_readiness(self):
        forest = split_forest(keep_simplex_diff=True, compute_barcode=False)
        with patch.object(forest, "get_root", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                forest.compute_barcode()
        self.assertFalse(forest._barcode_computed)
        self.assertFalse(forest._barcode_diffs_ready)

    def test_only_internal_builders_are_available_and_used_by_dispatch(self):
        for diffs, name in ((False, "_compute_barcode_cycles"), (True, "_compute_barcode_diff")):
            with self.subTest(diffs=diffs):
                self.assertFalse(hasattr(PersistenceForest, name[1:]))
                original = getattr(PersistenceForest, name)
                with patch.object(PersistenceForest, name, autospec=True,
                                  side_effect=original) as internal:
                    forest = split_forest(keep_simplex_diff=diffs)
                    internal.assert_called_once_with(forest, print_info=False)
                delayed = split_forest(keep_simplex_diff=diffs, compute_barcode=False)
                with patch.object(delayed, name, wraps=getattr(delayed, name)) as internal:
                    delayed.compute_barcode()
                    internal.assert_called_once_with(print_info=False)


if __name__ == "__main__":
    unittest.main()
