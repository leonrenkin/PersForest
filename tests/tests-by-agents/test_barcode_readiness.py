"""Barcode capabilities and internal builder dispatch after the readiness fix."""

import unittest
from unittest.mock import patch

import numpy as np

from persforest import PersistenceForest
from test_interior_simplex_activity import split_forest


class BarcodeReadinessTests(unittest.TestCase):
    def test_dispatch_marks_completed_barcodes_ready(self):
        for storage in ("diffs", "cycles"):
            with self.subTest(storage=storage):
                forest = split_forest(storage=storage)
                self.assertTrue(forest._barcode_computed)
                self.assertTrue(forest._barcode_diffs_ready)
                with self.assertRaisesRegex(RuntimeError, "already been called"):
                    forest.compute_barcode()
                self.assertTrue(forest._barcode_diffs_ready)

    def test_empty_and_delayed_results_have_explicit_readiness(self):
        for storage in ("diffs", "cycles"):
            with self.subTest(storage=storage):
                forest = PersistenceForest(np.empty((0, 2)), storage=storage, compute_barcode=False)
                self.assertFalse(forest._barcode_diffs_ready)
                with self.assertRaisesRegex(RuntimeError, "call compute_barcode"):
                    forest.interior_simplex_activity()
                forest.compute_barcode()
                self.assertTrue(forest._barcode_diffs_ready)
                self.assertEqual(forest.interior_simplex_activity(), {})

    def test_delayed_cycles_storage_attaches_interiors(self):
        delayed = split_forest(storage="cycles", compute_barcode=False)
        with self.assertRaisesRegex(RuntimeError, "call compute_barcode"):
            delayed._add_interior_to_barcode()
        delayed.compute_barcode()
        eager = split_forest(storage="cycles")
        snapshot = lambda forest: sorted(
            (bar.birth, bar.death, [(r.active_start, r.active_end, frozenset(r.interior))
                                    for r in bar.cycle_reps])
            for bar in forest.barcode)
        self.assertTrue(all(r.interior_available for bar in delayed.barcode for r in bar.cycle_reps))
        self.assertEqual(snapshot(delayed), snapshot(eager))

    def test_failed_builder_does_not_publish_readiness(self):
        forest = split_forest(storage="cycles", compute_barcode=False)
        with patch.object(forest, "get_root", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                forest.compute_barcode()
        self.assertFalse(forest._barcode_computed)
        self.assertFalse(forest._barcode_diffs_ready)

    def test_only_internal_builder_is_available_and_used_by_dispatch(self):
        self.assertFalse(hasattr(PersistenceForest, "_compute_barcode_cycles"))
        self.assertFalse(hasattr(PersistenceForest, "compute_barcode_diff"))
        for storage in ("diffs", "cycles"):
            with self.subTest(storage=storage):
                original = PersistenceForest._compute_barcode_diff
                with patch.object(PersistenceForest, "_compute_barcode_diff", autospec=True,
                                  side_effect=original) as internal:
                    forest = split_forest(storage=storage)
                    internal.assert_called_once_with(forest, print_info=False)
                delayed = split_forest(storage=storage, compute_barcode=False)
                with patch.object(delayed, "_compute_barcode_diff",
                                  wraps=delayed._compute_barcode_diff) as internal:
                    delayed.compute_barcode()
                    internal.assert_called_once_with(print_info=False)

    def test_invalid_storage_is_rejected(self):
        for storage in ("stored", None, "diff_only"):
            with self.subTest(storage=storage), self.assertRaisesRegex(ValueError, "storage must be"):
                PersistenceForest(np.empty((0, 2)), storage=storage)


if __name__ == "__main__":
    unittest.main()
