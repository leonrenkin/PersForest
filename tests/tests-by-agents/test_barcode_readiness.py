"""Barcode readiness: computed, delayed, empty and failed barcodes."""

import unittest
from unittest.mock import patch

import numpy as np

from persforest import PersistenceForest
from persforest.barcode import PFBar
from test_interior_simplex_activity import split_forest


class BarcodeReadinessTests(unittest.TestCase):
    def test_computed_barcode_is_ready_and_cannot_be_recomputed(self):
        forest = split_forest()
        self.assertTrue(forest._barcode_computed)
        self.assertTrue(forest._barcode_diffs_ready)
        with self.assertRaisesRegex(RuntimeError, "already been called"):
            forest.compute_barcode()
        self.assertTrue(forest._barcode_diffs_ready)

    def test_empty_and_delayed_results_have_explicit_readiness(self):
        forest = PersistenceForest(np.empty((0, 2)), compute_barcode=False)
        self.assertFalse(forest._barcode_diffs_ready)
        with self.assertRaisesRegex(RuntimeError, "call compute_barcode"):
            forest.interior_simplex_activity()
        forest.compute_barcode()
        self.assertTrue(forest._barcode_diffs_ready)
        self.assertEqual(forest.interior_simplex_activity(), {})

    def test_delayed_barcode_matches_eager_barcode(self):
        delayed = split_forest(compute_barcode=False)
        with self.assertRaisesRegex(RuntimeError, "call compute_barcode"):
            delayed.interior_simplex_activity()
        delayed.compute_barcode()
        eager = split_forest()

        def representatives(forest):
            return sorted((bar.birth, bar.death, [(r.active_start, r.active_end, frozenset(r.signed_simplices),
                                                   frozenset(r.interior))
                                                  for r in forest.iter_bar_cycle_reps(bar)])
                          for bar in forest.barcode)

        self.assertEqual(representatives(delayed), representatives(eager))

    def test_failed_bar_creation_does_not_publish_readiness(self):
        forest = split_forest(compute_barcode=False)
        with patch.object(PFBar, "_all_from_forest", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                forest.compute_barcode()
        self.assertFalse(forest._barcode_computed)
        self.assertFalse(forest._barcode_diffs_ready)
        self.assertEqual(forest.barcode, set())

    def test_storage_option_no_longer_exists(self):
        with self.assertRaises(TypeError):
            PersistenceForest(np.empty((0, 2)), storage="diffs")


if __name__ == "__main__":
    unittest.main()
