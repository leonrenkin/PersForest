"""Explicit outcomes of ``PersistenceForest.max_bar`` for empty and uncomputed barcodes."""

import unittest

import numpy as np

from persforest import PersistenceForest
from test_interior_simplex_activity import split_forest


class MaxBarTests(unittest.TestCase):
    def test_returns_a_longest_bar(self):
        forest = split_forest()
        longest = max(bar.lifespan() for bar in forest.barcode)
        self.assertIn(forest.max_bar(), forest.barcode)
        self.assertEqual(forest.max_bar().lifespan(), longest)

    def test_computed_empty_barcode_raises_value_error(self):
        forest = PersistenceForest(np.empty((0, 2)))
        self.assertTrue(forest._barcode_computed)
        with self.assertRaisesRegex(ValueError, "Barcode is empty"):
            forest.max_bar()

    def test_uncomputed_barcode_raises_runtime_error(self):
        forest = split_forest(compute_barcode=False)
        with self.assertRaisesRegex(RuntimeError, "call compute_barcode"):
            forest.max_bar()
        forest.compute_barcode()
        self.assertIn(forest.max_bar(), forest.barcode)


if __name__ == "__main__":
    unittest.main()
