"""Cached root lookups in the sweep and barcode extraction equal the uncached walk."""
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from persforest import PersistenceForest
from persforest._forest import build, compute_barcode, queries
from persforest._forest.graph_model import PFNode
from engine_golden import CASES


def smooth_image(shape, seed):
    """Seeded noise smoothed by repeated neighbor averaging: deep forest trees."""
    values = np.random.default_rng(seed).random(shape)
    for _ in range(20):
        for axis in range(values.ndim):
            values = (values + np.roll(values, 1, axis) + np.roll(values, -1, axis)) / 3
    return values


def uncached_root_id(nodes, node_id):
    while nodes[node_id].parent is not None:
        node_id = nodes[node_id].parent
    return node_id


INPUTS = dict(CASES)
INPUTS["cubical_smooth_2d"] = lambda **o: PersistenceForest.from_cubical_grid(smooth_image((40, 40), 1), **o)
INPUTS["cubical_smooth_3d"] = lambda **o: PersistenceForest.from_cubical_grid(smooth_image((12, 12, 12), 2), **o)


class CachedRootLookupTests(unittest.TestCase):
    def test_every_cached_lookup_matches_the_uncached_walk_in_both_phases(self):
        original = queries._cached_root_id
        for name, make in INPUTS.items():
            for storage in ("diffs", "cycles"):
                for reduce in (False, True):
                    with self.subTest(input=name, storage=storage, reduce=reduce):
                        checks = {"sweep": [], "barcode": []}

                        def checked(phase):
                            def lookup(nodes, ancestors, node_id):
                                result = original(nodes, ancestors, node_id)
                                checks[phase].append(result == uncached_root_id(nodes, node_id))
                                return result
                            return lookup

                        with patch.object(build, "_cached_root_id", side_effect=checked("sweep")), \
                             patch.object(compute_barcode, "_cached_root_id", side_effect=checked("barcode")), \
                             warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            forest = make(storage=storage, reduce=reduce)
                        for phase, results in checks.items():
                            if forest.barcode:
                                self.assertTrue(results, phase)
                            self.assertTrue(all(results), phase)
                        for bar in forest.barcode:
                            leaf = forest.nodes[bar._node_progression[-1]]
                            self.assertFalse(leaf.children)
                            self.assertEqual(bar.root_id, forest.get_root(leaf).id)

    def test_path_compression_points_the_walked_path_at_the_root(self):
        # Chain 1 -> 2 -> 3 -> 4 (root); a stale cache entry 1 -> 2 is resumed.
        nodes = {i: PFNode(id=i, filt_val=float(-i), cycle=None, children=set(),
                           parent=(i + 1 if i < 4 else None)) for i in range(1, 5)}
        ancestors = {1: 2}
        self.assertEqual(queries._cached_root_id(nodes, ancestors, 1), 4)
        self.assertEqual(ancestors, {1: 4, 2: 4, 3: 4})
        # A node gaining a parent (as in the sweep) is found through the cache.
        nodes[5] = PFNode(id=5, filt_val=-5.0, cycle=None, children={4})
        nodes[4].parent = 5
        self.assertEqual(queries._cached_root_id(nodes, ancestors, 1), 5)
        self.assertEqual(ancestors[1], 5)
        self.assertEqual(queries._cached_root_id(nodes, ancestors, 5), 5)


if __name__ == "__main__":
    unittest.main()
