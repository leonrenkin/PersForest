import unittest
from types import SimpleNamespace

import numpy as np

from persforest.PersistenceForest import PersistenceForest
from persforest.forest_landscapes import _build_step_function_data


class BarCycleIteratorTests(unittest.TestCase):
    def test_diff_only_matches_reconstruction_and_stored_cycles(self):
        for dim in (2, 3):
            for reduce in (False, True):
                with self.subTest(dim=dim, reduce=reduce):
                    points = np.random.default_rng(42).random((30, dim))
                    forest = PersistenceForest(
                        points, reduce=reduce, diff_only_mode=True,
                        keep_simplex_diff=True,
                    )
                    full = PersistenceForest(points, reduce=reduce)
                    full_bars = {b._node_progression: b for b in full.barcode}
                    self.assertTrue(any(
                        n._barcode_interior_diff for n in forest.nodes.values()
                    ))
                    for bar in forest.barcode:
                        stored = list(bar.cycle_reps)
                        reps = list(forest.iter_bar_cycle_reps(bar))
                        self.assertEqual(bar.cycle_reps, stored)
                        self.assertTrue(all(r.interior_available for r in reps))
                        reference = list(full.iter_bar_cycle_reps(full_bars[bar._node_progression]))
                        self.assertEqual(
                            [(r.active_start, r.active_end, r.signed_simplices)
                             for r in reps],
                            [(r.active_start, r.active_end, r.signed_simplices)
                             for r in reference],
                        )
                        self.assertEqual(len({id(r.interior) for r in reps}), len(reps))
                        self.assertEqual(len({id(r.signed_simplices) for r in reps}), len(reps))

                        def measurement(chain, points):
                            return sum(
                                np.linalg.norm(points[simplex[0]] - points[simplex[1]])
                                for simplex, _ in chain.signed_simplices
                            )

                        # Compare both iterator paths with the original stored
                        # representative path, including value/interval alignment.
                        full_bar = full_bars[bar._node_progression]
                        original = SimpleNamespace(
                            starts=[r.active_start for r in full_bar.cycle_reps],
                            ends=[r.active_end for r in full_bar.cycle_reps],
                            vals=[measurement(r, points) for r in full_bar.cycle_reps],
                        )
                        for source, source_bar in ((forest, bar), (full, full_bar)):
                            sf = _build_step_function_data(source, source_bar, measurement)
                            self.assertTrue(np.all(np.diff(sf.starts) >= 0))
                            self.assertTrue(np.all(np.diff(sf.ends) >= 0))
                            np.testing.assert_array_equal(sf.starts, original.starts)
                            np.testing.assert_array_equal(sf.ends, original.ends)
                            np.testing.assert_allclose(sf.vals, original.vals)

    def test_empty_step_function(self):
        bar = SimpleNamespace(birth=1., death=2., cycle_reps=[])
        forest = SimpleNamespace(
            barcode=[bar], point_cloud=[], iter_bar_cycle_reps=lambda bar: iter(())
        )
        sf = _build_step_function_data(forest, bar, lambda *_: 1., baseline=3.)
        self.assertEqual(sf.starts.size, 0)
        self.assertEqual(sf.ends.size, 0)
        self.assertEqual(sf.vals.size, 0)
        self.assertEqual(sf.domain, (1., 2.))
        self.assertEqual(sf.baseline, 3.)

    def test_normal_mode_yields_existing_objects_in_descending_order(self):
        forest = PersistenceForest(np.random.default_rng(42).random((15, 2)))
        for bar in forest.barcode:
            reps = list(forest.iter_bar_cycle_reps(bar))
            self.assertEqual(len(reps), len(bar.cycle_reps))
            self.assertTrue(all(a is b for a, b in zip(reps, reversed(bar.cycle_reps))))
            starts = [r.active_start for r in reps]
            self.assertEqual(starts, sorted(starts, reverse=True))

    def test_rejects_foreign_bar_and_missing_diffs(self):
        forest = PersistenceForest(
            np.random.default_rng(42).random((15, 2)),
            diff_only_mode=True, keep_simplex_diff=True,
        )
        with self.assertRaisesRegex(ValueError, "not in barcode"):
            next(forest.iter_bar_cycle_reps(object()))
        bar = next(iter(forest.barcode))
        forest.nodes[bar._node_progression[-1]]._simplex_diff_available = False
        with self.assertRaisesRegex(ValueError, "simplex_diff"):
            next(forest.iter_bar_cycle_reps(bar))


if __name__ == '__main__':
    unittest.main()
