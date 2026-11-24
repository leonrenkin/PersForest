import unittest
from types import SimpleNamespace

import numpy as np

from persforest import PersistenceForest
from persforest.landscapes.profiles import _build_step_function_data
from reference_engine import ReferenceForest


def chain_records(chains):
    return [(r.active_start, r.active_end, r.signed_simplices, r.interior) for r in chains]


class BarCycleIteratorTests(unittest.TestCase):
    def test_iterator_matches_point_queries_and_reference_engine(self):
        for dim in (2, 3):
            for reduce in (False, True):
                with self.subTest(dim=dim, reduce=reduce):
                    points = np.random.default_rng(42).random((30, dim))
                    forest = PersistenceForest(points, reduce=reduce)
                    reference = ReferenceForest(forest._cells, reduce=reduce, filtration_tol=forest.filtration_tol)
                    reference_bars = {(b.birth, b.death, b.critical_values): b for b in reference.barcode}
                    for bar in forest.barcode:
                        reps = list(forest.iter_bar_cycle_reps(bar))
                        self.assertEqual(len(reps), len(bar.critical_values))
                        self.assertTrue(all(r.interior_available for r in reps))
                        # Each yielded chain is independent of the iterator's live sets.
                        self.assertEqual(len({id(r.interior) for r in reps}), len(reps))
                        self.assertEqual(len({id(r.signed_simplices) for r in reps}), len(reps))
                        # Incremental replay and single-slice reconstruction agree.
                        point_queries = [forest.cycle_for_bar_at(bar, r.active_start) for r in reps
                                         if r.active_start < r.active_end]
                        self.assertEqual(chain_records(point_queries),
                                         chain_records(r for r in reps if r.active_start < r.active_end))
                        expected = reference.iter_bar_cycle_reps(
                            reference_bars[bar.birth, bar.death, bar.critical_values])
                        self.assertEqual(chain_records(reps), chain_records(expected))

                        def measurement(chain, points):
                            return sum(np.linalg.norm(points[simplex[0]] - points[simplex[1]])
                                       for simplex, _ in chain.signed_simplices)

                        ascending = list(reversed(reps))
                        sf = _build_step_function_data(forest, bar, measurement, signed=True)
                        self.assertTrue(np.all(np.diff(sf.starts) >= 0))
                        self.assertTrue(np.all(np.diff(sf.ends) >= 0))
                        np.testing.assert_array_equal(sf.starts, [r.active_start for r in ascending])
                        np.testing.assert_array_equal(sf.ends, [r.active_end for r in ascending])
                        np.testing.assert_allclose(sf.vals, [measurement(r, points) for r in ascending])

    def test_empty_step_function(self):
        bar = SimpleNamespace(birth=1., death=2.)
        forest = SimpleNamespace(
            barcode=[bar], point_cloud=[], iter_bar_cycle_reps=lambda bar: iter(())
        )
        sf = _build_step_function_data(forest, bar, lambda *_: 1., baseline=3., signed=True)
        self.assertEqual(sf.starts.size, 0)
        self.assertEqual(sf.ends.size, 0)
        self.assertEqual(sf.vals.size, 0)
        self.assertEqual(sf.domain, (1., 2.))
        self.assertEqual(sf.baseline, 3.)

    def test_representatives_come_in_descending_order(self):
        forest = PersistenceForest(np.random.default_rng(42).random((15, 2)))
        for bar in forest.barcode:
            reps = list(forest.iter_bar_cycle_reps(bar))
            starts = [r.active_start for r in reps]
            self.assertEqual(starts, sorted(starts, reverse=True))
            self.assertEqual(reps[0].active_end, bar.death)
            self.assertEqual(reps[-1].active_start, bar.birth)

    def test_rejects_foreign_bar(self):
        forest = PersistenceForest(np.random.default_rng(42).random((15, 2)))
        with self.assertRaisesRegex(ValueError, "not in barcode"):
            next(forest.iter_bar_cycle_reps(object()))


if __name__ == '__main__':
    unittest.main()
