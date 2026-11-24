"""Construction resources: reduction on flat regions and the paused cyclic collector."""
import gc
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from persforest import PersistenceForest
from persforest._forest.reduction import merge_owned_sets
from fixtures_small_filtrations import active_reps


def flat_hole(side: int) -> np.ndarray:
    """A side x side region of value 1 enclosed by a two-pixel frame of value 0."""
    return np.pad(np.ones((side, side)), 2)


def nested_plateaus() -> np.ndarray:
    """Value-3 annulus around a value-1 block whose value-2 centre is a nested hole."""
    values = np.zeros((12, 12))
    values[2:10, 2:10] = 3.0
    values[4:8, 4:8] = 1.0
    values[5:7, 5:7] = 2.0
    return values


def active_signature(forest, time):
    return sorted(tuple(sorted(rep.signed_simplices)) for rep in active_reps(forest, time))


class FlatRegionReductionTests(unittest.TestCase):
    def test_flat_hole_has_one_bar_with_square_boundary_and_full_interior(self):
        forest = PersistenceForest.from_cubical_grid(flat_hole(30))
        self.assertEqual([(bar.birth, bar.death) for bar in forest.barcode], [(0.0, 1.0)])
        chain = forest.cycle_for_bar_at(next(iter(forest.barcode)), 0.5)
        # Interior walls entered with the plateau, so only the 4 * 30 outer facets remain.
        self.assertEqual(len(chain.signed_simplices), 120)
        self.assertEqual(chain.unsigned().signed_simplices, chain.signed_simplices)
        self.assertEqual(len(chain.interior), 900)

    def test_nested_flat_holes_match_unreduced_forest(self):
        reduced = PersistenceForest.from_cubical_grid(nested_plateaus())
        unreduced = PersistenceForest.from_cubical_grid(nested_plateaus(), reduce=False)
        self.assertEqual(sorted((bar.birth, bar.death) for bar in reduced.barcode), [(0.0, 3.0), (1.0, 2.0)])
        for time in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            with self.subTest(time=time):
                self.assertEqual(active_signature(reduced, time), active_signature(unreduced, time))

    def test_reduction_memory_stays_small_on_flat_hole(self):
        # Building a new diff-set union per collapse kept every intermediate set
        # alive: this 30 x 30 flat hole peaked at about 150 MiB of traced
        # allocations, growing quadratically with the area.
        forest = PersistenceForest.from_cubical_grid(flat_hole(30), reduce=False, compute_barcode=False)
        tracemalloc.start()
        try:
            forest._reduce_forest()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 10 * 2**20)
        self.assertEqual(len(forest.nodes), 2)

    def test_merge_owned_sets_reuses_the_larger_set(self):
        larger = {1, 2, 3}
        self.assertIs(merge_owned_sets({4}, larger), larger)
        self.assertEqual(larger, {1, 2, 3, 4})
        kept = {1, 2}
        self.assertIs(merge_owned_sets(kept, {3}), kept)


class CyclicCollectorTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(gc.enable if gc.isenabled() else gc.disable)
        gc.enable()

    def recorder(self, method):
        """Patch a PersistenceForest method to record whether the collector runs."""
        original = getattr(PersistenceForest, method)
        states = []

        def recording(forest, *args, **kwargs):
            states.append(gc.isenabled())
            return original(forest, *args, **kwargs)

        return states, patch.object(PersistenceForest, method, recording)

    def test_build_runs_with_collector_paused_and_restores_it(self):
        factories = {
            'cubical': lambda: PersistenceForest.from_cubical_grid(nested_plateaus()),
            'alpha': lambda: PersistenceForest.from_points(np.random.default_rng(0).random((20, 2))),
        }
        for name, factory in factories.items():
            with self.subTest(factory=name):
                sweep_states, sweep_patch = self.recorder('_compute_forest')
                barcode_states, barcode_patch = self.recorder('_compute_barcode_diff')
                with sweep_patch, barcode_patch:
                    factory()
                self.assertEqual(sweep_states, [False])
                self.assertEqual(barcode_states, [False])
                self.assertTrue(gc.isenabled())

    def test_delayed_barcode_runs_with_collector_paused(self):
        forest = PersistenceForest.from_cubical_grid(nested_plateaus(), compute_barcode=False)
        self.assertTrue(gc.isenabled())
        states, barcode_patch = self.recorder('_compute_barcode_diff')
        with barcode_patch:
            forest.compute_barcode()
        self.assertEqual(states, [False])
        self.assertTrue(gc.isenabled())

    def test_failed_build_restores_collector(self):
        def fail(forest, *args, **kwargs):
            raise RuntimeError('injected failure')

        with patch.object(PersistenceForest, '_reduce_forest', fail):
            with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                PersistenceForest.from_cubical_grid(nested_plateaus())
        self.assertTrue(gc.isenabled())

    def test_collector_disabled_by_caller_stays_disabled(self):
        gc.disable()
        PersistenceForest.from_cubical_grid(nested_plateaus())
        self.assertFalse(gc.isenabled())


if __name__ == '__main__':
    unittest.main()
