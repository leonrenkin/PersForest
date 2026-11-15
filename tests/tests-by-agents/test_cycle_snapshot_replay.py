"""Copy-free snapshot replay: cycle_for_bar_at equals the iterated representatives."""
import unittest

import numpy as np

from persforest import PersistenceForest
from fixtures_small_filtrations import annulus, build, disconnected, fan, mode_options, split


def _forests():
    rng = np.random.default_rng(7)
    yield "alpha2", PersistenceForest.from_points(rng.uniform(0, 1, (300, 2)))
    yield "alpha3", PersistenceForest.from_points(rng.uniform(0, 1, (120, 3)))
    yield "cubical2", PersistenceForest.from_cubical_grid(rng.uniform(0, 1, (20, 20)))
    yield "cubical2-vertices", PersistenceForest.from_cubical_grid(
        rng.uniform(0, 1, (15, 15)), filtration_type="vertices")
    yield "cubical3", PersistenceForest.from_cubical_grid(rng.uniform(0, 1, (6, 6, 6)))
    for fixture in (split, fan, annulus, disconnected):
        for name, options in mode_options():
            yield f"{fixture.__name__}-{name}", build(fixture(), **options)


def _key(chain):
    return (frozenset(chain.signed_simplices), frozenset(chain.interior or ()),
            chain.active_start, chain.active_end, chain.interior_available)


class SnapshotReplayTests(unittest.TestCase):
    def test_snapshot_equals_first_active_iterated_representative(self):
        for name, forest in _forests():
            with self.subTest(forest=name):
                for bar in forest.barcode:
                    reps = list(forest.iter_bar_cycle_reps(bar))
                    times = {r.active_start for r in reps} | {r.active_end for r in reps}
                    for time in sorted(times):
                        if not bar.birth <= time < bar.death:
                            continue
                        expected = next(r for r in reps if r.active_start <= time < r.active_end)
                        self.assertEqual(_key(forest.cycle_for_bar_at(bar, time)), _key(expected))

    def test_half_open_activity_and_lifespan_errors(self):
        forest = PersistenceForest.from_points(np.random.default_rng(3).uniform(0, 1, (200, 2)))
        bar = forest.max_bar()
        reps = list(forest.iter_bar_cycle_reps(bar))
        inner = [r for r in reps if r.active_end > r.active_start]
        for rep in inner:
            # Start included, end excluded: the end belongs to the next later representative.
            self.assertEqual(_key(forest.cycle_for_bar_at(bar, rep.active_start)), _key(rep))
        for value in (bar.death, bar.birth - 1.0):
            with self.assertRaises(ValueError):
                forest.cycle_for_bar_at(bar, value)

    def test_returned_chains_are_independent(self):
        forest = PersistenceForest.from_points(np.random.default_rng(4).uniform(0, 1, (200, 2)))
        bar = forest.max_bar()
        time = bar.birth
        first = forest.cycle_for_bar_at(bar, time)
        reference = _key(first)
        first.signed_simplices.clear()
        first.interior.clear()
        self.assertEqual(_key(forest.cycle_for_bar_at(bar, time)), reference)

    def test_snapshot_matches_stored_cycles_storage(self):
        points = np.random.default_rng(5).uniform(0, 1, (150, 2))
        diffs = PersistenceForest(points, storage="diffs")
        cycles = PersistenceForest(points, storage="cycles")
        stored = {bar._node_progression: bar for bar in cycles.barcode}
        for bar in diffs.barcode:
            other = stored[bar._node_progression]
            for time in {bar.birth, (bar.birth + bar.death) / 2}:
                if bar.birth <= time < bar.death:
                    self.assertEqual(
                        frozenset(diffs.cycle_for_bar_at(bar, time).signed_simplices),
                        frozenset(cycles.cycle_for_bar_at(other, time).signed_simplices))


if __name__ == "__main__":
    unittest.main()
