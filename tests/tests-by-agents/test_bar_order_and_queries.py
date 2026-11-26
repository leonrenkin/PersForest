"""Reproducible bar order across processes and validated representative queries."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

import persforest
from persforest import PersistenceForest
from persforest.measurements.cubical import cubical_chain_volume


def tied_grid() -> np.ndarray:
    """Two holes born and dying at the same values, with different perimeters (4 and 6)."""
    grid = np.zeros((5, 7))
    grid[1, 1] = 1
    grid[3, 4] = 1
    grid[3, 5] = 1
    return grid


PROBE = """
import json, numpy as np
from persforest import PersistenceForest
from persforest.measurements.cubical import cubical_chain_volume
grid = np.zeros((5, 7)); grid[1, 1] = 1; grid[3, 4] = 1; grid[3, 5] = 1
forest = PersistenceForest.from_cubical_grid(grid)
profiles = forest.compute_measurement_profiles(cubical_chain_volume, label="perimeter")
print(json.dumps({
    "barcode": [bar.index for bar in forest.barcode],
    "profiles": [float(profiles.step_functions[i].vals[0]) for i in sorted(profiles.step_functions)],
    "longest": [bar.index for bar in forest.longest_bars(2)],
    "max": forest.max_bar().index,
}))
"""


class BarOrderTests(unittest.TestCase):
    def test_barcode_iterates_in_index_order(self):
        rng = np.random.default_rng(3)
        forest = PersistenceForest.from_cubical_grid(rng.integers(0, 4, size=(12, 12)).astype(float))
        self.assertGreater(len(forest.barcode), 3)
        self.assertEqual([bar.index for bar in forest.barcode], list(range(len(forest.barcode))))

    def test_ties_are_broken_by_index(self):
        forest = PersistenceForest.from_cubical_grid(tied_grid())
        bars = sorted(forest.barcode, key=lambda bar: bar.index)
        self.assertEqual(len(bars), 2)
        self.assertEqual((bars[0].birth, bars[0].death), (bars[1].birth, bars[1].death))
        self.assertEqual(bars[0].critical_values, bars[1].critical_values)
        self.assertIs(forest.max_bar(), bars[0])
        self.assertEqual(forest.longest_bars(2), bars)
        profiles = forest.compute_measurement_profiles(cubical_chain_volume, label="perimeter")
        self.assertEqual([profiles.bars[i].tree_id for i in range(2)], [bar.tree_id for bar in bars])

    def test_profiles_are_identical_across_processes(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(persforest.__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
        outputs = []
        for seed in range(6):
            env["PYTHONHASHSEED"] = str(seed)
            result = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True,
                                    text=True, check=True)
            outputs.append(json.loads(result.stdout))
        self.assertTrue(all(output == outputs[0] for output in outputs), outputs)
        self.assertEqual(sorted(outputs[0]["profiles"]), [4.0, 6.0])


class CycleQueryValidationTests(unittest.TestCase):
    def setUp(self):
        self.forest = PersistenceForest.from_cubical_grid(tied_grid())
        self.bar = min(self.forest.barcode, key=lambda bar: bar.index)

    def test_values_outside_the_half_open_lifespan_raise_value_error(self):
        for value in (math.nan, self.bar.death, np.nextafter(self.bar.birth, -np.inf), math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.forest.cycle_for_bar_at(self.bar, value)

    def test_birth_is_included(self):
        rep = self.forest.cycle_for_bar_at(self.bar, self.bar.birth)
        self.assertEqual(rep.active_start, self.bar.birth)
        self.assertGreater(rep.active_end, self.bar.birth)

    def test_each_critical_value_selects_the_next_representative(self):
        rng = np.random.default_rng(5)
        forest = PersistenceForest.from_cubical_grid(rng.integers(0, 6, size=(16, 16)).astype(float))
        checked = 0
        for bar in forest.barcode:
            values = bar.critical_values
            starts = (bar.birth,) + values[:-1]
            for start, end in zip(starts, values):
                if start == end:
                    continue
                for value in (start, np.nextafter(end, -np.inf)):
                    rep = forest.cycle_for_bar_at(bar, value)
                    self.assertEqual((rep.active_start, rep.active_end), (start, end))
                    checked += 1
        self.assertGreater(checked, 10)


if __name__ == "__main__":
    unittest.main()
