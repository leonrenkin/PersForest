"""Construction output matches snapshots recorded before the cell-interface refactor."""
import json
import unittest

from engine_golden import GOLDEN_PATH, compute_all


class EngineGoldenTests(unittest.TestCase):
    def test_bars_representatives_interiors_and_activity_match_snapshots(self):
        expected = json.loads(GOLDEN_PATH.read_text())
        actual = compute_all()
        self.assertEqual(sorted(actual), sorted(expected))
        for case, record in expected.items():
            with self.subTest(case=case):
                self.assertEqual(actual[case], record)


if __name__ == '__main__':
    unittest.main()
