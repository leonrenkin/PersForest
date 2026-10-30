"""Validation of explicitly supplied filtrations."""

import unittest
import warnings

from persforest import PersistenceForest
from persforest.PersistenceForest import _validate_filtration
from fixtures_small_filtrations import positive_bars, triangle


class ManualFiltrationValidationTests(unittest.TestCase):
    def test_validator_checks_order_and_repeated_vertices(self):
        entries = [((0,), 0), ((1,), 0), ((0, 1), 0)]
        self.assertEqual(_validate_filtration(iter(entries), 2), entries)
        with self.assertRaisesRegex(ValueError, "Face"):
            _validate_filtration([entries[2], *entries[:2]], 2)
        with self.assertRaisesRegex(ValueError, "repeated vertices"):
            _validate_filtration([((0, 0), 0)], 2)

    def setUp(self):
        self.fixture = triangle()
        self.entries = sorted(
            self.fixture.values.items(),
            key=lambda item: (item[1], len(item[0]), item[0]),
        )

    def test_valid_manual_filtration_preserves_barcode_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forest = PersistenceForest(self.fixture.points, filtration=iter(self.entries))

        self.assertEqual(positive_bars(forest), [(2, 6)])
        self.assertEqual(len(caught), 1)
        self.assertIn("embedded and contractible terminal complex", str(caught[0].message))

    def test_invalid_order_or_simplex_is_rejected(self):
        decreasing_values = self.entries.copy()
        decreasing_values[3], decreasing_values[5] = decreasing_values[5], decreasing_values[3]
        cases = [
            (decreasing_values, "nondecreasing"),
            ([(s, t) for s, t in self.entries if s != (0, 1)], "Face"),
            (self.entries[:-1] + [((0, 0, 2), 6)], "repeated vertices"),
            (self.entries + [((2, 1, 0), 7)], "more than once"),
            (self.entries[:-1] + [((0, 1, 2), float("inf"))], "finite"),
            (self.entries[:-1] + [((0, 1, 3), 6)], "invalid vertex"),
        ]
        for entries, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                PersistenceForest(self.fixture.points, filtration=entries)

    def test_equal_value_coface_must_follow_face(self):
        entries = self.entries[:-1] + [((0, 1, 2), 2)]
        entries.remove(((1, 2), 2))
        entries.append(((1, 2), 2))
        with self.assertRaisesRegex(ValueError, "Face"):
            PersistenceForest(self.fixture.points, filtration=entries)


if __name__ == "__main__":
    unittest.main()
