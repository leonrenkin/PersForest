"""Validation of explicitly supplied filtrations."""

import unittest
import warnings

import numpy as np

from persforest import PersistenceForest
from persforest.filtrations._validation import _validate_filtration
from fixtures_small_filtrations import mode_options, positive_bars, reps, triangle
from oracle_chains import boundary, coefficients


class ManualFiltrationValidationTests(unittest.TestCase):
    def test_validator_checks_order_and_repeated_vertices(self):
        entries = [((0,), 0), ((1,), 0), ((0, 1), 0)]
        self.assertEqual(_validate_filtration(iter(entries), 2, 2), entries)
        with self.assertRaisesRegex(ValueError, "Face"):
            _validate_filtration([entries[2], *entries[:2]], 2, 2)
        with self.assertRaisesRegex(ValueError, "repeated vertices"):
            _validate_filtration([((0, 0), 0)], 2, 2)

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
        self.assertIn("contractible terminal complex", str(caught[0].message))
        self.assertIn("meaningless", str(caught[0].message))

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
        points = np.vstack((self.fixture.points, [[1, 1]]))
        above_ambient = [((3,), 0)] + self.entries + [((0, 3), 7), ((1, 3), 7), ((2, 3), 7),
                                        ((0, 1, 3), 7), ((0, 2, 3), 7), ((1, 2, 3), 7),
                                        ((0, 1, 2, 3), 8)]
        with self.assertRaisesRegex(ValueError, "above the ambient dimension 2"):
            PersistenceForest(points, filtration=above_ambient)
        for entries, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                PersistenceForest(self.fixture.points, filtration=entries)

    def test_equal_value_coface_must_follow_face(self):
        entries = self.entries[:-1] + [((0, 1, 2), 2)]
        entries.remove(((1, 2), 2))
        entries.append(((1, 2), 2))
        with self.assertRaisesRegex(ValueError, "Face"):
            PersistenceForest(self.fixture.points, filtration=entries)


class VertexOrderTests(unittest.TestCase):
    """Custom simplices are canonicalized; their caller vertex order is irrelevant."""

    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def entries(self, permute):
        values = [((i,), 0.0) for i in range(4)]
        values += [(edge, 1.0) for edge in [(0, 1), (1, 2), (2, 3), (0, 3)]] + [((0, 2), 1.5)]
        values += [(permute((0, 1, 2)), 2.0), (permute((0, 2, 3)), 3.0)]
        return values

    def snapshot(self, forest):
        return sorted(
            (bar.birth, bar.death, tuple(
                (r.active_start, r.active_end, frozenset(r.signed_simplices),
                 frozenset(r.interior) if r.interior_available else None)
                for r in reps(forest, bar)))
            for bar in forest.barcode)

    def test_unsorted_vertex_tuples_give_the_sorted_result_in_every_mode(self):
        permutations = {"sorted": lambda s: s, "swapped": lambda s: (s[1], s[0], s[2]),
                        "reversed": lambda s: tuple(reversed(s))}
        for mode, options in mode_options():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                results = {name: PersistenceForest(self.points, filtration=self.entries(permute), **options)
                           for name, permute in permutations.items()}
            expected = self.snapshot(results["sorted"])
            self.assertEqual(positive_bars(results["sorted"]), [(1.0, 3.0), (1.5, 2.0)])
            for name, forest in results.items():
                with self.subTest(mode=mode, order=name):
                    self.assertEqual(self.snapshot(forest), expected)
                    self.assertTrue(all(len(simplex) == 2 and simplex[0] < simplex[1]
                                        for simplex, _ in forest.filtration if len(simplex) == 2))
                    for bar in forest.barcode:
                        for rep in reps(forest, bar):
                            self.assertEqual(boundary(coefficients(rep.signed_simplices)), {})
                            if rep.interior_available:
                                self.assertEqual(boundary(coefficients(rep.interior)),
                                                 coefficients(rep.signed_simplices))


if __name__ == "__main__":
    unittest.main()
