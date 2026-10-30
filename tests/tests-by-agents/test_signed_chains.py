"""C1–C8: signs, projection, cancellation, and geometric representations."""
from itertools import permutations
import unittest
import numpy as np
from persforest.PersistenceForest import (
    SignedChain, signed_boundary, simplex_orientation, sign_of_determinant,
    merge_at_simplex, update_chain_with_diff,
    key, are_dict_keys_sorted, union_optional_sets,
)
from fixtures_small_filtrations import chain, fan
from oracle_chains import coefficients, boundary, canonical


class SignedChainTests(unittest.TestCase):
    def test_literal_boundaries_and_sign_reversal(self):
        for simplex, expected in [([0, 1, 2], chain('+12 -02 +01')),
                                  ([0, 1, 2, 3], chain('+123 -023 +013 -012'))]:
            self.assertEqual(signed_boundary(simplex, 1), expected)
            self.assertEqual(signed_boundary(simplex, -1), {(s, -v) for s, v in expected})
            self.assertEqual(boundary(coefficients(expected)), {})

    def test_orientation_under_all_vertex_permutations(self):
        for points in (np.array([[0, 0], [3, 0], [0, 4]]),
                       np.vstack((np.zeros(3), np.eye(3)))):
            simplex = tuple(range(len(points)))
            for perm in permutations(simplex):
                with self.subTest(perm=perm):
                    _, sign = canonical(perm)
                    self.assertEqual(simplex_orientation(perm, points), sign)
                    projected = coefficients(signed_boundary(list(perm), sign))
                    self.assertEqual(projected, boundary({simplex: 1}))

    def test_determinant_sign_and_degeneracy(self):
        for matrix, expected in [(np.eye(3), 1), ([[0, 1], [1, 0]], -1),
                                 ([[2, 3], [0, -4]], -1)]:
            self.assertEqual(sign_of_determinant(matrix), expected)
        with self.assertRaises(ValueError):
            sign_of_determinant([[1, 2], [2, 4]])
        with self.assertRaisesRegex(ValueError, r'simplex \(0, 1, 2\).*degenerate'):
            simplex_orientation([0, 1, 2], np.array([[0, 0], [1, 0], [2, 0]]))

    def test_cancel_projection_double_part_and_merge_do_not_mutate(self):
        original = chain('+01 -02 +12 +03 -03')
        value = SignedChain(set(original), 2, 3, True, {((0, 1, 2), 1)})
        for actual in (value.unsigned(), value.cancel_simplex([0, 3])):
            self.assertEqual(actual.signed_simplices, chain('+01 -02 +12'))
        self.assertEqual(value.only_double_simplices().signed_simplices, chain('+03 -03'))
        self.assertEqual(value.unsigned().active_start, 2)
        self.assertEqual(value.unsigned().active_end, 3)
        self.assertTrue(value.unsigned().interior_available)
        self.assertEqual(value.signed_simplices, original)
        a, b = SignedChain(set(chain('+01 -02 +12'))), SignedChain(set(chain('-01 +03 -13')))
        self.assertEqual(merge_at_simplex(a, b, [0, 1]).signed_simplices, chain('-02 +12 +03 -13'))
        self.assertIn(((0, 1), 1), a.signed_simplices)
        self.assertIn(((0, 1), -1), b.signed_simplices)

    def test_diff_update_adds_signed_boundaries_and_removes_only_requested_terms(self):
        initial = set(chain('+34'))
        result = update_chain_with_diff(initial, set(chain('+012 -013')), set(chain('+01 -01')))
        self.assertIs(result, initial)
        self.assertEqual(result, chain('+34 -02 +12 +03 -13'))
        self.assertIs(update_chain_with_diff(result, None, None), result)

    def test_segments_simplices_coordinates_and_dimensions(self):
        points = fan().points
        value = SignedChain(set(chain('+01 -02 +03 -03')))
        segments = {tuple(map(tuple, segment)) for segment in value.segments(points)}
        self.assertEqual(segments, {(tuple(points[0]), tuple(points[1])),
                                    (tuple(points[2]), tuple(points[0])),
                                    (tuple(points[0]), tuple(points[3])),
                                    (tuple(points[3]), tuple(points[0]))})
        self.assertCountEqual(value.simplices(False), [(0, 1), (0, 2)])
        self.assertEqual(len(value.simplices(True)), 4)
        np.testing.assert_array_equal(value.vertex_coordinates(points), points)
        np.testing.assert_array_equal(value.vertex_coordinates(points, signed=False), points[:3])
        self.assertEqual(value.dim(), 1)
        self.assertEqual(SignedChain(set(chain('+012'))).dim(), 2)

    def test_empty_projection_and_small_set_utilities(self):
        empty = SignedChain(set())
        self.assertEqual(empty.unsigned().signed_simplices, set())
        self.assertEqual(empty.only_double_simplices().signed_simplices, set())
        self.assertEqual(boundary(coefficients(empty.signed_simplices)), {})
        self.assertEqual(key([3, 1, 2]), (1, 2, 3))
        self.assertTrue(are_dict_keys_sorted({}))
        self.assertTrue(are_dict_keys_sorted({1: 0, 3: 0}))
        self.assertFalse(are_dict_keys_sorted({3: 0, 1: 0}))
        self.assertIsNone(union_optional_sets(None, None))
        self.assertEqual(union_optional_sets({1}, None), {1})
        self.assertEqual(union_optional_sets(None, {2}), {2})
        self.assertEqual(union_optional_sets({1}, {2}), {1, 2})
