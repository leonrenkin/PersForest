"""Scale-aware orientation checks and fail-fast construction regressions."""

from itertools import combinations
import unittest
import warnings

import numpy as np

from persforest import PersistenceForest
from persforest.geometry.simplicial import boundary_faces_from_tetrahedra, simplex_orientation, simplex_orientations


class OrientationNumericsTests(unittest.TestCase):
    def test_uniform_scaling_preserves_sign_and_reversal(self):
        for dim in (1, 2, 3, 4):
            points = np.vstack((np.zeros(dim), np.eye(dim)))
            simplex = list(range(dim + 1))
            reversed_simplex = [1, 0] + simplex[2:]
            for scale in (1e-200, 1e-13, 1., 1e13, 1e200):
                with self.subTest(dim=dim, scale=scale):
                    self.assertEqual(simplex_orientation(simplex, points * scale), 1)
                    self.assertEqual(simplex_orientation(reversed_simplex, points * scale), -1)

    def test_degenerate_and_nearly_flat_simplices_raise_at_every_scale(self):
        for dim in (2, 3):
            for height in (0., 1e-14):
                points = np.vstack((np.zeros(dim), np.eye(dim)))
                points[-1] = points[1]
                points[-1, -1] = height
                for scale in (1e-100, 1., 1e100):
                    with self.subTest(dim=dim, height=height, scale=scale):
                        with self.assertRaisesRegex(ValueError, 'simplex .*near-degenerate'):
                            simplex_orientation(list(range(dim + 1)), points * scale)

    def test_invalid_and_nonfinite_simplices_raise(self):
        # Edge vectors as rows: a zero matrix, an infinite and a NaN entry.
        for edges in (np.zeros((2, 2)), [[1, 0], [0, np.inf]], [[1, np.nan], [0, 1]]):
            with self.subTest(edges=edges), self.assertRaisesRegex(ValueError, r'simplex \(0, 1, 2\)'):
                simplex_orientation([0, 1, 2], np.vstack((np.zeros(2), edges)))
        points = np.vstack((np.zeros(2), np.eye(2)))
        for simplex in ([], [0, 1], [0, 1, 2, 2], [0, 1, 3], [-1, 1, 2]):
            with self.subTest(simplex=simplex), self.assertRaises(ValueError):
                simplex_orientation(simplex, points)

    def test_batched_orientations_match_single_simplices_and_name_the_first_failure(self):
        rng = np.random.default_rng(5)
        for dim in (1, 2, 3, 4, 5):
            points = rng.normal(size=(30, dim))
            simplices = np.array([rng.choice(30, dim + 1, replace=False) for _ in range(200)])
            with self.subTest(dim=dim):
                expected = [simplex_orientation(simplex.tolist(), points) for simplex in simplices]
                np.testing.assert_array_equal(simplex_orientations(simplices, points), expected)
                # The sign agrees with NumPy's determinant of the edge vectors.
                edges = points[simplices[:, 1:]] - points[simplices[:, :1]]
                np.testing.assert_array_equal(expected, np.sign(np.linalg.det(edges)))
        self.assertEqual(simplex_orientations(np.zeros((0, 3), dtype=int), np.zeros((4, 2))).shape, (0,))
        points = np.array([[0., 0], [1, 0], [2, 0], [0, 1]])
        with self.assertRaisesRegex(ValueError, r'simplex \(0, 1, 2\)'):
            simplex_orientations([[0, 1, 3], [0, 1, 2], [1, 2, 0]], points)

    def test_overflowing_edges_raise_without_runtime_warning(self):
        points = np.array([[-1e308, 0], [1e308, 0], [0, 1e308]])
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with self.assertRaisesRegex(ValueError, 'simplex .*non-finite'):
                simplex_orientation([0, 1, 2], points)

    def test_custom_filtration_rejects_bad_top_simplex(self):
        filtration = [(simplex, float(size)) for size in (1, 2, 3)
                      for simplex in combinations(range(3), size)]
        for height in (0., 1e-14):
            points = np.array([[0., 0], [1, 0], [2, height]])
            for options in ({}, {'reduce': False}, {'compute_barcode': False}):
                with self.subTest(height=height, options=options), warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    with self.assertRaisesRegex(ValueError, r'simplex \(0, 1, 2\)'):
                        PersistenceForest(points, filtration=filtration, **options)

    def test_tetrahedron_boundary_does_not_fabricate_orientation(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        with self.assertRaisesRegex(ValueError, r'simplex \(0, 1, 2, 3\)'):
            boundary_faces_from_tetrahedra([(0, 1, 2, 3)], points)
