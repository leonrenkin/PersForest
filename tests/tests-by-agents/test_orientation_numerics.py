"""Scale-aware orientation checks and fail-fast construction regressions."""

from itertools import combinations
import unittest
import warnings

import numpy as np

from persforest import PersistenceForest
from persforest.PersistenceForest import (
    boundary_faces_from_tetrahedra, sign_of_determinant, simplex_orientation,
)


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

    def test_invalid_and_nonfinite_matrices_raise(self):
        for matrix in ([], np.zeros((2, 2)), np.zeros((2, 3)),
                       [[1, 0], [0, np.inf]], [[1, np.nan], [0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                sign_of_determinant(matrix)

    def test_overflowing_edges_raise_without_runtime_warning(self):
        points = np.array([[-1e308, 0], [1e308, 0], [0, 1e308]])
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with self.assertRaisesRegex(ValueError, 'simplex .*non-finite'):
                simplex_orientation([0, 1, 2], points)

    def test_custom_filtration_rejects_bad_top_simplex_in_each_storage_mode(self):
        filtration = [(simplex, float(size)) for size in (1, 2, 3)
                      for simplex in combinations(range(3), size)]
        for height in (0., 1e-14):
            points = np.array([[0., 0], [1, 0], [2, height]])
            for options in ({}, {'keep_simplex_diff': True},
                            {'diff_only_mode': True, 'keep_simplex_diff': True}):
                with self.subTest(height=height, options=options), warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    with self.assertRaisesRegex(ValueError, r'simplex \(0, 1, 2\)'):
                        PersistenceForest(points, filtration=filtration, **options)

    def test_tetrahedron_boundary_does_not_fabricate_orientation(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        with self.assertRaisesRegex(ValueError, r'simplex \(0, 1, 2, 3\)'):
            boundary_faces_from_tetrahedra([(0, 1, 2, 3)], points)
