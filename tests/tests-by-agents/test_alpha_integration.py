"""A1–A6: real GUDHI alpha geometry, radii, and independent persistence pairs."""
import math
from pathlib import Path
import subprocess
import sys
import unittest
import gudhi
import numpy as np
from persforest import PersistenceForest
from persforest.cycle_rep_vectorisations import signed_chain_area, signed_chain_edge_length, signed_chain_interior_volume
from fixtures_small_filtrations import positive_bars, active_reps
from oracle_chains import coefficients, boundary, homology_data, rank, sample_times


class AlphaIntegrationTests(unittest.TestCase):
    def test_a1_equilateral_triangle_radius_units_and_geometry(self):
        points = np.array([[0, 0], [2, 0], [1, math.sqrt(3)]])
        forest = PersistenceForest(points, keep_simplex_diff=True, compute_interior=True)
        np.testing.assert_allclose(positive_bars(forest), [(1, 2 / math.sqrt(3))], rtol=1e-12, atol=1e-12)
        [bar] = forest.barcode
        [rep] = bar.cycle_reps
        self.assertEqual(rep.signed_simplices, {((0, 1), 1), ((0, 2), -1), ((1, 2), 1)})
        self.assertAlmostEqual(signed_chain_area(rep, points), math.sqrt(3))
        self.assertAlmostEqual(signed_chain_edge_length(rep, points), 6)
        self.assertEqual(len(forest.cycle_reps_at(bar.birth)), 1)
        self.assertEqual(forest.cycle_reps_at(bar.death), [])

    def test_a2_right_triangle_has_no_positive_h1_interval(self):
        forest = PersistenceForest([[0, 0], [1, 0], [0, 1]])
        self.assertEqual(positive_bars(forest), [])
        self.assertEqual(forest.nodes, {})

    def test_a3_square_tie_preserves_square_boundary_not_diagonal(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        for permutation in ([0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]):
            forest = PersistenceForest(points[permutation])
            np.testing.assert_allclose(positive_bars(forest), [(.5, 1 / math.sqrt(2))], rtol=1e-12, atol=1e-12)
            [bar] = forest.barcode
            for rep in bar.cycle_reps:
                support = {tuple(sorted(permutation[i] for i in s)) for s, _ in rep.signed_simplices}
                self.assertEqual(support, {(0, 1), (1, 2), (2, 3), (0, 3)})
                self.assertAlmostEqual(signed_chain_area(rep, forest.point_cloud), 1)

    def test_a4_regular_tetrahedron_radius_barcode_and_volume(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [.5, math.sqrt(3) / 2, 0],
                           [.5, math.sqrt(3) / 6, math.sqrt(2 / 3)]])
        forest = PersistenceForest(points, keep_simplex_diff=True, compute_interior=True)
        np.testing.assert_allclose(positive_bars(forest), [(1 / math.sqrt(3), math.sqrt(3 / 8))], rtol=1e-12, atol=1e-12)
        [bar] = forest.barcode
        [rep] = bar.cycle_reps
        self.assertEqual(len(rep.signed_simplices), 4)
        self.assertEqual(boundary(coefficients(rep.signed_simplices)), {})
        self.assertAlmostEqual(signed_chain_interior_volume(rep, points), math.sqrt(2) / 12)

    def test_a5_seeded_barcode_matches_gudhi_over_two_fields(self):
        for count, dim in [(12, 2), (10, 3)]:
            points = np.random.default_rng(17).random((count, dim))
            forest = PersistenceForest(points, keep_simplex_diff=True, compute_interior=True)
            for field in (2, 3):
                # Keep the owner alive while its iterators or persistence data are used.
                alpha = gudhi.AlphaComplex(points=points)
                tree = alpha.create_simplex_tree(output_squared_values=False)
                pairs = tree.persistence(homology_coeff_field=field, min_persistence=0)
                expected = sorted(interval for dimension, interval in pairs
                                  if dimension == dim - 1 and np.isfinite(interval[1]) and interval[1] > interval[0] + 1e-12)
                np.testing.assert_allclose(positive_bars(forest), expected, rtol=1e-10, atol=1e-12)
            values = {tuple(s): t for s, t in forest.simplex_tree.get_filtration()}
            # Slab midpoints avoid conflating tolerance-collapsed critical values
            # with exact backend events; exact fixture endpoints are tested separately.
            levels = sorted(set(values.values()))
            for time in [(a + b) / 2 for a, b in zip(levels, levels[1:]) if b - a > 2e-12]:
                active = active_reps(forest, time)
                rows, boundaries, betti = homology_data(values, dim, time)
                columns = [coefficients(r.signed_simplices) for r in active]
                self.assertEqual(len(active), betti)
                self.assertEqual(rank(boundaries + columns, rows) - rank(boundaries, rows), betti)
                for rep, column in zip(active, columns):
                    self.assertEqual(boundary(column), {})
                    self.assertEqual(boundary(coefficients(rep.interior)), column)

    def test_alpha_translation_reflection_scaling_and_local_rng(self):
        points = np.array([[0, 0], [2, 0], [1, math.sqrt(3)]])
        # Capture global state without reseeding it.
        state = np.random.get_state()
        first = PersistenceForest(points)
        for transformed, scale in [(points + [7, -3], 1), (points * [-1, 1], 1), (2 * points, 2)]:
            other = PersistenceForest(transformed)
            np.testing.assert_allclose(positive_bars(other), np.array(positive_bars(first)) * scale, rtol=1e-12, atol=1e-12)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        grid = np.linspace(.8, 1.3, 31)
        original = first.compute_measurement_landscapes(signed_chain_edge_length, 'length', x_grid=grid, max_k=1)
        doubled = PersistenceForest(2 * points).compute_measurement_landscapes(signed_chain_edge_length, 'length', x_grid=2 * grid, max_k=1)
        np.testing.assert_allclose(doubled.evaluate_on_grid(2 * grid), 4 * original.evaluate_on_grid(grid), atol=1e-12)

    def test_degenerate_clouds_have_no_codimension_one_bars(self):
        for points in (np.empty((0, 2)), np.array([[0., 0]]), np.array([[0., 0], [1, 0]]),
                       np.array([[0., 0], [1, 0], [2, 0]]),
                       np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]),
                       np.array([[0., 0], [1, 0], [0, 1], [0, 1]])):
            with self.subTest(points=points.tolist()):
                self.assertEqual(positive_bars(PersistenceForest(points)), [])

    def test_d8_malformed_point_array_has_value_error(self):
        with self.assertRaises(ValueError):
            PersistenceForest([0, 1, 2])

    def assert_nonfinite_rejected_in_child(self, expression):
        # GUDHI can raise a native signal, so never send these inputs to the
        # backend in the test runner's own process. No core dumps are needed.
        code = f'''
try:
    import resource
except ImportError:
    pass
else:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import numpy as np
from persforest import PersistenceForest
try:
    PersistenceForest([[0, 0], [{expression}, 1], [1, 0]])
except ValueError:
    print('rejected')
else:
    raise AssertionError('nonfinite coordinates were accepted')
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=15)
        self.assertEqual(result.returncode, 0, f'exit={result.returncode}: {result.stderr}')
        self.assertEqual(result.stdout.strip(), 'rejected')

    def test_d8_nan_points_reject_before_native_backend(self):
        self.assert_nonfinite_rejected_in_child('np.nan')

    def test_d8_infinite_points_reject_before_native_backend(self):
        self.assert_nonfinite_rejected_in_child('np.inf')
