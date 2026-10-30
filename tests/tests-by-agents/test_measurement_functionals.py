"""G1–G9: analytic values, signed multiplicity, shape scores, and invariance."""
from collections import Counter
import math
import unittest
import numpy as np
from persforest.PersistenceForest import SignedChain
from persforest import cycle_rep_vectorisations as v
from fixtures_geometry import square, ell, ring, disjoint_squares, touching_triangles, doubled, polygon
from fixtures_small_filtrations import fan, build, chain
from oracle_chains import simplex_volume


class PolygonMeasurementTests(unittest.TestCase):
    def test_g1_all_polygon_ratios_have_explicit_formula(self):
        points, _ = square()
        cases = [(v.polygon_length, 8), (v.polygon_area, 4),
                 (v.polygon_area_length_ratio, 1 / 16),
                 (v.polygon_area_length_squared_ratio, .5),
                 (v.polygon_length_area_ratio, 2),
                 (v.polygon_length_squared_area_ratio_normalized, 16 - 4 * math.pi),
                 (v.polygon_length_squared_area_ratio, 16),
                 (v.total_curvature, 2 * math.pi), (v.curvature_excess, 0)]
        for func, expected in cases:
            for coords in (points, points[::-1], np.vstack((points, points[0]))):
                with self.subTest(function=func.__name__, points=coords.tolist()):
                    self.assertAlmostEqual(func(coords), expected, places=12)
        self.assertEqual(v.polygon_area(points, signed=True), 4)
        self.assertEqual(v.polygon_area(points[::-1], signed=True), -4)

    def test_g2_concave_polygon_curvature(self):
        points, _ = ell()
        self.assertEqual(v.polygon_area(points), 3)
        self.assertEqual(v.polygon_length(points), 8)
        self.assertAlmostEqual(v.total_curvature(points), 3 * math.pi)
        self.assertAlmostEqual(v.curvature_excess(points), .5)
        self.assertAlmostEqual(v.curvature_excess(points, normalize=False), math.pi)

    def test_area_threshold_is_strict_and_bad_polygon_rejected(self):
        # Binary-exact triangle area 1/2, so threshold equality is unambiguous.
        points = np.array([[0., 0], [1, 0], [0, 1]])
        perimeter = 2 + math.sqrt(2)
        for tol in (.25, .5, .75):
            self.assertAlmostEqual(v.polygon_length_area_ratio(points, tol=tol),
                                   0 if tol > .5 else 2 * perimeter)
            self.assertAlmostEqual(v.polygon_length_squared_area_ratio_normalized(points, tol=tol),
                                   0 if tol > .5 else 2 * perimeter**2 - 4 * math.pi)
        with self.assertRaises(ValueError):
            v.polygon_area(np.zeros((4, 3)))
        with self.assertRaisesRegex(ValueError, 'Zero-length'):
            v.total_curvature([(0, 0), (1, 0), (1, 0), (0, 1)])


class SignedMeasurementTests(unittest.TestCase):
    def test_g1_square_all_shape_functionals(self):
        points, rep = square()
        circularity = math.pi / 4
        cases = [(v.signed_chain_area, 4), (v.signed_chain_edge_length, 8),
                 (v.constant_one_functional, 1), (v.signed_chain_connected_components, 1),
                 (v.signed_chain_excess_connected_components, 0),
                 (v.signed_chain_connected_components_only_signed_simplices, 0),
                 (v.signed_chain_avg_tendril_length, 0),
                 (v.signed_chain_num_of_branching_points, 0),
                 (v.signed_chain_num_of_branching_points_only_signed_simplices, 0),
                 (v.signed_chain_tendril_branching_ratio, 0),
                 (v.signed_chain_convex_hull_area_deficit, 0),
                 (v.signed_chain_convex_hull_perimeter_deficit, 0),
                 (v.signed_chain_max_convexity_defect_depth, 0),
                 (v.signed_chain_excess_curvature, 0),
                 (v.signed_chain_excess_curvature_normalized, 0),
                 (v.signed_chain_excess_curvature_diff_to_unsigned, 0),
                 (v.signed_chain_circularity, circularity),
                 (v.signed_chain_circularity_complement, 1 - circularity),
                 (v.signed_chain_circularity_squared_complement, 1 - circularity**2),
                 (v.signed_chain_circularity_complement_squared, (1 - circularity)**2),
                 (v.signed_chain_non_circularity, 1 / circularity - 1)]
        for func, expected in cases:
            with self.subTest(function=func.__name__):
                self.assertAlmostEqual(func(rep, points), expected, places=12)

    def test_g2_concavity_has_exact_scores(self):
        points, rep = ell()
        for func, expected in [(v.signed_chain_area, 3), (v.signed_chain_edge_length, 8),
                               (v.signed_chain_convex_hull_area_deficit, 1 / 7),
                               (v.signed_chain_convex_hull_perimeter_deficit, 1 - (6 + math.sqrt(2)) / 8),
                               (v.signed_chain_max_convexity_defect_depth, .25),
                               (v.signed_chain_excess_curvature, math.pi),
                               (v.signed_chain_excess_curvature_normalized, .5),
                               (v.signed_chain_circularity, 3 * math.pi / 16)]:
            with self.subTest(function=func.__name__):
                self.assertAlmostEqual(func(rep, points), expected, places=12)
        self.assertAlmostEqual(v.signed_chain_max_convexity_defect_depth(rep, points, normalize=False), 1 / math.sqrt(2))

    def test_g3_hole_subtracts_area_but_adds_boundary_length(self):
        points, rep = ring()
        for func, expected in [(v.signed_chain_area, 12), (v.signed_chain_edge_length, 24),
                               (v.signed_chain_connected_components, 2),
                               (v.signed_chain_excess_connected_components, 1),
                               (v.signed_chain_excess_curvature_normalized, 0),
                               (v.signed_chain_circularity, math.pi / 12),
                               (v.signed_chain_convex_hull_area_deficit, .25),
                               (v.signed_chain_convex_hull_perimeter_deficit, 1 / 3)]:
            self.assertAlmostEqual(func(rep, points), expected, places=12)

    def test_g4_disjoint_boundary_length_and_components(self):
        points, rep = disjoint_squares()
        self.assertEqual(v.signed_chain_edge_length(rep, points), 8)
        self.assertEqual(v.signed_chain_connected_components(rep, points), 2)
        self.assertEqual(v.signed_chain_excess_connected_components(rep, points), 1)
        self.assertAlmostEqual(v.signed_chain_excess_curvature(rep, points), 0)

    def test_g6_to_g8_signed_incidence_and_geometric_tendril_convention(self):
        for kind, length, branch in [('edge', 2, 0), ('path', 4, 1), ('y', 6, 1)]:
            points, rep = doubled(kind)
            with self.subTest(kind=kind):
                self.assertAlmostEqual(v.signed_chain_edge_length(rep, points), length)
                self.assertEqual(rep.unsigned().signed_simplices, set())
                self.assertAlmostEqual(v.signed_chain_area(rep, points), 0)
                self.assertEqual(v.signed_chain_connected_components_only_signed_simplices(rep, points), 1)
                self.assertEqual(v.signed_chain_num_of_branching_points(rep, points), branch)
                self.assertEqual(v.signed_chain_num_of_branching_points_only_signed_simplices(rep, points), branch)
                self.assertEqual(v.signed_chain_tendril_branching_ratio(rep, points), branch)
                # Compatibility: current documented geometric, not signed-volume, average (D5).
                self.assertAlmostEqual(v.signed_chain_avg_tendril_length(rep, points), length / 2)
        points, rep = doubled()
        self.assertEqual(v.signed_chain_circularity(rep, points), 0)

    def test_f3_signed_curvature_difference_is_two_pi(self):
        forest = build(fan())
        rep = next(r for b in forest.barcode for r in b.cycle_reps if r.active_start == 2)
        self.assertAlmostEqual(v.signed_chain_excess_curvature(rep, forest.point_cloud), 2 * math.pi)
        self.assertAlmostEqual(v.signed_chain_excess_curvature_normalized(rep, forest.point_cloud), 1)
        self.assertAlmostEqual(v.signed_chain_excess_curvature_diff_to_unsigned(rep, forest.point_cloud), 2 * math.pi)

    def test_rigid_motions_reflection_and_scaling(self):
        points, rep = ell()
        transforms = [(np.eye(2), np.array([7, -3]), 1),
                      (np.array([[0, -1], [1, 0]]), np.zeros(2), 1),
                      (np.diag([-1, 1]), np.zeros(2), 1),
                      (2 * np.eye(2), np.array([7, -3]), 2)]
        functions = [(v.signed_chain_area, 3, 2), (v.signed_chain_edge_length, 8, 1),
                     (v.signed_chain_circularity, 3 * math.pi / 16, 0),
                     (v.signed_chain_convex_hull_area_deficit, 1 / 7, 0),
                     (v.signed_chain_convex_hull_perimeter_deficit, 1 - (6 + math.sqrt(2)) / 8, 0),
                     (v.signed_chain_max_convexity_defect_depth, .25, 0),
                     (v.signed_chain_excess_curvature_normalized, .5, 0)]
        for transform, offset, scale in transforms:
            transformed = points @ transform.T + offset
            sign = int(np.sign(np.linalg.det(transform)))
            oriented = SignedChain({(s, sign * value) for s, value in rep.signed_simplices})
            for func, expected, degree in functions:
                with self.subTest(function=func.__name__, transform=transform.tolist()):
                    self.assertAlmostEqual(func(oriented, transformed), expected * scale**degree, places=11)
            self.assertAlmostEqual(v.signed_chain_max_convexity_defect_depth(oriented, transformed, normalize=False),
                                   scale / math.sqrt(2))

    def test_wrong_dimension_and_missing_interiors_reject(self):
        rep = SignedChain(set(chain('+012')))
        for func in (v.signed_chain_edge_length, v.signed_chain_area, v.signed_chain_circularity,
                     v.signed_chain_connected_components, v.signed_chain_excess_curvature):
            with self.subTest(function=func.__name__), self.assertRaises(ValueError):
                func(rep, np.eye(3))
        with self.assertRaisesRegex(ValueError, 'Interior is unavailable'):
            v.signed_chain_interior_volume(rep, np.eye(3))

    def test_g9_interior_volumes_translation_scaling_orientation_degeneracy(self):
        for dim in range(1, 5):
            points = np.vstack((np.zeros(dim), np.eye(dim)))
            for orientation in (-1, 1):
                rep = SignedChain(set(), interior_available=True,
                                  interior={(tuple(range(dim + 1)), orientation)})
                self.assertAlmostEqual(v.signed_chain_interior_volume(rep, points), 1 / math.factorial(dim))
                self.assertAlmostEqual(v.signed_chain_interior_volume(rep, 2 * points + 7), 2**dim / math.factorial(dim))
                self.assertEqual(v.signed_chain_interior_volume(rep, np.zeros_like(points)), 0)
        rep = SignedChain(set(), interior_available=True, interior=set())
        self.assertEqual(v.signed_chain_interior_volume(rep, np.zeros((0, 2))), 0)
        rep.interior = {((0, 1), 1)}
        with self.assertRaisesRegex(ValueError, 'full-dimensional'):
            v.signed_chain_interior_volume(rep, np.eye(3))


    def test_d3_simplex_support_volume_is_translation_invariant(self):
        points, rep = polygon([(0, 0), (3, 0), (0, 4)])
        self.assertAlmostEqual(v.signed_chain_volume(rep, points + 10), 12)


    def test_d3_surface_volume_uses_gram_formula(self):
        points = np.vstack((np.zeros(3), np.eye(3)))
        rep = SignedChain(set(chain('+123 -023 +013 -012')))
        expected = (3 + math.sqrt(3)) / 2
        self.assertAlmostEqual(sum(simplex_volume(points, s) for s, _ in rep.signed_simplices), expected)
        self.assertAlmostEqual(v.signed_chain_volume(rep, points), expected)

    def test_d4_constant_one_vanishes_on_zero_chain(self):
        self.assertEqual(v.constant_one_functional(SignedChain(set()), np.zeros((0, 2))), 0)

    def test_d5_components_are_support_components_not_path_count(self):
        points, rep = touching_triangles()
        self.assertEqual(v.signed_chain_connected_components(rep, points), 1)
        self.assertEqual(v.signed_chain_excess_connected_components(rep, points), 0)

    def test_d5_doubled_loops_use_geometric_length_per_support_component(self):
        for fixture, components, average in ((square, 1, 8), (disjoint_squares, 2, 4)):
            points, rep = fixture()
            doubled_rep = SignedChain({(simplex, sign)
                                       for simplex, _ in rep.signed_simplices
                                       for sign in (-1, 1)})
            with self.subTest(fixture=fixture.__name__):
                self.assertEqual(v.signed_chain_connected_components_only_signed_simplices(
                    doubled_rep, points), components)
                self.assertAlmostEqual(v.signed_chain_avg_tendril_length(doubled_rep, points), average)

    def test_d5_empty_support_has_zero_components_and_tendril_length(self):
        rep = SignedChain(set())
        points = np.empty((0, 2))
        for func in (v.signed_chain_connected_components,
                     v.signed_chain_excess_connected_components,
                     v.signed_chain_connected_components_only_signed_simplices,
                     v.signed_chain_avg_tendril_length):
            with self.subTest(function=func.__name__):
                self.assertEqual(func(rep, points), 0)

    def test_d5_support_connectivity_ignores_embedding_and_orientation(self):
        points, rep = touching_triangles()
        reversed_rep = SignedChain({(simplex, -sign) for simplex, sign in rep.signed_simplices})
        embedded_points = np.column_stack((points, np.zeros(len(points))))
        self.assertEqual(v.signed_chain_connected_components(reversed_rep, embedded_points), 1)
        # An ordinary edge joins signed support, but does not join doubled support.
        rep = SignedChain({((0, 1), 1), ((0, 1), -1), ((1, 2), 1),
                           ((2, 3), 1), ((2, 3), -1)})
        self.assertEqual(v.signed_chain_connected_components(rep, points), 1)
        self.assertEqual(v.signed_chain_connected_components_only_signed_simplices(rep, points), 2)

    def test_hull_deficits_use_one_hull_for_outer_and_hole_boundaries(self):
        points, rep = ring()
        # A connected square ring: area 12, global hull area 16,
        # total boundary length 24, global hull perimeter 16.
        self.assertAlmostEqual(v.signed_chain_convex_hull_area_deficit(rep, points), .25)
        self.assertAlmostEqual(v.signed_chain_convex_hull_perimeter_deficit(rep, points), 1 / 3)

    def test_perimeter_hull_deficit_clips_negative_ratio(self):
        points, rep = disjoint_squares()
        # Exercise the clipping extension only; disconnected enclosed-area
        # computations remain outside the supported measurement domain.
        # Global hull perimeter 10 exceeds total boundary length 8.
        self.assertEqual(v.signed_chain_convex_hull_perimeter_deficit(rep, points), 0)

    def test_hull_deficits_vanish_on_empty_chain(self):
        rep = SignedChain(set())
        points = np.empty((0, 2))
        self.assertEqual(v.signed_chain_convex_hull_area_deficit(rep, points), 0)
        self.assertEqual(v.signed_chain_convex_hull_perimeter_deficit(rep, points), 0)


# Each zero-chain contract is an individually reported failure, not one matrix
# whose first failure hides the others. D4 requires a separate API correction.
def _zero_measurement_case(func):
    def test(self):
        self.assertEqual(func(SignedChain(set()), np.zeros((0, 2))), 0)
    test.__doc__ = f'D4: {func.__name__} must vanish on the zero chain.'
    return test


for _function in (v.signed_chain_edge_length, v.signed_chain_area, v.signed_chain_circularity,
                  v.signed_chain_circularity_complement, v.signed_chain_circularity_squared_complement,
                  v.signed_chain_circularity_complement_squared, v.signed_chain_excess_curvature):
    setattr(SignedMeasurementTests, f'test_d4_zero_{_function.__name__}', _zero_measurement_case(_function))


class PathAndGeometryHelperTests(unittest.TestCase):
    def test_paths_cover_each_directed_edge_once_and_close(self):
        for factory, count in [(square, 1), (ell, 1), (ring, 2), (touching_triangles, 2), (doubled, 1)]:
            points, rep = factory()
            paths = v.signed_chain_to_polyhedral_paths(rep, points)
            self.assertEqual(len(paths), count)
            directed = Counter((s[0], s[1]) if sign == 1 else (s[1], s[0]) for s, sign in rep.signed_simplices)
            actual = Counter()
            for path in paths:
                self.assertGreaterEqual(len(path), 2)
                self.assertNotEqual(path[0], path[-1])
                actual.update(zip(path, np.roll(path, -1)))
            self.assertEqual(actual, directed)
            reverse = SignedChain({(s, -sign) for s, sign in rep.signed_simplices})
            reversed_paths = v.signed_chain_to_polyhedral_paths(reverse, points)
            reversed_edges = Counter(edge for path in reversed_paths for edge in zip(path, np.roll(path, -1)))
            self.assertEqual(reversed_edges, Counter({(b, a): n for (a, b), n in directed.items()}))

    def test_paths_reject_open_invalid_and_nonplanar_chains(self):
        for rep, points in [(SignedChain({((0, 1), 1)}), np.array([[0, 0], [1, 0]])),
                            (SignedChain({((0, 1), 0)}), np.array([[0, 0], [1, 0]])),
                            (SignedChain(set(chain('+01 -02 +12'))), np.eye(3))]:
            with self.assertRaises(ValueError):
                v.signed_chain_to_polyhedral_paths(rep, points)
        self.assertEqual(v.signed_chain_to_polyhedral_paths(SignedChain(set()), np.zeros((0, 2))), [])

    def test_convex_hull_duplicates_collinearity_and_diameter(self):
        points = np.array([[0, 0], [3, 0], [3, 4], [0, 4], [1, 1], [3, 0], [1, 0]], float)
        hull = points[v._convex_hull_indices(points)]
        self.assertEqual(set(map(tuple, hull)), {(0, 0), (3, 0), (3, 4), (0, 4)})
        self.assertEqual(v._convex_polygon_diameter(hull), 5)
        line = np.array([[0, 0], [1, 0], [2, 0], [1, 0]], float)
        self.assertEqual(set(map(tuple, line[v._convex_hull_indices(line)])), {(0, 0), (2, 0)})
        self.assertEqual(v._convex_polygon_diameter(line[[0, 2]]), 2)
        self.assertEqual(v._convex_polygon_diameter(line[:1]), 0)
        with self.assertRaises(ValueError):
            v._convex_hull_indices(np.array([[np.nan, 0]]))

    def test_distances_clamp_to_segment_and_zero_length_rejects(self):
        points = np.array([[-1., 0], [.5, 2], [2, 0]])
        start, end = np.array([0., 0]), np.array([1., 0])
        np.testing.assert_allclose(v._point_to_segment_distances(points, start, end), [1, 2, 1])
        with self.assertRaises(ValueError):
            v._point_to_segment_distances(points, start, start)
        self.assertAlmostEqual(v._polygon_max_convexity_defect_depth(ell()[0]), 1 / math.sqrt(2))
        self.assertEqual(v._polygon_max_convexity_defect_depth(np.array([[0, 0], [1, 0], [2, 0]])), 0)

    def test_unit_interval_clamps_roundoff_not_geometry_errors(self):
        epsilon = np.finfo(float).eps
        self.assertEqual(v._require_unit_interval(-epsilon, 'test'), 0)
        self.assertEqual(v._require_unit_interval(1 + epsilon, 'test'), 1)
        for value in (-.01, 1.01, np.nan, np.inf):
            with self.assertRaises(ValueError):
                v._require_unit_interval(value, 'test')
