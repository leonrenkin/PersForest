"""Compiled alpha builder (geogram's Delaunay triangulation) against GUDHI's exact alpha complex.

GUDHI runs with ``precision='exact'``; its default ``'safe'`` precision is only
accurate to about 1e-5 relative. Degenerate inputs may be triangulated
differently, so there only the bars of positive length are compared.
"""
from fractions import Fraction
import unittest

import gudhi as gd
import numpy as np

from persforest import _engine
from persforest._forest.arrays import build_forest_arrays
from persforest.filtrations.delaunay import alpha_filtration
from persforest.filtrations.simplicial import SimplicialFiltration, simplicial_boundary
from persforest.geometry.simplicial import simplex_orientations

FILTRATION_TOL = 1e-12
# Observed differences to GUDHI's exact values are below 1e-14; the builder's
# documented first-order bound is about 5e-13 for radii.
VALUE_RTOL = 1e-13


def gudhi_exact(points):
    """GUDHI's exact alpha complex as a SimplicialFiltration (top simplices and facets)."""
    tree = gd.AlphaComplex(points=points, precision='exact').create_simplex_tree(output_squared_values=False)
    return SimplicialFiltration.from_events(points, tree.get_filtration())


def values_by_key(cells):
    return dict(zip(cells.cell_keys(cells.event_cells), cells.event_values.tolist()))


def positive_bars(cells):
    forest = build_forest_arrays(cells, reduce=True, filtration_tol=FILTRATION_TOL)
    bars = sorted(zip(forest.bar_birth.tolist(), forest.bar_death.tolist()))
    return np.array([bar for bar in bars if bar[1] - bar[0] > FILTRATION_TOL]).reshape(-1, 2)


def exact_orientation(points, simplex):
    """Sign of det(p[v_k] - p[v_0]) in rational arithmetic."""
    rows = [[Fraction(x) - Fraction(y) for x, y in zip(points[v], points[simplex[0]])] for v in simplex[1:]]
    n = len(rows)
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if rows[r][col] != 0), None)
        if pivot is None:
            return 0
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
            det = -det
        det *= rows[col][col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col])]
    return (det > 0) - (det < 0)


def grid(*sides):
    return np.stack(np.meshgrid(*[np.arange(side, dtype=float) for side in sides], indexing='ij'), -1).reshape(-1, len(sides))


COCIRCULAR = np.array([[3, 4], [4, 3], [5, 0], [0, 5], [-3, 4], [-4, 3], [-5, 0], [0, -5],
                       [3, -4], [4, -3], [-3, -4], [-4, -3]], dtype=float)


class FiltrationContractMixin:
    """Structural properties every builder output must have."""

    def assert_valid_filtration(self, cells):
        dim = cells.top_dimension
        n_top, n_cells = cells.n_top, cells.n_cells
        values, ids = cells.event_values, cells.event_cells
        self.assertEqual(cells.top_vertices.shape, (n_top, dim + 1))
        self.assertEqual(cells.facet_vertices.shape, (n_cells - n_top, dim))
        self.assertEqual(cells.top_facets.shape, (n_top, dim + 1))
        self.assertEqual(cells.top_orientation.shape, (n_top,))
        self.assertTrue(np.all(np.isfinite(values)) and np.all(values >= 0))
        self.assertTrue(np.all(np.diff(values) >= 0), 'values must be nondecreasing')
        # IDs in event order, top simplices first.
        np.testing.assert_array_equal(ids[ids < n_top], np.arange(n_top))
        np.testing.assert_array_equal(ids[ids >= n_top], np.arange(n_top, n_cells))
        for rows in (cells.top_vertices, cells.facet_vertices):
            self.assertTrue(np.all(np.diff(rows, axis=1) > 0), 'vertices must increase within a row')
        keys = cells.cell_keys(range(n_cells))
        self.assertEqual(len(set(keys)), n_cells)
        # Facet j of a top simplex omits vertex j, and enters no later than the top simplex.
        position = np.empty(n_cells, dtype=np.int64)
        position[ids] = np.arange(n_cells)
        cofaces = np.bincount(cells.top_facets.ravel() - n_top, minlength=n_cells - n_top)
        self.assertTrue(np.all((cofaces >= 1) & (cofaces <= 2)))
        for top in range(n_top):
            facets = cells.top_facets[top].tolist()
            self.assertEqual(cells.cell_keys(facets), [facet for facet, _ in simplicial_boundary(keys[top])])
            self.assertTrue(np.all(position[facets] < position[top]))
        # Ties: facets before top simplices, then increasing vertex tuples.
        sort_keys = [(value, 0 if cell >= n_top else 1, keys[cell]) for cell, value in zip(ids.tolist(), values.tolist())]
        for before, after in zip(sort_keys, sort_keys[1:]):
            if before[0] == after[0]:
                self.assertLess(before[1:], after[1:])


class GeneralPositionTests(FiltrationContractMixin, unittest.TestCase):
    def test_same_cells_values_and_bars_as_gudhi_exact(self):
        rng = np.random.default_rng(5)
        for name, points in (('uniform 2D', rng.random((800, 2))), ('uniform 3D', rng.random((400, 3))),
                             ('gaussian 3D', rng.normal(size=(300, 3)))):
            with self.subTest(name):
                ours, theirs = alpha_filtration(points), gudhi_exact(points)
                self.assert_valid_filtration(ours)
                mine, reference = values_by_key(ours), values_by_key(theirs)
                self.assertEqual(mine.keys(), reference.keys())
                keys = list(reference)
                np.testing.assert_allclose([mine[k] for k in keys], [reference[k] for k in keys],
                                           rtol=VALUE_RTOL, atol=0)
                ours_bars, theirs_bars = positive_bars(ours), positive_bars(theirs)
                self.assertEqual(ours_bars.shape, theirs_bars.shape)
                np.testing.assert_allclose(ours_bars, theirs_bars, rtol=VALUE_RTOL, atol=0)

    def test_orientations_agree_with_simplex_orientations(self):
        rng = np.random.default_rng(6)
        for dim in (2, 3):
            with self.subTest(dim=dim):
                points = rng.random((300, dim))
                cells = alpha_filtration(points)
                np.testing.assert_array_equal(cells.top_orientation,
                                              simplex_orientations(cells.top_vertices, points))
                self.assertEqual([cells.orientation(top) for top in range(cells.n_top)],
                                 cells.top_orientations().tolist())


class DegenerateInputTests(FiltrationContractMixin, unittest.TestCase):
    def assert_same_positive_bars(self, points):
        ours = alpha_filtration(points)
        self.assert_valid_filtration(ours)
        ours_bars, theirs_bars = positive_bars(ours), positive_bars(gudhi_exact(points))
        self.assertEqual(ours_bars.shape, theirs_bars.shape)
        np.testing.assert_allclose(ours_bars, theirs_bars, rtol=VALUE_RTOL, atol=0)
        return ours

    def test_degenerate_inputs_have_gudhis_positive_bars(self):
        rng = np.random.default_rng(7)
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        lines = np.vstack([np.c_[np.linspace(0, 1, 15), np.full(15, y)] for y in (0.0, 0.3, 0.7)])
        cases = {
            'grid 12x12': grid(12, 12),
            'grid 5x5x5': grid(5, 5, 5),
            'grid 7x4x3 anisotropic': grid(7, 4, 3) * [1.0, 2.0, 0.5],
            'exactly cocircular': COCIRCULAR,
            'cocircular and center': np.vstack([COCIRCULAR, [[0.0, 0.0]]]),
            'cospherical and center': np.vstack([np.r_[COCIRCULAR.T, [np.zeros(12)]].T,
                                                 [[0, 0, 5], [0, 0, -5], [0, 0, 0]]]),
            'rounded circle': np.c_[np.cos(t), np.sin(t)],
            'collinear subsets': np.vstack([lines, rng.random((20, 2))]),
            'triangle': np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 1.0]]),
            'four points 2D': np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 1.0], [1.0, -3.0]]),
            'tetrahedron': np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'five points 3D': np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]),
        }
        for name, points in cases.items():
            with self.subTest(name):
                self.assert_same_positive_bars(points)

    def test_grids_use_the_exact_gabriel_test(self):
        for points in (grid(6, 6), grid(4, 4, 4)):
            with self.subTest(dim=points.shape[1]):
                self.assertGreater(_engine.alpha_filtration(points)['n_exact_gabriel_tests'], 0)

    def test_first_of_identical_points_is_the_vertex(self):
        rng = np.random.default_rng(8)
        for dim in (2, 3):
            with self.subTest(dim=dim):
                unique = rng.random((80, dim))
                copies = rng.integers(0, 80, 30)
                points = np.vstack([unique, unique[copies], unique[copies[:5]]])
                cells = self.assert_same_positive_bars(points)
                self.assertTrue(np.all(cells.top_vertices < 80))
                reference = alpha_filtration(unique)
                np.testing.assert_array_equal(cells.top_vertices, reference.top_vertices)
                np.testing.assert_array_equal(cells.event_values, reference.event_values)

    def test_too_few_independent_points_give_an_empty_filtration(self):
        cases = {
            'no points 2D': np.zeros((0, 2)), 'one point': np.array([[1.0, 2.0]]),
            'two points': np.array([[0.0, 0.0], [1.0, 1.0]]),
            'collinear 2D': np.c_[np.arange(5.0), 2 * np.arange(5.0)],
            'identical 2D': np.ones((4, 2)), 'origin only': np.zeros((3, 3)),
            'three points 3D': np.eye(3), 'coplanar 3D': np.c_[grid(4, 4), np.ones(16)],
            'collinear 3D': np.c_[np.arange(5.0), np.arange(5.0), np.zeros(5)],
        }
        for name, points in cases.items():
            with self.subTest(name):
                cells = alpha_filtration(points)
                self.assertEqual((cells.n_top, cells.n_cells), (0, 0))
                self.assert_valid_filtration(cells)
                self.assertEqual(positive_bars(cells).shape, (0, 2))


class ExactPathTests(unittest.TestCase):
    def test_nearly_flat_cells_have_exact_values_and_orientations(self):
        rng = np.random.default_rng(9)
        # Points close to a line (2D) or a plane (3D) make many cells nearly flat.
        cases = {2: np.c_[rng.random(60), 1e-9 * rng.random(60)],
                 3: np.c_[rng.random((60, 2)), 1e-9 * rng.random(60)]}
        for dim, points in cases.items():
            with self.subTest(dim=dim):
                self.assertGreater(_engine.alpha_filtration(points)['n_exact_radii'], 0)
                ours = alpha_filtration(points)
                mine, reference = values_by_key(ours), values_by_key(gudhi_exact(points))
                self.assertEqual(mine.keys(), reference.keys())
                np.testing.assert_allclose([mine[k] for k in reference], list(reference.values()),
                                           rtol=VALUE_RTOL, atol=0)
                expected = [exact_orientation(points, top) for top in ours.top_vertices.tolist()]
                self.assertEqual(ours.top_orientation.tolist(), expected)

    def test_cells_far_smaller_than_the_cloud(self):
        # A cluster of extent 1e-20 near the origin, inside a cloud of extent 1:
        # its cells are rescaled locally before their radii are computed.
        rng = np.random.default_rng(10)
        for dim in (2, 3):
            with self.subTest(dim=dim):
                far = 0.1 + rng.random((8, dim))  # in general position, unlike cube corners
                points = np.vstack([far, 1e-20 * (1 + rng.random((25, dim)))])
                mine, reference = values_by_key(alpha_filtration(points)), values_by_key(gudhi_exact(points))
                self.assertEqual(mine.keys(), reference.keys())
                np.testing.assert_allclose([mine[k] for k in reference], list(reference.values()),
                                           rtol=VALUE_RTOL, atol=0)
                self.assertLess(min(reference.values()), 1e-19)

    def test_cells_a_few_ulps_wide(self):
        # Near 2^-100, the smallest accepted magnitude relative to the largest
        # coordinate, points can differ by a few ulps (2^-152). Without local
        # rescaling the degree-8 radius formulas of such cells underflow.
        rng = np.random.default_rng(16)
        for dim in (2, 3):
            with self.subTest(dim=dim):
                lattice = rng.integers(0, 64, (40, dim)).astype(float)
                cluster = 2.0 ** -100 + np.ldexp(lattice, -152)  # exact: the ulp of 2^-100 is 2^-152
                far = 0.25 + rng.random((6, dim))
                cells = alpha_filtration(np.vstack([far, cluster]))
                tops = {tuple(v - 6 for v in key): value for key, value in
                        zip(cells.cell_keys(range(cells.n_top)), cells.event_values[np.argsort(cells.event_cells)])
                        if min(key) >= 6}
                alone = alpha_filtration(lattice)
                expected = dict(zip(alone.cell_keys(range(alone.n_top)),
                                    alone.event_values[np.argsort(alone.event_cells)][:alone.n_top]))
                common = tops.keys() & expected.keys()
                self.assertGreater(len(common), len(expected) // 2)
                # Power-of-two scaling and translation by 2^-100 are exact, so radii scale exactly.
                for key in common:
                    self.assertEqual(tops[key], np.ldexp(expected[key], -152))


class ScaleTests(unittest.TestCase):
    def test_power_of_two_scaling_scales_values_exactly(self):
        rng = np.random.default_rng(12)
        for dim in (2, 3):
            points = rng.random((200, dim)) + 0.25  # coordinates in [0.25, 1.25): scaling stays exact
            reference = _engine.alpha_filtration(points)
            for exponent in (-900, -300, 300, 900):
                with self.subTest(dim=dim, exponent=exponent):
                    scaled = _engine.alpha_filtration(np.ldexp(points, exponent))
                    for key in ('top_vertices', 'facet_vertices', 'top_facets', 'top_orientation', 'event_cells'):
                        np.testing.assert_array_equal(scaled[key], reference[key])
                    np.testing.assert_array_equal(scaled['event_values'], np.ldexp(reference['event_values'], exponent))

    def test_extreme_decimal_scales(self):
        # GUDHI returns 0 or inf here; the builder rescales exactly.
        rng = np.random.default_rng(13)
        points = rng.random((150, 2))
        reference = alpha_filtration(points)
        for factor in (1e-200, 1e200):
            with self.subTest(factor=factor):
                cells = alpha_filtration(points * factor)
                np.testing.assert_array_equal(cells.top_vertices, reference.top_vertices)
                np.testing.assert_allclose(cells.event_values / factor, reference.event_values, rtol=VALUE_RTOL)

    def test_coordinates_spanning_too_many_orders_of_magnitude_are_rejected(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1e-40, 0.5]])
        with self.assertRaisesRegex(ValueError, 'orders of magnitude'):
            alpha_filtration(points)
        points[3, 0] = 1e-29  # above 2^-100 times the largest coordinate
        self.assertGreater(alpha_filtration(points).n_top, 0)

    def test_invalid_input_is_rejected(self):
        for points in (np.zeros((3, 1)), np.zeros((5, 4)), np.array([[0.0, 0.0], [np.nan, 1.0], [1.0, 0.0]]),
                       np.array([[0.0, 0.0], [np.inf, 1.0], [1.0, 0.0]])):
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    _engine.alpha_filtration(points)


class DeterminismTests(unittest.TestCase):
    def test_repeated_builds_are_identical(self):
        rng = np.random.default_rng(14)
        for points in (grid(9, 9, 9), rng.random((3000, 3)), grid(30, 30), np.round(rng.random((2000, 2)) * 8)):
            with self.subTest(shape=points.shape):
                first = _engine.alpha_filtration(points)
                for _ in range(3):
                    again = _engine.alpha_filtration(points)
                    for key, value in first.items():
                        np.testing.assert_array_equal(np.asarray(again[key]), np.asarray(value), err_msg=key)

    def test_concurrent_builds_from_threads(self):
        # The binding releases the GIL; geogram's global state is guarded by a lock.
        from concurrent.futures import ThreadPoolExecutor
        rng = np.random.default_rng(17)
        inputs = [rng.random((4000, 3)), grid(8, 8, 8), rng.random((6000, 2)), grid(25, 25)] * 2
        expected = [_engine.alpha_filtration(points) for points in inputs]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_engine.alpha_filtration, inputs))
        for result, reference in zip(results, expected):
            for key, value in reference.items():
                np.testing.assert_array_equal(np.asarray(result[key]), np.asarray(value), err_msg=key)

    def test_cells_do_not_depend_on_the_order_of_the_points(self):
        # Degenerate configurations are perturbed by coordinates, not by index.
        rng = np.random.default_rng(15)
        for points in (grid(6, 6, 6), grid(14, 14), np.vstack([COCIRCULAR, [[0.0, 0.0]]])):
            with self.subTest(shape=points.shape):
                permutation = rng.permutation(len(points))

                def cells_by_coordinates(pts):
                    cells = alpha_filtration(pts)
                    keys = cells.cell_keys(cells.event_cells)
                    return {frozenset(map(tuple, pts[list(key)].tolist())): value
                            for key, value in zip(keys, cells.event_values.tolist())}

                self.assertEqual(cells_by_coordinates(points), cells_by_coordinates(points[permutation]))


if __name__ == '__main__':
    unittest.main()
