"""Cell-filtration contract: simplicial boundaries, engine isolation, non-simplicial cells."""
import ast
from collections import defaultdict
from pathlib import Path
import unittest
from unittest.mock import patch

import gudhi
import numpy as np

from persforest import PersistenceForest
from persforest.filtrations._interface import oriented_boundary
from persforest.filtrations.simplicial import SimplicialFiltration
from fixtures_small_filtrations import mode_options, reps

ENGINE = Path(__file__).resolve().parents[2] / 'src' / 'persforest' / '_forest'
ALLOWED_ENGINE_IMPORTS = ('persforest._forest', 'persforest.barcode', 'persforest.chains',
                          'persforest.filtrations._interface')


def linear_boundary(cells, chain):
    """Boundary of a coefficient chain {cell: coefficient}, cancelling zeros."""
    result = defaultdict(int)
    for cell, coefficient in chain.items():
        for facet, incidence in cells.boundary(cell):
            result[facet] += coefficient * incidence
    return {facet: c for facet, c in result.items() if c}


def coefficients(signed_terms):
    result = defaultdict(int)
    for cell, sign in signed_terms:
        result[cell] += sign
    return {cell: c for cell, c in result.items() if c}


class SimplicialAdapterTests(unittest.TestCase):
    cells = SimplicialFiltration(np.zeros((5, 3)), [])

    def test_literal_boundaries(self):
        self.assertEqual(self.cells.boundary((4,)), ())
        self.assertEqual(self.cells.boundary((1, 3)), (((3,), 1), ((1,), -1)))
        self.assertEqual(self.cells.boundary((0, 1, 2)), (((1, 2), 1), ((0, 2), -1), ((0, 1), 1)))
        self.assertEqual(self.cells.boundary((0, 1, 2, 3)),
                         (((1, 2, 3), 1), ((0, 2, 3), -1), ((0, 1, 3), 1), ((0, 1, 2), -1)))
        self.assertEqual([self.cells.dimension(c) for c in [(0,), (0, 1), (0, 1, 2, 3)]], [0, 1, 3])

    def test_boundary_of_boundary_and_orientation_reversal(self):
        for cell in [(0, 1), (0, 2, 4), (0, 1, 3, 4)]:
            with self.subTest(cell=cell):
                self.assertEqual(linear_boundary(self.cells, dict(self.cells.boundary(cell))), {})
                self.assertEqual(oriented_boundary(self.cells, cell, -1),
                                 tuple((f, -c) for f, c in self.cells.boundary(cell)))

    def test_orientation_is_geometric(self):
        points = np.array([[0, 0], [3, 0], [0, 4]], dtype=float)
        self.assertEqual(SimplicialFiltration(points, []).orientation((0, 1, 2)), 1)
        self.assertEqual(SimplicialFiltration(points[[0, 2, 1]], []).orientation((0, 1, 2)), -1)

    def test_shared_facet_cancels_between_coherently_oriented_triangles(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cells = SimplicialFiltration(points, [])
        total = defaultdict(int)
        for triangle in [(0, 1, 2), (0, 2, 3)]:
            for facet, c in oriented_boundary(cells, triangle, cells.orientation(triangle)):
                total[facet] += c
        self.assertEqual({f: c for f, c in total.items() if c},
                         {(0, 1): 1, (1, 2): 1, (2, 3): 1, (0, 3): -1})


class EngineIsolationTests(unittest.TestCase):
    def test_engine_imports_only_cell_neutral_modules(self):
        for path in sorted(ENGINE.glob('*.py')):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or '']
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    with self.subTest(file=path.name, module=module):
                        if module.startswith('persforest'):
                            self.assertTrue(module.startswith(ALLOWED_ENGINE_IMPORTS), module)
                        else:
                            self.assertNotIn(module.split('.')[0], {'gudhi', 'numpy', 'matplotlib', 'plotly'})


class GridCells:
    """Test-only cubical cells of a full 2D grid with values on squares.

    A cell is its tuple of doubled coordinates; odd entries are the axes the cell
    extends along. Faces enter at the minimum value of their incident squares.
    """

    top_dimension = 2

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        rows, cols = self.values.shape
        squares = {(2 * i + 1, 2 * j + 1): self.values[i, j] for i in range(rows) for j in range(cols)}
        edges = {}
        for square, value in squares.items():
            for edge, _ in self.boundary(square):
                edges[edge] = min(edges.get(edge, np.inf), value)
        # Ties: an edge enters with its first square, so faces precede cofaces.
        self.events = sorted(list(edges.items()) + list(squares.items()),
                             key=lambda item: (item[1], self.dimension(item[0]), item[0]))

    def dimension(self, cell):
        return sum(coordinate % 2 for coordinate in cell)

    def boundary(self, cell):
        facets = []
        odd_axes = [axis for axis, coordinate in enumerate(cell) if coordinate % 2]
        for position, axis in enumerate(odd_axes):
            sign = -1 if position % 2 else 1
            for step, incidence in ((1, sign), (-1, -sign)):
                facet = list(cell)
                facet[axis] += step
                facets.append((tuple(facet), incidence))
        return tuple(facets)

    def orientation(self, cell):
        return 1


def grid_forest(values, **options):
    cells = GridCells(values)
    with patch('persforest.forest.prepare_alpha_filtration', return_value=(None, cells)):
        return cells, PersistenceForest(np.zeros((1, 2)), **options)


def gudhi_h1(values):
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=np.asarray(values, dtype=float))
    complex_.compute_persistence()
    return sorted((float(b), float(d)) for b, d in complex_.persistence_intervals_in_dimension(1) if d > b)


class NonSimplicialCellTests(unittest.TestCase):
    def test_grid_cells_satisfy_the_boundary_contract(self):
        cells = GridCells(np.zeros((2, 2)))
        self.assertEqual(cells.boundary((1, 1)), (((2, 1), 1), ((0, 1), -1), ((1, 2), -1), ((1, 0), 1)))
        for cell in [(1, 1), (1, 0), (2, 1), (0, 0)]:
            self.assertEqual(linear_boundary(cells, dict(cells.boundary(cell))), {})

    def test_two_squares_by_hand(self):
        # Squares enter at 1 and 2; the shared edge and the outer edges enter at 1.
        _, forest = grid_forest([[2.0, 1.0]], keep_simplex_diff=True, compute_interior=True)
        self.assertEqual(sorted((b.birth, b.death) for b in forest.barcode if b.death > b.birth), [])
        _, forest = grid_forest([[2.0, 3.0, 1.0]], keep_simplex_diff=True, compute_interior=True)
        self.assertEqual(sorted((b.birth, b.death) for b in forest.barcode if b.death > b.birth), [])
        values = [[1.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 1.0]]
        _, forest = grid_forest(values, keep_simplex_diff=True, compute_interior=True)
        [bar] = [b for b in forest.barcode if b.death > b.birth]
        self.assertEqual((bar.birth, bar.death), (1.0, 3.0))
        [rep] = reps(forest, bar)
        self.assertEqual(coefficients(rep.signed_simplices),
                         {(4, 3): 1, (2, 3): -1, (3, 4): -1, (3, 2): 1})
        self.assertEqual(rep.interior, {((3, 3), 1)})

    def test_random_grids_match_gudhi_and_representatives_are_bounded_cycles(self):
        rng = np.random.default_rng(2026)
        grids = [rng.random((9, 11)) for _ in range(4)] + [rng.integers(0, 4, (10, 9)).astype(float)]
        for index, values in enumerate(grids):
            expected = gudhi_h1(values)
            snapshots = {}
            for mode, options in mode_options():
                with self.subTest(grid=index, mode=mode):
                    cells, forest = grid_forest(values, **options)
                    bars = sorted((b.birth, b.death) for b in forest.barcode if b.death > b.birth)
                    self.assertEqual(bars, expected)
                    record = []
                    for bar in forest.barcode:
                        for rep in reps(forest, bar):
                            support = coefficients(rep.signed_simplices)
                            self.assertEqual(linear_boundary(cells, support), {})
                            if rep.interior_available:
                                self.assertEqual(linear_boundary(cells, coefficients(rep.interior)), support)
                            record.append((bar.birth, bar.death, rep.active_start, rep.active_end,
                                           frozenset(rep.signed_simplices)))
                    snapshots[mode] = sorted(record, key=repr)
            with self.subTest(grid=index, check='modes agree'):
                self.assertEqual(snapshots['stored/reduce=True'], snapshots['diff_only/reduce=True'])
                self.assertEqual(snapshots['stored/reduce=False'], snapshots['interiors/reduce=False'])


if __name__ == '__main__':
    unittest.main()
