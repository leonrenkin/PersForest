"""Cell-filtration contract: simplicial and cubical boundaries, engine isolation."""
import ast
from collections import defaultdict
from pathlib import Path
import unittest

import numpy as np

from persforest.filtrations._interface import oriented_boundary
from persforest.filtrations.cubical import CubicalFiltration
from persforest.filtrations.simplicial import SimplicialFiltration

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


class CubicalAdapterTests(unittest.TestCase):
    def test_literal_boundaries_and_boundary_of_boundary(self):
        cells = CubicalFiltration(3, [])
        self.assertEqual(cells.boundary((0, 0, 0)), ())
        self.assertEqual(cells.boundary((1, 0, 2)), (((2, 0, 2), 1), ((0, 0, 2), -1)))
        # Square oriented by (x, y): right - left - top + bottom.
        self.assertEqual(cells.boundary((1, 1, 0)),
                         (((2, 1, 0), 1), ((0, 1, 0), -1), ((1, 2, 0), -1), ((1, 0, 0), 1)))
        for cell in [(1, 1, 0), (1, 0, 1), (1, 1, 1), (3, 5, 1)]:
            with self.subTest(cell=cell):
                self.assertEqual(linear_boundary(cells, dict(cells.boundary(cell))), {})
        self.assertEqual([cells.dimension(c) for c in [(0, 2, 4), (1, 2, 4), (1, 3, 5)]], [0, 1, 3])

    def test_adjacent_top_cells_cancel_on_their_shared_facet(self):
        cells = CubicalFiltration(2, [])
        total = linear_boundary(cells, {(1, 1): 1, (3, 1): 1})
        self.assertNotIn((2, 1), total)
        self.assertEqual(len(total), 6)


if __name__ == '__main__':
    unittest.main()
