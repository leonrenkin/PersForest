"""Cell-filtration contract: boundaries, integer cell IDs, event order, engine isolation."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from persforest.filtrations._interface import facet_ids, signed_incidences
from persforest.filtrations.cubical import cubical_boundary, cubical_dimension, prepare_cubical_filtration
from persforest.filtrations.delaunay import prepare_point_filtration
from persforest.filtrations.simplicial import SimplicialFiltration, simplicial_boundary

ENGINE = Path(__file__).resolve().parents[2] / 'src' / 'persforest' / '_forest'
ALLOWED_ENGINE_IMPORTS = ('persforest._forest', 'persforest._engine', 'persforest.barcode', 'persforest.chains',
                          'persforest.filtrations._interface')
SIMPLICIAL = SimpleNamespace(boundary=simplicial_boundary)
CUBICAL = SimpleNamespace(boundary=cubical_boundary)


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


def closed_triangles(points, triangles):
    """Adapter over 2D triangles and their edges (edges first, value 0)."""
    edges = sorted({tuple(sorted(edge)) for t in triangles for edge in ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))})
    return SimplicialFiltration.from_events(np.asarray(points, dtype=float),
                                            [(e, 0.0) for e in edges] + [(t, 0.0) for t in triangles])


def oriented_facets(cells, top_id, orientation):
    """Keyed boundary of a top cell as the engine assembles it from IDs."""
    return tuple(zip(cells.cell_keys(facet_ids(cells, top_id)), signed_incidences(cells, orientation)))


class SimplicialAdapterTests(unittest.TestCase):
    def test_literal_boundaries(self):
        self.assertEqual(simplicial_boundary((4,)), ())
        self.assertEqual(simplicial_boundary((1, 3)), (((3,), 1), ((1,), -1)))
        self.assertEqual(simplicial_boundary((0, 1, 2)), (((1, 2), 1), ((0, 2), -1), ((0, 1), 1)))
        self.assertEqual(simplicial_boundary((0, 1, 2, 3)),
                         (((1, 2, 3), 1), ((0, 2, 3), -1), ((0, 1, 3), 1), ((0, 1, 2), -1)))

    def test_boundary_of_boundary_and_orientation_reversal(self):
        for cell in [(0, 1), (0, 2, 4), (0, 1, 3, 4)]:
            with self.subTest(cell=cell):
                self.assertEqual(linear_boundary(SIMPLICIAL, dict(simplicial_boundary(cell))), {})
        cells = closed_triangles([[0, 0], [1, 0], [0, 1]], [(0, 1, 2)])
        self.assertEqual(oriented_facets(cells, 0, -1),
                         tuple((f, -c) for f, c in oriented_facets(cells, 0, 1)))

    def test_orientation_is_geometric(self):
        points = np.array([[0, 0], [3, 0], [0, 4]], dtype=float)
        self.assertEqual(closed_triangles(points, [(0, 1, 2)]).orientation(0), 1)
        self.assertEqual(closed_triangles(points[[0, 2, 1]], [(0, 1, 2)]).orientation(0), -1)

    def test_shared_facet_cancels_between_coherently_oriented_triangles(self):
        cells = closed_triangles([[0, 0], [1, 0], [1, 1], [0, 1]], [(0, 1, 2), (0, 2, 3)])
        total = defaultdict(int)
        for top_id in range(cells.n_top):
            for facet, c in oriented_facets(cells, top_id, cells.orientation(top_id)):
                total[facet] += c
        self.assertEqual({f: c for f, c in total.items() if c},
                         {(0, 1): 1, (1, 2): 1, (2, 3): 1, (0, 3): -1})

    def test_ids_facets_and_event_order(self):
        for dim, n_points in ((2, 40), (3, 25)):
            points = np.random.default_rng(dim).random((n_points, dim))
            for filtration_type in ('alpha', 'delaunay_cech'):
                with self.subTest(dim=dim, filtration_type=filtration_type):
                    cells = prepare_point_filtration(points, filtration_type)
                    events = list(zip(cells.cell_keys(cells.event_cells), cells.event_values.tolist()))
                    # The compiled builders: by value, facets first, then vertex tuple.
                    self.assertEqual(events, sorted(events, key=lambda e: (e[1], len(e[0]) == dim + 1, e[0])))
                    self.assertEqual(sorted(cells.event_cells.tolist()), list(range(cells.n_cells)))
                    keys = cells.cell_keys(range(cells.n_cells))
                    self.assertTrue(all(len(k) == dim + 1 for k in keys[:cells.n_top]))
                    self.assertTrue(all(len(k) == dim for k in keys[cells.n_top:]))
                    for top_id in range(cells.n_top):
                        self.assertEqual(oriented_facets(cells, top_id, 1), simplicial_boundary(keys[top_id]))

    def test_missing_facet_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            SimplicialFiltration.from_events(np.zeros((3, 2)), [((0, 1), 0.0), ((0, 1, 2), 0.0)])


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
                        if module == 'persforest':
                            self.assertEqual([alias.name for alias in node.names], ['_engine'])
                        elif module.startswith('persforest'):
                            self.assertTrue(module.startswith(ALLOWED_ENGINE_IMPORTS), module)
                        else:
                            # NumPy is allowed: the engine's arrays are NumPy arrays.
                            self.assertNotIn(module.split('.')[0], {'gudhi', 'matplotlib', 'plotly'})


class CubicalAdapterTests(unittest.TestCase):
    def test_literal_boundaries_and_boundary_of_boundary(self):
        self.assertEqual(cubical_boundary((0, 0, 0)), ())
        self.assertEqual(cubical_boundary((1, 0, 2)), (((2, 0, 2), 1), ((0, 0, 2), -1)))
        # Square oriented by (x, y): right - left - top + bottom.
        self.assertEqual(cubical_boundary((1, 1, 0)),
                         (((2, 1, 0), 1), ((0, 1, 0), -1), ((1, 2, 0), -1), ((1, 0, 0), 1)))
        for cell in [(1, 1, 0), (1, 0, 1), (1, 1, 1), (3, 5, 1)]:
            with self.subTest(cell=cell):
                self.assertEqual(linear_boundary(CUBICAL, dict(cubical_boundary(cell))), {})
        self.assertEqual([cubical_dimension(c) for c in [(0, 2, 4), (1, 2, 4), (1, 3, 5)]], [0, 1, 3])

    def test_adjacent_top_cells_cancel_on_their_shared_facet(self):
        total = linear_boundary(CUBICAL, {(1, 1): 1, (3, 1): 1})
        self.assertNotIn((2, 1), total)
        self.assertEqual(len(total), 6)

    def test_ids_facets_and_event_order_match_doubled_coordinates(self):
        rng = np.random.default_rng(11)
        for shape in ((5,), (4, 6), (3, 4, 5)):
            for filtration_type in ('top_cells', 'vertices'):
                for values in (rng.random(shape), rng.integers(0, 3, shape).astype(float)):
                    with self.subTest(shape=shape, filtration_type=filtration_type):
                        _, cells = prepare_cubical_filtration(values, filtration_type)
                        d = len(shape)
                        keys = cells.cell_keys(range(cells.n_cells))
                        self.assertEqual(len(set(keys)), cells.n_cells)
                        # Short lookups take a pure-Python path; it must agree.
                        self.assertEqual([cells.cell_keys([i])[0] for i in range(cells.n_cells)], keys)
                        self.assertTrue(all(cubical_dimension(k) == d for k in keys[:cells.n_top]))
                        self.assertTrue(all(cubical_dimension(k) == d - 1 for k in keys[cells.n_top:]))
                        for top_id in range(cells.n_top):
                            self.assertEqual(oriented_facets(cells, top_id, 1), cubical_boundary(keys[top_id]))
                        # Forward order: value, then dimension (faces first), then doubled coordinates.
                        order = cells.event_cells.tolist()
                        self.assertEqual(sorted(order), list(range(cells.n_cells)))
                        values_by_id = dict(zip(order, cells.event_values.tolist()))
                        if filtration_type == 'top_cells':
                            # Each top cell takes the value of its pixel.
                            self.assertEqual([values_by_id[t] for t in range(cells.n_top)],
                                             [values[tuple((c - 1) // 2 for c in keys[t])] for t in range(cells.n_top)])
                        expected = sorted(range(cells.n_cells),
                                          key=lambda i: (values_by_id[i], cubical_dimension(keys[i]), keys[i]))
                        self.assertEqual(order, expected)


if __name__ == '__main__':
    unittest.main()
