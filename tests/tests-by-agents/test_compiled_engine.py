"""Compiled engine: forests agree with the frozen Python reference engine."""
from collections import defaultdict
import json
import unittest
import warnings

import numpy as np

from persforest import PersistenceForest
from persforest._forest.arrays import build_forest_arrays
from engine_golden import snapshot
from fixtures_small_filtrations import annulus, build, disconnected, fan, split, tetrahedra, triangle
from reference_engine import ReferenceForest


def tree_partition(forest) -> list:
    """Bars grouped by tree, as sorted JSON keys (independent of tree numbering)."""
    trees = defaultdict(list)
    for bar in forest.barcode:
        trees[bar.tree_id].append(json.dumps([bar.birth, bar.death, list(bar.critical_values)]))
    return sorted(sorted(group) for group in trees.values())


def random_inputs():
    rng = np.random.default_rng(2026)
    for shape in ((40,), (9, 11), (12, 12), (5, 6, 7)):
        for filtration_type in ('top_cells', 'vertices'):
            for kind in ('float', 'ties'):
                values = rng.random(shape) if kind == 'float' else rng.integers(0, 4, shape).astype(float)
                yield (f'cubical {shape} {filtration_type} {kind}',
                       lambda v=values, t=filtration_type, **o: PersistenceForest.from_cubical_grid(
                           v, filtration_type=t, **o))
    for dim, n in ((2, 60), (3, 30)):
        for filtration_type in ('alpha', 'delaunay_cech'):
            points = rng.random((n, dim))
            yield (f'{filtration_type} {dim}d n={n}',
                   lambda p=points, t=filtration_type, **o: PersistenceForest.from_points(
                       p, filtration_type=t, **o))
    lattice = np.column_stack([a.ravel() for a in np.meshgrid(np.arange(5.0), np.arange(5.0))])
    yield 'alpha lattice ties', lambda **o: PersistenceForest.from_points(lattice, **o)
    for factory in (triangle, split, fan, disconnected, tetrahedra, annulus):
        yield f'fixture {factory.__name__}', lambda f=factory, **o: build(f(), **o)


class CompiledEngineTests(unittest.TestCase):
    def test_forests_match_reference_engine_on_random_inputs(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for name, make in random_inputs():
                for reduce in (False, True):
                    for tol in (0.0, 1e-12):
                        with self.subTest(case=name, reduce=reduce, tol=tol):
                            forest = make(reduce=reduce, filtration_tol=tol)
                            reference = ReferenceForest(forest._cells, reduce=reduce, filtration_tol=tol)
                            self.assertEqual(snapshot(forest), snapshot(reference))
                            self.assertEqual(tree_partition(forest), tree_partition(reference))

    def test_subtrees_are_contiguous_post_order_ranges(self):
        values = np.random.default_rng(3).random((15, 15))
        forest = PersistenceForest.from_cubical_grid(values)
        arrays = build_forest_arrays(forest._cells, reduce=True, filtration_tol=forest.filtration_tol)
        parent, first = arrays.node_parent, arrays.node_first
        for node in range(len(parent)):
            if parent[node] >= 0:
                self.assertLess(node, parent[node])
                self.assertLessEqual(first[parent[node]], first[node])
            self.assertLessEqual(first[node], node)

    def test_small_and_large_chain_paths_agree(self):
        from unittest.mock import patch
        import persforest._forest.arrays as arrays_module
        for make in (lambda: PersistenceForest.from_cubical_grid(np.random.default_rng(4).random((12, 12))),
                     lambda: PersistenceForest.from_points(np.random.default_rng(5).random((40, 3)))):
            forest = make()
            arrays = build_forest_arrays(forest._cells, reduce=True, filtration_tol=forest.filtration_tol)
            for node in range(len(arrays.node_value)):
                entries = arrays.subtree_entries(node)
                with patch.object(arrays_module, "_SMALL_CHAIN", 10 ** 9):
                    small = arrays_module.chain_ids(forest._cells, *entries)
                with patch.object(arrays_module, "_SMALL_CHAIN", -1):
                    large = arrays_module.chain_ids(forest._cells, *entries)
                self.assertEqual(set(zip(small[0], small[1])), set(zip(large[0], large[1])))
                self.assertEqual(sorted(zip(small[2], small[3])), sorted(zip(large[2], large[3])))
                self.assertEqual(len(small[0]), len(set(zip(small[0], small[1]))))

    def test_invalid_engine_input_raises_value_error(self):
        from persforest import _engine
        empty_facets = np.zeros((1, 2), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            _engine.build_forest(np.array([5], dtype=np.int64), np.zeros(1), 1, 3, empty_facets,
                                 np.ones(1, dtype=np.int8), True, 0.0)
        three_tops = np.array([[3, 4], [3, 5], [3, 6]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'more than two top cells'):
            _engine.build_forest(np.array([3, 4, 5, 6, 0, 1, 2], dtype=np.int64), np.zeros(7), 3, 7,
                                 three_tops, np.ones(3, dtype=np.int8), True, 0.0)
        # Node and cell IDs are 32-bit inside the engine.
        with self.assertRaisesRegex(ValueError, '32-bit IDs'):
            _engine.build_forest(np.zeros(0, dtype=np.int64), np.zeros(0), 0, 2 ** 31,
                                 np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int8), True, 0.0)


if __name__ == '__main__':
    unittest.main()
