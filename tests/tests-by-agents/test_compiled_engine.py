"""Compiled engine: forest arrays agree with the golden snapshots and the Python engine."""
from collections import defaultdict
import json
import unittest
import warnings

import numpy as np

from persforest import PersistenceForest
from persforest._forest.arrays import build_forest_arrays, chain_ids
from engine_golden import CASES, GOLDEN_PATH, MODES, _signed, digest, snapshot
from fixtures_small_filtrations import annulus, build, disconnected, fan, split, tetrahedra, triangle


def keyed(cells, ids, signs):
    return set(zip(cells.cell_keys(ids.tolist()), signs.tolist()))


def arrays_snapshot(arrays, cells) -> dict:
    """The golden-snapshot record, computed from the engine arrays alone."""
    def bar_key(bar):
        return [float(arrays.bar_birth[bar]), float(arrays.bar_death[bar]),
                arrays.critical_values(bar).tolist()]

    bars = []
    activity = defaultdict(list)
    for bar in range(arrays.n_bars):
        reps = []
        for node in arrays.progression(bar).tolist():
            facets, facet_signs, tops, orientations = chain_ids(cells, *arrays.subtree_entries(node))
            start, end = arrays.active_interval(node)
            reps.append([start, end, _signed(keyed(cells, facets, facet_signs)),
                         _signed(keyed(cells, tops, orientations))])
        parent = int(arrays.bar_parent[bar])
        bars.append(bar_key(bar) + [None if parent < 0 else bar_key(parent), reps])
        # A top cell's interior activity ends at the first node, walking down
        # from the death, whose subtree contains it.
        seen = {}
        previous = None
        for node in reversed(arrays.progression(bar).tolist()):
            cells_, signs_ = (arrays.subtree_entries(node) if previous is None
                              else arrays.step_entries(previous, node))
            for top in cells_[cells_ < cells.n_top].tolist():
                seen.setdefault(top, float(arrays.node_value[node]))
            previous = node
        birth, death = float(arrays.bar_birth[bar]), float(arrays.bar_death[bar])
        for top, end in seen.items():
            activity[cells.cell_keys([top])[0]].append([birth, death, birth, end])
    bars.sort(key=lambda item: json.dumps(item))
    return {'bars': bars,
            'interior_activity': sorted([list(key), sorted(rows)] for key, rows in activity.items())}


def tree_partition(groups) -> list:
    return sorted(sorted(json.dumps(item) for item in group) for group in groups)


def python_trees(forest):
    trees = defaultdict(list)
    for bar in forest.barcode:
        values = [forest.nodes[n].filt_val for n in bar._node_progression]
        trees[bar.root_id].append([bar.birth, bar.death, values])
    return trees.values()


def compiled_trees(arrays):
    trees = defaultdict(list)
    for bar in range(arrays.n_bars):
        trees[int(arrays.bar_tree[bar])].append(
            [float(arrays.bar_birth[bar]), float(arrays.bar_death[bar]), arrays.critical_values(bar).tolist()])
    return trees.values()


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
    def test_arrays_reproduce_golden_snapshots(self):
        expected = json.loads(GOLDEN_PATH.read_text())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for name, make in CASES.items():
                for mode, options in MODES.items():
                    with self.subTest(case=name, mode=mode):
                        forest = make(**options)
                        arrays = build_forest_arrays(forest._cells, reduce=options['reduce'],
                                                     filtration_tol=forest.filtration_tol)
                        self.assertEqual(digest(arrays_snapshot(arrays, forest._cells)),
                                         expected[f'{name}/{mode}']['digest'])

    def test_arrays_match_python_engine_on_random_inputs(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for name, make in random_inputs():
                for reduce in (False, True):
                    for tol in (0.0, 1e-12):
                        with self.subTest(case=name, reduce=reduce, tol=tol):
                            forest = make(reduce=reduce, filtration_tol=tol)
                            arrays = build_forest_arrays(forest._cells, reduce=reduce, filtration_tol=tol)
                            self.assertEqual(arrays_snapshot(arrays, forest._cells), snapshot(forest))
                            self.assertEqual(tree_partition(compiled_trees(arrays)),
                                             tree_partition(python_trees(forest)))

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


if __name__ == '__main__':
    unittest.main()
