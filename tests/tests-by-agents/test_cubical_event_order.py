"""Cubical events: the engine's counting sort equals a lexicographic sort of all cells."""
import unittest

import numpy as np

from persforest import _engine
from persforest.filtrations.cubical import _cell_values, cubical_boundary, prepare_cubical_filtration


def lexsort_events(values, filtration_type):
    """Reference: cell keys and values sorted by (value, dimension, doubled coordinates in row-major order)."""
    grid = np.asarray(values, dtype=float)
    blocks = _cell_values(grid, filtration_type)
    doubled_shape = tuple(2 * n + 1 for n in blocks[0][0].shape)
    keys, row_major, cell_values, dimensions = [], [], [], []
    for block_values, block_parities in blocks:
        index = np.indices(block_values.shape).reshape(grid.ndim, -1)
        coordinates = 2 * index + np.array(block_parities)[:, None]
        keys.append(coordinates.T)
        row_major.append(np.ravel_multi_index(coordinates, doubled_shape))
        cell_values.append(block_values.reshape(-1))
        dimensions.append(np.full(block_values.size, sum(block_parities)))
    cell_values = np.concatenate(cell_values)
    order = np.lexsort((np.concatenate(row_major), np.concatenate(dimensions), cell_values))
    return list(map(tuple, np.concatenate(keys)[order].tolist())), cell_values[order]


class CubicalEventTests(unittest.TestCase):
    def test_events_match_lexicographic_sort_and_ids_follow_event_order(self):
        rng = np.random.default_rng(11)
        shapes = [(1,), (7,), (1, 1), (1, 5), (6, 1), (5, 8), (3, 4, 5), (2, 3, 2, 3)]
        for shape in shapes:
            inputs = {'float': rng.random(shape), 'ties': rng.integers(0, 3, shape).astype(float),
                      'constant': np.zeros(shape)}
            for filtration_type in ('top_cells', 'vertices'):
                if filtration_type == 'vertices' and min(shape) < 2:
                    continue
                for kind, values in inputs.items():
                    with self.subTest(shape=shape, filtration_type=filtration_type, kind=kind):
                        _, cells = prepare_cubical_filtration(values, filtration_type)
                        keys, event_values = lexsort_events(values, filtration_type)
                        self.assertEqual(cells.cell_keys(cells.event_cells), keys)
                        np.testing.assert_array_equal(cells.event_values, event_values)
                        # Top cells 0 .. n_top - 1, then facets, each in event order.
                        is_top = cells.event_cells < cells.n_top
                        np.testing.assert_array_equal(cells.event_cells[is_top], np.arange(cells.n_top))
                        np.testing.assert_array_equal(cells.event_cells[~is_top],
                                                      np.arange(cells.n_top, cells.n_cells))
                        all_keys = cells.cell_keys(np.arange(cells.n_cells))
                        self.assertEqual([cells._cell_key(cell) for cell in range(cells.n_cells)], all_keys)
                        for top in range(cells.n_top):
                            self.assertEqual([all_keys[facet] for facet in cells.top_facets[top].tolist()],
                                             [facet for facet, _ in cubical_boundary(all_keys[top])])

    def test_signed_zeros_tie_and_keep_their_values(self):
        values = np.array([[0.0, -0.0, 1.0], [-0.0, 0.0, -0.0]])
        _, cells = prepare_cubical_filtration(values)
        keys, event_values = lexsort_events(values, 'top_cells')
        self.assertEqual(cells.cell_keys(cells.event_cells), keys)
        np.testing.assert_array_equal(np.signbit(cells.event_values), np.signbit(event_values))

    def test_invalid_input_raises_value_error(self):
        ranks, values = np.zeros(5, dtype=np.int32), np.zeros(5)  # a 1x1 grid: 1 top cell, 4 facets
        with self.assertRaisesRegex(ValueError, 'cell count'):
            _engine.cubical_events([1, 2], ranks, values, 1)
        with self.assertRaisesRegex(ValueError, 'rank out of range'):
            _engine.cubical_events([1, 1], np.array([0, 0, 1, 0, 0], dtype=np.int32), values, 1)
        with self.assertRaisesRegex(ValueError, 'extents'):
            _engine.cubical_events([0, 2], ranks, values, 1)
        with self.assertRaisesRegex(ValueError, 'differ in length'):
            _engine.cubical_events([1, 1], ranks, values[:4], 1)


if __name__ == '__main__':
    unittest.main()
