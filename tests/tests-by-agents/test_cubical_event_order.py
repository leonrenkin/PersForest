"""Cubical event order: the engine's counting sort equals a lexicographic sort of all cells."""
import unittest

import numpy as np

from persforest import _engine
from persforest.filtrations.cubical import _cell_values, prepare_cubical_filtration


def lexsort_events(values, filtration_type):
    """Reference: cell IDs sorted by (value, dimension, doubled coordinates in row-major order)."""
    grid = np.asarray(values, dtype=float)
    blocks = _cell_values(grid, filtration_type)
    doubled_shape = tuple(2 * n + 1 for n in blocks[0][0].shape)
    row_major, cell_values, dimensions = [], [], []
    for block_values, block_parities in blocks:
        index = np.indices(block_values.shape).reshape(grid.ndim, -1)
        row_major.append(np.ravel_multi_index(2 * index + np.array(block_parities)[:, None], doubled_shape))
        cell_values.append(block_values.reshape(-1))
        dimensions.append(np.full(block_values.size, sum(block_parities)))
    cell_values = np.concatenate(cell_values)
    order = np.lexsort((np.concatenate(row_major), np.concatenate(dimensions), cell_values))
    return order, cell_values[order]


class CubicalEventOrderTests(unittest.TestCase):
    def test_matches_lexicographic_sort(self):
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
                        order, event_values = lexsort_events(values, filtration_type)
                        np.testing.assert_array_equal(cells.event_cells, order)
                        np.testing.assert_array_equal(cells.event_values, event_values)

    def test_signed_zeros_tie_and_keep_their_values(self):
        values = np.array([[0.0, -0.0, 1.0], [-0.0, 0.0, -0.0]])
        _, cells = prepare_cubical_filtration(values)
        order, event_values = lexsort_events(values, 'top_cells')
        np.testing.assert_array_equal(cells.event_cells, order)
        np.testing.assert_array_equal(np.signbit(cells.event_values), np.signbit(event_values))

    def test_invalid_input_raises_value_error(self):
        ranks = np.zeros(4, dtype=np.int64)  # a 1x1 grid has 1 top cell and 4 facets
        with self.assertRaisesRegex(ValueError, 'cell count'):
            _engine.cubical_event_order([1, 1], ranks, 1)
        with self.assertRaisesRegex(ValueError, 'rank out of range'):
            _engine.cubical_event_order([1, 1], np.array([0, 0, 1, 0, 0], dtype=np.int64), 1)
        with self.assertRaisesRegex(ValueError, 'extents'):
            _engine.cubical_event_order([0, 2], ranks, 1)


if __name__ == '__main__':
    unittest.main()
