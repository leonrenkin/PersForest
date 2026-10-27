"""Cubical grids: GUDHI barcodes, representatives, conventions, geometry and validation."""
import unittest

import gudhi
import numpy as np

from persforest import PersistenceForest
from engine_golden import snapshot
from fixtures_small_filtrations import mode_options, reps
from test_cell_interface import coefficients, linear_boundary

GUDHI_KEYWORD = {"top_cells": "top_dimensional_cells", "vertices": "vertices"}
# Shapes per dimension, chosen so that every dimension has finite bars.
SHAPES = {1: (30,), 2: (9, 11), 3: (5, 6, 5), 4: (4, 4, 4, 4)}


def gudhi_bars(values, filtration_type):
    complex_ = gudhi.CubicalComplex(**{GUDHI_KEYWORD[filtration_type]: np.asarray(values, dtype=float)})
    complex_.compute_persistence()
    return sorted((float(b), float(d)) for b, d in
                  complex_.persistence_intervals_in_dimension(np.ndim(values) - 1)
                  if np.isfinite(d) and d > b)


def bars(forest):
    return sorted((b.birth, b.death) for b in forest.barcode if b.death > b.birth)


def grids(dim, filtration_type="top_cells"):
    rng = np.random.default_rng(100 + dim)
    # A vertex grid has one top cell fewer per axis than entries.
    shape = tuple(n + (filtration_type == "vertices") for n in SHAPES[dim])
    return [rng.random(shape), rng.random(shape), rng.integers(0, 4, shape).astype(float)]


class CubicalBarcodeTests(unittest.TestCase):
    def test_barcodes_match_gudhi_for_both_conventions_in_one_to_four_dimensions(self):
        for filtration_type in ("top_cells", "vertices"):
            for dim in SHAPES:
                total = 0
                for index, values in enumerate(grids(dim, filtration_type)):
                    expected = gudhi_bars(values, filtration_type)
                    total += len(expected)
                    for mode, options in mode_options():
                        with self.subTest(filtration_type=filtration_type, dim=dim, grid=index, mode=mode):
                            forest = PersistenceForest.from_cubical_grid(
                                values, filtration_type=filtration_type, **options)
                            self.assertEqual(bars(forest), expected)
                with self.subTest(filtration_type=filtration_type, dim=dim, check="nonempty"):
                    self.assertGreater(total, 0)

    def test_representatives_are_bounded_cycles_and_storages_agree(self):
        for filtration_type in ("top_cells", "vertices"):
            for dim in (1, 2, 3):
                for index, values in enumerate(grids(dim, filtration_type)):
                    with self.subTest(filtration_type=filtration_type, dim=dim, grid=index):
                        records = {}
                        for storage in ("diffs", "cycles"):
                            forest = PersistenceForest.from_cubical_grid(
                                values, filtration_type=filtration_type, storage=storage)
                            for bar in forest.barcode:
                                for rep in reps(forest, bar):
                                    support = coefficients(rep.signed_simplices)
                                    self.assertEqual(linear_boundary(forest._cells, support), {})
                                    self.assertEqual(
                                        linear_boundary(forest._cells, coefficients(rep.interior)), support)
                            records[storage] = snapshot(forest)
                        self.assertEqual(records["diffs"], records["cycles"])


class CubicalConventionTests(unittest.TestCase):
    def test_top_cells_hand_case(self):
        values = [[1, 1, 1], [1, 3, 1], [1, 1, 1]]
        forest = PersistenceForest.from_cubical_grid(values)
        [bar] = forest.barcode
        self.assertEqual((bar.birth, bar.death), (1.0, 3.0))
        [rep] = reps(forest, bar)
        # Boundary of the central pixel in the (axis 0, axis 1) frame.
        self.assertEqual(coefficients(rep.signed_simplices), {(4, 3): 1, (2, 3): -1, (3, 4): -1, (3, 2): 1})
        self.assertEqual(rep.interior, {((3, 3), 1)})

    def test_vertices_hand_case(self):
        values = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        forest = PersistenceForest.from_cubical_grid(values, filtration_type="vertices")
        [bar] = forest.barcode
        self.assertEqual((bar.birth, bar.death), (0.0, 5.0))
        [rep] = reps(forest, bar)
        squares = {(1, 1): 1, (1, 3): 1, (3, 1): 1, (3, 3): 1}
        self.assertEqual(coefficients(rep.signed_simplices), linear_boundary(forest._cells, squares))
        self.assertEqual(len(coefficients(rep.signed_simplices)), 8)
        self.assertEqual(rep.interior, {(square, 1) for square in squares})

    def test_conventions_assign_min_and_max_values(self):
        values = np.array([[0.0, 2.0], [5.0, 1.0]])
        top = dict(PersistenceForest.from_cubical_grid(values).filtration)
        self.assertEqual(top[(1, 1)], 0.0)
        self.assertEqual(top[(2, 1)], 0.0)    # between pixels (0, 0) and (1, 0)
        self.assertEqual(top[(2, 3)], 1.0)    # between pixels (0, 1) and (1, 1)
        self.assertEqual(top[(0, 3)], 2.0)    # domain boundary of pixel (0, 1)
        vertex = dict(PersistenceForest.from_cubical_grid(values, filtration_type="vertices").filtration)
        self.assertEqual(vertex, {(1, 1): 5.0, (0, 1): 2.0, (2, 1): 5.0, (1, 0): 5.0, (1, 2): 2.0})

    def test_events_are_ordered_by_value_dimension_and_cell(self):
        forest = PersistenceForest.from_cubical_grid(np.zeros((2, 3)))
        keys = [(value, sum(c & 1 for c in cell), cell) for cell, value in forest.filtration]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(forest.filtration), 6 + 3 * 3 + 2 * 4)

    def test_transposition_reverses_orientation_and_maps_cells(self):
        values = np.random.default_rng(8).random((6, 7))
        forest = PersistenceForest.from_cubical_grid(values, storage="cycles")
        transposed = PersistenceForest.from_cubical_grid(values.T, storage="cycles")
        self.assertEqual(bars(forest), bars(transposed))
        swap = lambda cell: (cell[1], cell[0])
        expected = sorted((b.birth, b.death, tuple(sorted((swap(c), -v) for c, v in
                                                          coefficients(r.signed_simplices).items())))
                          for b in forest.barcode for r in reps(forest, b))
        actual = sorted((b.birth, b.death, tuple(sorted(coefficients(r.signed_simplices).items())))
                        for b in transposed.barcode for r in reps(transposed, b))
        self.assertEqual(actual, expected)


class CubicalInputTests(unittest.TestCase):
    def test_attributes_and_geometry(self):
        values = np.random.default_rng(1).random((4, 5))
        forest = PersistenceForest.from_cubical_grid(values, spacing=(0.5, 2.0), origin=1.0)
        self.assertIsNone(forest.point_cloud)
        self.assertIsNone(forest.simplex_tree)
        self.assertEqual(forest.dim, 2)
        self.assertEqual((forest.filtration_type, forest.filtration_units), ("top_cells", "grid values"))
        self.assertEqual(forest.storage, "diffs")
        np.testing.assert_array_equal(forest.grid_values, values)
        self.assertFalse(forest.grid_values.flags.writeable)
        values[0, 0] = 99
        self.assertNotEqual(forest.grid_values[0, 0], 99)
        geometry = forest.geometry
        self.assertEqual((geometry.shape, geometry.spacing, geometry.origin), ((4, 5), (0.5, 2.0), (1.0, 1.0)))
        np.testing.assert_allclose(geometry.coordinates((1, 1)), [1.25, 2.0])
        self.assertEqual(geometry.cell_measure((1, 1)), 1.0)
        self.assertEqual(geometry.cell_measure((1, 2)), 0.5)
        self.assertEqual(geometry.cell_measure((2, 3)), 2.0)

    def test_spacing_and_origin_do_not_change_the_barcode(self):
        values = np.random.default_rng(4).random((7, 6))
        reference = snapshot(PersistenceForest.from_cubical_grid(values))
        self.assertEqual(snapshot(PersistenceForest.from_cubical_grid(values, spacing=(3.0, 0.1), origin=(-2, 5))),
                         reference)

    def test_invalid_input_is_rejected(self):
        cases = [
            (dict(values=[[0, np.nan]]), "finite"),
            (dict(values=[[0, np.inf]]), "finite"),
            (dict(values=3.0), "at least one axis"),
            (dict(values=np.zeros((0, 3))), "at least 1 entries"),
            (dict(values=np.zeros((1, 3)), filtration_type="vertices"), "at least 2 entries"),
            (dict(values=[["a", "b"]]), "numeric"),
            (dict(values=np.zeros((2, 2)), filtration_type="lower_star"), "filtration_type must be one of"),
            (dict(values=np.zeros((2, 2)), spacing=0), "positive"),
            (dict(values=np.zeros((2, 2)), spacing=(1, 2, 3)), "one value per axis"),
            (dict(values=np.zeros((2, 2)), origin=(0, np.nan)), "finite"),
            (dict(values=np.zeros((2, 2)), storage="stored"), "storage must be"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                PersistenceForest.from_cubical_grid(**kwargs)

    def test_delayed_barcode_and_interior_activity(self):
        values = [[1, 1, 1], [1, 3, 1], [1, 1, 1]]
        delayed = PersistenceForest.from_cubical_grid(values, compute_barcode=False)
        delayed.compute_barcode()
        self.assertEqual(bars(delayed), [(1.0, 3.0)])
        [(cell, rows)] = delayed.interior_simplex_activity().items()
        self.assertEqual(cell, (3, 3))
        self.assertEqual([(start, end) for _, start, end in rows], [(1.0, 3.0)])


if __name__ == "__main__":
    unittest.main()
