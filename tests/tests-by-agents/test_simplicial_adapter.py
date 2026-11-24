"""Array-based simplicial adapter: agrees with a dict-based numbering of the same events."""
import unittest

import numpy as np

from persforest.filtrations.delaunay import prepare_point_filtration
from persforest.filtrations.simplicial import SimplicialFiltration, prepare_simplicial_filtration, simplicial_boundary


def reference_numbering(events, top_dimension):
    """Former adapter: tops then facets in order of appearance, facet IDs looked up by key."""
    events = [(tuple(simplex), value) for simplex, value in events if len(simplex) >= top_dimension]
    tops = [simplex for simplex, _ in events if len(simplex) == top_dimension + 1]
    simplices = tops + [simplex for simplex, _ in events if len(simplex) == top_dimension]
    ids = {simplex: index for index, simplex in enumerate(simplices)}
    top_facets = [[ids[facet] for facet, _ in simplicial_boundary(top)] for top in tops]
    return simplices, [ids[simplex] for simplex, _ in events], [value for _, value in events], top_facets


class SimplicialAdapterArrayTests(unittest.TestCase):
    def assert_matches_reference(self, cells, events, points):
        simplices, event_cells, values, top_facets = reference_numbering(events, points.shape[1])
        self.assertEqual(cells.cell_keys(np.arange(cells.n_cells)), simplices)
        self.assertEqual(cells.n_top, len(top_facets))
        self.assertEqual(cells.event_cells.tolist(), event_cells)
        self.assertEqual(cells.event_values.tolist(), values)
        self.assertEqual(cells.top_facets.tolist(), top_facets)
        if cells.n_top:
            edges = points[cells.top_vertices[:, 1:]] - points[cells.top_vertices[:, :1]]
            np.testing.assert_array_equal(cells.top_orientations(), np.sign(np.linalg.det(edges)))
            self.assertEqual([cells.orientation(top) for top in range(cells.n_top)], cells.top_orientations().tolist())

    def test_point_filtrations_match_the_reference(self):
        rng = np.random.default_rng(3)
        for dim, n_points in ((1, 12), (2, 60), (3, 40), (4, 25)):
            points = rng.random((n_points, dim))
            for filtration_type in ('alpha', 'delaunay_cech'):
                with self.subTest(dim=dim, filtration_type=filtration_type):
                    tree, cells = prepare_point_filtration(points, filtration_type)
                    self.assert_matches_reference(cells, list(tree.get_filtration()), points)

    def test_custom_filtration_matches_the_reference(self):
        points = np.array([[0., 0], [2, 0], [1, 2], [3, 2], [1, -2]])
        filtration = [((0,), 0.), ((1,), 0.), ((2,), 0.), ((3,), 0.), ((4,), 0.),
                      ((0, 1), 1.), ((0, 2), 1.), ((1, 2), 1.), ((1, 3), 2.), ((2, 3), 2.),
                      ((0, 1, 2), 3.), ((0, 4), 3.), ((1, 4), 3.), ((1, 2, 3), 3.), ((0, 1, 4), 4.)]
        cells = prepare_simplicial_filtration(filtration, points)
        self.assert_matches_reference(cells, filtration, points)

    def test_cell_keys_agree_on_every_input_form(self):
        tree, cells = prepare_point_filtration(np.random.default_rng(4).random((30, 2)), 'alpha')
        ids = [5, cells.n_top, 0, cells.n_cells - 1, cells.n_top - 1, 7]
        keys = reference_numbering(list(tree.get_filtration()), 2)[0]
        expected = [keys[cell_id] for cell_id in ids]
        long_ids = ids * 10
        for form in (ids, tuple(ids), np.array(ids), long_ids, np.array(long_ids)):
            with self.subTest(form=type(form).__name__, n=len(form)):
                self.assertEqual(cells.cell_keys(form), (expected * 10)[:len(form)])
        self.assertEqual(cells.cell_keys([]), [])
        self.assertEqual(cells.cell_keys(np.array([], dtype=int)), [])
        for bad in ([cells.n_cells], [-1], list(range(20)) + [cells.n_cells], np.array([-1])):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'out of range'):
                cells.cell_keys(bad)

    def test_inconsistent_events_raise(self):
        points = np.zeros((3, 2))
        cases = [([((0, 1, 2, 3), 0.)], 'at most 3 vertices'),
                 ([((0, 1), 0.), ((0, 2), 0.), ((1, 2), 0.), ((0, 2, 1), 0.)], 'strictly increasing'),
                 ([((0, 5), 0.)], 'out of range'),
                 ([((0, 1), 0.), ((0, 1), 1.)], r'Facet \(0, 1\) appears more than once'),
                 ([((0, 1), 0.), ((1, 2), 0.), ((0, 1, 2), 1.)], r'Facet \(0, 2\) .*missing')]
        for events, message in cases:
            with self.subTest(events=events), self.assertRaisesRegex(ValueError, message):
                SimplicialFiltration.from_events(points, events)

    def test_empty_and_facet_only_events(self):
        for events in ([], [((0,), 0.)], [((0, 1), 0.)]):
            with self.subTest(events=events):
                cells = SimplicialFiltration.from_events(np.zeros((2, 2)), events)
                self.assertEqual(cells.n_top, 0)
                self.assertEqual(cells.top_facets.shape, (0, 3))
                self.assertEqual(cells.top_orientations().shape, (0,))
                self.assertEqual(cells.n_cells, len(cells.event_cells))


if __name__ == '__main__':
    unittest.main()
