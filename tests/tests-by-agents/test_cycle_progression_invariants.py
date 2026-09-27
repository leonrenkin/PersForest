"""Exact homology/boundary/progression checks independent of forest internals."""
import unittest
import numpy as np
from fixtures_small_filtrations import (
    triangle, split, fan, disconnected, tetrahedra, annulus, build, reps, active_reps,
)
from oracle_chains import coefficients, boundary, rank, homology_data, sample_times, weighted_minimum_basis
from oracle_chains import canonical
from fixtures_small_filtrations import Fixture


class ProgressionInvariantTests(unittest.TestCase):
    def test_all_fixture_representatives_are_a_basis_modulo_boundaries(self):
        for factory in (triangle, split, fan, disconnected, tetrahedra, annulus):
            fixture = factory()
            forest = build(fixture, keep_simplex_diff=True, compute_interior=True)
            for time in sample_times(fixture.values):
                with self.subTest(fixture=factory.__name__, time=time):
                    active = active_reps(forest, time)
                    rows, boundaries, betti = homology_data(fixture.values, forest.dim, time)
                    columns = [coefficients(r.signed_simplices) for r in active]
                    self.assertEqual(len(active), betti)
                    self.assertEqual(rank(boundaries + columns, rows) - rank(boundaries, rows), betti)
                    interior_supports = []
                    for rep, column in zip(active, columns):
                        self.assertEqual(boundary(column), {})
                        self.assertEqual(boundary(coefficients(rep.interior)), column)
                        self.assertTrue(all(fixture.values[s] <= time for s, _ in rep.signed_simplices))
                        support = {s for s, _ in rep.interior}
                        self.assertTrue(all(fixture.values[s] > time for s in support))
                        self.assertTrue(all(support.isdisjoint(previous) for previous in interior_supports))
                        interior_supports.append(support)

    def test_own_progression_coefficient_is_one_not_homology_equality(self):
        for factory in (split, fan, tetrahedra):
            fixture = factory()
            forest = build(fixture, keep_simplex_diff=True, compute_interior=True)
            for bar in forest.barcode:
                representatives = reps(forest, bar)
                for i, earlier in enumerate(representatives):
                    for later in representatives[i:]:
                        time = later.active_start
                        rows, boundaries, _ = homology_data(fixture.values, forest.dim, time)
                        others = [coefficients(r.signed_simplices) for other in forest.barcode if other is not bar
                                  for r in reps(forest, other) if r.active_start <= time < r.active_end]
                        difference = coefficients(list(earlier.signed_simplices)
                                                  + [(s, -v) for s, v in later.signed_simplices])
                        self.assertEqual(rank(boundaries + others + [difference], rows),
                                         rank(boundaries + others, rows))
                        self.assertTrue(later.interior.issubset(earlier.interior))
        # F2: before the split the class is A+B; B alone is not its image.
        fixture = split()
        forest = build(fixture)
        bar = next(b for b in forest.barcode if b.birth == 1)
        difference = coefficients(list(bar.cycle_reps[0].signed_simplices)
                                  + [(s, -v) for s, v in bar.cycle_reps[1].signed_simplices])
        rows, boundaries, _ = homology_data(fixture.values, 2, 4)
        self.assertGreater(rank(boundaries + [difference], rows), rank(boundaries, rows))

    def test_tiny_exhaustive_weight_optimality(self):
        for factory in (split, fan):
            fixture = factory()
            forest = build(fixture, keep_simplex_diff=True, compute_interior=True)
            top = sorted(s for s in fixture.values if len(s) == 3)
            for weights in (dict.fromkeys(top, 1), {s: 2 * i + 1 for i, s in enumerate(top)}):
                for time in sample_times(fixture.values):
                    with self.subTest(fixture=factory.__name__, weights=weights, time=time):
                        actual = sum(weights[s] for rep in active_reps(forest, time) for s, _ in rep.interior)
                        self.assertEqual(actual, weighted_minimum_basis(fixture.values, fixture.points, time, weights))

    def test_oracle_rank_and_boundary_on_independent_known_matrices(self):
        self.assertEqual(rank([{(0,): 1}, {(1,): 1}, {(0,): 2, (1,): 3}], [(0,), (1,)]), 2)
        self.assertEqual(rank([{(0,): 2, (1,): 4}, {(0,): 3, (1,): 6}], [(0,), (1,)]), 1)
        self.assertEqual(boundary({(0, 1, 2): 1}), {(1, 2): 1, (0, 2): -1, (0, 1): 1})
        self.assertEqual(boundary(boundary({(0, 1, 2, 3): 1})), {})

    def test_distinct_death_progressions_are_equivariant_under_vertex_relabeling(self):
        for factory in (split, fan, tetrahedra):
            fixture = factory()
            count = len(fixture.points)
            for permutation in (list(reversed(range(count))), list(range(1, count)) + [0]):
                inverse = {old: new for new, old in enumerate(permutation)}
                relabeled = Fixture(fixture.points[permutation],
                    {tuple(sorted(inverse[i] for i in s)): value for s, value in fixture.values.items()}, [])
                for diff_only in (False, True):
                    forest = build(relabeled, keep_simplex_diff=True, diff_only_mode=diff_only,
                                   compute_interior=not diff_only)
                    for bar in forest.barcode:
                        expected = next(e for e in fixture.expected if e[:2] == (bar.birth, bar.death))
                        for rep, literal in zip(reps(forest, bar), expected[2]):
                            mapped = set()
                            for simplex, sign in rep.signed_simplices:
                                old, parity = canonical(tuple(permutation[i] for i in simplex))
                                mapped.add((old, sign * parity))
                            self.assertEqual(mapped, literal.simplices)
                            self.assertEqual((rep.active_start, rep.active_end), (literal.start, literal.end))
