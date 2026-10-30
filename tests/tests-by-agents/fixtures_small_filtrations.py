"""Explicit embedded fixtures F1–F6 from TEST_PLAN.md (no alpha geometry)."""
from dataclasses import dataclass, replace
import importlib
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch

import gudhi
import numpy as np

from persforest.PersistenceForest import PFBar, SignedChain

MODULE = importlib.import_module('persforest.PersistenceForest')
Simplex = tuple[int, ...]
SignedSet = frozenset[tuple[Simplex, int]]


def chain(text: str) -> SignedSet:
    """Read literal signed simplices with single-digit vertex indices."""
    return frozenset((tuple(map(int, term[1:])), 1 if term[0] == '+' else -1)
                     for term in text.split())


@dataclass(frozen=True)
class ExpectedRep:
    start: float
    end: float
    simplices: SignedSet
    interior: SignedSet


@dataclass
class Fixture:
    points: np.ndarray
    values: dict[Simplex, float]
    # Duplicate intervals are allowed; this is deliberately not a dict.
    expected: list[tuple[float, float, tuple[ExpectedRep, ...]]]

    def changed(self, updates: dict[Simplex, float]) -> 'Fixture':
        return replace(self, values=self.values | updates, expected=[])


def make_fixture(points, values, expected=()) -> Fixture:
    """Add the explicitly specified vertex-at-zero default, but no other faces."""
    return Fixture(np.array(points, dtype=float),
                   {(i,): 0. for i in range(len(points))} | values, list(expected))


def triangle() -> Fixture:
    return make_fixture([(0, 0), (3, 0), (0, 4)],
                        {(0, 1): 1, (0, 2): 1, (1, 2): 2, (0, 1, 2): 6},
                        [(2, 6, (ExpectedRep(2, 6, chain('+01 -02 +12'), chain('+012')),))])


def split() -> Fixture:
    a, b, o = chain('+01 -02 +12'), chain('-01 +03 -13'), chain('-02 +12 +03 -13')
    return make_fixture([(0, 0), (4, 0), (0, 2), (0, -1)],
        {(0, 2): 1, (1, 2): 1, (0, 3): 1, (1, 3): 1,
         (0, 1): 3, (0, 1, 2): 5, (0, 1, 3): 7},
        [(1, 7, (ExpectedRep(1, 3, o, chain('+012 -013')),
                 ExpectedRep(3, 7, b, chain('-013')))),
         (3, 5, (ExpectedRep(3, 5, a, chain('+012')),))])


def fan() -> Fixture:
    o = chain('+01 -02 +12')
    interior = chain('+013 -023 +123')
    return make_fixture([(0, 0), (4, 0), (0, 4), (1, 1)],
        {(0, 1): 1, (0, 2): 1, (1, 2): 1, (0, 3): 2, (1, 3): 3,
         (2, 3): 4, (0, 1, 3): 5, (1, 2, 3): 6, (0, 2, 3): 7},
        [(1, 7, (ExpectedRep(1, 2, o, interior),
                 ExpectedRep(2, 3, o | chain('+03 -03'), interior),
                 ExpectedRep(3, 4, chain('-02 +03 +12 -13'), chain('-023 +123')),
                 ExpectedRep(4, 7, chain('-02 +03 -23'), chain('-023')))),
         (3, 5, (ExpectedRep(3, 5, chain('+01 -03 +13'), chain('+013')),)),
         (4, 6, (ExpectedRep(4, 6, chain('+12 -13 +23'), chain('+123')),))])


def disconnected(shift_time: float = 0) -> Fixture:
    first = triangle()
    values = first.values | {tuple(v + 3 for v in s): t + shift_time
                             for s, t in first.values.items()}
    moved = lambda terms: frozenset((tuple(v + 3 for v in s), sign) for s, sign in terms)
    reps = tuple(ExpectedRep(r.start + shift_time, r.end + shift_time,
                             moved(r.simplices), moved(r.interior))
                 for r in first.expected[0][2])
    return Fixture(np.vstack((first.points, first.points + [10, 0])), values,
                   first.expected + [(2 + shift_time, 6 + shift_time, reps)])


def tetrahedra() -> Fixture:
    top, bottom = (0, 1, 2, 3), (0, 1, 2, 4)
    values = {s: (0 if n < 3 else 3 if s == (0, 1, 2) else 1)
              for tet in (top, bottom) for n in (1, 2, 3) for s in combinations(tet, n)}
    values |= {top: 5, bottom: 7}
    a, b = chain('+123 -023 +013 -012'), chain('-124 +024 -014 +012')
    o = chain('+123 -023 +013 -124 +024 -014')
    return make_fixture([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -2)],
        values, [(1, 7, (ExpectedRep(1, 3, o, chain('+0123 -0124')),
                         ExpectedRep(3, 7, b, chain('-0124')))),
                 (3, 5, (ExpectedRep(3, 5, a, chain('+0123')),))])


def annulus() -> Fixture:
    ring = [(0, 1, 5), (0, 4, 5), (1, 2, 6), (1, 5, 6),
            (2, 3, 7), (2, 6, 7), (0, 3, 4), (3, 4, 7)]
    inner = [(4, 5, 6), (4, 6, 7)]
    values = {s: (0 if n == 1 else 3 if n == 2 else 5)
              for tri in ring + inner for n in (1, 2, 3) for s in combinations(tri, n)}
    for tri in inner:
        for n in (2, 3):
            for s in combinations(tri, n):
                values[s] = 1
    for s in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        values[s] = 2
    return make_fixture([(0, 0), (4, 0), (4, 4), (0, 4),
                         (1, 1), (3, 1), (3, 3), (1, 3)], values)


def simplex_tree(fixture: Fixture) -> gudhi.SimplexTree:
    """Validate closure first so GUDHI cannot silently repair a bad fixture."""
    for simplex, value in fixture.values.items():
        if not simplex or tuple(sorted(set(simplex))) != simplex or not np.isfinite(value):
            raise ValueError('Fixture must use canonical nonempty simplices and finite values')
        for face in combinations(simplex, len(simplex) - 1):
            if face and (face not in fixture.values or fixture.values[face] > value):
                raise ValueError(f'Invalid fixture face {face} of {simplex}')
    tree = gudhi.SimplexTree()
    for simplex, value in sorted(fixture.values.items(), key=lambda item: (item[1], len(item[0]), item[0])):
        tree.insert(simplex, filtration=value)
    actual = {tuple(s): t for s, t in tree.get_filtration()}
    if actual != fixture.values:
        raise AssertionError('GUDHI altered fixture values')
    return tree


def build(fixture: Fixture, *, order=None, **options):
    """Inject only alpha construction; run the complete production algorithm."""
    tree = simplex_tree(fixture)
    if order is not None:
        entries = [(tuple(s), float(t)) for s, t in order]
        if len(entries) != len(fixture.values) or dict(entries) != fixture.values:
            raise ValueError('Order must contain every fixture simplex exactly once')
        positions = {s: i for i, (s, _) in enumerate(entries)}
        for i, (s, t) in enumerate(entries):
            if i and t < entries[i - 1][1]:
                raise ValueError('Order decreases in filtration')
            if any(face and positions[face] >= i for face in combinations(s, len(s) - 1)):
                raise ValueError('Order puts coface before face')
        tree = SimpleNamespace(get_filtration=lambda: iter(entries))
    fake_alpha = SimpleNamespace(create_simplex_tree=lambda **kwargs: tree)
    with patch.object(MODULE.gd, 'AlphaComplex', return_value=fake_alpha):
        return MODULE.PersistenceForest(fixture.points, **options)


def positive_bars(forest) -> list[tuple[float, float]]:
    return sorted((b.birth, b.death) for b in forest.barcode if b.death > b.birth)


def reps(forest, bar) -> list[SignedChain]:
    return list(reversed(list(forest.iter_bar_cycle_reps(bar))))


def active_reps(forest, time: float) -> list[SignedChain]:
    return [r for b in forest.barcode if b.birth <= time < b.death
            for r in reps(forest, b) if r.active_start <= time < r.active_end]


def mode_options():
    for reduced in (False, True):
        for name, options in [('stored', {}), ('diffs', {'keep_simplex_diff': True}),
                              ('interiors', {'keep_simplex_diff': True, 'compute_interior': True}),
                              ('diff_only', {'keep_simplex_diff': True, 'diff_only_mode': True})]:
            yield f'{name}/reduce={reduced}', {'reduce': reduced, **options}


def profile_forest(specs):
    """Create independent bar profiles: (birth, death, [(start,end,value), ...])."""
    bars = []
    for index, (birth, death, records) in enumerate(specs):
        cycles = [SimpleNamespace(active_start=s, active_end=e, value=v) for s, e, v in records]
        bars.append(PFBar(birth, death, tuple(range(len(cycles))), cycles))
    return SimpleNamespace(barcode=bars, point_cloud=np.zeros((0, 2)),
                           iter_bar_cycle_reps=lambda bar: iter(reversed(bar.cycle_reps)),
                           barcode_functionals={}, landscape_families={},
                           max_bar=lambda: max(bars, key=lambda bar: bar.lifespan()))


def measured(rep, points) -> float:
    return rep.value
