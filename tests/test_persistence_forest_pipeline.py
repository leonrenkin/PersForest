"""A square triangulation for tests with manually assigned filtrations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import combinations
import unittest

import numpy as np

from persforest import PersistenceForest
from persforest.cycle_rep_vectorisations import signed_chain_area, signed_chain_edge_length
from persforest.forest_landscapes import compute_barcode_functionals


point_cloud = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]]

# The perimeter is ordered counterclockwise around the center vertex 4.
outer_vertices = (0, 1, 2, 5, 8, 7, 6, 3)
outer_edges = tuple(
    tuple(sorted((start, end)))
    for start, end in zip(outer_vertices, outer_vertices[1:] + outer_vertices[:1])
)
spokes = tuple((min(4, vertex), max(4, vertex)) for vertex in outer_vertices)
triangles = tuple(tuple(sorted((*edge, 4))) for edge in outer_edges)

# Explicit closure makes it straightforward to assign filtration values later.
simplicial_complex = (
    {(vertex,) for vertex in range(len(point_cloud))}
    | set(outer_edges)
    | set(spokes)
    | set(triangles)
)


def make_filtration_values(
    edge_values: Mapping[tuple[int, int], float],
    triangle_values: Mapping[tuple[int, int, int], float],
) -> dict[tuple[int, ...], float]:
    """Combine edge and triangle values with vertices born at zero.

    For assigning values, edges are listed in ``outer_edges + spokes``:
    perimeter edges follow ``outer_vertices``, then spokes follow the same
    vertex order. ``triangles`` follows ``outer_edges`` one for one. Each
    simplex tuple has sorted vertex indices; mapping insertion order is ignored.
    """
    return {(vertex,): 0.0 for vertex in range(len(point_cloud))} | dict(edge_values) | dict(triangle_values)


def order_filtration(
    values: Mapping[tuple[int, ...], float],
) -> list[tuple[tuple[int, ...], float]]:
    """Sort for ``PersistenceForest(..., filtration=...)``, with stable ties."""
    if set(values) != simplicial_complex:
        raise ValueError("Filtration values must cover exactly the simplicial complex")
    if any(values[face] > value for simplex, value in values.items() if len(simplex) > 1
           for face in combinations(simplex, len(simplex) - 1)):
        raise ValueError("A face has a later filtration value than its coface")
    return sorted(values.items(), key=lambda item: (item[1], len(item[0]), item[0]))


@dataclass(frozen=True)
class ExpectedRep:
    start: float
    end: float
    signed_simplices: frozenset[tuple[tuple[int, ...], int]]
    signed_value: float
    unsigned_value: float


@dataclass(frozen=True)
class ExpectedBar:
    birth: float
    death: float
    reps: tuple[ExpectedRep, ...]  # Ascending activity order.


@dataclass(frozen=True)
class ManualCase:
    name: str
    edge_values: Mapping[tuple[int, int], float]
    triangle_values: Mapping[tuple[int, int, int], float]
    cycle_func: Callable[..., float]
    forest: tuple  # Nested (filtration value, signed cycle, children) trees.
    bars: tuple[ExpectedBar, ...]
    grid: tuple[float, ...]
    signed_landscapes: tuple[tuple[float, ...], ...]  # Rows are λ₁, λ₂, ...
    unsigned_landscapes: tuple[tuple[float, ...], ...]


def forest_snapshot(forest: PersistenceForest) -> tuple:
    """Represent rooted trees without depending on generated node IDs."""
    def node_snapshot(node_id):
        node = forest.nodes[node_id]
        cycle = None if node.cycle is None else tuple(sorted(node.cycle.signed_simplices))
        children = tuple(sorted((node_snapshot(child) for child in node.children), key=repr))
        return node.filt_val, cycle, children

    return tuple(sorted((node_snapshot(node.id) for node in forest.nodes.values()
                         if node.parent is None), key=repr))


def bar_signature(bar) -> tuple:
    """Identify a bar by its endpoints and complete signed progression."""
    return (bar.birth, bar.death, tuple(
        (rep.active_start, rep.active_end, frozenset(rep.signed_simplices))
        for rep in bar.cycle_reps
    ))


def check_manual_case(test: unittest.TestCase, case: ManualCase) -> None:
    """Compare one manual filtration with forest, barcode, and measurements."""
    values = make_filtration_values(case.edge_values, case.triangle_values)
    forest = PersistenceForest(point_cloud, filtration=order_filtration(values))
    test.assertEqual(forest_snapshot(forest), case.forest)

    expected_signatures = [(
        bar.birth, bar.death,
        tuple((rep.start, rep.end, rep.signed_simplices) for rep in bar.reps),
    ) for bar in case.bars]
    expected = dict(zip(expected_signatures, case.bars))
    test.assertCountEqual([bar_signature(bar) for bar in forest.barcode], expected_signatures)

    for signed, landscape_rows in ((True, case.signed_landscapes),
                                   (False, case.unsigned_landscapes)):
        label = f"{case.name}-{signed}"
        if landscape_rows:
            family = forest.compute_measurement_landscapes(
                case.cycle_func, label, signed=signed, x_grid=np.asarray(case.grid),
                max_k=len(landscape_rows), cache_functionals=True,
            )
            profiles = forest.barcode_functionals[label]
        else:
            profiles = compute_barcode_functionals(
                forest,
                lambda chain, points: case.cycle_func(chain if signed else chain.unsigned(), points),
                label, cache=False,
            )
        test.assertEqual(len(profiles.bars), len(case.bars))
        for bar in profiles.bars:
            reps = expected[bar_signature(bar)].reps
            step = profiles[bar]
            np.testing.assert_allclose(step.starts, [rep.start for rep in reps])
            np.testing.assert_allclose(step.ends, [rep.end for rep in reps])
            np.testing.assert_allclose(
                step.vals,
                [rep.signed_value if signed else rep.unsigned_value for rep in reps],
                rtol=1e-12, atol=1e-12,
            )
        if landscape_rows:
            np.testing.assert_allclose(
                family.evaluate_on_grid(case.grid, levels=len(landscape_rows)),
                landscape_rows, rtol=1e-12, atol=1e-12,
            )

outer_cycle = frozenset(
    (tuple(sorted((start, end))), 1 if start < end else -1)
    for start, end in zip(outer_vertices, outer_vertices[1:] + outer_vertices[:1])
)

#only outer circle
expected_landscapes_1 = ((0, 1, 2, 3, 4, 3, 2, 1, 0), (0,) * 9)
manual_case_1 = ManualCase(
    name="outer_cycle_filled_at_2",
    edge_values={edge: 0.0 for edge in outer_edges} | {edge: 2.0 for edge in spokes}, # type: ignore
    triangle_values={triangle: 2.0 for triangle in triangles}, # type: ignore
    cycle_func=signed_chain_area,
    forest=((0.0, tuple(sorted(outer_cycle)), ((2.0, tuple(sorted(outer_cycle)), ()),)),),
    bars=(ExpectedBar(0.0, 2.0, (ExpectedRep(0.0, 2.0, outer_cycle, 4.0, 4.0),)),),
    grid=tuple(i / 4 for i in range(9)),
    signed_landscapes=expected_landscapes_1,
    unsigned_landscapes=expected_landscapes_1,
)

#outer circle with spoke does not change area
spoke_1_4 = (1, 4)
outer_cycle_with_spoke = outer_cycle | {(spoke_1_4, 1), (spoke_1_4, -1)}
expected_landscapes_2 = ((0, 1, 2, 3, 4, 3, 2, 1, 0), (0,) * 9)
manual_case_2 = ManualCase(
    name="outer_cycle_with_doubled_spoke",
    edge_values={edge: 0.0 for edge in outer_edges}
    | {edge: (0.0 if edge == spoke_1_4 else 2.0) for edge in spokes},
    triangle_values={triangle: 2.0 for triangle in triangles},
    cycle_func=signed_chain_area,
    forest=((0.0, tuple(sorted(outer_cycle_with_spoke)),
             ((2.0, tuple(sorted(outer_cycle_with_spoke)), ()),)),),
    bars=(ExpectedBar(0.0, 2.0, (
        ExpectedRep(0.0, 2.0, outer_cycle_with_spoke, 4.0, 4.0),
    )),),
    grid=tuple(i / 4 for i in range(9)),
    signed_landscapes=expected_landscapes_2,
    unsigned_landscapes=expected_landscapes_2,
)

#outer circle with spoke does change length between signed and unsigned.
manual_case_3 = ManualCase(
    name="outer_cycle_with_doubled_spoke_length",
    edge_values={edge: 0.0 for edge in outer_edges}
    | {edge: (0.0 if edge == spoke_1_4 else 2.0) for edge in spokes},
    triangle_values={triangle: 2.0 for triangle in triangles},
    cycle_func=signed_chain_edge_length,
    forest=((0.0, tuple(sorted(outer_cycle_with_spoke)),
             ((2.0, tuple(sorted(outer_cycle_with_spoke)), ()),)),),
    bars=(ExpectedBar(0.0, 2.0, (
        ExpectedRep(0.0, 2.0, outer_cycle_with_spoke, 10.0, 8.0),
    )),),
    grid=tuple(i / 4 for i in range(9)),
    unsigned_landscapes=((0, 2, 4, 6, 8, 6, 4, 2, 0), (0,) * 9),
    signed_landscapes=((0, 2.5, 5, 7.5, 10, 7.5, 5, 2.5, 0), (0,) * 9),
)


def oriented_edge(start: int, end: int):
    """Encode a directed edge using the signed-simplex convention."""
    return tuple(sorted((start, end))), 1 if start < end else -1


triangle_cycles = {
    tuple(sorted((start, end, 4))): frozenset((
        oriented_edge(start, end), oriented_edge(end, 4), oriented_edge(4, start),
    ))
    for start, end in zip(outer_vertices, outer_vertices[1:] + outer_vertices[:1])
}
triangle_014 = (0, 1, 4)
leaf_nodes = tuple(sorted((
    (3.0 if triangle == triangle_014 else 2.0, tuple(sorted(cycle)), ())
    for triangle, cycle in triangle_cycles.items()
), key=repr))

expected_landscapes_4 = ((0,2,2.25,2.5,0.5,0.25,0), *((0,0,0,0.25,0.0,0.0,0.0) for i in range(7)), (0,0,0,0,0,0,0))
manual_case_4 = ManualCase(
    name="spokes_at_1_one_triangle_at_3",
    edge_values={edge: 0.0 for edge in outer_edges} | {edge: 1.0 for edge in spokes},
    triangle_values={triangle: (3.0 if triangle == triangle_014 else 2.0)
                     for triangle in triangles},
    cycle_func=signed_chain_area,
    forest=((0.0, tuple(sorted(outer_cycle)),
             ((1.0, tuple(sorted(outer_cycle)), leaf_nodes),)),),
    bars=(
        ExpectedBar(0.0, 3.0, (
            ExpectedRep(0.0, 1.0, outer_cycle, 4.0, 4.0),
            ExpectedRep(1.0, 3.0, triangle_cycles[triangle_014], 0.5, 0.5),
        )),
        *(ExpectedBar(1.0, 2.0, (
            ExpectedRep(1.0, 2.0, cycle, 0.5, 0.5),
        )) for triangle, cycle in triangle_cycles.items() if triangle != triangle_014),
    ),
    grid=tuple(i / 2 for i in range(7)),
    signed_landscapes=expected_landscapes_4,
    unsigned_landscapes=expected_landscapes_4,
)

CASES: tuple[ManualCase, ...] = (manual_case_1, manual_case_2, manual_case_3, manual_case_4)


class ManualPipelineTests(unittest.TestCase):
    def test_manual_cases(self):
        for case in CASES:
            with self.subTest(case=case.name):
                check_manual_case(self, case)
