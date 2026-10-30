"""Planar loops of 1-cycles: raw, midpoint, smoothed, averaged and taut curves, loop functionals and plotting."""
import math
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from persforest import PersistenceForest
from persforest.chains import SignedChain
from persforest.filtrations.cubical import CubicalFiltration
from persforest.geometry.cubical import cubical_chain_to_vertex_chain
from persforest.geometry.cubical import make_cubical_geometry
from persforest.geometry.loops import PlanarLoops
from persforest.geometry.loops import averaged_loops
from persforest.geometry.loops import midpoint_loops
from persforest.geometry.loops import raw_loops
from persforest.geometry.loops import smoothed_loops
from persforest.geometry.loops import taut_loops
from persforest.geometry.planar import polygon_area
from persforest.geometry.planar import polygon_length
from persforest.measurements import cubical as cubical_measurements
from persforest.measurements.basic import signed_chain_area
from persforest.measurements.basic import signed_chain_connected_components
from persforest.measurements.cubical import cubical_chain_volume
from persforest.measurements.loops import loops_area
from persforest.measurements.loops import loops_circularity
from persforest.measurements.loops import loops_convex_hull_area_deficit
from persforest.measurements.loops import loops_excess_curvature
from persforest.measurements.loops import loops_length
from persforest.measurements.loops import loops_max_convexity_defect_depth
from persforest.measurements.loops import on_averaged_loops
from persforest.measurements.loops import on_midpoint_loops
from persforest.measurements.loops import on_raw_loops
from persforest.measurements.loops import on_smoothed_loops
from persforest.measurements.loops import on_taut_loops
from persforest.measurements import shape
from persforest.plotting.loops import plot_planar_loops

_GRID = CubicalFiltration(2, [])


def region_chain(mask, spacing=(1.0, 1.0)):
    """Boundary chain of the pixels in ``mask`` and the grid geometry."""
    coefficients = {}
    for i, j in zip(*np.nonzero(mask)):
        for facet, sign in _GRID.boundary((2 * i + 1, 2 * j + 1)):
            coefficients[facet] = coefficients.get(facet, 0) + sign
    chain = SignedChain({(facet, int(np.sign(c))) for facet, c in coefficients.items() if c})
    return chain, make_cubical_geometry(mask.shape, spacing, 0.0, "top_cells")


def disk(radius, center=(0.3, -0.2), size=None):
    size = size or int(2 * radius + 6)
    x, y = np.mgrid[:size, :size] + 0.5 - size / 2
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2


def ellipse(a, b, angle, size=60):
    x, y = np.mgrid[:size, :size] + 0.5 - size / 2 + 0.17
    u = math.cos(angle) * x + math.sin(angle) * y
    v = -math.sin(angle) * x + math.cos(angle) * y
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def pad(mask):
    return np.pad(np.asarray(mask, bool), 2)


def distances_to_staircase(curve, staircase):
    """Distance of every curve vertex to the closed staircase polyline."""
    start, end = staircase, np.roll(staircase, -1, axis=0)
    step = end - start
    t = np.clip(((curve[:, None] - start) * step).sum(2) / (step**2).sum(1), 0, 1)
    return np.linalg.norm(curve[:, None] - (start + t[..., None] * step), axis=2).min(1)


def shortest_crossing_length(staircase, sweeps):
    """Independent check of taut loops: minimise the closed length over one point per portal.

    Each portal joins the pixel centers on both sides of a staircase edge
    (doubled coordinates). Jacobi sweeps move every other point to the exact
    minimiser of its two adjacent segments; the problem is convex.
    """
    vertices = staircase.astype(float)
    edges = np.roll(vertices, -1, axis=0) - vertices
    middle = vertices + edges / 2
    across = edges / np.linalg.norm(edges, axis=1)[:, None]
    along = np.column_stack((-across[:, 1], across[:, 0]))
    half = np.abs(np.column_stack((-edges[:, 1], edges[:, 0])) / 2).sum(1)
    t = np.zeros(len(vertices))
    for _ in range(sweeps):
        for parity in (0, 1):
            i = np.arange(parity, len(vertices), 2)
            points = middle + t[:, None] * along
            a, b = points[i - 1] - middle[i], points[(i + 1) % len(vertices)] - middle[i]
            a_u, a_w = (a * along[i]).sum(1), (a * across[i]).sum(1)
            b_u, b_w = (b * along[i]).sum(1), (b * across[i]).sum(1)
            b_w = np.where(a_w * b_w > 0, -b_w, b_w)  # reflect onto the other side
            gap = a_w - b_w
            share = np.divide(a_w, gap, out=np.zeros_like(gap), where=gap != 0)
            # Both neighbours on the portal line: any point between them is optimal.
            u = np.where(gap != 0, a_u + share * (b_u - a_u), (a_u + b_u) / 2)
            t[i] = np.clip(u, -half[i], half[i])
    return polygon_length(middle + t[:, None] * along)


def silently(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)



class RawLoopTests(unittest.TestCase):
    def test_pixel_staircase_is_counterclockwise_with_physical_size(self):
        chain, geometry = region_chain(pad(np.ones((1, 1))), spacing=(2.0, 3.0))
        loops = raw_loops(chain, geometry)
        self.assertEqual((loops.kind, len(loops)), ("staircase", 1))
        self.assertEqual(polygon_area(loops.loops[0], signed=True), 6.0)
        self.assertEqual(loops_length(loops), 10.0)
        self.assertEqual(loops_length(loops), cubical_chain_volume(chain, geometry))

    def test_vertex_chain_has_simplicial_orientation_and_matches_forest_reps(self):
        forest = PersistenceForest.from_cubical_grid(np.where(pad(disk(4)), 1.0, 0.0), storage="cycles")
        bar = max(forest.barcode, key=lambda b: b.death - b.birth)
        rep = forest.cycle_for_bar_at(bar, bar.birth)
        vertex_chain, points = cubical_chain_to_vertex_chain(rep, forest.geometry)
        self.assertEqual(len(vertex_chain.signed_simplices), len(rep.signed_simplices))
        self.assertTrue(all(i < j for (i, j), _ in vertex_chain.signed_simplices))
        self.assertEqual(signed_chain_area(vertex_chain, points), disk(4).sum())
        self.assertEqual(loops_area(raw_loops(rep, forest.geometry)), disk(4).sum())

    def test_holes_subtract_and_raw_keeps_doubled_edges(self):
        ring = np.ones((5, 5), bool)
        ring[2, 2] = False
        chain, geometry = region_chain(pad(ring))
        loops = raw_loops(chain, geometry)
        self.assertEqual(sorted(polygon_area(loop, signed=True) for loop in loops), [-1.0, 25.0])
        self.assertEqual(loops_area(loops), 24.0)
        # A tendril: grid edge (3, 6) hangs below the outer boundary vertex (4, 6).
        tendril = SignedChain(set(chain.signed_simplices) | {((3, 6), 1), ((3, 6), -1)})
        self.assertEqual(loops_length(raw_loops(tendril, geometry)), loops_length(loops) + 2)
        self.assertEqual(loops_area(raw_loops(tendril, geometry)), 24.0)
        for build in (midpoint_loops, taut_loops, averaged_loops):
            with self.subTest(build=build.__name__):
                self.assertEqual(loops_length(build(tendril, geometry)), loops_length(build(chain, geometry)))

    def test_rejections(self):
        chain, geometry = region_chain(pad(np.ones((1, 1))))
        for sigma in (0.0, -1.0, math.inf):
            with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, "positive"):
                smoothed_loops(chain, geometry, sigma)
        for passes, error in ((0, ValueError), (1.5, TypeError), (True, TypeError)):
            with self.subTest(passes=passes), self.assertRaises(error):
                averaged_loops(chain, geometry, passes)
        with self.assertRaisesRegex(ValueError, "2D grids"):
            raw_loops(SignedChain({((1, 2, 2), 1)}), make_cubical_geometry((2, 2, 2), None, None, "top_cells"))
        with self.assertRaisesRegex(ValueError, "Expected edges"):
            raw_loops(SignedChain({((1, 1), 1)}), geometry)
        points = np.random.default_rng(0).random((10, 2))
        for build in (midpoint_loops, taut_loops):
            with self.subTest(build=build.__name__), self.assertRaisesRegex(TypeError, "needs the CubicalGeometry"):
                build(SignedChain({((0, 1), 1)}), points)

    def test_zero_chain_gives_no_loops_and_zero_measurements(self):
        _, geometry = region_chain(pad(np.ones((1, 1))))
        builds = (raw_loops, midpoint_loops, taut_loops, averaged_loops,
                  lambda chain, geometry: smoothed_loops(chain, geometry, 1.0))
        for build in builds:
            loops = build(SignedChain(set()), geometry)
            self.assertEqual(len(loops), 0)
            for functional in (loops_length, loops_area, loops_circularity, loops_excess_curvature,
                               loops_convex_hull_area_deficit, loops_max_convexity_defect_depth):
                self.assertEqual(functional(loops), 0.0)


class MidpointLoopTests(unittest.TestCase):
    def test_pixel_becomes_a_diamond(self):
        chain, geometry = region_chain(pad(np.ones((1, 1))), spacing=(2.0, 3.0))
        loops = midpoint_loops(chain, geometry)
        self.assertEqual(loops.kind, "midpoint")
        self.assertAlmostEqual(loops_length(loops), 2 * math.hypot(2.0, 3.0))
        self.assertAlmostEqual(loops_area(loops), 3.0)

    def test_each_loop_loses_half_a_pixel(self):
        for mask, spacing in ((disk(7), (1.0, 1.0)), (ellipse(12, 5, 0.4), (0.5, 2.0))):
            chain, geometry = region_chain(pad(mask), spacing)
            expected = mask.sum() * spacing[0] * spacing[1] - spacing[0] * spacing[1] / 2
            self.assertAlmostEqual(loops_area(midpoint_loops(chain, geometry)), expected)
        ring = np.ones((5, 5), bool)
        ring[2, 2] = False  # the hole shrinks as well, so the region keeps its area
        chain, geometry = region_chain(pad(ring))
        self.assertAlmostEqual(loops_area(midpoint_loops(chain, geometry)), 24.0)


class SmoothedLoopTests(unittest.TestCase):
    def test_area_is_kept_and_a_pixel_becomes_a_circle(self):
        chain, geometry = region_chain(pad(np.ones((1, 1))), spacing=(2.0, 2.0))
        for sigma in (0.5, 3.0, 1e3):
            loops = smoothed_loops(chain, geometry, sigma)
            self.assertEqual((loops.kind, dict(loops.params)), ("smoothed", {"sigma": sigma}))
            self.assertAlmostEqual(loops_area(loops), 4.0)
            # Smoothing well beyond the loop length leaves the circle of equal area.
            circle = 2 * math.sqrt(4 * math.pi)
            self.assertLess(loops_length(loops) / circle - 1, 0.01 if sigma < 1 else 1e-3)
        ring = np.ones((9, 9), bool)
        ring[3:5, 3:6] = False
        chain, geometry = region_chain(pad(ring))
        self.assertAlmostEqual(loops_area(smoothed_loops(chain, geometry, 1.0)), 75.0)

    def test_disks_are_round_and_flat_at_two_pixels(self):
        for radius in (3, 8, 20):
            chain, geometry = region_chain(pad(disk(radius)), (0.5, 0.5))
            loops = smoothed_loops(chain, geometry, 1.0)
            with self.subTest(radius=radius):
                self.assertGreater(loops_circularity(loops), 0.99)
                self.assertLess(loops_excess_curvature(loops), 0.1)
                self.assertLess(silently(loops_circularity, raw_loops(chain, geometry)), 0.65)

    def test_rotation_and_anisotropic_sampling_agree(self):
        circularity = []
        for angle in np.linspace(0, math.pi / 2, 5):
            chain, geometry = region_chain(pad(ellipse(20, 10, angle)))
            circularity.append(loops_circularity(smoothed_loops(chain, geometry, 2.0)))
        x, y = np.mgrid[:60, :120] + 0.5
        chain, geometry = region_chain(pad(((x - 30.2) / 20) ** 2 + ((y / 2 - 30.1) / 10) ** 2 <= 1), (1.0, 0.5))
        circularity.append(loops_circularity(smoothed_loops(chain, geometry, 2.0)))
        true_perimeter = math.pi * (3 * 30 - math.sqrt(70 * 50))
        self.assertLess(np.ptp(circularity), 0.02)
        self.assertLess(abs(np.mean(circularity) - 4 * math.pi * 200 * math.pi / true_perimeter**2), 0.02)

    def test_result_does_not_depend_on_chain_construction(self):
        chain, geometry = region_chain(pad(ellipse(9, 4, 0.7, size=30)))
        reordered = SignedChain(set(sorted(chain.signed_simplices, reverse=True)))
        for build in (lambda c: smoothed_loops(c, geometry, 1.5), lambda c: taut_loops(c, geometry)):
            first, second = build(chain), build(reordered)
            self.assertAlmostEqual(loops_length(first), loops_length(second), places=12)
            self.assertAlmostEqual(loops_area(first), loops_area(second), places=12)

    def test_tendrils_are_dropped_and_vertices_filtrations_work(self):
        forest = PersistenceForest.from_cubical_grid(
            np.random.default_rng(1).random((25, 25)), filtration_type="vertices", storage="cycles")
        with_tendrils = 0
        builds = (midpoint_loops, taut_loops, averaged_loops, lambda c, g: smoothed_loops(c, g, 1.0))
        for bar in forest.barcode:
            for rep in forest.iter_bar_cycle_reps(bar):
                with_tendrils += bool(rep.only_double_simplices().signed_simplices)
                for build in builds:
                    signed = build(rep, forest.geometry)
                    self.assertEqual(loops_length(signed), loops_length(build(rep.unsigned(), forest.geometry)))
                    if len(signed) and signed.kind != "taut":
                        self.assertTrue(0 < loops_circularity(signed) <= 1 + 1e-12)
        self.assertGreater(with_tendrils, 0)

    def test_simplicial_cycles_can_be_smoothed_and_averaged(self):
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        radii = 1 + 0.08 * np.random.default_rng(2).standard_normal(40)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        chain = SignedChain({((min(i, (i + 1) % 40), max(i, (i + 1) % 40)), 1 if i < (i + 1) % 40 else -1)
                             for i in range(40)})
        raw = raw_loops(chain, points)
        self.assertEqual(raw.kind, "polygon")
        self.assertAlmostEqual(loops_area(raw), signed_chain_area(chain, points))
        for loops in (smoothed_loops(chain, points, 0.2), averaged_loops(chain, points, 2)):
            with self.subTest(kind=loops.kind):
                self.assertAlmostEqual(loops_area(loops), loops_area(raw))
                self.assertGreater(loops_circularity(loops), loops_circularity(raw))


class AveragedLoopTests(unittest.TestCase):
    def test_area_is_kept_and_curve_stays_close(self):
        for passes, reach in ((1, 0.2), (4, 0.35)):
            for mask in (disk(16), ellipse(30, 15, math.radians(30), size=70)):
                chain, geometry = region_chain(pad(mask))
                loops = averaged_loops(chain, geometry, passes)
                [staircase] = raw_loops(chain, geometry)
                with self.subTest(passes=passes):
                    self.assertEqual((loops.kind, dict(loops.params)), ("averaged", {"passes": passes}))
                    self.assertAlmostEqual(loops_area(loops), mask.sum())
                    self.assertLess(distances_to_staircase(loops.loops[0], staircase).max(), reach)
        chain, geometry = region_chain(pad(disk(16)))
        circularity = [loops_circularity(averaged_loops(chain, geometry, passes)) for passes in (1, 4)]
        self.assertGreater(circularity[1], circularity[0])
        self.assertGreater(circularity[1], 0.99)

    def test_pixel_keeps_its_area_as_a_square(self):
        chain, geometry = region_chain(pad(np.ones((1, 1))), spacing=(2.0, 2.0))
        loops = averaged_loops(chain, geometry, 3)
        self.assertAlmostEqual(loops_area(loops), 4.0)
        self.assertAlmostEqual(loops_circularity(loops), math.pi / 4)


class TautLoopTests(unittest.TestCase):
    def test_thin_regions_collapse_onto_pixel_centers(self):
        chain, geometry = region_chain(pad(np.ones((1, 1))), spacing=(2.0, 3.0))
        [point] = taut_loops(chain, geometry)
        np.testing.assert_array_equal(point, [[5.0, 7.5]])
        bar = taut_loops(*region_chain(pad(np.ones((1, 4)))))
        np.testing.assert_array_equal(bar.loops[0], [[2.5, 2.5], [2.5, 5.5]])
        for loops in (taut_loops(chain, geometry), bar):
            self.assertEqual(loops.kind, "taut")
            self.assertEqual(loops_area(loops), 0.0)
            self.assertEqual(loops_circularity(loops), 0.0)
            self.assertEqual(loops_excess_curvature(loops), 0.0)
        self.assertEqual(loops_length(bar), 6.0)

    def test_convex_regions_give_convex_polygons_on_pixel_centers(self):
        for mask in (disk(12), ellipse(20, 9, 0.5), np.ones((6, 9), bool)):
            chain, geometry = region_chain(pad(mask), (0.5, 2.0))
            loops = taut_loops(chain, geometry)
            [loop] = loops
            doubled = 2 * loop / np.array([0.5, 2.0])
            np.testing.assert_array_equal(doubled % 2, 1)  # pixel centers
            self.assertGreater(polygon_area(loop, signed=True), 0)
            self.assertAlmostEqual(loops_excess_curvature(loops), 0.0, places=9)
            self.assertAlmostEqual(loops_convex_hull_area_deficit(loops), 0.0, places=12)
        chain, geometry = region_chain(pad(disk(30)))
        self.assertGreater(loops_circularity(taut_loops(chain, geometry)), 0.99)

    def test_holes_stay_clockwise_and_inside(self):
        ring = disk(10)
        ring[disk(4, size=ring.shape[0])] = False
        chain, geometry = region_chain(pad(ring))
        loops = taut_loops(chain, geometry)
        areas = sorted(polygon_area(loop, signed=True) for loop in loops)
        self.assertEqual(len(areas), 2)
        self.assertLess(areas[0], 0)
        self.assertGreater(areas[1], -areas[0])
        self.assertAlmostEqual(loops_area(loops), ring.sum(), delta=0.1 * ring.sum())

    def test_matches_a_convex_solver_on_random_regions(self):
        rng = np.random.default_rng(8)
        checked = 0
        for _ in range(12):
            size = rng.integers(3, 7)
            chain, geometry = region_chain(pad(rng.random((size, size)) > 0.4))
            if not chain.signed_simplices:
                continue
            for staircase, taut in zip(raw_loops(chain, geometry), taut_loops(chain, geometry)):
                doubled = np.rint(2 * staircase).astype(np.int64)
                expected = shortest_crossing_length(doubled, sweeps=3000) / 2
                length = polygon_length(taut) if len(taut) > 1 else 0.0
                self.assertAlmostEqual(length, expected, places=5)
                checked += 1
        self.assertGreater(checked, 15)

    def test_anisotropic_spacing_scales_the_unit_result(self):
        mask = pad(ellipse(11, 5, 0.9, size=30))
        unit = taut_loops(*region_chain(mask))
        scaled = taut_loops(*region_chain(mask, (0.5, 3.0)))
        np.testing.assert_allclose(scaled.loops[0], unit.loops[0] * [0.5, 3.0])


class LoopFunctionalTests(unittest.TestCase):
    def test_area_sums_signed_loops(self):
        square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        hole = 0.5 + square[::-1] / 4
        self.assertEqual(loops_area(PlanarLoops((square, hole), "polygon")), 3.75)
        # One boundary split into two counterclockwise loops at a shared vertex.
        touching = PlanarLoops((square, square + 2.0), "polygon")
        self.assertEqual(loops_area(touching), 8.0)

    def test_curvature_warns_only_for_grid_loops(self):
        chain, geometry = region_chain(pad(disk(5)))
        cases = ((raw_loops, True), (midpoint_loops, True), (taut_loops, False), (averaged_loops, False),
                 (lambda c, g: smoothed_loops(c, g, 1.0), False))
        for build, warns in cases:
            loops = build(chain, geometry)
            with self.subTest(kind=loops.kind), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loops_excess_curvature(loops)
            self.assertEqual(any("taut or smoothed" in str(w.message) for w in caught), warns)

    def test_simplicial_wrappers_measure_polygon_loops(self):
        forest = PersistenceForest.from_points(np.random.default_rng(4).random((80, 2)), storage="cycles")
        points = forest.geometry
        for bar in forest.barcode:
            rep = forest.cycle_for_bar_at(bar, bar.birth)
            loops = raw_loops(rep, points)
            self.assertEqual(shape.signed_chain_circularity(rep, points), loops_circularity(loops))
            self.assertEqual(signed_chain_area(rep, points), loops_area(loops))


class OnLoopsTests(unittest.TestCase):
    def test_measurements_apply_functional_to_loops(self):
        chain, geometry = region_chain(pad(disk(6)))
        cases = (
            (on_raw_loops(loops_circularity), raw_loops(chain, geometry), "loops_circularity[raw]", {"curve": "raw"}),
            (on_midpoint_loops(loops_circularity), midpoint_loops(chain, geometry),
             "loops_circularity[midpoint]", {"curve": "midpoint"}),
            (on_taut_loops(loops_circularity), taut_loops(chain, geometry), "loops_circularity[taut]", {"curve": "taut"}),
            (on_averaged_loops(loops_circularity), averaged_loops(chain, geometry, 4),
             "loops_circularity[averaged, passes=4]", {"curve": "averaged", "passes": 4}),
        )
        for measurement, loops, name, params in cases:
            with self.subTest(name=name):
                self.assertEqual(measurement(chain, geometry), loops_circularity(loops))
                self.assertEqual(measurement.measurement_name, name)
                self.assertEqual(measurement.__name__, name)
                self.assertEqual(dict(measurement.measurement_params), params)
        measurement = on_smoothed_loops(loops_max_convexity_defect_depth, sigma=2.0, normalize=False)
        self.assertEqual(measurement(chain, geometry),
                         loops_max_convexity_defect_depth(smoothed_loops(chain, geometry, 2.0), normalize=False))
        self.assertEqual(measurement.measurement_name,
                         "loops_max_convexity_defect_depth[smoothed, sigma=2.0, normalize=False]")
        self.assertEqual(dict(measurement.measurement_params),
                         {"curve": "smoothed", "sigma": 2.0, "normalize": False})

    def test_rejections_happen_when_wrapping(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            on_smoothed_loops(loops_circularity, sigma=0.0)
        with self.assertRaises(ValueError):
            on_averaged_loops(loops_circularity, passes=0)
        with self.assertRaises(TypeError):
            on_raw_loops("loops_circularity")

    def test_profiles_record_the_measurement_description(self):
        values = np.where(pad(disk(5)), 1.0, 0.0)
        forest = PersistenceForest.from_cubical_grid(values, spacing=0.5)
        measurement = on_smoothed_loops(loops_circularity, sigma=1.0)
        profiles = forest.compute_measurement_profiles(measurement, "circularity")
        [values_at] = profiles.evaluate_on_grid(np.array([0.5]))
        self.assertGreater(values_at[0], 0.99)
        self.assertEqual(profiles.measurement_name, "loops_circularity[smoothed, sigma=1.0]")
        self.assertEqual(dict(profiles.measurement_params), {"curve": "smoothed", "sigma": 1.0})
        explicit = forest.compute_measurement_profiles(measurement, "other", measurement_name="mine",
                                                       measurement_params={"a": 1}, cache=False)
        self.assertEqual((explicit.measurement_name, dict(explicit.measurement_params)), ("mine", {"a": 1}))
        plain = forest.compute_measurement_profiles(lambda chain, geometry: 1.0, "plain", cache=False)
        self.assertEqual((plain.measurement_name, dict(plain.measurement_params)), ("plain", {}))


class CubicalTopologyTests(unittest.TestCase):
    def test_counterparts_match_simplicial_functionals_on_the_vertex_chain(self):
        forest = PersistenceForest.from_cubical_grid(
            np.random.default_rng(5).random((20, 20)), filtration_type="vertices", storage="cycles")
        pairs = [(name, getattr(cubical_measurements, "cubical_chain_" + name.removeprefix("signed_chain_")))
                 for name in ("signed_chain_connected_components", "signed_chain_excess_connected_components",
                              "signed_chain_connected_components_only_signed_simplices",
                              "signed_chain_num_of_branching_points",
                              "signed_chain_num_of_branching_points_only_signed_simplices",
                              "signed_chain_avg_tendril_length", "signed_chain_tendril_branching_ratio")]
        tendril_lengths = []
        for bar in forest.barcode:
            for rep in forest.iter_bar_cycle_reps(bar):
                vertex_chain, points = cubical_chain_to_vertex_chain(rep, forest.geometry)
                for name, cubical in pairs:
                    simplicial = getattr(shape, name, None) or getattr(cubical_measurements, name)
                    self.assertEqual(cubical(rep, forest.geometry), simplicial(vertex_chain, points))
                tendril_lengths.append(cubical_measurements.cubical_chain_avg_tendril_length(rep, forest.geometry))
        self.assertGreater(max(tendril_lengths), 0)

    def test_simplicial_topology_functionals_reject_grids(self):
        chain, geometry = region_chain(pad(disk(3)))
        functions = (signed_chain_connected_components, shape.signed_chain_num_of_branching_points,
                     shape.signed_chain_avg_tendril_length, shape.signed_chain_tendril_branching_ratio)
        for function in functions:
            with self.subTest(function=function.__name__), self.assertRaisesRegex(TypeError, "cubical"):
                function(chain, geometry)
        for function in (shape.signed_chain_circularity, signed_chain_area):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(TypeError, "on_raw_loops"):
                function(chain, geometry)


class LoopPlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_loops_are_drawn_closed_in_display_frame(self):
        ring = np.ones((5, 7), bool)
        ring[2, 2:4] = False
        chain, geometry = region_chain(pad(ring), (1.0, 2.0))
        loops = midpoint_loops(chain, geometry)
        for image_orientation in (False, True):
            ax = plot_planar_loops(loops, show=False, label="midpoint", show_orientation=True,
                                   image_orientation=image_orientation)
            self.assertEqual(len(ax.lines), 2)
            self.assertEqual(len({line.get_color() for line in ax.lines}), 1)
            self.assertEqual([line.get_label() for line in ax.lines][0], "midpoint")
            data = ax.lines[0].get_xydata()
            np.testing.assert_array_equal(data[0], data[-1])
            expected = loops.loops[0][0][::-1] if image_orientation else loops.loops[0][0]
            np.testing.assert_array_equal(data[0], expected)
            self.assertEqual(ax.yaxis_inverted(), image_orientation)

    def test_loops_overlay_forest_plots(self):
        forest = PersistenceForest.from_cubical_grid(np.where(pad(disk(4)), 1.0, 0.0), storage="cycles")
        bar = max(forest.barcode, key=lambda b: b.death - b.birth)
        rep = forest.cycle_for_bar_at(bar, bar.birth)
        ax = forest.plot_at_filtration(bar.birth, show=False, show_cycles=False)
        limits = ax.get_xlim(), ax.get_ylim()
        plot_planar_loops(taut_loops(rep, forest.geometry), ax=ax, show=False, color="red")
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        with self.assertRaises(TypeError):
            plot_planar_loops(rep, show=False)


if __name__ == "__main__":
    unittest.main()
