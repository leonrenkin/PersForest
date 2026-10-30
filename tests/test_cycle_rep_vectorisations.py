import unittest
from unittest.mock import patch

import numpy as np

from persforest.chains import SignedChain
from persforest.measurements.shape import (
    signed_chain_circularity,
    signed_chain_circularity_complement,
    signed_chain_circularity_complement_squared,
    signed_chain_circularity_squared_complement,
    signed_chain_convex_hull_area_deficit,
    signed_chain_convex_hull_perimeter_deficit,
    signed_chain_max_convexity_defect_depth,
)


def closed_path_chain(vertex_indices: list[int]) -> SignedChain:
    """Build an oriented signed chain following the given closed path."""
    signed_edges = set()
    for start, end in zip(vertex_indices, vertex_indices[1:] + vertex_indices[:1]):
        if start < end:
            signed_edges.add(((start, end), 1))
        else:
            signed_edges.add(((end, start), -1))
    return SignedChain(signed_edges)


class CircularityTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.point_cloud = np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
        )
        self.cycle = SignedChain(
            {
                ((0, 1), 1),
                ((1, 2), 1),
                ((2, 3), 1),
                ((0, 3), -1),
            }
        )

    def test_circularity_transforms(self) -> None:
        circularity = signed_chain_circularity(self.cycle, self.point_cloud)
        complement = signed_chain_circularity_complement(
            self.cycle, self.point_cloud
        )

        self.assertAlmostEqual(
            signed_chain_circularity_squared_complement(
                self.cycle, self.point_cloud
            ),
            1 - circularity**2,
        )
        self.assertAlmostEqual(
            signed_chain_circularity_complement_squared(
                self.cycle, self.point_cloud
            ),
            complement**2,
        )

    def test_transforms_are_scale_invariant(self) -> None:
        scaled_point_cloud = 10.0 * self.point_cloud

        self.assertAlmostEqual(
            signed_chain_circularity_squared_complement(
                self.cycle, scaled_point_cloud
            ),
            signed_chain_circularity_squared_complement(
                self.cycle, self.point_cloud
            ),
        )
        self.assertAlmostEqual(
            signed_chain_circularity_complement_squared(
                self.cycle, scaled_point_cloud
            ),
            signed_chain_circularity_complement_squared(
                self.cycle, self.point_cloud
            ),
        )


class ConvexityDeficitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notched_square = np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [3.0, 4.0],
                [3.0, 1.0],
                [1.0, 1.0],
                [1.0, 4.0],
                [0.0, 4.0],
            ]
        )
        self.notched_cycle = closed_path_chain(list(range(len(self.notched_square))))

    def test_convex_cycle_has_zero_deficits(self) -> None:
        point_cloud = np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
        )
        cycle = closed_path_chain([0, 1, 2, 3])

        self.assertAlmostEqual(
            signed_chain_convex_hull_area_deficit(cycle, point_cloud),
            0.0,
        )
        self.assertAlmostEqual(
            signed_chain_convex_hull_perimeter_deficit(cycle, point_cloud),
            0.0,
        )
        self.assertAlmostEqual(
            signed_chain_max_convexity_defect_depth(cycle, point_cloud),
            0.0,
        )

    def test_rotated_convex_cycle_has_exactly_zero_deficits(self) -> None:
        point_cloud = np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]
        )
        cycle = closed_path_chain([0, 1, 2, 3])
        angle = 0.06911503837897545
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        transformed = 7.0 * point_cloud @ rotation.T + np.array([2.0, -5.0])

        self.assertEqual(
            signed_chain_convex_hull_area_deficit(cycle, transformed),
            0.0,
        )
        self.assertEqual(
            signed_chain_convex_hull_perimeter_deficit(cycle, transformed),
            0.0,
        )

    def test_notch_has_expected_measurements(self) -> None:
        self.assertAlmostEqual(
            signed_chain_convex_hull_area_deficit(
                self.notched_cycle,
                self.notched_square,
            ),
            6.0 / 16.0,
        )
        self.assertAlmostEqual(
            signed_chain_convex_hull_perimeter_deficit(
                self.notched_cycle,
                self.notched_square,
            ),
            1.0 - 16.0 / 22.0,
        )
        self.assertAlmostEqual(
            signed_chain_max_convexity_defect_depth(
                self.notched_cycle,
                self.notched_square,
                normalize=False,
            ),
            3.0,
        )
        self.assertAlmostEqual(
            signed_chain_max_convexity_defect_depth(
                self.notched_cycle,
                self.notched_square,
            ),
            3.0 / np.sqrt(32.0),
        )

    def test_measurements_are_rigid_motion_and_scale_invariant(self) -> None:
        angle = 0.37
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        scale = 7.0
        transformed = scale * self.notched_square @ rotation.T + np.array([2.0, -5.0])

        self.assertAlmostEqual(
            signed_chain_convex_hull_area_deficit(
                self.notched_cycle,
                transformed,
            ),
            signed_chain_convex_hull_area_deficit(
                self.notched_cycle,
                self.notched_square,
            ),
        )
        self.assertAlmostEqual(
            signed_chain_convex_hull_perimeter_deficit(
                self.notched_cycle,
                transformed,
            ),
            signed_chain_convex_hull_perimeter_deficit(
                self.notched_cycle,
                self.notched_square,
            ),
        )
        self.assertAlmostEqual(
            signed_chain_max_convexity_defect_depth(
                self.notched_cycle,
                transformed,
            ),
            signed_chain_max_convexity_defect_depth(
                self.notched_cycle,
                self.notched_square,
            ),
        )
        self.assertAlmostEqual(
            signed_chain_max_convexity_defect_depth(
                self.notched_cycle,
                transformed,
                normalize=False,
            ),
            scale
            * signed_chain_max_convexity_defect_depth(
                self.notched_cycle,
                self.notched_square,
                normalize=False,
            ),
        )

    def test_defect_depth_is_independent_of_path_orientation(self) -> None:
        reversed_cycle = closed_path_chain(
            list(reversed(range(len(self.notched_square))))
        )

        self.assertAlmostEqual(
            signed_chain_max_convexity_defect_depth(
                reversed_cycle,
                self.notched_square,
                normalize=False,
            ),
            3.0,
        )

    def test_multiple_convex_paths_have_no_pathwise_depth_defect(self) -> None:
        point_cloud = np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [0.0, 4.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [1.5, 2.0],
            ]
        )
        outer_cycle = closed_path_chain([0, 1, 2, 3])
        inner_cycle = closed_path_chain([4, 6, 5])  # clockwise: a hole
        cycle = SignedChain(outer_cycle.signed_simplices | inner_cycle.signed_simplices)

        self.assertAlmostEqual(
            signed_chain_convex_hull_area_deficit(cycle, point_cloud),
            0.5 / 16.0,
        )
        self.assertAlmostEqual(
            signed_chain_convex_hull_perimeter_deficit(cycle, point_cloud),
            1.0 - 16.0 / (17.0 + np.sqrt(5.0)),
        )
        self.assertAlmostEqual(
            signed_chain_max_convexity_defect_depth(cycle, point_cloud),
            0.0,
        )

    def test_zero_area_collinear_cycle_is_treated_as_convex(self) -> None:
        point_cloud = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        cycle = SignedChain(
            {
                ((0, 1), 1),
                ((0, 1), -1),
                ((1, 2), 1),
                ((1, 2), -1),
            }
        )

        self.assertEqual(
            signed_chain_convex_hull_area_deficit(cycle, point_cloud),
            0.0,
        )
        self.assertEqual(
            signed_chain_convex_hull_perimeter_deficit(cycle, point_cloud),
            0.0,
        )
        self.assertEqual(
            signed_chain_max_convexity_defect_depth(cycle, point_cloud),
            0.0,
        )

    def test_empty_cycle_is_treated_as_convex(self) -> None:
        point_cloud = np.empty((0, 2))
        cycle = SignedChain(set())

        self.assertEqual(
            signed_chain_convex_hull_area_deficit(cycle, point_cloud),
            0.0,
        )
        self.assertEqual(
            signed_chain_convex_hull_perimeter_deficit(cycle, point_cloud),
            0.0,
        )
        self.assertEqual(
            signed_chain_max_convexity_defect_depth(cycle, point_cloud),
            0.0,
        )

    def test_non_planar_point_cloud_is_rejected(self) -> None:
        point_cloud = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        cycle = closed_path_chain([0, 1, 2])

        for functional in (
            signed_chain_convex_hull_area_deficit,
            signed_chain_convex_hull_perimeter_deficit,
            signed_chain_max_convexity_defect_depth,
        ):
            with self.assertRaisesRegex(ValueError, "shape \\(n_points, 2\\)"):
                functional(cycle, point_cloud)

    def test_out_of_range_deficit_raises_instead_of_being_clamped(self) -> None:
        point_cloud = np.array(
            [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        )
        cycle = closed_path_chain([0, 1, 2, 3])

        with patch(
            "persforest.measurements.loops.loops_area",
            return_value=17.0,
        ):
            with self.assertRaisesRegex(
                ValueError, "area deficit must lie in \\[0, 1\\]"
            ):
                signed_chain_convex_hull_area_deficit(cycle, point_cloud)

    def test_roundoff_sized_deficit_errors_are_clamped(self) -> None:
        point_cloud = np.array(
            [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        )
        cycle = closed_path_chain([0, 1, 2, 3])
        roundoff = 8 * np.finfo(np.float64).eps

        with patch(
            "persforest.measurements.loops.loops_area",
            return_value=16.0 * (1.0 + roundoff),
        ):
            self.assertEqual(
                signed_chain_convex_hull_area_deficit(cycle, point_cloud),
                0.0,
            )

        with patch(
            "persforest.measurements.loops.loops_area",
            return_value=-16.0 * roundoff,
        ):
            self.assertEqual(
                signed_chain_convex_hull_area_deficit(cycle, point_cloud),
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
