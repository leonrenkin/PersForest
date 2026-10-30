from dataclasses import dataclass
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from persforest import PersistenceForest
from persforest.plotting._geometry import (
    _activity_face_records,
    _validated_activity_tetrahedra,
)
from persforest.plotting._scene import _activity_rows_for_plotting
from persforest.plotting.interior_mpl import (
    plot_interior_simplex_activity,
    plot_interior_simplex_activity_gradient,
)
from persforest.plotting.interior_plotly import (
    plot_interior_simplex_activity_plotly,
)


@dataclass(frozen=True)
class FakeBar:
    birth: float
    death: float

    def lifespan(self) -> float:
        return self.death - self.birth


class FakeForest3D:
    dim = 3
    point_cloud = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    filtration = [
        ((0, 1, 2), 0.1),
        ((0, 1, 2, 3), 0.2),
        ((1, 2, 3, 4), 0.4),
    ]
    long_bar = FakeBar(0.0, 2.0)
    short_bar = FakeBar(0.25, 1.0)

    def interior_simplex_activity(self):
        return {
            (0, 1, 2, 3): [
                (self.long_bar, 0.0, 1.0),
                (self.short_bar, 0.0, 0.25),
            ],
            (1, 2, 3, 4): [(self.long_bar, 0.0, 0.5)],
        }

    def _get_color_map(self, coloring):
        if coloring not in ("forest", "bars"):
            raise ValueError
        return {
            self.long_bar: "#1f77b4",
            self.short_bar: "#d62728",
        }


class FakeForest2D:
    dim = 2
    point_cloud = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    filtration = [
        ((0, 1), 0.1),
        ((0, 2), 0.1),
        ((1, 2), 0.2),
    ]
    bar = FakeBar(0.0, 1.0)

    def interior_simplex_activity(self):
        return {(0, 1, 2): [(self.bar, 0.0, 1.0)]}

    def _get_color_map(self, coloring):
        return {self.bar: "#1f77b4"}


class InteriorActivityGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.forest = FakeForest3D()

    def tearDown(self) -> None:
        plt.close("all")

    def test_longest_overlap_selects_one_bar_per_tetrahedron(self) -> None:
        rows = _activity_rows_for_plotting(
            self.forest,
            overlap="longest",
            min_activity_length=0.0,
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row[1] is self.forest.long_bar for row in rows))

    def test_cell_and_surface_geometry_have_expected_face_counts(self) -> None:
        rows = _activity_rows_for_plotting(
            self.forest,
            overlap="longest",
            min_activity_length=0.0,
        )
        rows = _validated_activity_tetrahedra(self.forest.point_cloud, rows)

        cells = _activity_face_records(
            self.forest.point_cloud,
            rows,
            geometry="cells",
            cell_shrink=1.0,
        )
        surface = _activity_face_records(
            self.forest.point_cloud,
            rows,
            geometry="surface",
            cell_shrink=1.0,
        )

        self.assertEqual(len(cells), 8)
        self.assertEqual(len(surface), 6)

    def test_layered_surface_retains_differently_colored_region_boundary(self) -> None:
        rows = _activity_rows_for_plotting(
            self.forest,
            overlap="layer",
            min_activity_length=0.0,
        )
        rows = _validated_activity_tetrahedra(self.forest.point_cloud, rows)
        surface = _activity_face_records(
            self.forest.point_cloud,
            rows,
            geometry="surface",
            cell_shrink=1.0,
        )

        self.assertEqual(len(surface), 10)

    def test_degenerate_tetrahedra_are_explicitly_skipped(self) -> None:
        rows = [((0, 0, 1, 2), self.forest.long_bar, 1.0)]

        with self.assertWarnsRegex(RuntimeWarning, "Skipped 1 degenerate"):
            valid = _validated_activity_tetrahedra(
                self.forest.point_cloud,
                rows,
            )

        self.assertEqual(valid, [])

    def test_degeneracy_check_is_scale_invariant(self) -> None:
        rows = [((0, 1, 2, 3), self.forest.long_bar, 1.0)]

        for scale in (1e-12, 1e12):
            valid = _validated_activity_tetrahedra(
                scale * self.forest.point_cloud,
                rows,
            )
            self.assertEqual(valid, rows)

    def test_static_3d_dispatch_returns_3d_axes_and_activity_mesh(self) -> None:
        axes = plot_interior_simplex_activity(
            self.forest,
            show=False,
            geometry="surface",
            style={"cell_shrink": 0.96},
        )

        self.assertTrue(hasattr(axes, "zaxis"))
        activity_meshes = [
            collection
            for collection in axes.collections
            if isinstance(collection, Poly3DCollection)
        ]
        self.assertEqual(len(activity_meshes), 1)

    def test_plotly_renderer_groups_activity_by_bar(self) -> None:
        figure = plot_interior_simplex_activity_plotly(
            self.forest,
            show=False,
            geometry="cells",
            cell_shrink=0.96,
        )

        self.assertEqual(
            [trace.type for trace in figure.data],
            ["mesh3d", "scatter3d"],
        )
        self.assertEqual(len(figure.data[0].i), 8)
        self.assertEqual(figure.data[0].legendgroup, "activity-bar-0")

    def test_plotly_defaults_to_clean_surface_geometry(self) -> None:
        figure = plot_interior_simplex_activity_plotly(
            self.forest,
            show=False,
        )

        self.assertEqual(
            [trace.type for trace in figure.data],
            ["mesh3d", "scatter3d"],
        )
        self.assertEqual(len(figure.data[0].i), 6)
        self.assertEqual(
            set(figure.data[0].facecolor),
            {
                "rgba(31, 119, 180, 0.65)",
                "rgba(31, 119, 180, 0.95)",
            },
        )

    def test_plotly_surface_edge_mode_uses_subdued_bar_colored_edges(self) -> None:
        figure = plot_interior_simplex_activity_plotly(
            self.forest,
            show=False,
            geometry="cells",
            cell_shrink=0.9,
            edge_mode="surface",
        )

        self.assertEqual(
            [trace.type for trace in figure.data],
            ["mesh3d", "scatter3d", "scatter3d"],
        )
        self.assertEqual(
            figure.data[1].line.color,
            "rgba(20, 77, 117, 0.35)",
        )
        mesh_x = {float(value) for value in figure.data[0].x}
        edge_x = {
            float(value)
            for value in figure.data[1].x
            if value is not None
        }
        self.assertTrue(edge_x.issubset(mesh_x))

    def test_plotly_legacy_activity_edge_style_remains_available(self) -> None:
        figure = plot_interior_simplex_activity_plotly(
            self.forest,
            show=False,
            geometry="cells",
            style={"show_activity_edges": True},
        )

        self.assertEqual(
            [trace.type for trace in figure.data],
            ["mesh3d", "scatter3d", "scatter3d"],
        )

    def test_plotly_rejects_unknown_edge_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "edge_mode"):
            plot_interior_simplex_activity_plotly(
                self.forest,
                show=False,
                edge_mode="visible",
            )


class InteriorActivityDispatchTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_2d_dispatch_keeps_existing_artist_structure(self) -> None:
        axes = plot_interior_simplex_activity(
            FakeForest2D(),
            show=False,
            show_complex=True,
        )

        self.assertEqual(
            [type(collection).__name__ for collection in axes.collections],
            ["PolyCollection", "LineCollection", "PathCollection"],
        )

    def test_gradient_dispatch_rejects_3d(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports 2D"):
            plot_interior_simplex_activity_gradient(FakeForest3D(), show=False)

    def test_persistence_forest_exposes_thin_public_wrappers(self) -> None:
        self.assertTrue(hasattr(PersistenceForest, "plot_interior_simplex_activity"))
        self.assertTrue(
            hasattr(PersistenceForest, "plot_interior_simplex_activity_gradient")
        )
        self.assertTrue(
            hasattr(PersistenceForest, "plot_interior_simplex_activity_plotly")
        )


if __name__ == "__main__":
    unittest.main()
