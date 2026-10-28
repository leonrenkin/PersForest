"""Cubical renderers: geometry, orientation, image background, 3D, Plotly, animation and gating."""
import importlib.util
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from persforest import PersistenceForest
from persforest.measurements.cubical import cubical_chain_volume
from persforest.plotting._cubical_scene import chain_face_quads
from persforest.plotting.animation.filtration import _animate_filtration_generic, animate_filtration_pair

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def centered(shape):
    """Grid of ones with one high center cell: a single bar [1, 3)."""
    values = np.ones(shape)
    values[tuple(n // 2 for n in shape)] = 3
    return values


def line_collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def poly_collections(ax):
    # 3D collections project their paths at draw time.
    ax.figure.canvas.draw()
    return [c for c in ax.collections if isinstance(c, Poly3DCollection)]


def signed_area(segments):
    """Shoelace sum over oriented segments; positive for counterclockwise cycles."""
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in segments) / 2


def signed_volume(quads):
    """Divergence-theorem volume enclosed by oriented quadrilaterals."""
    return sum(np.dot(q[0], np.cross(q[i], q[i + 1])) for q in quads for i in (1, 2)) / 6


class CubicalFiltrationPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_cycle_follows_the_center_pixel_at_physical_coordinates(self):
        for storage in ("diffs", "cycles"):
            with self.subTest(storage=storage):
                forest = PersistenceForest.from_cubical_grid(
                    centered((3, 3)), spacing=(2.0, 3.0), origin=(10.0, 20.0), storage=storage)
                ax = forest.plot_at_filtration(2, show=False)
                [collection] = line_collections(ax)
                points = np.concatenate(collection.get_segments())
                # Pixel (1, 1) spans [12, 14] along axis 0 and [23, 26] along axis 1.
                np.testing.assert_allclose(points.min(axis=0), [12, 23])
                np.testing.assert_allclose(points.max(axis=0), [14, 26])
                self.assertEqual(len(collection.get_segments()), 4)

    def test_signed_orientation_is_the_boundary_of_the_interior(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)), spacing=(2.0, 3.0), storage="cycles")
        [bar] = forest.barcode
        [(_, interior_sign)] = forest.cycle_for_bar_at(bar, 2).interior
        for image_orientation, reflection in ((False, 1), (True, -1)):
            with self.subTest(image_orientation=image_orientation):
                ax = forest.plot_at_filtration(2, show=False, signed=True, image_orientation=image_orientation)
                [collection] = line_collections(ax)
                self.assertAlmostEqual(signed_area(collection.get_segments()), reflection * interior_sign * 6.0)

    def test_image_orientation_draws_rows_downward(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 5)))
        physical = forest.plot_at_filtration(2, show=False)
        image = forest.plot_at_filtration(2, show=False, image_orientation=True)
        self.assertFalse(physical.yaxis_inverted())
        self.assertTrue(image.yaxis_inverted())
        np.testing.assert_allclose(
            np.concatenate(line_collections(image)[0].get_segments()),
            np.concatenate(line_collections(physical)[0].get_segments())[:, ::-1])
        self.assertEqual((image.get_xlabel(), image.get_ylabel()), ("axis 1", "axis 0"))

    def test_complex_tints_exactly_the_present_pixels(self):
        values = np.arange(12.0).reshape(3, 4)
        forest = PersistenceForest.from_cubical_grid(values)
        for image_orientation, expected in ((False, (values <= 5).T), (True, values <= 5)):
            with self.subTest(image_orientation=image_orientation):
                ax = forest.plot_at_filtration(5, show=False, show_cycles=False, show_image=False,
                                               image_orientation=image_orientation)
                [image] = ax.get_images()
                np.testing.assert_array_equal(image.get_array()[..., 3] > 0, expected)

    def test_vertex_grids_span_one_cell_less_per_axis(self):
        forest = PersistenceForest.from_cubical_grid(
            centered((5, 5)), filtration_type="vertices", spacing=0.5, origin=1.0)
        ax = forest.plot_at_filtration(2, show=False, show_image=False)
        [image] = ax.get_images()
        np.testing.assert_allclose(image.get_extent(), [1.0, 3.0, 1.0, 3.0])
        self.assertEqual(image.get_array().shape[:2], (4, 4))

    def test_cycle_reps_draw_one_collection_per_selected_bar(self):
        forest = PersistenceForest.from_cubical_grid(np.random.default_rng(3).random((10, 10)))
        threshold = sorted(bar.lifespan() for bar in forest.barcode)[-3]
        ax = forest.plot_barcode_cycle_reps(min_bar_length=threshold, show=False, show_orientation_arrows=True)
        self.assertEqual(len(line_collections(ax)), 3)


class CubicalImageBackgroundTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_2d_grids_show_the_veiled_image_by_default(self):
        values = np.arange(12.0).reshape(3, 4)
        forest = PersistenceForest.from_cubical_grid(values)
        ax = forest.plot_at_filtration(5, show=False)
        value_image, veil = ax.get_images()
        np.testing.assert_array_equal(value_image.get_array(), values.T)
        np.testing.assert_array_equal(veil.get_array()[..., 3] > 0, (values > 5).T)
        self.assertTrue(hasattr(ax, "_persforest_colorbar"))
        self.assertEqual(len(forest.plot_barcode_cycle_reps(show=False).get_images()), 1)
        three_d = PersistenceForest.from_cubical_grid(centered((3, 3, 3)))
        self.assertEqual(three_d.plot_at_filtration(2, show=False).get_images(), [])
        points = PersistenceForest.from_points(np.random.default_rng(0).random((20, 2)))
        points.plot_at_filtration(0.1, show=False, show_image=False)

    def test_image_shows_all_values_and_veils_absent_pixels(self):
        values = np.arange(12.0).reshape(3, 4)
        forest = PersistenceForest.from_cubical_grid(values)
        ax = forest.plot_at_filtration(5, show=False, show_cycles=False, show_image=True, image_orientation=True)
        value_image, veil = ax.get_images()
        self.assertLess(value_image.get_zorder(), veil.get_zorder())
        self.assertAlmostEqual(value_image.get_alpha(), 0.75)
        np.testing.assert_array_equal(value_image.get_array(), values)
        self.assertEqual(value_image.get_clim(), (0.0, 11.0))
        np.testing.assert_array_equal(veil.get_array()[..., 3] > 0, values > 5)

    def test_image_alone_and_vertex_pixels_centered_on_vertices(self):
        forest = PersistenceForest.from_cubical_grid(centered((5, 5)), filtration_type="vertices", spacing=0.5)
        ax = forest.plot_at_filtration(2, show=False, show_complex=False, show_image=True,
                                       style_2d={"image_cmap": "Greys", "image_alpha": 0.5})
        [image] = ax.get_images()
        np.testing.assert_allclose(image.get_extent(), [-0.25, 2.25, -0.25, 2.25])
        self.assertAlmostEqual(image.get_alpha(), 0.5)
        self.assertEqual(image.get_cmap().name, "Greys")
        self.assertEqual(len(line_collections(ax)), 1)

    def test_colorbar_shows_the_values_and_is_reused_on_redraw(self):
        forest = PersistenceForest.from_cubical_grid(np.arange(12.0).reshape(3, 4))
        ax = forest.plot_at_filtration(5, show=False, show_image=True)
        colorbar = ax._persforest_colorbar
        self.assertEqual(colorbar.ax.get_ylabel(), "value")
        self.assertEqual(colorbar.mappable.get_clim(), (0.0, 11.0))
        axes_count = len(ax.figure.axes)
        ax.clear()
        forest.plot_at_filtration(7, ax=ax, show=False, show_image=True)
        self.assertEqual(len(ax.figure.axes), axes_count)
        self.assertIs(ax._persforest_colorbar.mappable, ax.get_images()[0])
        ax = forest.plot_at_filtration(5, show=False, show_image=True,
                                       style_2d={"image_colorbar": False})
        self.assertEqual(len(ax.figure.axes), 1)

    def test_axes_end_at_the_grid(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 5)), spacing=(2.0, 1.0), origin=(1.0, 0.0))
        ax = forest.plot_at_filtration(2, show=False)
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), ((1.0, 7.0), (0.0, 5.0)))
        ax = forest.plot_at_filtration(2, show=False, image_orientation=True)
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), ((0.0, 5.0), (7.0, 1.0)))
        self.assertFalse(line_collections(ax)[0].get_clip_on())

    def test_cycle_reps_draw_cycles_above_the_image(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)))
        ax = forest.plot_barcode_cycle_reps(show=False, show_image=True)
        [image] = ax.get_images()
        [collection] = line_collections(ax)
        self.assertGreater(collection.get_zorder(), image.get_zorder())


class Cubical3DPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_complex_surface_and_cycle_faces(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3, 3)), spacing=(1.0, 2.0, 3.0), storage="cycles")
        ax = forest.plot_at_filtration(2, show=False, signed=True)
        complex_surface, cycle = poly_collections(ax)
        # The 26 present voxels expose their 54 outer faces and the 6 cavity faces.
        self.assertEqual(len(complex_surface.get_paths()), 60)
        [bar] = forest.barcode
        quads = chain_face_quads(forest.cycle_for_bar_at(bar, 2), forest.geometry)
        self.assertEqual(len(cycle.get_paths()), 6)
        np.testing.assert_allclose(quads.reshape(-1, 3).min(axis=0), [1, 2, 3])
        np.testing.assert_allclose(quads.reshape(-1, 3).max(axis=0), [2, 4, 6])
        # Consistently oriented faces enclose the voxel volume 1 * 2 * 3.
        self.assertAlmostEqual(abs(signed_volume(quads)), 6.0)
        self.assertEqual((ax.get_xlabel(), ax.get_zlabel()), ("axis 0", "axis 2"))

    def test_interior_activity_cells_and_surface(self):
        values = np.ones((4, 3, 3))
        values[1:3, 1, 1] = 3
        forest = PersistenceForest.from_cubical_grid(values)
        for geometry, faces in (("cells", 12), ("surface", 10)):
            with self.subTest(geometry=geometry):
                ax = forest.plot_interior_simplex_activity(show=False, geometry=geometry, dpi=50)
                [collection] = poly_collections(ax)
                self.assertEqual(len(collection.get_paths()), faces)

    def test_3d_specific_options_and_style(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3, 3)))
        ax = forest.plot_at_filtration(2, show=False, style_3d={"camera_eye": (10, 20), "remove_axes": True})
        self.assertAlmostEqual(ax.elev, 10)
        self.assertAlmostEqual(ax.azim, 20)
        with self.assertRaisesRegex(ValueError, "image_orientation"):
            forest.plot_at_filtration(2, show=False, image_orientation=True)
        with self.assertRaisesRegex(ValueError, "show_image"):
            forest.plot_at_filtration(2, show=False, show_image=True)


class CubicalInteriorActivityPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_only_the_enclosed_pixel_is_colored(self):
        for storage in ("diffs", "cycles"):
            for overlap in ("longest", "layer"):
                with self.subTest(storage=storage, overlap=overlap):
                    forest = PersistenceForest.from_cubical_grid(centered((3, 3)), storage=storage)
                    ax = forest.plot_interior_simplex_activity(show=False, overlap=overlap)
                    [image] = ax.get_images()
                    alpha = image.get_array()[..., 3]
                    expected = np.zeros((3, 3), dtype=bool)
                    expected[1, 1] = True
                    np.testing.assert_array_equal(alpha > 0, expected)

    def test_show_complex_adds_the_sublevel_pixels_beneath(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)))
        ax = forest.plot_interior_simplex_activity(show=False, show_complex=True, complex_max_filtration=1)
        complex_image, activity_image = ax.get_images()
        self.assertLess(complex_image.get_zorder(), activity_image.get_zorder())
        self.assertEqual(int((complex_image.get_array()[..., 3] > 0).sum()), 8)

    def test_gradient_is_strongest_at_the_enclosed_pixel(self):
        forest = PersistenceForest.from_cubical_grid(centered((5, 5)), spacing=2.0)
        ax = forest.plot_interior_simplex_activity_gradient(show=False, dpi=50, style={"resolution": 50})
        [image] = ax.get_images()
        np.testing.assert_allclose(image.get_extent(), [0, 10, 0, 10])
        rgb = image.get_array()[..., :3]
        self.assertEqual(rgb.shape, (5, 5, 3))
        whiteness = rgb.sum(axis=2)
        self.assertEqual(np.unravel_index(np.argmin(whiteness), whiteness.shape), (2, 2))
        self.assertLess(whiteness[2, 2], whiteness[0, 0])


@unittest.skipUnless(HAS_PLOTLY, "Plotly optional extra is not installed")
class CubicalPlotlyTests(unittest.TestCase):
    def test_2d_snapshot_matches_the_matplotlib_geometry(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 5)), spacing=(2.0, 1.0))
        for image_orientation in (False, True):
            with self.subTest(image_orientation=image_orientation):
                fig = forest.plot_at_filtration_plotly(2, show=False, show_image=True,
                                                       image_orientation=image_orientation)
                image, veil, cycle = fig.data
                self.assertEqual(np.asarray(image.z).shape, (3, 5) if image_orientation else (5, 3))
                self.assertEqual(int(np.isfinite(np.asarray(veil.z, dtype=float)).sum()), 1)
                points = np.array([(x, y) for x, y in zip(cycle.x, cycle.y) if x is not None and np.isfinite(x)])
                expected = [[2, 2], [4, 3]] if not image_orientation else [[2, 2], [3, 4]]
                np.testing.assert_allclose([points.min(axis=0), points.max(axis=0)], expected)
                self.assertEqual(fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1], image_orientation)

    def test_slider_starts_at_the_smallest_value_with_fixed_trace_slots(self):
        forest = PersistenceForest.from_cubical_grid(np.random.default_rng(0).random((8, 8)) - 2)
        fig = forest.plot_filtration_interactive(show=False, resolution=5)
        self.assertEqual(len(fig.frames), 6)
        self.assertAlmostEqual(float(fig.frames[0].name), float(forest.grid_values.min()))
        self.assertEqual(len({len(frame.data) for frame in fig.frames}), 1)

    def test_2d_slider_frames_move_only_lines(self):
        values = np.arange(12.0).reshape(3, 4)
        forest = PersistenceForest.from_cubical_grid(values)
        fig = forest.plot_filtration_interactive(show=False, resolution=4, filt_max=11, show_image=True)
        self.assertEqual(fig.data[0].type, "heatmap")
        self.assertTrue(fig.data[0].showscale)
        self.assertGreater(fig.layout.legend.x, fig.data[0].colorbar.x)
        for frame in fig.frames:
            self.assertEqual({trace.type for trace in frame.data}, {"scatter"})
            self.assertNotIn(0, frame.traces)
        self.assertFalse(fig.layout.sliders[0].steps[0].args[1]["frame"]["redraw"])
        # At t = 5.5 the pixels 0..5 form a staircase of 4 + 2 cells with 12 boundary edges.
        frame = next(frame for frame in fig.frames if abs(float(frame.name) - 5.5) < 1e-9)
        outline = frame.data[0]
        self.assertEqual(int(np.isnan(np.asarray(outline.x, dtype=float)).sum()), 12)
        fig3d = PersistenceForest.from_cubical_grid(centered((3, 3, 3))).plot_filtration_interactive(
            show=False, resolution=2)
        self.assertTrue(fig3d.layout.sliders[0].steps[0].args[1]["frame"]["redraw"])

    def test_2d_plotly_views_show_the_image_by_default(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 5)))
        self.assertEqual(forest.plot_at_filtration_plotly(2, show=False).data[0].type, "heatmap")
        self.assertEqual(forest.plot_filtration_interactive(show=False, resolution=2).data[0].type, "heatmap")
        three_d = PersistenceForest.from_cubical_grid(centered((3, 3, 3)))
        self.assertNotIn("heatmap", {trace.type for trace in three_d.plot_at_filtration_plotly(2, show=False).data})

    def test_3d_meshes_and_interior_activity(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3, 3)))
        complex_mesh, cycle_mesh = forest.plot_at_filtration_plotly(2, show=False).data
        self.assertEqual(len(complex_mesh.i), 120)
        self.assertEqual(len(cycle_mesh.i), 12)
        fig = forest.plot_interior_simplex_activity_plotly(show=False, edge_mode="surface", show_complex=True)
        mesh, edges, complex_edges = fig.data
        self.assertEqual(len(mesh.i), 12)
        self.assertEqual(edges.mode, "lines")
        self.assertEqual(complex_edges.name, "complex")
        with self.assertRaisesRegex(ValueError, "3D cubical"):
            PersistenceForest.from_cubical_grid(centered((3, 3))).plot_interior_simplex_activity_plotly(show=False)

    def test_point_cloud_options_are_rejected(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)))
        with self.assertRaisesRegex(ValueError, "fill_triangles"):
            forest.plot_at_filtration_plotly(2, show=False, fill_triangles=False)
        with self.assertRaisesRegex(ValueError, "vertex_size"):
            forest.plot_filtration_interactive(show=False, vertex_size=5)
        with self.assertRaisesRegex(ValueError, "point_color"):
            PersistenceForest.from_cubical_grid(centered((3, 3, 3))).plot_interior_simplex_activity_plotly(
                show=False, style={"point_color": "k"})


class CubicalAnimationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def draw(animation):
        animation._init_draw()
        animation._func(1)

    def test_filtration_animations_start_at_the_smallest_value(self):
        for shape in ((5, 5), (3, 3, 3)):
            with self.subTest(shape=shape):
                forest = PersistenceForest.from_cubical_grid(centered(shape) - 5)
                # The public 3D path saves an MP4 by default; render without saving.
                animation, fig = _animate_filtration_generic(
                    forest, filename=None, frames=3, with_barcode=True,
                    filtration_kwargs={"camera_mode": "orbit"} if len(shape) == 3 else None)
                animation._init_draw()
                cloud, barcode = fig.axes[:2]
                self.assertEqual(barcode.get_xlabel(), "Filtration value t")
                self.assertEqual(cloud.artists[-1].txt.get_text(), "$t = -4$")
                animation._func(2)

    def test_measurement_and_pair_animations(self):
        forest = PersistenceForest.from_cubical_grid(centered((5, 5)), storage="cycles")
        animation, fig = forest.animate_barcode_measurement(cubical_chain_volume, frames=3)
        self.draw(animation)
        self.assertEqual(fig.axes[0].get_title(), "Cubical filtration")
        animation, fig = animate_filtration_pair(forest, forest, frames=3)
        self.draw(animation)


class CubicalPlotGatingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def calls(forest):
        return {
            "plot_at_filtration": lambda **kw: forest.plot_at_filtration(0.5, show=False, **kw),
            "plot_barcode_cycle_reps": lambda **kw: forest.plot_barcode_cycle_reps(show=False, **kw),
            "plot_interior_simplex_activity": lambda **kw: forest.plot_interior_simplex_activity(show=False, **kw),
            "plot_interior_simplex_activity_gradient":
                lambda **kw: forest.plot_interior_simplex_activity_gradient(show=False, **kw),
        }

    def test_one_dimensional_grids_raise_value_error(self):
        forest = PersistenceForest.from_cubical_grid(np.random.default_rng(0).random(12))
        for name, call in self.calls(forest).items():
            with self.subTest(method=name), self.assertRaisesRegex(ValueError, "1D cubical"):
                call()

    def test_unsupported_axis_counts_raise_value_error(self):
        four_d = PersistenceForest.from_cubical_grid(centered((3, 3, 3, 3)))
        with self.assertRaisesRegex(ValueError, "2D and 3D cubical"):
            four_d.plot_at_filtration(2, show=False)
        three_d = PersistenceForest.from_cubical_grid(centered((3, 3, 3)))
        for name in ("plot_barcode_cycle_reps", "plot_interior_simplex_activity_gradient"):
            with self.subTest(method=name), self.assertRaisesRegex(ValueError, "2D cubical"):
                self.calls(three_d)[name]()

    def test_point_cloud_options_and_keys_are_rejected_for_grids(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)))
        for name, call in self.calls(forest).items():
            with self.subTest(method=name), self.assertRaisesRegex(ValueError, "vertex_size"):
                call(vertex_size=5)
        with self.assertRaisesRegex(ValueError, "point_color"):
            forest.plot_at_filtration(2, show=False, style_2d={"point_color": "red"})
        with self.assertRaisesRegex(ValueError, "activity_edge_color"):
            forest.plot_interior_simplex_activity(show=False, style={"activity_edge_color": "w"})

    def test_cubical_options_are_rejected_for_point_clouds(self):
        forest = PersistenceForest.from_points(np.random.default_rng(0).random((20, 2)))
        for name, call in self.calls(forest).items():
            with self.subTest(method=name), self.assertRaisesRegex(ValueError, "from_cubical_grid"):
                call(image_orientation=True)
        with self.assertRaisesRegex(ValueError, "show_image"):
            forest.plot_at_filtration(0.1, show=False, show_image=True)

    def test_simplex_snapshots_refuse_cubical_cells(self):
        forest = PersistenceForest.from_cubical_grid(centered((3, 3)))
        for helper in (forest._simplices_present_at_filtration, forest._complex_snapshot_at_filtration):
            with self.subTest(helper=helper.__name__), self.assertRaises(NotImplementedError):
                helper(2)


if __name__ == "__main__":
    unittest.main()
