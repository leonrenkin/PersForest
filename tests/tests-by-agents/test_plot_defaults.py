"""Shared defaults of the static plots: no implicit show, rcParams dpi, strict style keys."""
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from persforest import PersistenceForest
from persforest.measurements.basic import signed_chain_area
from persforest.measurements.loops import loops_area, on_raw_loops


def _square_with_hole():
    rng = np.random.default_rng(3)
    angles = rng.uniform(0, 2 * np.pi, 40)
    ring = np.c_[np.cos(angles), np.sin(angles)] + rng.normal(0, 0.03, (40, 2))
    return PersistenceForest(ring)


class PlotDefaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forest = _square_with_hole()
        cls.forest.compute_measurement_landscapes(signed_chain_area, "area", max_k=2)
        cls.radius = 0.5 * cls.forest.max_bar().death

    def tearDown(self):
        plt.close('all')

    def test_no_plot_calls_show_by_default(self):
        forest = self.forest
        calls = {
            "plot_barcode": lambda: forest.plot_barcode(),
            "plot_persistence_forest": lambda: forest.plot_persistence_forest(),
            "plot_at_filtration": lambda: forest.plot_at_filtration(self.radius),
            "plot_barcode_cycle_reps": lambda: forest.plot_barcode_cycle_reps(),
            "plot_interior_simplex_activity": lambda: forest.plot_interior_simplex_activity(),
            "plot_measurement_landscapes": lambda: forest.plot_measurement_landscapes("area"),
            "plot_barcode_measurement": lambda: forest.plot_barcode_measurement(signed_chain_area),
        }
        for name, call in calls.items():
            with self.subTest(name=name), mock.patch.object(plt, "show") as show:
                call()
                show.assert_not_called()
            plt.close('all')

    def test_show_true_calls_show_once(self):
        with mock.patch.object(plt, "show") as show:
            self.forest.plot_barcode(show=True)
        show.assert_called_once()

    def test_new_figures_use_rcparams_dpi(self):
        with matplotlib.rc_context({"figure.dpi": 72}):
            ax = self.forest.plot_interior_simplex_activity()
            self.assertEqual(ax.figure.dpi, 72)
            ax = self.forest.plot_at_filtration(self.radius, dpi=150)
            self.assertEqual(ax.figure.dpi, 150)

    def test_unknown_style_keys_are_rejected(self):
        forest = self.forest
        with self.assertRaisesRegex(ValueError, r"Unknown style_2d keys \['cycle_width'\]"):
            forest.plot_at_filtration(self.radius, style_2d={"cycle_width": 2})
        with self.assertRaisesRegex(ValueError, r"Unknown style keys \['edge_colour'\]"):
            forest.plot_interior_simplex_activity(style={"edge_colour": "k"})
        # Supported keys still apply.
        ax = forest.plot_at_filtration(self.radius, style_2d={"cycle_edge_width": 3.0})
        self.assertTrue(any(np.allclose(c.get_linewidths(), 3.0) for c in ax.collections
                            if len(c.get_linewidths())))

    def test_charts_have_open_frame_and_landscapes_run_dark_to_light(self):
        from matplotlib.colors import to_rgb
        forest = self.forest
        charts = [forest.plot_measurement_landscapes("area"),
                  forest.plot_barcode_measurement(signed_chain_area)[0],
                  forest.plot_landscape_comparison_between_measurements(["area"])]
        for ax in charts:
            self.assertFalse(ax.spines["top"].get_visible())
            self.assertFalse(ax.spines["right"].get_visible())
            self.assertTrue(ax.spines["left"].get_visible())
            legend = ax.get_legend()
            if legend is not None:
                self.assertFalse(legend.get_frame_on())
        lines = charts[0].get_lines()
        self.assertEqual([line.get_label() for line in lines], [r"$\lambda_{1}$", r"$\lambda_{2}$"])
        luminance = [np.dot(to_rgb(line.get_color()), (0.2126, 0.7152, 0.0722)) for line in lines]
        self.assertLess(luminance[0], luminance[1])

    def test_unknown_barcode_coloring_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coloring must be"):
            self.forest.plot_barcode(coloring="rainbow")

    def test_measurement_plot_names_measurement_and_bar(self):
        forest = self.forest
        bar = forest.max_bar()
        interval = f"[{bar.birth:.4g}, {bar.death:.4g})"

        ax, _ = forest.plot_barcode_measurement(signed_chain_area)
        self.assertEqual(ax.get_title(), f"signed_chain_area along bar {interval}")
        self.assertEqual(ax.get_ylabel(), "signed_chain_area")
        self.assertEqual(ax.get_xlabel(), "filtration value")

        ax, _ = forest.plot_barcode_measurement(lambda chain, points: 1.0)
        self.assertEqual(ax.get_title(), f"Measurement along bar {interval}")
        self.assertEqual(ax.get_ylabel(), "measurement")

        measurement = on_raw_loops(loops_area)
        ax, _ = forest.plot_barcode_measurement(measurement, label="area")
        self.assertEqual(ax.get_title(), f"area along bar {interval}")

        ax, _ = forest.plot_barcode_measurement(measurement)
        self.assertIn(measurement.measurement_name, ax.get_title())
        self.assertNotIn("None", ax.get_title())


if __name__ == "__main__":
    unittest.main()


class ThreeDDefaultTests(unittest.TestCase):
    """3D plots: shell of the complex, opaque shaded cycles, solid long-bar interiors."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        def sphere(centre, radius, n):
            v = rng.normal(size=(n, 3))
            return np.asarray(centre) + radius * v / np.linalg.norm(v, axis=1)[:, None]
        cls.forest = PersistenceForest(np.vstack([sphere((0, 0, 0), 1.0, 160), sphere((3, 0, 0), 0.6, 80),
                                                  rng.uniform(-1.5, 4, (12, 3))]))
        top = sorted(cls.forest.barcode, key=lambda b: b.lifespan(), reverse=True)[:2]
        cls.radius = max(b.birth for b in top) + 0.2 * (min(b.death for b in top) - max(b.birth for b in top))
        cls.top = top

    def tearDown(self):
        plt.close('all')

    def test_filtration_draws_shaded_opaque_cycles_in_one_collection(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax = self.forest.plot_at_filtration(self.radius)
        shell, cycles = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        active = self.forest._active_bars_with_cycles_at(self.radius)
        self.assertGreaterEqual(len(active), 2)
        faces = cycles.get_facecolors()
        self.assertEqual(len(faces), sum(len(c.simplices(signed=False)) for _, c in active))
        np.testing.assert_allclose(faces[:, 3], 1.0)
        self.assertGreater(len({tuple(np.round(f[:3], 4)) for f in faces}), len(active))  # shading varies
        self.assertTrue(np.all(shell.get_facecolors()[:, 3] < 0.1))
        self.assertEqual(ax.get_xlabel(), "")
        self.assertTrue(ax.get_title().startswith("r = "))

    def test_interior_regions_of_long_bars_are_opaque(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax = self.forest.plot_interior_simplex_activity()
        [collection] = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        alphas = set(np.round(collection.get_facecolors()[:, 3], 6))
        self.assertLessEqual(alphas, {0.04, 1.0})
        self.assertIn(1.0, alphas)
        with self.assertRaisesRegex(ValueError, "opacity_by"):
            self.forest.plot_interior_simplex_activity(style={"opacity_by": "lifespan"})
