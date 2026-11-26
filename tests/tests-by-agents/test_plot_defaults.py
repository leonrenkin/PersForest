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
