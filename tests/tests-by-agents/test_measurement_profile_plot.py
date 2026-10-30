"""Overlay plot of measurement profiles: bar selection, colors and axis alignment."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from persforest.measurements.basic import signed_chain_area
from fixtures_small_filtrations import split, build


class MeasurementProfilePlotTests(unittest.TestCase):
    def setUp(self):
        self.forest = build(split())
        self.profiles = self.forest.compute_measurement_profiles(signed_chain_area, "area", signed=True)

    def tearDown(self):
        plt.close("all")

    def test_one_curve_per_bar_in_barcode_colors_and_limits(self):
        fig, (bar_ax, ax) = plt.subplots(1, 2)
        self.forest.plot_barcode(ax=bar_ax, coloring="forest")
        self.forest.plot_measurement_profiles(self.profiles, ax=ax)
        bars = sorted(self.forest.barcode, key=lambda b: b.birth)
        self.assertEqual(len(ax.lines), len(bars))
        for line, bar in zip(ax.lines, bars):
            self.assertEqual(to_rgba(line.get_color()), to_rgba(self.forest.color_map_forest[bar]))
        self.assertEqual(ax.get_xlim(), bar_ax.get_xlim())
        self.assertEqual(ax.get_ylabel(), "area")

        # Bar [1, 7): area 6 on [1, 3), 2 on [3, 7).
        xs, ys = ax.lines[0].get_data()
        np.testing.assert_array_equal(xs, [1, 3, 3, 7])
        np.testing.assert_array_equal(ys, [6, 6, 2, 2])

    def test_bar_filters_match_plot_barcode(self):
        ax = self.forest.plot_measurement_profiles("area", min_bar_length=5, coloring="grey", legend=True)
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(ax.lines[0].get_label(), "[1, 7)")
        self.assertEqual(to_rgba(ax.lines[0].get_color()), to_rgba("black"))
        ax = self.forest.plot_measurement_profiles(self.profiles, max_bars=1, color="red")
        self.assertEqual(len(ax.lines), 1)
        np.testing.assert_array_equal(ax.lines[0].get_xdata()[[0, -1]], [1, 7])
        self.assertEqual(to_rgba(ax.lines[0].get_color()), to_rgba("red"))

    def test_only_profiled_bars_are_drawn(self):
        long_only = self.forest.compute_measurement_profiles(
            signed_chain_area, "long", signed=True, min_bar_length=5, cache=False)
        ax = self.forest.plot_measurement_profiles(long_only)
        self.assertEqual(len(ax.lines), 1)

    def test_measurement_axis_starts_at_zero_and_bars_restrict(self):
        short = min(self.forest.barcode, key=lambda b: b.lifespan())
        ax = self.forest.plot_measurement_profiles(self.profiles, bars=[short])
        self.assertEqual(len(ax.lines), 1)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [4, 4])
        self.assertEqual(ax.get_ylim()[0], 0)
        negative = self.forest.compute_measurement_profiles(
            lambda chain, points: -signed_chain_area(chain, points), "negative", signed=True)
        self.assertEqual(self.forest.plot_measurement_profiles(negative).get_ylim()[0], -6)

    def test_extend_baseline_draws_zero_outside_the_bar(self):
        ax = self.forest.plot_measurement_profiles(self.profiles, extend_baseline=True)
        lower, upper = ax.get_xlim()
        xs, ys = ax.lines[0].get_data()  # bar [1, 7): area 6 on [1, 3), 2 on [3, 7)
        np.testing.assert_array_equal(xs, [lower, 1, 1, 3, 3, 7, 7, upper])
        np.testing.assert_array_equal(ys, [0, 0, 6, 6, 2, 2, 0, 0])
        anim, fig = self.forest.animate_barcode_measurement(
            signed_chain_area, frames=2, t_min=-1, t_max=9,
            filtration_kwargs={"show_complex": False}, measurement_kwargs={"extend_baseline": True})
        panel = fig.axes[1]
        self.assertEqual(panel.get_xlim()[0], -1)
        self.assertGreaterEqual(panel.get_xlim()[1], 9)
        np.testing.assert_array_equal(panel.lines[0].get_xdata()[[0, -1]], panel.get_xlim())

    def test_animation_shows_filtered_bars_or_one_chosen_bar(self):
        options = dict(frames=2, t_max=7, filtration_kwargs={"show_complex": False})
        anim, fig = self.forest.animate_barcode_measurement(signed_chain_area, **options)
        self.assertEqual(len(fig.axes[1].lines), 3)  # two profiles and the time line
        plt.close(fig)
        anim, fig = self.forest.animate_barcode_measurement(
            signed_chain_area, frames=2, t_max=7,
            filtration_kwargs={"show_complex": False, "min_bar_length": 5})
        self.assertEqual(len(fig.axes[1].lines), 2)
        plt.close(fig)
        short = min(self.forest.barcode, key=lambda b: b.lifespan())
        anim, fig = self.forest.animate_barcode_measurement(
            signed_chain_area, bar=short, frames=2, t_max=7,
            filtration_kwargs={"show_complex": False, "min_bar_length": 5})
        np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), [4, 4])
        self.assertEqual(len(fig.axes[1].lines), 2)

    def test_rejects_unknown_labels_and_foreign_profiles(self):
        with self.assertRaises(KeyError):
            self.forest.plot_measurement_profiles("missing")
        with self.assertRaises(ValueError):
            build(split()).plot_measurement_profiles(self.profiles)
        with self.assertRaises(ValueError):
            self.forest.plot_measurement_profiles(self.profiles, min_bar_length=10)


if __name__ == "__main__":
    unittest.main()
