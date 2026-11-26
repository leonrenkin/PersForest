"""Cycle queries, plots and animations work on representatives rebuilt from the forest."""

import importlib.util
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from fixtures_small_filtrations import build, split, tetrahedra
from line_helpers import drawn_edges
from persforest.plotting.animation.filtration import animate_filtration_pair


class CycleQueryTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_point_and_forest_queries_match_the_bar_iterator(self):
        for fixture in (split(), tetrahedra()):
            forest = build(fixture)
            with self.assertRaisesRegex(ValueError, "not in barcode"):
                forest.cycle_for_bar_at(next(iter(build(fixture).barcode)), 3.)
            for bar in forest.barcode:
                for cycle in forest.iter_bar_cycle_reps(bar):
                    time = cycle.active_start
                    if time >= cycle.active_end:
                        continue
                    actual = forest.cycle_for_bar_at(bar, time)
                    self.assertEqual(actual.signed_simplices, cycle.signed_simplices)
                    self.assertEqual((actual.active_start, actual.active_end),
                                     (cycle.active_start, cycle.active_end))
                with self.assertRaises(ValueError):
                    forest.cycle_for_bar_at(bar, bar.death)

            for time in (1., 3., 5., 7., np.nextafter(3., np.inf)):
                expected = [rep.signed_simplices for bar in forest.barcode
                            for rep in forest.iter_bar_cycle_reps(bar)
                            if rep.active_start <= time < rep.active_end]
                for method in ("active_cycles_at", "cycle_reps_at"):
                    actual = [rep.signed_simplices for rep in getattr(forest, method)(time)]
                    self.assertCountEqual(actual, expected)

        delayed = build(split(), compute_barcode=False)
        with self.assertRaisesRegex(RuntimeError, "compute_barcode"):
            delayed.active_cycles_at(3.)

    def test_static_cycle_plots_draw_the_active_representatives(self):
        forest = build(split())
        ax = forest.plot_at_filtration(3., show=False, show_complex=False)
        counts = sorted(len(drawn_edges(c)) for c in ax.collections if isinstance(c, LineCollection))
        self.assertEqual(counts, [3, 3])
        plt.close(ax.figure)

        # In 3D all active cycles share one depth-sorted collection.
        forest = build(tetrahedra())
        active = forest._active_bars_with_cycles_at(3.)
        self.assertEqual(len(active), 2)
        ax = forest.plot_at_filtration(3., show=False, show_complex=False)
        [surface] = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        self.assertEqual(len(surface.get_facecolors()),
                         sum(len(cycle.simplices(signed=False)) for _, cycle in active))
        plt.close(ax.figure)

        ax = build(split()).plot_barcode_cycle_reps(show=False)
        self.assertTrue(any(isinstance(c, LineCollection) for c in ax.collections))
        plt.close(ax.figure)

    def test_animations_render_every_frame(self):
        forest = build(split())
        anim, fig = forest.animate_filtration(frames=4, t_min=0, t_max=7)
        drawn = []
        for index in range(4):
            anim._func(index)
            drawn.append(sorted(len(c.get_segments()) for ax in fig.axes for c in ax.collections
                                if isinstance(c, LineCollection)))
        plt.close(fig)
        self.assertEqual(len(drawn), 4)
        anim, fig = forest.animate_barcode_measurement(lambda cycle, points: 1., frames=2)
        for index in range(2):
            anim._func(index)
        plt.close(fig)
        anim, fig = animate_filtration_pair(forest, build(split()), frames=2)
        for index in range(2):
            anim._func(index)
        plt.close(fig)

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "plotly is not installed")
    def test_interactive_filtration_plot(self):
        figure = build(split()).plot_filtration_interactive(resolution=3, show=False)
        self.assertTrue(figure.frames)


if __name__ == "__main__":
    unittest.main()
