"""Cycle queries, plots and animations reconstruct representatives with storage="diffs"."""

import importlib.util
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from fixtures_small_filtrations import build, split, tetrahedra
from persforest.plotting.animation.filtration import animate_filtration_pair


class DiffOnlyCycleQueryTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_bar_query_and_forest_queries_match_stored_cycles(self):
        for fixture in (split(), tetrahedra()):
            stored = build(fixture, storage="cycles")
            diff = build(fixture, storage="diffs")
            stored_bars = {(bar.birth, bar.death): bar for bar in stored.barcode}
            with self.assertRaisesRegex(ValueError, "not in barcode"):
                diff.cycle_for_bar_at(next(iter(stored.barcode)), 3.)
            for bar in diff.barcode:
                reference = stored_bars[(bar.birth, bar.death)]
                for cycle in stored.iter_bar_cycle_reps(reference):
                    time = cycle.active_start
                    if time >= cycle.active_end:
                        continue
                    actual = diff.cycle_for_bar_at(bar, time)
                    self.assertEqual(actual.signed_simplices, cycle.signed_simplices)
                    self.assertEqual((actual.active_start, actual.active_end),
                                     (cycle.active_start, cycle.active_end))
                with self.assertRaises(ValueError):
                    diff.cycle_for_bar_at(bar, bar.death)
                with self.assertRaisesRegex(ValueError, 'storage="diffs"'):
                    bar.cycle_at_filtration_value(bar.birth)

            for time in (1., 3., 5., 7., np.nextafter(3., np.inf)):
                for method in ("active_cycles_at", "cycle_reps_at"):
                    expected = [rep.signed_simplices for rep in getattr(stored, method)(time)]
                    actual = [rep.signed_simplices for rep in getattr(diff, method)(time)]
                    self.assertCountEqual(actual, expected)
            self.assertCountEqual(
                [rep.signed_simplices for rep in diff.barcode_cycle_reps()],
                [rep.signed_simplices for rep in stored.barcode_cycle_reps()],
            )

        delayed = build(split(), storage="diffs", compute_barcode=False)
        with self.assertRaisesRegex(RuntimeError, "compute_barcode"):
            delayed.active_cycles_at(3.)

    def test_static_cycle_plots_match_stored_mode(self):
        for fixture, time, collection_type in (
            (split(), 3., LineCollection),
            (tetrahedra(), 3., Poly3DCollection),
        ):
            stored = build(fixture, storage="cycles")
            diff = build(fixture, storage="diffs")
            for forest in (stored, diff):
                ax = forest.plot_at_filtration(time, show=False, show_complex=False)
                counts = sorted(
                    len(collection.get_segments()) if isinstance(collection, LineCollection)
                    else 1
                    for collection in ax.collections
                    if isinstance(collection, collection_type)
                )
                if forest is stored:
                    expected = counts
                else:
                    self.assertEqual(counts, expected)
                plt.close(ax.figure)

        stored = build(split(), storage="cycles")
        diff = build(split(), storage="diffs")
        for forest in (stored, diff):
            ax = forest.plot_barcode_cycle_reps(show=False)
            counts = sorted(len(c.get_segments()) for c in ax.collections
                            if isinstance(c, LineCollection))
            if forest is stored:
                expected = counts
            else:
                self.assertEqual(counts, expected)
            plt.close(ax.figure)

    def test_animations_render_every_frame_like_cycle_storage(self):
        forests = {storage: build(split(), storage=storage) for storage in ("cycles", "diffs")}
        frames = {}
        for storage, forest in forests.items():
            anim, fig = forest.animate_filtration(frames=4, t_min=0, t_max=7)
            drawn = []
            for index in range(4):
                anim._func(index)
                drawn.append(sorted(len(c.get_segments()) for ax in fig.axes for c in ax.collections
                                    if isinstance(c, LineCollection)))
            frames[storage] = drawn
            plt.close(fig)
            anim, fig = forest.animate_barcode_measurement(lambda cycle, points: 1., frames=2)
            for index in range(2):
                anim._func(index)
            plt.close(fig)
        self.assertEqual(frames["diffs"], frames["cycles"])
        anim, fig = animate_filtration_pair(forests["cycles"], forests["diffs"], frames=2)
        for index in range(2):
            anim._func(index)
        plt.close(fig)

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "plotly is not installed")
    def test_interactive_filtration_plot_accepts_diff_storage(self):
        figures = [build(split(), storage=storage).plot_filtration_interactive(resolution=3, show=False)
                   for storage in ("cycles", "diffs")]
        self.assertEqual(len(figures[0].frames), len(figures[1].frames))
        for cycles_frame, diffs_frame in zip(figures[0].frames, figures[1].frames):
            self.assertEqual(len(cycles_frame.data), len(diffs_frame.data))


if __name__ == "__main__":
    unittest.main()
