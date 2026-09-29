"""Cycle queries and static plots reconstruct representatives in diff-only mode."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from fixtures_small_filtrations import build, split, tetrahedra
from persforest.simplicial_filtration_animation import animate_filtration_pair


class DiffOnlyCycleQueryTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_bar_query_and_forest_queries_match_stored_cycles(self):
        for fixture in (split(), tetrahedra()):
            stored = build(fixture)
            diff = build(fixture, diff_only_mode=True)
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
                with self.assertRaisesRegex(ValueError, "diff_only_mode"):
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

        delayed = build(split(), diff_only_mode=True, compute_barcode=False)
        with self.assertRaisesRegex(RuntimeError, "compute_barcode"):
            delayed.active_cycles_at(3.)

    def test_static_cycle_plots_match_stored_mode(self):
        for fixture, time, collection_type in (
            (split(), 3., LineCollection),
            (tetrahedra(), 3., Poly3DCollection),
        ):
            stored = build(fixture)
            diff = build(fixture, diff_only_mode=True)
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

        stored = build(split())
        diff = build(split(), diff_only_mode=True)
        for forest in (stored, diff):
            ax = forest.plot_barcode_cycle_reps(show=False)
            counts = sorted(len(c.get_segments()) for c in ax.collections
                            if isinstance(c, LineCollection))
            if forest is stored:
                expected = counts
            else:
                self.assertEqual(counts, expected)
            plt.close(ax.figure)

    def test_animations_reject_diff_only_mode_before_rendering(self):
        diff = build(split(), diff_only_mode=True)
        for animate in (
            lambda: diff.animate_filtration(frames=2),
            lambda: diff.animate_barcode_measurement(lambda cycle, points: 1., frames=2),
            lambda: diff.plot_filtration_interactive(resolution=2, show=False),
            lambda: animate_filtration_pair(build(split()), diff, frames=2),
        ):
            with self.subTest(animate=animate), self.assertRaisesRegex(ValueError, "diff_only_mode=True"):
                animate()


if __name__ == "__main__":
    unittest.main()
