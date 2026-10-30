"""V1–V5/V7: test plotted scientific data, not backend-dependent pixels."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import is_color_like
from persforest.cycle_rep_vectorisations import signed_chain_area, constant_one_functional
from persforest.forest_landscapes import (plot_barcode_measurement_generic, plot_landscape_comparison,
                                        plot_landscape_comparison_between_functionals)
from persforest.interior_activity_plotting import _activity_rows_for_plotting
from persforest.simplicial_filtration_animation import _compute_frame_times
from persforest.color_scheme import build_color_scheme
from fixtures_small_filtrations import triangle, split, fan, build


class PlottingContractTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_snapshot_2d_closed_sublevel_selection(self):
        fixture = split()
        forest = build(fixture)
        for time in (0, 1, 3, 5, 7):
            snapshot = forest._complex_snapshot_at_filtration(time)
            self.assertEqual(set(snapshot['edges']), {s for s, v in fixture.values.items() if len(s) == 2 and v <= time})
            self.assertEqual(set(snapshot['triangles']), {s for s, v in fixture.values.items() if len(s) == 3 and v <= time})
            np.testing.assert_array_equal(snapshot['points'], fixture.points)

    def test_filtration_cycle_artists_observe_birth_split_and_death(self):
        forest = build(split())
        before = deepcopy([b.cycle_reps for b in forest.barcode])
        for time, counts in [(0, []), (1, [4]), (3, [3, 3]), (5, [3]), (7, [])]:
            ax = forest.plot_at_filtration(time, show=False, show_complex=False,
                                           style_2d={'show_orientation_arrows': False})
            cycles = [c for c in ax.collections if isinstance(c, LineCollection)]
            self.assertEqual(sorted(len(c.get_segments()) for c in cycles), counts)
            for collection in cycles:
                self.assertTrue(np.isfinite(collection.get_segments()).all())
            plt.close(ax.figure)
        self.assertEqual(before, [b.cycle_reps for b in forest.barcode])

    def test_signed_plot_preserves_doubled_edge(self):
        forest = build(fan())
        for signed, expected in [(False, 3), (True, 5)]:
            ax = forest.plot_at_filtration(2, signed=signed, show=False, show_complex=False,
                                           style_2d={'show_orientation_arrows': False})
            [collection] = [c for c in ax.collections if isinstance(c, LineCollection)]
            self.assertEqual(len(collection.get_segments()), expected)

    def test_dual_edge_disappears_when_shared_primal_edge_arrives(self):
        forest = build(split())
        ax = forest.plot_at_filtration_with_dual(1, show=False, show_cycles=False)
        dual = [c for c in ax.collections if c.get_label() == 'dual edges']
        self.assertEqual(len(dual), 1)
        [segment] = dual[0].get_segments()
        np.testing.assert_allclose(sorted(map(tuple, segment)), [(4 / 3, -1 / 3), (4 / 3, 2 / 3)])
        ax = forest.plot_at_filtration_with_dual(3, show=False, show_cycles=False)
        self.assertFalse(any(c.get_label() == 'dual edges' for c in ax.collections))

    def test_barcode_representative_plot_respects_relative_position_and_filter(self):
        forest = build(split())
        for position, expected in [(0, [3, 4]), (.5, [3, 3])]:
            ax = forest.plot_barcode_cycle_reps(relative_position=position, show=False)
            collections = [c for c in ax.collections if isinstance(c, LineCollection)]
            self.assertEqual(sorted(len(c.get_segments()) for c in collections), expected)
        ax = forest.plot_barcode_cycle_reps(relative_position=0, min_bar_length=6, show=False)
        collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        self.assertEqual([len(c.get_segments()) for c in collections], [4])

    def test_profile_and_landscape_artist_values_and_cache_nonmutation(self):
        forest = build(split())
        ax, profile = plot_barcode_measurement_generic(forest, signed_chain_area, show=False)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 3, 3, 7])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [6, 6, 2, 2])
        grid = np.arange(9.)
        family = forest.compute_measurement_landscapes(signed_chain_area, 'area', x_grid=grid, max_k=2)
        before = family.evaluate_on_grid(grid, levels=2).copy()
        ax = forest.plot_measurement_landscapes('area', ks=[2, 1], show=False)
        for line, level in zip(ax.lines, [2, 1]):
            np.testing.assert_array_equal(line.get_xdata(), grid)
            np.testing.assert_array_equal(line.get_ydata(), before[level - 1])
        np.testing.assert_array_equal(family.evaluate_on_grid(grid, levels=2), before)
        forest.compute_measurement_landscapes(constant_one_functional, 'one', x_grid=grid)
        ax = plot_landscape_comparison_between_functionals(forest, ['area', 'one'])
        self.assertEqual([line.get_label() for line in ax.lines], ['area', 'one'])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), before[0])
        other = build(split())
        other.compute_measurement_landscapes(lambda r, p: 2 * signed_chain_area(r, p), 'area', x_grid=grid)
        ax = plot_landscape_comparison([forest, other], 'area', forest_labels=['a', 'b'])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), 2 * ax.lines[0].get_ydata())

    def test_interior_activity_rows_faces_and_real_artists(self):
        forest = build(split(), keep_simplex_diff=True, compute_interior=True)
        rows = _activity_rows_for_plotting(forest, 'layer', 0)
        self.assertEqual(sorted((s, b.birth, length) for s, b, length in rows),
                         [((0, 1, 2), 1, 2), ((0, 1, 2), 3, 2), ((0, 1, 3), 1, 6)])
        self.assertEqual(len(_activity_rows_for_plotting(forest, 'longest', 0)), 2)
        self.assertEqual(len(_activity_rows_for_plotting(forest, 'layer', 3)), 1)
        ax = forest.plot_interior_simplex_activity(overlap='layer', show=False)
        polygons = [c for c in ax.collections if isinstance(c, PolyCollection)]
        self.assertEqual(sum(len(c.get_paths()) for c in polygons), 3)
        ax = forest.plot_interior_simplex_activity_gradient(show=False, dpi=40, figsize=(2, 2))
        self.assertTrue(ax.collections or ax.images)
        with self.assertRaises(ValueError):
            _activity_rows_for_plotting(forest, 'invalid', 0)

    def test_frame_times_have_exact_bounds_and_finite_fallback(self):
        forest = build(triangle())
        np.testing.assert_array_equal(_compute_frame_times(forest, frames=4, t_min=None, t_max=None), [0, 2, 4, 6])
        np.testing.assert_array_equal(_compute_frame_times(forest, frames=3, t_min=2, t_max=6), [2, 4, 6])
        fallback = SimpleNamespace(barcode=[], filtration=[([0], 1), ([1], 3)])
        np.testing.assert_array_equal(_compute_frame_times(fallback, frames=2, t_min=None, t_max=None), [0, 3])
        for args in [dict(frames=0, t_min=0, t_max=1), dict(frames=2, t_min=2, t_max=1)]:
            with self.assertRaises(ValueError):
                _compute_frame_times(forest, **args)

    def test_animation_frames_update_cycle_artists_without_stale_lines(self):
        forest = build(triangle())
        anim, fig = forest.animate_filtration(frames=4, t_min=0, t_max=6, dpi=40, figsize=(2, 2),
                                              show_complex=False, filtration_kwargs={'style_2d': {'show_orientation_arrows': False}})
        # Explicit draw triggers initialization; callbacks are inspected without a GUI.
        fig.canvas.draw()
        for frame, expected in [(0, 0), (1, 1), (2, 1), (3, 0), (1, 1)]:
            anim._func(frame)
            cycles = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
            self.assertEqual(len(cycles), expected)
            if cycles:
                self.assertEqual(len(cycles[0].get_segments()), 3)

    def test_color_scheme_repeatability_valid_colors_and_no_global_rng_effect(self):
        before = np.random.get_state()
        first = build_color_scheme([2, 1], ['a', 'b'], seed=17, num_hues=24)
        build_color_scheme([1], ['other'], seed=99, num_hues=24)
        second = build_color_scheme([2, 1], ['a', 'b'], seed=17, num_hues=24)
        self.assertEqual(first, second)
        self.assertEqual(set(first['sets']), {'a', 'b'})
        for name, count in [('a', 2), ('b', 1)]:
            self.assertEqual(len(first['sets'][name]['colors']), count)
            self.assertTrue(all(is_color_like(c) for c in first['sets'][name]['colors']))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(build_color_scheme([], [], seed=17)['sets'], {})
