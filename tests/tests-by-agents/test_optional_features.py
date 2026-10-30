"""V5–V6: optional backends, tiny export, and actionable missing-feature errors."""
import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from persforest.simplicial_filtration_animation import _ffmpeg_path_or_raise
from persforest.forest_landscapes import animate_barcode_measurement_generic
from persforest.cycle_rep_vectorisations import signed_chain_area
from fixtures_small_filtrations import triangle, split, build

HAS_PLOTLY = importlib.util.find_spec('plotly') is not None
HAS_PILLOW = importlib.util.find_spec('PIL') is not None


class OptionalFeatureTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_missing_plotly_reports_install_hint(self):
        forest = build(triangle())
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name == 'plotly' or name.startswith('plotly.'):
                raise ImportError('Simulated absent optional dependency')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=blocked):
            with self.assertRaisesRegex(RuntimeError, 'pip install.*plotly'):
                forest.plot_at_filtration_plotly(3, show=False)

    def test_core_import_in_fresh_process_without_plotly_or_notebook_packages(self):
        code = '''
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'plotly', 'nbformat', 'ipywidgets', 'anywidget'}:
        raise ImportError('Optional backend blocked in test')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from persforest import PersistenceForest
f = PersistenceForest([[0,0], [1,0], [0,1]])
assert not f.barcode
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_ffmpeg_is_actionable(self):
        with patch('persforest.simplicial_filtration_animation.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Install ffmpeg'):
                _ffmpeg_path_or_raise()

    @unittest.skipUnless(HAS_PLOTLY, 'Plotly optional extra is not installed')
    def test_plotly_snapshot_and_slider_geometry_without_rendering(self):
        forest = build(triangle())
        figure = forest.plot_at_filtration_plotly(3, show=False)
        self.assertTrue(any(trace.type == 'scatter' and trace.mode == 'markers' for trace in figure.data))
        lines = [trace for trace in figure.data if trace.type == 'scatter' and trace.mode == 'lines']
        self.assertGreaterEqual(len(lines), 1)
        slider = forest.plot_filtration_interactive(filt_max=6, resolution=3, show=False)
        self.assertEqual(len(slider.frames), 4)
        self.assertEqual(len(slider.layout.sliders[0].steps), 4)
        self.assertEqual(len(slider.frames[0].data), len(slider.frames[-1].data))
        self.assertTrue(all(frame.name for frame in slider.frames))

    @unittest.skipUnless(HAS_PILLOW, 'Pillow animation extra is not installed')
    def test_tiny_gif_export_is_readable_and_has_expected_frames(self):
        from PIL import Image
        forest = build(triangle())
        with tempfile.TemporaryDirectory(prefix='persforest-test-') as directory:
            path = Path(directory) / 'triangle.gif'
            anim, fig = forest.animate_filtration(filename=str(path), frames=3, fps=2,
                t_min=0, t_max=6, dpi=40, figsize=(2, 2), show_complex=False,
                filtration_kwargs={'style_2d': {'show_orientation_arrows': False}})
            with Image.open(path) as image:
                self.assertEqual(image.format, 'GIF')
                self.assertEqual(image.n_frames, 3)
                self.assertEqual(image.size, (80, 80))

    def test_measurement_animation_tracks_same_time_as_profile(self):
        forest = build(split())
        anim, fig = animate_barcode_measurement_generic(forest, signed_chain_area, frames=3,
            t_min=1, t_max=7, dpi=40, total_figsize=(4, 2),
            filtration_kwargs={'show': False, 'show_complex': False,
                               'style_2d': {'show_orientation_arrows': False}})
        fig.canvas.draw()
        for frame, time in enumerate([1, 4, 7]):
            anim._func(frame)
            np.testing.assert_array_equal(fig.axes[1].lines[-1].get_xdata(), [time, time])
        # The animation panel extends beyond the bar, displaying zero baseline.
        np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), [0, 0, 6, 6, 2, 2, 0, 0])
