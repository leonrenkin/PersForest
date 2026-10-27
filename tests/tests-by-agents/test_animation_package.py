"""Animation extraction contracts: imports, writers, and synchronized frames."""

import importlib
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from fixtures_small_filtrations import build, tetrahedra, triangle



class AnimationPackageTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')


    def test_mp4_writers_preserve_distinct_encoder_and_canvas_settings(self):
        export = importlib.import_module('persforest.plotting.animation._export')
        paired_options = dict(fps=3, codec='libx264', extra_args=[
            '-pix_fmt', 'yuv420p', '-vf', 'scale=1920:-2', '-profile:v', 'baseline',
            '-level', '3.1', '-movflags', '+faststart',
        ])
        for name in ('filtration', 'paired', 'measurement'):
            with self.subTest(path=name):
                anim, fig = Mock(), Mock()
                fig.get_facecolor.return_value = (1., 1., 1., 1.)
                helper = getattr(export, '_save_' + name + '_animation')
                arguments = (anim, fig, 'movie.MP4') if name == 'filtration' else (anim, 'movie.MP4')
                with patch.object(export, '_ffmpeg_path_or_raise', return_value='/fake/ffmpeg'), \
                     patch('matplotlib.animation.FFMpegWriter') as writer:
                    helper(*arguments, fps=3, dpi=40)
                writer.assert_called_once_with(**(paired_options if name == 'paired' else dict(fps=3, bitrate=2000)))
                expected = dict(writer=writer.return_value, dpi=40)
                if name == 'filtration':
                    expected['savefig_kwargs'] = dict(pad_inches=0., facecolor=(1., 1., 1., 1.))
                    fig.canvas.draw.assert_called_once_with()
                anim.save.assert_called_once_with('movie.MP4', **expected)

    def test_mp4_writers_reject_other_formats_without_saving(self):
        export = importlib.import_module('persforest.plotting.animation._export')
        for name in ('filtration', 'paired', 'measurement'):
            for suffix in ('gif', 'gifv', 'html', 'avi', ''):
                with self.subTest(path=name, suffix=suffix):
                    anim, fig = Mock(), Mock()
                    filename = 'movie.' + suffix
                    arguments = (anim, fig, filename) if name == 'filtration' else (anim, filename)
                    with self.assertRaisesRegex(ValueError, 'only .mp4'):
                        getattr(export, '_save_' + name + '_animation')(*arguments, fps=3, dpi=40)
                    anim.save.assert_not_called()
                    fig.canvas.draw.assert_not_called()

    def test_public_dispatch_rejects_gif_for_2d_and_3d(self):
        for fixture in (triangle, tetrahedra):
            forest = build(fixture())
            for suffix in ('gif', 'gifv'):
                with self.subTest(dimension=forest.dim, suffix=suffix):
                    with self.assertRaisesRegex(ValueError, '.mp4 filename'):
                        forest.animate_filtration(filename='movie.' + suffix)

    def test_mp4_writers_report_missing_ffmpeg(self):
        export = importlib.import_module('persforest.plotting.animation._export')
        for name in ('filtration', 'paired', 'measurement'):
            with self.subTest(path=name), patch.object(export.shutil, 'which', return_value=None):
                anim, fig = Mock(), Mock()
                args = (anim, fig, 'movie.mp4') if name == 'filtration' else (anim, 'movie.mp4')
                with self.assertRaisesRegex(RuntimeError, 'Install ffmpeg'):
                    getattr(export, '_save_' + name + '_animation')(*args, fps=3, dpi=40)
                anim.save.assert_not_called()

    def test_ffmpeg_errors_preserve_the_process_diagnostic(self):
        export = importlib.import_module('persforest.plotting.animation._export')
        failure = subprocess.CalledProcessError(1, 'ffmpeg', stderr=b'encoder unavailable\n')
        with patch.object(export.shutil, 'which', return_value='/fake/ffmpeg'), \
             patch.object(export.subprocess, 'run', side_effect=failure) as run:
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg failed.*\nencoder unavailable') as caught:
                export._assemble_mp4_with_ffmpeg(frame_dir=Path('/frames'), fps=3, output_file=Path('/movie.mp4'))
        self.assertIs(caught.exception.__cause__, failure)
        command = run.call_args.args[0]
        self.assertEqual(command[0], '/fake/ffmpeg')
        self.assertEqual(command[command.index('-r') + 1], '3')
        self.assertEqual(command[-1], '/movie.mp4')

    def test_paired_frames_keep_both_barcode_markers_synchronized(self):
        from persforest.plotting.animation.filtration import animate_filtration_pair
        anim, fig = animate_filtration_pair(build(triangle()), build(triangle()),
            frames=3, t_min=0, t_max=6, total_figsize=(4, 3))
        fig.canvas.draw()
        for index, value in enumerate([0., 3.15, 6.3]):
            anim._func(index)
            for ax in fig.axes[2:]:
                np.testing.assert_allclose(ax.lines[-1].get_xdata(), [value, value])

    @patch('persforest.plotting.animation.filtration._save_filtration_animation')
    def test_3d_frames_keep_cycles_camera_and_barcode_in_sync(self, save_animation):
        forest = build(tetrahedra())
        anim, fig = forest.animate_filtration(frames=4, t_min=0, t_max=7,
            with_barcode=True, show_complex=False, camera_mode='orbit',
            figsize=(4, 2), dpi=40,
            filtration_kwargs={'style_3d': {'camera_eye': {'elev': 12, 'azim': 20}}})
        fig.canvas.draw()
        scene, barcode = fig.axes
        self.assertEqual(scene.name, '3d')
        for index, (value, count) in enumerate([(0., 0), (7 / 3, 1), (14 / 3, 2), (7., 0)]):
            anim._func(index)
            self.assertEqual(sum(isinstance(c, Poly3DCollection) for c in scene.collections), count)
            self.assertAlmostEqual(scene.elev, 12.)
            self.assertAlmostEqual(scene.azim, 20. + 120 * index)
            np.testing.assert_allclose(barcode.lines[-1].get_xdata(), [value, value])


if __name__ == '__main__':
    unittest.main()
