"""The droplet segmentation application in applications/droplet_segmentation: API, tuner, files and CLI."""
import importlib.util
import io
import json
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np

APPLICATION = Path(__file__).resolve().parents[2] / "applications" / "droplet_segmentation"
MISSING = [name for name in ("scipy", "PIL", "matplotlib", "pandas") if importlib.util.find_spec(name) is None]
HAS_WIDGETS = importlib.util.find_spec("ipywidgets") is not None

if not MISSING:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    sys.path.insert(0, str(APPLICATION))
    import dropseg
    from dropseg.__main__ import main as cli
    from dropseg.simulate import Droplet, count_found, droplet_image, true_labels
    from dropseg.stages import Stages


# The former fixed defaults of the intensity parameters, given explicitly where a test relies on them.
LEVELS = {"min_bar_length": 0.2, "split_level": 0.3, "min_level": 0.1}


def scene():
    """Two isolated droplets and a pair without a valley between them (an 8-shaped outline)."""
    droplets = [Droplet(14, 14, 6, 0.8), Droplet(34, 50, 5, 0.6), Droplet(30, 18, 6, 0.7), Droplet(30, 30, 6, 0.7)]
    shape = (48, 64)
    image = np.clip(droplet_image(droplets, shape, blur=1.0, background=0.02, noise=0.005, seed=1), 0, 1)
    return image, true_labels(droplets, shape)


@unittest.skipIf(MISSING, f"needs {MISSING}")
class SegmentImageTests(unittest.TestCase):
    def test_finds_every_droplet_and_splits_the_pair(self):
        image, truth = scene()
        segmentation = dropseg.segment_image(image, dropseg.Parameters(**LEVELS))
        self.assertEqual(segmentation.n_segments, 4)
        self.assertEqual(count_found(segmentation.labels, truth), 4)
        # The pair was cut by the lobe splitting: its two segments share one outline group.
        self.assertEqual(len(np.unique(segmentation.groups[segmentation.groups > 0])), 3)

    def test_without_lobe_splitting_the_pair_is_one_segment(self):
        image, truth = scene()
        segmentation = dropseg.segment_image(image, dropseg.Parameters(**LEVELS, split_lobes=False))
        self.assertEqual(segmentation.n_segments, 3)

    def test_upsampled_segmentation_agrees_on_original_pixels(self):
        image, truth = scene()
        segmentation = dropseg.segment_image(image, dropseg.Parameters(**LEVELS, upsample=2), tile_size=32)
        self.assertEqual(segmentation.factor, 2)
        self.assertEqual(segmentation.labels.shape, (96, 128))
        self.assertEqual(segmentation.original_labels.shape, image.shape)
        self.assertEqual(count_found(segmentation.original_labels, truth), 4)

    def test_large_images_are_tiled_at_the_original_resolution(self):
        image, truth = scene()
        from dropseg import whole_image
        limit = whole_image.LARGE_IMAGE
        whole_image.LARGE_IMAGE = 100
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # no droplet is dropped by the tiling
                segmentation = dropseg.segment_image(image, dropseg.Parameters(**LEVELS), tile_size=32, tile_margin=12)
        finally:
            whole_image.LARGE_IMAGE = limit
        self.assertIsNone(segmentation.result)
        self.assertEqual(len(segmentation.tiles), 4)
        self.assertEqual(count_found(segmentation.labels, truth), 4)

    def test_warns_when_tiles_still_cut_droplets_after_the_last_retry(self):
        # A cut that survives every widening of the margin is rare in practice, so the tiled
        # segmentation is replaced by one whose tiles report two cut droplets.
        from unittest import mock
        from dropseg import whole_image
        image, _ = scene()
        real = whole_image.segment_tiled

        def cutting(*args, **kwargs):
            labels, tile_results = real(*args, **kwargs)
            tile_results[0].cut, tile_results[-1].cut = 1, 1
            return labels, tile_results

        with mock.patch.object(whole_image, "segment_tiled", cutting):
            with self.assertWarnsRegex(UserWarning, r"tiles cut 2 droplet piece\(s\) .* tile_margin = 12 px"):
                dropseg.segment_image(image, dropseg.Parameters(**LEVELS, upsample=2), tile_size=32, tile_margin=12)

    def tiled(self, image, parameters, tile_size, tile_margin):
        """``segment_image`` forced into tiles at 1x, with the warnings it gave."""
        from unittest import mock
        import gc
        from dropseg import whole_image
        # Garbage of earlier tests (e.g. unrendered Matplotlib animations) warns
        # when collected; collect it now so that only segment_image's warnings are recorded.
        gc.collect()
        with mock.patch.object(whole_image, "LARGE_IMAGE", 1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            segmentation = dropseg.segment_image(image, parameters, tile_size=tile_size, tile_margin=tile_margin)
        return segmentation, [str(warning.message) for warning in caught]

    def test_a_droplet_reaching_a_tile_edge_is_found_by_the_retry(self):
        # Radius 20 in tiles of 16 with a margin of 6: the first tiles cut the droplet. Without the
        # dark frame it merged with the bright outside, formed no bar, and vanished uncounted.
        image = np.clip(droplet_image([Droplet(30.3, 29.7, 20, 0.8)], (64, 64), 1.0, 0.02, 0.005, 1), 0, 1)
        segmentation, caught = self.tiled(image, dropseg.Parameters(**LEVELS), 16, 6)
        self.assertEqual(caught, [])
        self.assertEqual(segmentation.n_segments, 1)
        np.testing.assert_array_equal(segmentation.labels > 0, dropseg.segment_image(
            image, dropseg.Parameters(**LEVELS)).labels > 0)

    def test_a_droplet_cut_after_every_retry_is_reported_not_lost_silently(self):
        image = np.clip(droplet_image([Droplet(30.3, 29.7, 10, 0.8)], (64, 64), 1.0, 0.02, 0.005, 1), 0, 1)
        segmentation, caught = self.tiled(image, dropseg.Parameters(**LEVELS), 16, 2)
        self.assertEqual(segmentation.n_segments, 0)
        self.assertEqual(len(caught), 1)
        self.assertRegex(caught[0], r"tiles cut \d+ droplet piece\(s\)")

    def test_tiles_reproduce_the_whole_image_on_a_crowded_scene(self):
        from dropseg.simulate import sample_droplets
        droplets = sample_droplets(np.random.default_rng([1, 9]), (160, 160), 25, (3.0, 12.0))
        image = np.clip(droplet_image(droplets, (160, 160), 1.0, 0.15, 0.02, 1), 0, 1)
        parameters = dropseg.Parameters(min_bar_length=0.2, split_level=0.49, min_level=0.29)  # about the inferred levels
        whole = dropseg.segment_image(image, parameters)
        segmentation, caught = self.tiled(image, parameters, 32, 4)
        self.assertEqual(caught, [])
        self.assertEqual(segmentation.n_segments, whole.n_segments)
        self.assertEqual(count_found(segmentation.labels, whole.labels), whole.n_segments)

    def test_runs_are_identical(self):
        image, _ = scene()
        first = dropseg.segment_image(image, dropseg.Parameters(**LEVELS))
        second = dropseg.segment_image(image, dropseg.Parameters(**LEVELS))
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_stages_on_the_whole_image_match_segment_image(self):
        image, _ = scene()
        parameters = dropseg.Parameters(**{**LEVELS, "min_level": 0.08})
        stages = Stages(image, (slice(0, image.shape[0]), slice(0, image.shape[1])))
        np.testing.assert_array_equal(stages.result(parameters).labels,
                                      dropseg.segment_image(image, parameters).labels)

    def test_rejects_unscaled_images(self):
        image, _ = scene()
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            dropseg.segment_image(image * 255, dropseg.Parameters(**LEVELS))
        with self.assertRaisesRegex(ValueError, "2D"):
            dropseg.segment_image(image[None], dropseg.Parameters(**LEVELS))


@unittest.skipIf(MISSING, f"needs {MISSING}")
class ParametersTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "below split_level"):
            dropseg.Parameters(min_level=0.3, split_level=0.3)
        with self.assertRaisesRegex(ValueError, "upsample"):
            dropseg.Parameters(upsample=0)

    def test_record_round_trip(self):
        parameters = dropseg.Parameters(upsample=4, smooth_radius=0.5, convexity_tol=0.4)
        self.assertEqual(dropseg.Parameters.from_record(json.loads(json.dumps(parameters.record()))), parameters)
        with self.assertRaisesRegex(ValueError, "unknown"):
            dropseg.Parameters.from_record({"min_bar": 0.1})

    def test_lengths_scale_with_the_upsampling(self):
        settings = dropseg.Parameters(**LEVELS, upsample=4, smooth_radius=0.5, min_area=2.0).settings()
        self.assertEqual(settings.smooth_radius, 2.0)
        self.assertEqual(settings.min_area, 32.0)
        self.assertAlmostEqual(settings.birth_offset, 0.1 * 0.2)


@unittest.skipIf(MISSING, f"needs {MISSING}")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        # Inferring warns; the tests below that do not check the warning ignore it.
        catcher = warnings.catch_warnings()
        catcher.__enter__()
        self.addCleanup(catcher.__exit__, None, None, None)
        warnings.simplefilter("ignore", dropseg.InferredLevelsWarning)

    def test_inferring_warns_with_the_values_and_recommends_the_tuner(self):
        image, _ = scene()
        warnings.simplefilter("error", dropseg.InferredLevelsWarning)
        with self.assertWarnsRegex(dropseg.InferredLevelsWarning, r"min_bar_length = .*Tuner") as caught:
            segmentation = dropseg.segment_image(image, dropseg.Parameters())
        self.assertIn(f"{segmentation.parameters.min_level:.3g}", str(caught.warning))
        with self.assertWarnsRegex(dropseg.InferredLevelsWarning, r"inferred: split_level = 0.3\."):
            dropseg.infer_parameters(image, dropseg.Parameters(min_level=0.1, min_bar_length=0.2))
        dropseg.segment_image(image, dropseg.Parameters(**LEVELS))  # given levels: no warning (it would raise)

    def test_background_and_noise_are_estimated_from_the_dark_half(self):
        droplets = [Droplet(20, 20, 8, 0.6), Droplet(40, 60, 10, 0.5)]
        image = np.clip(droplet_image(droplets, (64, 96), blur=1.0, background=0.2, noise=0.02, seed=3), 0, 1)
        background = dropseg.estimate_background(image)
        self.assertAlmostEqual(background.level, 0.2, delta=0.005)
        # The droplets (about a tenth of the pixels) lift the median and widen its dark half a little:
        # the noise comes out slightly high, which keeps the inferred levels on the safe side.
        self.assertGreaterEqual(background.noise, 0.02 * 0.95)
        self.assertLessEqual(background.noise, 0.02 * 1.2)

    def test_a_flat_quantized_background_is_not_estimated(self):
        from dropseg.simulate import camera
        # Noise of half a gray level: most background pixels share one value, so the noise measures as 0.
        image = camera(droplet_image([Droplet(20, 20, 8, 0.6)], (48, 48), blur=1.0, background=3 / 255,
                                     noise=0.5 / 255, seed=2))
        with self.assertRaisesRegex(ValueError, "too flat or too coarsely quantized.*Tuner\\(image, window"):
            dropseg.estimate_background(image)
        # One gray level of noise is measurable: at least one step.
        image = camera(droplet_image([Droplet(20, 20, 8, 0.6)], (48, 48), blur=1.0, background=3 / 255,
                                     noise=1.5 / 255, seed=2))
        self.assertGreaterEqual(dropseg.estimate_background(image).noise, 1 / 255)

    def test_an_inferred_min_level_above_every_pixel_is_refused(self):
        # Strong noise on a bright background: b + 5 sigma lies above the brightest pixel.
        image = np.clip(0.6 + 0.05 * np.random.default_rng(0).standard_normal((32, 32)), 0, 1)
        with self.assertRaisesRegex(ValueError, "above the brightest pixel.*dropseg.Tuner"):
            dropseg.infer_parameters(image)
        self.assertEqual(dropseg.infer_parameters(image, dropseg.Parameters(min_level=0.7)).min_level, 0.7)

    def test_a_constant_image_is_not_estimated(self):
        with self.assertRaisesRegex(ValueError, "cannot be estimated"):
            dropseg.estimate_background(np.full((8, 8), 0.3))

    def test_a_clipped_background_is_not_estimated(self):
        # Most pixels at zero, as after a black-level subtraction: the noise is hidden.
        image = np.zeros((32, 32))
        image[8:16, 8:16] = 0.5
        image[::3, ::5] = 0.01
        with self.assertRaisesRegex(ValueError, "100%|clipped"):
            dropseg.estimate_background(image)
        with self.assertRaisesRegex(ValueError, "give min_level and min_bar_length"):
            dropseg.segment_image(image, dropseg.Parameters())
        # With both given, split_level is derived without the background.
        segmentation = dropseg.segment_image(image, dropseg.Parameters(min_level=0.1, min_bar_length=0.2))
        self.assertAlmostEqual(segmentation.parameters.split_level, 0.1 + 0.2)
        self.assertIsNone(segmentation.background)
        self.assertEqual(segmentation.n_segments, 1)

    def test_only_unset_levels_are_inferred_and_split_level_follows_the_final_levels(self):
        from dropseg import inference
        image, _ = scene()
        background = dropseg.estimate_background(image)
        noise = background.noise
        inferred = dropseg.infer_parameters(image)
        self.assertAlmostEqual(inferred.min_level, background.level + inference.LEVEL_NOISE * noise)
        self.assertAlmostEqual(inferred.min_bar_length, inference.BAR_NOISE * noise)
        self.assertAlmostEqual(inferred.split_level,
                               inferred.min_level + inferred.min_bar_length)
        given = dropseg.infer_parameters(image, dropseg.Parameters(min_level=0.3, upsample=2))
        self.assertEqual((given.min_level, given.upsample), (0.3, 2))
        self.assertAlmostEqual(given.split_level, 0.3 + inference.BAR_NOISE * noise)
        explicit = dropseg.Parameters(**LEVELS)
        self.assertIs(dropseg.infer_parameters(image, explicit), explicit)
        with self.assertRaisesRegex(ValueError, "inferred from a background"):
            dropseg.infer_parameters(image, dropseg.Parameters(split_level=background.level))

    def test_unset_levels_cannot_reach_the_steps(self):
        with self.assertRaisesRegex(ValueError, "infer_parameters"):
            dropseg.Parameters(min_level=0.1).settings()

    def test_segment_image_infers_once_and_records_what_it_used(self):
        image, truth = scene()
        segmentation = dropseg.segment_image(image, dropseg.Parameters())
        self.assertEqual(segmentation.parameters, dropseg.infer_parameters(image))
        self.assertEqual(segmentation.background, dropseg.estimate_background(image))
        self.assertEqual(count_found(segmentation.labels, truth), 4)
        again = dropseg.segment_image(image, segmentation.parameters)
        np.testing.assert_array_equal(again.labels, segmentation.labels)
        self.assertIsNone(again.background)

    def test_inferred_levels_follow_a_rescaling_and_offset_of_the_intensities(self):
        image, _ = scene()
        first = dropseg.segment_image(image, dropseg.Parameters())
        second = dropseg.segment_image(0.5 * image + 0.3, dropseg.Parameters())
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_save_results_records_the_inferred_levels(self):
        image, _ = scene()
        parameters = dropseg.Parameters(min_level=0.05)
        segmentation = dropseg.segment_image(image, parameters)
        with tempfile.TemporaryDirectory() as folder:
            dropseg.save_results(folder, "scene", image, segmentation, parameters, scale=2)
            record = json.loads((Path(folder) / "scene_parameters.json").read_text())
        self.assertEqual(record["inferred"], ["min_bar_length", "split_level"])
        self.assertEqual(record["parameters"]["min_level"], 0.05)
        self.assertEqual(dropseg.Parameters.from_record(record["parameters"]), segmentation.parameters)
        self.assertAlmostEqual(record["background"]["noise"], segmentation.background.noise)


@unittest.skipIf(MISSING, f"needs {MISSING}")
class FilesAndCommandLineTests(unittest.TestCase):
    def test_save_results_writes_label_images_table_overlay_and_parameters(self):
        image, _ = scene()
        parameters = dropseg.Parameters(**LEVELS, upsample=2)
        segmentation = dropseg.segment_image(image, parameters, tile_size=32)
        with tempfile.TemporaryDirectory() as folder:
            written = dropseg.save_results(folder, "scene", image, segmentation, parameters, scale=3)
            names = sorted(path.name for path in written)
            self.assertEqual(names, ["scene_boundaries.png", "scene_labels.tif", "scene_labels_2x.tif",
                                     "scene_parameters.json", "scene_segments.csv"])
            labels = np.asarray(Image.open(Path(folder) / "scene_labels.tif"))
            np.testing.assert_array_equal(labels, segmentation.original_labels)
            # The overlay scale is rounded up to a multiple of the upsampling.
            self.assertEqual(Image.open(Path(folder) / "scene_boundaries.png").size, (64 * 4, 48 * 4))
            record = json.loads((Path(folder) / "scene_parameters.json").read_text())
            self.assertEqual(record["segments"], segmentation.n_segments)

    def test_command_line_segments_and_reruns_from_its_parameters(self):
        image, _ = scene()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            Image.fromarray(np.round(image * 255).astype(np.uint8)).save(folder / "scene.png")
            with redirect_stdout(io.StringIO()):
                cli([str(folder / "scene.png"), "--out", str(folder / "first"), "--min-area", "3", "--min-bar-length", "0.2",
                     "--split-level", "0.3", "--min-level", "0.1",
                     "--split-lobes", "false", "--workers", "1"])
                cli([str(folder), "--out", str(folder / "second"), "--parameters",
                     str(folder / "first" / "scene_parameters.json"), "--workers", "1"])
            record = json.loads((folder / "second" / "scene_parameters.json").read_text())
            self.assertEqual(record["parameters"]["min_area"], 3.0)
            self.assertFalse(record["parameters"]["split_lobes"])
            first, second = (np.asarray(Image.open(folder / run / "scene_labels.tif")) for run in ("first", "second"))
            np.testing.assert_array_equal(first, second)
            self.assertEqual(len(np.unique(first)) - 1, 3)

    def test_command_line_infers_unset_levels_and_reruns_them_exactly(self):
        image, _ = scene()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            Image.fromarray(np.round(image * 255).astype(np.uint8)).save(folder / "scene.png")
            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                cli([str(folder / "scene.png"), "--out", str(folder / "first")])
                cli([str(folder / "scene.png"), "--out", str(folder / "second"), "--parameters",
                     str(folder / "first" / "scene_parameters.json")])
            self.assertEqual(output.getvalue().count("inferred from background"), 1)
            self.assertEqual(errors.getvalue().count("dropseg.Tuner"), 1)  # only the run that inferred
            first, second = (json.loads((folder / run / "scene_parameters.json").read_text())
                             for run in ("first", "second"))
            self.assertEqual(first["inferred"], ["min_bar_length", "split_level", "min_level"])
            self.assertEqual(second["inferred"], [])
            self.assertEqual(first["parameters"], second["parameters"])
            labels = [np.asarray(Image.open(folder / run / "scene_labels.tif")) for run in ("first", "second")]
            np.testing.assert_array_equal(*labels)

    def _run_with_nondefault_options(self, folder):
        """Segment a noisy color scene with nondefault channel, preprocessing and tiling; returns the labels."""
        droplets = [Droplet(14, 14, 6, 0.8), Droplet(34, 50, 5, 0.6), Droplet(30, 18, 6, 0.7), Droplet(30, 30, 6, 0.7)]
        gray = np.clip(droplet_image(droplets, (48, 64), blur=1.0, background=0.05, noise=0.04, seed=1), 0, 1)
        rgb = np.zeros((48, 64, 3), np.uint8)
        rgb[..., 1] = np.round(gray * 200)
        rgb[..., 0] = 255 - rgb[..., 1]  # a different channel, so the channel must be restored too
        Image.fromarray(rgb).save(folder / "scene.png")
        labels, _ = self._cli(folder, "first", "--channel", "1", "--normalize", "percentile", "--percentiles", "0.5",
                              "99.5", "--blur", "1", "--upsample", "2", "--tile-size", "24", "--tile-margin", "10",
                              "--min-area", "3")
        return labels

    def _cli(self, folder, out, *options):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli([str(folder / "scene.png"), "--out", str(folder / out), *options])
        return (np.asarray(Image.open(folder / out / "scene_labels.tif")),
                json.loads((folder / out / "scene_parameters.json").read_text()))

    def test_replaying_parameters_restores_channel_preprocessing_and_tiling(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            first = self._run_with_nondefault_options(folder)
            saved = json.loads((folder / "first" / "scene_parameters.json").read_text())
            self.assertEqual(saved["preprocessing"], {"normalize": "percentile", "low_percentile": 0.5,
                                                      "high_percentile": 99.5, "crop": None, "blur": 1.0})
            self.assertEqual((saved["channel"], saved["tile_size"], saved["tile_margin"]), (1, 24, 10))
            replay, record = self._cli(folder, "replay", "--parameters", str(folder / "first" / "scene_parameters.json"))
            np.testing.assert_array_equal(first, replay)
            for key in ("parameters", "preprocessing", "channel", "tile_size", "tile_margin"):
                self.assertEqual(record[key], saved[key], key)

    def test_options_given_override_the_replayed_ones(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            first = self._run_with_nondefault_options(folder)
            labels, record = self._cli(folder, "override", "--parameters",
                                       str(folder / "first" / "scene_parameters.json"), "--blur", "0")
            self.assertEqual(record["preprocessing"]["blur"], 0.0)
            self.assertEqual(record["preprocessing"]["high_percentile"], 99.5)  # the others are still restored
            self.assertEqual((record["channel"], record["tile_size"], record["tile_margin"]), (1, 24, 10))
            self.assertFalse(np.array_equal(first, labels))  # the blur changes this segmentation

    def test_replaying_an_older_file_without_the_options_uses_the_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self._run_with_nondefault_options(folder)
            older = json.loads((folder / "first" / "scene_parameters.json").read_text())
            for key in ("preprocessing", "channel", "tile_size", "tile_margin"):
                del older[key]
            (folder / "older.json").write_text(json.dumps(older))
            _, record = self._cli(folder, "older", "--parameters", str(folder / "older.json"), "--channel", "1")
            self.assertEqual(record["preprocessing"], {"normalize": "full scale", "low_percentile": 1.0,
                                                       "high_percentile": 99.9, "crop": None, "blur": 0.0})
            self.assertEqual((record["tile_size"], record["tile_margin"]), (64, 12))
            self.assertEqual(record["parameters"], older["parameters"])


@unittest.skipIf(MISSING, f"needs {MISSING}")
class MergeSmallLobesTests(unittest.TestCase):
    def test_small_lobe_joins_the_sibling_with_the_brightest_shared_edge(self):
        from dropseg.segments import merge_small_lobes
        labels = np.array([[1, 1, 2, 2, 2],
                           [3, 3, 2, 2, 2]])
        image = np.array([[0.9, 0.9, 0.8, 0.5, 0.5],
                          [0.3, 0.3, 0.5, 0.5, 0.5]])
        merged = merge_small_lobes(labels, image, [[1, 2, 3]], min_area=3)
        # Lobe 1 meets 2 at 0.8 and 3 at 0.3, so it joins 2; lobe 3, still too small, then joins 2 as well.
        self.assertEqual(set(np.unique(merged)), {2})

    def test_lobes_of_other_outlines_and_large_lobes_are_kept(self):
        from dropseg.segments import merge_small_lobes
        labels = np.array([[1, 1, 1, 2, 4]])
        image = np.full(labels.shape, 0.5)
        merged = merge_small_lobes(labels, image, [[1, 2]], min_area=2)
        np.testing.assert_array_equal(merged, [[1, 1, 1, 1, 4]])  # 4 belongs to no split; the filter handles it


@unittest.skipIf(MISSING or not HAS_WIDGETS, "needs the droplets extra with ipywidgets")
class TunerTests(unittest.TestCase):
    def test_every_step_draws_and_compares(self):
        image, _ = scene()
        tuner = dropseg.Tuner(image, (slice(4, 44), slice(4, 60)), dropseg.Parameters(**LEVELS, smooth_radius=0.5))
        for step in dropseg.STEPS:
            plt.close(tuner.figure(step))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            widget = tuner.step("lobes")
        self.assertGreater(len(widget.children), 2)
        plt.close(tuner.compare("min_level", [0.05, 0.1]))

    def test_unset_levels_are_inferred_from_the_whole_image(self):
        image, _ = scene()
        with warnings.catch_warnings():
            warnings.simplefilter("error", dropseg.InferredLevelsWarning)  # the tuner does not warn
            tuner = dropseg.Tuner(image, (slice(4, 24), slice(4, 24)), dropseg.Parameters(smooth_radius=0.5))
        with self.assertWarns(dropseg.InferredLevelsWarning):
            expected = dropseg.infer_parameters(image, dropseg.Parameters(smooth_radius=0.5))
        self.assertEqual(tuner.parameters, expected)


if __name__ == "__main__":
    unittest.main()
