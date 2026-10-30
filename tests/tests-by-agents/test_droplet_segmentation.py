"""The droplet segmentation application in applications/droplet_segmentation: API, tuner, files and CLI."""
import importlib.util
import io
import json
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
        segmentation = dropseg.segment_image(image, dropseg.Parameters())
        self.assertEqual(segmentation.n_segments, 4)
        self.assertEqual(count_found(segmentation.labels, truth), 4)
        # The pair was cut by the lobe splitting: its two segments share one outline group.
        self.assertEqual(len(np.unique(segmentation.groups[segmentation.groups > 0])), 3)

    def test_without_lobe_splitting_the_pair_is_one_segment(self):
        image, truth = scene()
        segmentation = dropseg.segment_image(image, dropseg.Parameters(split_lobes=False))
        self.assertEqual(segmentation.n_segments, 3)

    def test_upsampled_segmentation_agrees_on_original_pixels(self):
        image, truth = scene()
        segmentation = dropseg.segment_image(image, dropseg.Parameters(upsample=2), tile_size=32)
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
            segmentation = dropseg.segment_image(image, dropseg.Parameters(), tile_size=32, tile_margin=12)
        finally:
            whole_image.LARGE_IMAGE = limit
        self.assertIsNone(segmentation.result)
        self.assertEqual(len(segmentation.tiles), 4)
        self.assertEqual(count_found(segmentation.labels, truth), 4)

    def test_runs_are_identical(self):
        image, _ = scene()
        first = dropseg.segment_image(image, dropseg.Parameters())
        second = dropseg.segment_image(image, dropseg.Parameters())
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_stages_on_the_whole_image_match_segment_image(self):
        image, _ = scene()
        parameters = dropseg.Parameters(min_level=0.08)
        stages = Stages(image, (slice(0, image.shape[0]), slice(0, image.shape[1])))
        np.testing.assert_array_equal(stages.result(parameters).labels,
                                      dropseg.segment_image(image, parameters).labels)

    def test_rejects_unscaled_images(self):
        image, _ = scene()
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            dropseg.segment_image(image * 255, dropseg.Parameters())
        with self.assertRaisesRegex(ValueError, "2D"):
            dropseg.segment_image(image[None], dropseg.Parameters())


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
        settings = dropseg.Parameters(upsample=4, smooth_radius=0.5, min_area=2.0).settings()
        self.assertEqual(settings.smooth_radius, 2.0)
        self.assertEqual(settings.min_area, 32.0)
        self.assertAlmostEqual(settings.birth_offset, 0.1 * 0.2)


@unittest.skipIf(MISSING, f"needs {MISSING}")
class FilesAndCommandLineTests(unittest.TestCase):
    def test_save_results_writes_label_images_table_overlay_and_parameters(self):
        image, _ = scene()
        parameters = dropseg.Parameters(upsample=2)
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
                cli([str(folder / "scene.png"), "--out", str(folder / "first"), "--min-area", "3",
                     "--split-lobes", "false", "--workers", "1"])
                cli([str(folder), "--out", str(folder / "second"), "--parameters",
                     str(folder / "first" / "scene_parameters.json"), "--workers", "1"])
            record = json.loads((folder / "second" / "scene_parameters.json").read_text())
            self.assertEqual(record["parameters"]["min_area"], 3.0)
            self.assertFalse(record["parameters"]["split_lobes"])
            first, second = (np.asarray(Image.open(folder / run / "scene_labels.tif")) for run in ("first", "second"))
            np.testing.assert_array_equal(first, second)
            self.assertEqual(len(np.unique(first)) - 1, 3)


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
        tuner = dropseg.Tuner(image, (slice(4, 44), slice(4, 60)), dropseg.Parameters(smooth_radius=0.5))
        for step in dropseg.STEPS:
            plt.close(tuner.figure(step))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            widget = tuner.step("lobes")
        self.assertGreater(len(widget.children), 2)
        plt.close(tuner.compare("min_level", [0.05, 0.1]))


if __name__ == "__main__":
    unittest.main()
