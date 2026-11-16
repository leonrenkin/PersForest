"""nucseg on the synthetic stack: 2D and 3D segmentation, reading and writing files, and the command line.

Run from the application folder with ``.venv/bin/python -m unittest discover tests``.
"""

from contextlib import redirect_stderr
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
import io
import json
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
import tifffile

import nucseg
from nucseg.__main__ import main as cli
from nucseg.synthetic import NUCLEI
from nucseg.synthetic import UM_PER_PIXEL
from nucseg.synthetic import Z_STEP_UM

# The defaults (bin_factor 4) give 0.26 µm binned pixels on the synthetic 0.065 µm grid.
PARAMETERS = nucseg.Parameters()


def nearest_nucleus(x_um: float, y_um: float) -> int:
    """Index in ``NUCLEI`` of the nucleus whose centre is nearest to ``(x, y)``."""
    return int(np.argmin([np.hypot(n.centre[2] - x_um, n.centre[1] - y_um) for n in NUCLEI]))


class Segment2DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = nucseg.synthetic_stack()
        cls.plain = nucseg.segment_2d(cls.stack, PARAMETERS)
        cls.framed = nucseg.segment_2d(cls.stack, replace(PARAMETERS, border_frame=True))

    def test_finds_every_nucleus_inside_the_image_once(self):
        rows = self.plain.table()
        self.assertEqual(len(rows), 4)
        found = sorted(nearest_nucleus(r["centroid_x_um"], r["centroid_y_um"]) for r in rows)
        self.assertEqual(found, [0, 1, 2, 3])
        for row in rows:
            centre = NUCLEI[nearest_nucleus(row["centroid_x_um"], row["centroid_y_um"])].centre
            self.assertLess(np.hypot(centre[2] - row["centroid_x_um"], centre[1] - row["centroid_y_um"]), 1.0)
            self.assertFalse(row["border"])

    def test_areas_lie_between_inner_and_outer_equator(self):
        # The outline runs on the envelope (sigma 0.2 µm, blurred), so the area
        # lies within a band of about +-1 µm around the equatorial ellipse.
        for row in self.plain.table():
            nucleus = NUCLEI[nearest_nucleus(row["centroid_x_um"], row["centroid_y_um"])]
            _, b, a = nucleus.semi_axes
            self.assertGreater(row["area_um2"], np.pi * (a - 1.2) * (b - 1.2))
            self.assertLess(row["area_um2"], np.pi * (a + 0.5) * (b + 0.5))

    def test_touching_nuclei_stay_apart(self):
        self.assertTrue(all(r["pieces"] == 1 for r in self.plain.table()))

    def test_border_frame_adds_the_cut_nucleus(self):
        rows = self.framed.table()
        border = [r for r in rows if r["border"]]
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(border), 1)
        self.assertEqual(nearest_nucleus(border[0]["centroid_x_um"], border[0]["centroid_y_um"]), 4)

    def test_labels_follow_the_table(self):
        labels = self.plain.labels()
        self.assertEqual(labels.shape, self.stack.data.shape[1:])
        self.assertEqual(sorted(np.unique(labels)), [0, 1, 2, 3, 4])
        pixel_area = UM_PER_PIXEL**2
        for row in self.plain.table():
            self.assertAlmostEqual((labels == row["label"]).sum() * pixel_area, row["area_um2"], delta=0.01)
        areas = [r["area_um2"] for r in self.plain.table()]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_postprocess_without_merging_keeps_the_pieces(self):
        unmerged = nucseg.postprocess(self.plain, merge=False)
        self.assertEqual(len(unmerged.nuclei), len(self.plain.pieces))
        with self.assertRaises(ValueError):
            nucseg.postprocess(self.plain, min_lifetime=0.1)
        with self.assertRaises(ValueError):
            nucseg.postprocess(self.plain, clean_outlines=True)

    def test_single_image(self):
        image = nucseg.Stack(self.stack.data[9], UM_PER_PIXEL)
        segmentation = nucseg.segment_2d(image, PARAMETERS)
        self.assertEqual(list(segmentation.images), [(0, 1)])
        self.assertEqual(len(segmentation.nuclei), 4)


class WindowTests(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(nucseg.Parameters(z_first=4, z_last=14).windows(20),
                         [(4, 7), (6, 9), (8, 11), (10, 13), (12, 15)])
        self.assertEqual(nucseg.Parameters().windows(5), [(0, 3), (2, 5)])
        self.assertEqual(nucseg.Parameters().windows(2), [(0, 2)])
        with self.assertRaises(ValueError):
            nucseg.Parameters(z_last=20).windows(20)


class Segment3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = nucseg.synthetic_stack()
        cls.segmentation = nucseg.segment_2d(cls.stack, replace(PARAMETERS, border_frame=True))
        cls.result = nucseg.segment_3d(cls.stack, cls.segmentation)

    def test_every_nucleus_closes_without_caps(self):
        rows = self.result.table()
        self.assertEqual([r["label"] for r in rows], [1, 2, 3, 4, 5])
        for row in rows:
            self.assertTrue(row["found"])
            self.assertFalse(row["top_cap"] or row["bottom_cap"])
            nucleus = NUCLEI[nearest_nucleus(row["centroid_x_um"], row["centroid_y_um"])]
            # The void ends at the inner edge of the blurred envelope: smaller
            # than the ellipsoid, centred on it in z.
            volume = 4 / 3 * np.pi * np.prod(nucleus.semi_axes)
            self.assertLess(row["volume_um3"], volume)
            self.assertGreater(row["volume_um3"], 0.15 * volume)
            self.assertLess(abs(row["centroid_z_um"] - nucleus.centre[0]), 0.8)

    def test_border_nucleus(self):
        border = [n for n in self.result.nuclei if n.border]
        self.assertEqual(len(border), 1)
        self.assertTrue(border[0].found)

    def test_labels(self):
        binned = self.result.labels()
        full = self.result.labels(full_resolution=True)
        self.assertEqual(full.shape, self.stack.data.shape)
        self.assertEqual(binned.shape, self.result.image.shape)
        self.assertEqual(set(np.unique(full)), {0, 1, 2, 3, 4, 5})
        np.testing.assert_array_equal(full[:, ::4, ::4], binned)

    def test_needs_a_z_step(self):
        flat = nucseg.Stack(self.stack.data, UM_PER_PIXEL, None)
        with self.assertRaisesRegex(ValueError, "z step"):
            nucseg.segment_3d(flat, self.segmentation)
        result = nucseg.segment_3d(flat, self.segmentation, nucseg.Parameters3D(z_step_um=Z_STEP_UM))
        self.assertEqual(result.z_step_um, Z_STEP_UM)

    def test_rejects_another_stack(self):
        other = nucseg.Stack(self.stack.data[:, :400], UM_PER_PIXEL, Z_STEP_UM)
        with self.assertRaises(ValueError):
            nucseg.segment_3d(other, self.segmentation)

    def test_surface_mesh_is_closed(self):
        from nucseg.surfaces import nucleus_mesh

        vertices, triangles = nucleus_mesh(self.result, self.result.nuclei[0], mesh_iterations=0)
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(counts % 2 == 0))  # every edge borders an even number of triangles


class ReadStackTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name)
        self.data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

    def tearDown(self):
        self.folder.cleanup()

    def test_imagej_voxel_size(self):
        file = self.path / "a.tif"
        tifffile.imwrite(file, self.data, imagej=True, resolution=(1 / 0.065, 1 / 0.065),
                         metadata={"unit": "micron", "spacing": 0.3, "axes": "ZYX"})
        stack = nucseg.read_stack(file)
        np.testing.assert_array_equal(stack.data, self.data)
        self.assertAlmostEqual(stack.um_per_pixel, 0.065, places=6)
        self.assertAlmostEqual(stack.z_step_um, 0.3)
        self.assertEqual(stack.name, "a")

    def test_ome_voxel_size(self):
        file = self.path / "a.ome.tif"
        tifffile.imwrite(file, self.data, ome=True, metadata={"axes": "ZYX", "PhysicalSizeX": 0.1,
                                                             "PhysicalSizeY": 0.1, "PhysicalSizeZ": 0.5})
        stack = nucseg.read_stack(file)
        self.assertAlmostEqual(stack.um_per_pixel, 0.1)
        self.assertAlmostEqual(stack.z_step_um, 0.5)

    def test_no_voxel_size(self):
        file = self.path / "plain.tif"
        tifffile.imwrite(file, self.data, photometric="minisblack")
        with self.assertRaisesRegex(ValueError, "um_per_pixel"):
            nucseg.read_stack(file)
        stack = nucseg.read_stack(file, um_per_pixel=0.2)
        self.assertEqual(stack.data.shape, (3, 4, 5))
        self.assertIsNone(stack.z_step_um)

    def test_channels_are_never_mixed_into_z(self):
        file = self.path / "hyper.tif"
        data = np.stack([self.data, self.data + 1000], axis=1)  # ZCYX
        tifffile.imwrite(file, data, imagej=True, resolution=(10, 10), metadata={"unit": "um", "axes": "ZCYX"})
        with self.assertRaisesRegex(ValueError, "2 channels"):
            nucseg.read_stack(file)
        np.testing.assert_array_equal(nucseg.read_stack(file, channel=1).data, self.data + 1000)
        with self.assertRaises(ValueError):
            nucseg.read_stack(file, channel=2)

    def test_time_series_is_rejected(self):
        file = self.path / "time.tif"
        tifffile.imwrite(file, np.stack([self.data, self.data]), imagej=True, resolution=(10, 10),
                         metadata={"unit": "um", "axes": "TZYX"})
        with self.assertRaisesRegex(ValueError, "time points"):
            nucseg.read_stack(file)


class FilesAndCommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.input = nucseg.write_synthetic_stack(cls.root / "data" / "synthetic.tif")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_cli(self, *args: str) -> int:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cli(list(args))

    def test_written_stack_reads_back(self):
        stack = nucseg.read_stack(self.input)
        np.testing.assert_array_equal(stack.data, nucseg.synthetic_stack().data)
        self.assertAlmostEqual(stack.um_per_pixel, UM_PER_PIXEL, places=6)
        self.assertAlmostEqual(stack.z_step_um, Z_STEP_UM)

    def test_command_line_2d_and_3d(self):
        out = self.root / "out"
        code = self.run_cli(str(self.input.parent), "--out", str(out), "--min-lifetime", "0.1",
                            "--border-frame", "true", "--3d", "--3d-margin-um", "2.5")
        self.assertEqual(code, 0)
        names = {p.name for p in out.iterdir()}
        expected = {f"synthetic{s}" for s in ("_labels.tif", "_nuclei.csv", "_parameters.json", "_overview.png",
                                              "_gallery.png", "_labels_3d.tif", "_nuclei_3d.csv",
                                              "_sections_3d.png")}
        self.assertEqual(names, expected)
        record = json.loads((out / "synthetic_parameters.json").read_text())
        self.assertEqual(record["parameters"]["min_lifetime"], 0.1)
        self.assertTrue(record["parameters"]["border_frame"])
        self.assertEqual(record["parameters_3d"]["margin_um"], 2.5)
        self.assertAlmostEqual(record["z_step_um"], Z_STEP_UM)

        with tifffile.TiffFile(out / "synthetic_labels_3d.tif") as tif:
            labels_3d = tif.asarray()
            self.assertAlmostEqual(tif.imagej_metadata["spacing"], Z_STEP_UM)
        self.assertEqual(labels_3d.shape, nucseg.synthetic_stack().data.shape)
        self.assertEqual(int(labels_3d.max()), 5)
        labels = nucseg.read_stack(out / "synthetic_labels.tif")
        self.assertAlmostEqual(labels.um_per_pixel, UM_PER_PIXEL, places=6)
        lines = (out / "synthetic_nuclei.csv").read_text().splitlines()
        self.assertEqual(len(lines), 6)

        # The parameters file reproduces the run; options override it.
        again = self.root / "again"
        self.assertEqual(self.run_cli(str(self.input), "--out", str(again), "--parameters",
                                      str(out / "synthetic_parameters.json"), "--no-figures", "--3d"), 0)
        self.assertEqual((again / "synthetic_nuclei.csv").read_text(), (out / "synthetic_nuclei.csv").read_text())
        self.assertEqual(json.loads((again / "synthetic_parameters.json").read_text())["parameters_3d"]["margin_um"],
                         2.5)

    def test_3d_options_need_3d(self):
        with self.assertRaises(SystemExit):
            self.run_cli(str(self.input), "--out", str(self.root / "x"), "--3d-margin-um", "2")

    def test_failures_are_reported(self):
        bad = self.root / "bad"
        bad.mkdir()
        tifffile.imwrite(bad / "no_size.tif", np.zeros((2, 8, 8), np.uint16), photometric="minisblack")
        self.assertEqual(self.run_cli(str(bad), "--out", str(self.root / "bad_out")), 1)


if __name__ == "__main__":
    unittest.main()
