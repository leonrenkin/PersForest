"""Benchmark plumbing only: no real alpha/forest/landscape computation."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "examples" / "performance"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("performance_runner_test", "run.py")
sampling = _load("performance_sampling_test", "sampling.py")
with patch.dict(sys.modules, {"run": runner}):
    plotting = _load("performance_plotting_test", "plot.py")


def _config(suite="barcode"):
    return dict(suite=suite, dimensions=[2, 3], sizes_2d=None, sizes_3d=None,
                samplers=["uniform", "noisy_sphere", "holes"], modes=list(runner.MODES),
                repeats=3, base_seed=12345, max_k=5, grid_points=512, signed=False)


class PerformanceHarnessTests(unittest.TestCase):
    def test_preview_and_resume_without_launching_scientific_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            arguments = ["run.py", "--dimensions", "2", "--sizes-2d", "3",
                         "--samplers", "uniform", "--repeats", "1",
                         "--output", str(output)]
            with patch.object(sys, "argv", arguments), \
                 patch.object(runner, "_supervise") as supervisor, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                supervisor.assert_not_called()
                self.assertFalse(output.exists())

            metadata = {key: "fixture" for key in
                        ("python", "executable", "platform", "machine", "packages",
                         "source_hashes", "sampler_parameters", "thread_policy")}

            def fake_run(case, directory, config):
                return dict(case, status="ok")

            with patch.object(runner, "_metadata", return_value=metadata), \
                 patch.object(runner, "_supervise", side_effect=fake_run) as supervisor, \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", arguments + ["--run"]):
                    runner.main()
                self.assertEqual(supervisor.call_count, 4)
                with patch.object(sys, "argv", arguments + ["--run", "--resume"]):
                    runner.main()
                self.assertEqual(supervisor.call_count, 4)
                with patch.object(sys, "argv", arguments + ["--run", "--resume", "--grid-points", "256"]), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        runner.main()

    def test_schedule_and_pairing(self):
        cases = runner.build_cases(_config())
        self.assertEqual(len(cases), 504)
        self.assertEqual(len({case["case_id"] for case in cases}), len(cases))
        self.assertEqual(cases, runner.build_cases(_config()))
        for dim, maximum in ((2, 150000), (3, 20000)):
            self.assertEqual(max(case["n_points"] for case in cases if case["dim"] == dim), maximum)
        cloud_seeds = {}
        for case in cases:
            key = (case["dim"], case["n_points"], case["sampler"], case["repeat"])
            cloud_seeds.setdefault(key, set()).add(case["seed"])
        self.assertTrue(all(len(seeds) == 1 for seeds in cloud_seeds.values()))
        landscape = runner.build_cases(_config("landscapes"))
        self.assertTrue(all(case["functional"] != "edge_length" for case in landscape if case["dim"] == 3))
        extension = runner.build_cases(_config("diff-scaling"))
        self.assertEqual({case["mode"] for case in extension}, {"diff_only_mode"})
        self.assertGreater(max(case["n_points"] for case in extension if case["dim"] == 3), 25000)

    def test_sampler_shape_seed_and_global_rng(self):
        before = np.random.get_state()
        for sampler in sampling.SAMPLER_PARAMETERS:
            for dim in (2, 3):
                points = sampling.sample_points(sampler, 23, dim, 91)
                self.assertEqual(points.shape, (23, dim))
                self.assertTrue(np.isfinite(points).all())
                np.testing.assert_array_equal(points, sampling.sample_points(sampler, 23, dim, 91))
                if sampler != "noisy_sphere":
                    self.assertTrue(((points >= 0) & (points <= 1)).all())
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_peak_units_and_truncated_result(self):
        with patch.object(runner.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=123)):
            with patch.object(runner.sys, "platform", "darwin"):
                self.assertEqual(runner._peak_bytes(), 123)
            with patch.object(runner.sys, "platform", "linux"):
                self.assertEqual(runner._peak_bytes(), 123 * 1024)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            path.write_text('{"case_id":"a","status":"error"}\n'
                            '{"case_id":"a","status":"ok"}\n{"case_id":')
            self.assertEqual(runner.load_results(path), [{"case_id": "a", "status": "ok"}])

    def test_worker_with_fake_scientific_objects(self):
        # Imports are allowed; real scientific constructors are never invoked.
        import gudhi
        import persforest

        calls = []

        class FakeAlpha:
            def __init__(self, points):
                calls.append("alpha")

            def create_simplex_tree(self, **kwargs):
                self.assert_radius = kwargs["output_squared_values"] is False
                return SimpleNamespace(num_simplices=lambda: 7)

        class FakeForest:
            def __init__(self, points, **kwargs):
                self.simplex_tree = gudhi.AlphaComplex(points).create_simplex_tree(output_squared_values=False)
                self._compute_forest()
                self._reduce_forest()
                if not kwargs.get("diff_only_mode"):
                    self._compute_loop_activity()
                self.compute_barcode_diff() if kwargs.get("keep_simplex_diff") else self.compute_barcode_cycles()
                if kwargs.get("compute_interior"):
                    self._add_interior_to_barcode()
                self.barcode = [SimpleNamespace(birth=0.1, death=0.5)]
                self.nodes = {1: None}
                self.filtration = [([0], 0)]

            def _compute_forest(self):
                calls.append("forest")

            def _reduce_forest(self):
                calls.append("reduce")

            def _compute_loop_activity(self):
                calls.append("activity")

            def compute_barcode_cycles(self):
                calls.append("cycles")

            def compute_barcode_diff(self):
                calls.append("diff")

            def _add_interior_to_barcode(self):
                calls.append("interior")

            def compute_measurement_landscapes(self, function, label, **kwargs):
                calls.append("landscape")
                if kwargs["cache"] or kwargs["cache_functionals"] or not kwargs["compute_functionals"]:
                    raise AssertionError("Unexpected functional reuse")
                return SimpleNamespace(landscapes={1: None}, x_grid=[0, 1])

        config = _config("landscapes")
        config.update(dimensions=[2], sizes_2d=[3], samplers=["uniform"], repeats=1)
        with tempfile.TemporaryDirectory() as temporary:
            spec_path = Path(temporary) / "spec.json"
            checkpoint = Path(temporary) / "checkpoint.json"
            for case in runner.build_cases(config):
                calls.clear()
                spec_path.write_text(json.dumps(case))
                with patch.object(persforest, "PersistenceForest", FakeForest), \
                     patch.object(gudhi, "AlphaComplex", FakeAlpha), \
                     patch.dict(sys.modules, {"sampling": sampling}):
                    self.assertEqual(runner._worker(spec_path, checkpoint), 0)
                result = json.loads(checkpoint.read_text())
                self.assertEqual(result["status"], "ok")
                self.assertEqual(calls.count("forest"), 1)
                self.assertEqual(calls.count("landscape"), 1)
                self.assertEqual("interior" in calls, case["mode"] == "compute_interior")
                self.assertEqual("activity" in calls, case["mode"] != "diff_only_mode")
                self.assertAlmostEqual(result["time_to_barcode_s"], result["time_alpha_s"] +
                                       result["time_forest_s"] + result["time_barcode_with_interior_s"])
                self.assertAlmostEqual(result["time_pipeline_s"], result["time_to_barcode_s"] + result["time_landscape_s"])
                self.assertGreaterEqual(result["peak_rss_bytes"], result["peak_rss_to_barcode_bytes"])

    def test_watchdog_stops_worker(self):
        class FakeProcess:
            pid = 123
            returncode = None

            def poll(self):
                return self.returncode

            def kill(self):
                self.returncode = -9

            def wait(self):
                return self.returncode

        case = runner.build_cases(_config())[0]
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runner.subprocess, "Popen", return_value=FakeProcess()), \
             patch.object(runner, "_rss_bytes", return_value=2 * 1024**3):
            result = runner._supervise(case, Path(temporary),
                                      dict(memory_limit_gib=1, timeout_s=10, poll_seconds=0.25))
        self.assertEqual(result["status"], "memory_limit")
        self.assertEqual(result["exit_code"], -9)

    def test_plots_and_failure_accounting_with_synthetic_records(self):
        cases = runner.build_cases(dict(_config("landscapes"), dimensions=[2],
                                        sizes_2d=[10], samplers=["uniform"], repeats=1))
        rows = []
        for case in cases:
            rows.append(dict(case, status="ok", time_to_barcode_s=4.0,
                             time_to_barcode_excluding_alpha_s=3.0, time_forest_s=2.0,
                             time_barcode_with_interior_s=1.0, time_landscape_s=1.0,
                             time_pipeline_s=5.0, alpha_pct_to_barcode=25.0,
                             alpha_pct_pipeline=20.0, landscape_pct_pipeline=20.0,
                             landscape_pct_excluding_alpha=25.0,
                             peak_rss_to_barcode_bytes=1024**3, peak_rss_bytes=2 * 1024**3,
                             point_cloud_sha256="cloud", barcode_endpoints_sha256="bars"))
        rows.append(dict(rows[0], status="memory_limit", repeat=1, case_id="failed"))
        self.assertEqual(sum(row["n_incomplete"] for row in plotting.summarize(rows)), 1)
        paired = plotting.paired_ratios(rows)
        self.assertEqual(len(paired), 2)
        self.assertTrue(all(row["time_forest_s_standard_over_diff"] == 1 for row in paired))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = plotting.make_plots(rows, output, 32)
            self.assertEqual(len(paths), 10)
            self.assertTrue(all(path.stat().st_size > 100 for path in paths))
            self.assertTrue((output / "summary.csv").exists())
            self.assertTrue((output / "failures.csv").exists())
            plotting._write_csv(output / "failures.csv", [])
            self.assertEqual((output / "failures.csv").read_text(), "")
        import matplotlib.pyplot as plt
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
