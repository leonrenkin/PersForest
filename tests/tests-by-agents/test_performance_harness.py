"""Benchmark plumbing only: no real alpha/forest/landscape computation."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "performance"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BENCHMARK_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("performance_runner_test", "run.py")
sampling = _load("performance_sampling_test", "sampling.py")
with patch.dict(sys.modules, {"run": runner}):
    plotting = _load("performance_plotting_test", "plot.py")


def _config(schedule="pilot", **overrides):
    config = dict(schedule=schedule, kinds=list(runner.KINDS), dimensions=[2, 3], samplers=None,
                  repeats=3, base_seed=12345, max_k=5,
                  grid_points=512, signed=False)
    for kind in runner.KINDS:
        for dim in (2, 3):
            config[f"{kind}_sizes_{dim}d"] = None
    config.update(overrides)
    return config


# The runner measures RSS with Unix-only APIs and supports only macOS and Linux.
@unittest.skipUnless(sys.platform == "darwin" or sys.platform.startswith("linux"),
                     "the benchmark runner supports macOS and Linux only")
class PerformanceHarnessTests(unittest.TestCase):
    def test_metadata_hashes_current_src_including_nested_modules(self):
        with patch.dict(sys.modules, {"sampling": sampling}):
            metadata = runner._metadata()
        hashes = metadata["source_hashes"]
        for name in ("src/persforest/forest.py",
                     "src/persforest/plotting/animation/_export.py"):
            self.assertEqual(hashes[name],
                             hashlib.sha256((runner.ROOT / name).read_bytes()).hexdigest())
        self.assertFalse(any(name.startswith("persforest/") for name in hashes))

    def test_preview_and_resume_without_launching_scientific_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            arguments = ["run.py", "--kinds", "points", "--dimensions", "2",
                         "--points-sizes-2d", "3", "--samplers", "uniform", "--repeats", "1",
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
                self.assertEqual(supervisor.call_count, 1)
                with patch.object(sys, "argv", arguments + ["--run", "--resume"]):
                    runner.main()
                self.assertEqual(supervisor.call_count, 1)
                with patch.object(sys, "argv", arguments + ["--run", "--resume", "--grid-points", "256"]), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        runner.main()

    def test_schedule_caps_and_pairing(self):
        for schedule in runner.SCHEDULES:
            cases = runner.build_cases(_config(schedule))
            self.assertEqual(len({case["case_id"] for case in cases}), len(cases))
            self.assertEqual(cases, runner.build_cases(_config(schedule)))
            for (kind, dim), entry in runner.SCHEDULES[schedule].items():
                group = [case for case in cases if (case["kind"], case["dim"]) == (kind, dim)]
                samplers = sampling.POINT_SAMPLERS if kind == "points" else sampling.GRID_SAMPLERS
                self.assertEqual({case["sampler"] for case in group}, set(samplers))
                # Barcodes run at every size; landscapes only up to their cap.
                self.assertEqual({case["size"] for case in group}, set(entry["sizes"]))
                for case in group:
                    self.assertEqual(case["landscape"], case["size"] <= entry["landscape_max"])
                    self.assertEqual(case["n_input"], case["size"] if kind == "points" else case["size"] ** dim)
            # One case, with its own seed, per input family, size and repeat.
            keys = [(case["kind"], case["dim"], case["size"], case["sampler"], case["repeat"]) for case in cases]
            self.assertEqual(len(keys), len(set(keys)))
        custom = runner.build_cases(_config(kinds=["cubical"], dimensions=[3],
                                            samplers=["holes"], repeats=1, cubical_sizes_3d=[20, 10]))
        self.assertEqual([case["size"] for case in custom], [10, 20])

    def test_sampler_shape_seed_and_global_rng(self):
        before = np.random.get_state()
        for sampler in sampling.POINT_SAMPLERS:
            for dim in (2, 3):
                points = sampling.sample_points(sampler, 23, dim, 91)
                self.assertEqual(points.shape, (23, dim))
                self.assertTrue(np.isfinite(points).all())
                np.testing.assert_array_equal(points, sampling.sample_points(sampler, 23, dim, 91))
                if sampler != "noisy_sphere":
                    self.assertTrue(((points >= 0) & (points <= 1)).all())
        for sampler in sampling.GRID_SAMPLERS:
            for dim in (2, 3):
                values = sampling.sample_grid(sampler, 9, dim, 91)
                self.assertEqual(values.shape, (9,) * dim)
                self.assertTrue(np.isfinite(values).all())
                np.testing.assert_array_equal(values, sampling.sample_grid(sampler, 9, dim, 91))
        # The point hole family keeps the September 2026 clouds for a given seed.
        rng = np.random.default_rng(5)
        radii = rng.uniform(0.0, 0.05, size=30)
        centers = rng.uniform(radii[:, None], 1.0 - radii[:, None], (30, 2))
        balls = sampling._hole_balls(np.random.default_rng(5), 2)
        np.testing.assert_array_equal(balls[0], centers)
        np.testing.assert_array_equal(balls[1], radii)
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
        import persforest
        import persforest._forest.arrays as arrays_module
        import persforest.forest as forest_module

        calls = []

        class FakeForest:
            @classmethod
            def _make(cls, data, prepare_name, **kwargs):
                forest = cls()
                getattr(forest_module, prepare_name)(data)
                arrays_module.build_forest_arrays(None, reduce=True, filtration_tol=1e-12)
                forest.compute_barcode()
                forest.barcode = [SimpleNamespace(birth=0.1, death=0.5)]
                forest._cells = SimpleNamespace(event_cells=[0])
                forest._forest = SimpleNamespace(node_value=[1.0])
                return forest

            @classmethod
            def from_points(cls, points, **kwargs):
                return cls._make(points, "prepare_point_filtration", **kwargs)

            @classmethod
            def from_cubical_grid(cls, values, **kwargs):
                return cls._make(values, "prepare_cubical_filtration", **kwargs)

            def compute_barcode(self):
                calls.append("barcode")

            def compute_measurement_landscapes(self, function, label, **kwargs):
                calls.append(("landscape", function.__name__))
                if kwargs["cache"] or kwargs["cache_profiles"]:
                    raise AssertionError("Unexpected functional reuse")
                return SimpleNamespace(landscapes={1: None}, x_grid=[0, 1])

        def fake_prepare(data):
            calls.append("filtration")

        def fake_engine(cells, **kwargs):
            calls.append("forest")

        config = _config(dimensions=[2], samplers=["uniform"], repeats=1,
                         points_sizes_2d=[3], cubical_sizes_2d=[3])
        with tempfile.TemporaryDirectory() as temporary:
            spec_path = Path(temporary) / "spec.json"
            checkpoint = Path(temporary) / "checkpoint.json"
            cases = runner.build_cases(config)
            self.assertEqual({case["kind"] for case in cases}, set(runner.KINDS))
            for case in cases:
                calls.clear()
                spec_path.write_text(json.dumps(case))
                with patch.object(persforest, "PersistenceForest", FakeForest), \
                     patch.object(forest_module, "prepare_point_filtration", fake_prepare), \
                     patch.object(forest_module, "prepare_cubical_filtration", fake_prepare), \
                     patch.object(arrays_module, "build_forest_arrays", fake_engine), \
                     patch.dict(sys.modules, {"sampling": sampling}):
                    self.assertEqual(runner._worker(spec_path, checkpoint), 0)
                result = json.loads(checkpoint.read_text())
                self.assertEqual(result["status"], "ok")
                self.assertEqual(calls.count("filtration"), 1)
                self.assertEqual(calls.count("forest"), 1)
                self.assertEqual(calls.count("barcode"), 1)
                functional = "signed_chain_volume" if case["kind"] == "points" else "cubical_chain_volume"
                self.assertEqual(calls.count(("landscape", functional)), 1)
                self.assertIn("time_forest_build_s", result)
                self.assertAlmostEqual(result["time_to_barcode_s"], result["time_filtration_s"] +
                                       result["time_forest_s"] + result["time_barcode_s"])
                self.assertAlmostEqual(result["time_persforest_s"],
                                       result["time_to_barcode_s"] - result["time_filtration_s"])
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

    def test_plots_and_summary_with_synthetic_records(self):
        cases = runner.build_cases(_config(dimensions=[2], repeats=2, points_sizes_2d=[10, 20],
                                           cubical_sizes_2d=[4, 8]))
        rows = []
        for case in cases:
            rows.append(dict(case, status="ok", time_filtration_s=1.0, time_forest_s=2.0,
                             time_barcode_s=1.0, time_persforest_s=3.0,
                             time_to_barcode_s=4.0, time_landscape_s=1.0,
                             time_persforest_and_landscape_s=4.0, n_bars=5, n_events=9,
                             peak_rss_to_barcode_bytes=1024**3, peak_rss_bytes=2 * 1024**3))
        # A landscape timeout keeps the barcode timing but not the landscape.
        rows.append(dict(rows[0], status="timeout", repeat=2, case_id="timed_out",
                         time_persforest_s=6.0, time_landscape_s=99.0))
        summary = plotting.summarize(rows)
        first = next(row for row in summary if all(row[key] == rows[0][key] for key in
                                                   ("kind", "dim", "sampler", "size")))
        self.assertEqual((first["n_attempts"], first["n_ok"]), (3, 2))
        self.assertEqual(first["time_persforest_s_max"], 6.0)
        self.assertEqual(first["time_landscape_s_max"], 1.0)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary)
            plotting.print_table(summary)
            paths = plotting.make_plots(summary, output, "synthetic")
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.stat().st_size > 100 for path in paths))
        import matplotlib.pyplot as plt
        self.assertEqual(plt.get_fignums(), [])

if __name__ == "__main__":
    unittest.main()
