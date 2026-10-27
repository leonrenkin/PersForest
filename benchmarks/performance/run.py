"""Serial, isolated runtime/RSS benchmarks. No computations without --run.

See README.md for definitions, schedules, watchdog limitations and examples.
Only the worker imports the scientific stack; the supervisor remains small.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
import traceback
from typing import Callable, Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MODES = {
    "diffs": {"storage": "diffs"},
    "cycles": {"storage": "cycles"},
}
SCHEDULES = {
    "barcode": {2: [1000, 5000, 10000, 25000, 50000, 100000],
                3: [500, 1000, 2500, 5000, 10000, 15000, 20000]},
    "landscapes": {2: [500, 1000, 2500, 5000, 10000, 20000],
                   3: [250, 500, 1000, 2500, 5000]},
    "diff-scaling": {2: [1000, 5000, 10000, 25000, 50000, 100000, 150000, 200000, 300000, 400000],
                     3: [500, 1000, 2500, 5000, 10000, 15000,20000, 25000, 50000, 75000]},
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _peak_bytes() -> int:
    # macOS returns bytes; Linux returns KiB. Both include native allocations.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class _Recorder:
    def __init__(self, path: Path, result: dict):
        self.path = path
        self.result = result
        self.overhead_s = 0.0

    def checkpoint(self, stage: str) -> None:
        start = time.perf_counter()
        self.result.update(stage=stage, peak_rss_bytes=_peak_bytes())
        _write_json(self.path, self.result)
        self.overhead_s += time.perf_counter() - start

    def call(self, stage: str, function: Callable, *args: Any, **kwargs: Any) -> Any:
        self.checkpoint(stage)
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            key = f"time_{stage}_s"
            self.result[key] = self.result.get(key, 0.0) + elapsed
            self.result[f"peak_rss_through_{stage}_bytes"] = _peak_bytes()
            self.checkpoint(stage + "_complete")


def _instrumented_class(base: type, recorder: _Recorder) -> type:
    """Wrap existing phase methods without changing their calls or arguments.

    Parameters
    ----------
    base : type
        PersistenceForest class.
    recorder : _Recorder
        Worker-local timing/checkpoint sink.

    Returns
    -------
    type
        Benchmark-only subclass, retaining the normal constructor path.
    """
    methods = {}
    for name, stage in {
        "_compute_forest": "forest_build",
        "_reduce_forest": "reduction",
        "_compute_loop_activity": "activity",
        "_compute_barcode_diff": "barcode",
        "_add_interior_to_barcode": "interior",
    }.items():
        original = getattr(base, name)

        def wrapped(self, *args, _original=original, _stage=stage, **kwargs):
            if _stage == "barcode":
                recorder.result["peak_rss_to_forest_bytes"] = _peak_bytes()
            return recorder.call(_stage, _original, self, *args, **kwargs)

        methods[name] = wrapped
    return type("MeasuredPersistenceForest", (base,), methods)


def _worker(spec_path: Path, checkpoint_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    result = dict(spec, status="running", worker_started_utc=_utc_now())
    recorder = _Recorder(checkpoint_path, result)
    recorder.checkpoint("imports")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import gudhi
        from persforest import PersistenceForest
        from persforest.measurements.basic import constant_one_functional, signed_chain_edge_length, signed_chain_volume
        # Load the lazy landscape module outside every scientific timer, with
        # the same imported baseline for barcode and landscape workers.
        import persforest.landscapes.family
        from sampling import sample_points
        from unittest.mock import patch

        result["baseline_import_peak_rss_bytes"] = _peak_bytes()
        points = recorder.call("sampling", sample_points, spec["sampler"],
                               spec["n_points"], spec["dim"], spec["seed"])
        result["point_cloud_sha256"] = hashlib.sha256(points.tobytes()).hexdigest()
        base_alpha = gudhi.AlphaComplex

        class MeasuredAlpha:
            def __init__(self, *args, **kwargs):
                self.alpha = recorder.call("alpha", base_alpha, *args, **kwargs)

            def create_simplex_tree(self, *args, **kwargs):
                return recorder.call("alpha", self.alpha.create_simplex_tree,
                                     *args, **kwargs)

        measured = _instrumented_class(PersistenceForest, recorder)
        recorder.checkpoint("constructor")
        overhead_before = recorder.overhead_s
        # Return the original GUDHI tree: no copied/manual filtration, changed
        # tie order, validation pass, or extra tree retention versus production.
        with patch.object(gudhi, "AlphaComplex", MeasuredAlpha):
            start = time.perf_counter()
            forest = measured(points, compute_barcode=True, print_info=False,
                              reduce=True, filtration_tol=spec["filtration_tol"],
                              **MODES[spec["mode"]])
            constructor_s = time.perf_counter() - start
        constructor_s -= recorder.overhead_s - overhead_before
        alpha_s = result["time_alpha_s"]
        barcode_s = result.get("time_barcode_s", 0.0)
        interior_s = result.get("time_interior_s", 0.0)
        result.update(
            time_to_barcode_s=constructor_s,
            time_to_barcode_excluding_alpha_s=constructor_s - alpha_s,
            time_forest_s=constructor_s - alpha_s - barcode_s - interior_s,
            time_barcode_with_interior_s=barcode_s + interior_s,
            peak_rss_to_barcode_bytes=_peak_bytes(),
            n_simplices=forest.simplex_tree.num_simplices(),
            n_relevant_simplices=len(forest.filtration),
            n_nodes=len(forest.nodes), n_bars=len(forest.barcode),
            alpha_pct_to_barcode=100 * alpha_s / constructor_s,
        )
        recorder.checkpoint("barcode_complete")
        # Endpoint summaries are outside timed regions; no representative
        # materialization (especially important for storage="diffs").
        endpoints = np.array(sorted((bar.birth, bar.death) for bar in forest.barcode))
        result["barcode_endpoints_sha256"] = hashlib.sha256(endpoints.tobytes()).hexdigest()
        del endpoints
        landscape_s = 0.0
        if spec["functional"] != "none":
            functional = {"constant_one": constant_one_functional,
                          "edge_length": signed_chain_edge_length,
                          "surface_area": signed_chain_volume}[spec["functional"]]
            family = recorder.call(
                "landscape", forest.compute_measurement_landscapes,
                functional, spec["functional"], max_k=spec["max_k"],
                num_grid_points=spec["grid_points"], min_bar_length=0.0,
                signed=spec["signed"], cache=False, cache_profiles=False,
            )
            landscape_s = result["time_landscape_s"]
            result["landscape_levels"] = len(family.landscapes)
            result["landscape_grid_points"] = len(family.x_grid)
        total = constructor_s + landscape_s
        result.update(status="ok", time_pipeline_s=total,
                      time_from_sampling_s=total + result["time_sampling_s"],
                      alpha_pct_pipeline=100 * alpha_s / total,
                      landscape_pct_pipeline=100 * landscape_s / total,
                      landscape_pct_excluding_alpha=100 * landscape_s / (total - alpha_s),
                      instrumentation_io_s=recorder.overhead_s)
        recorder.checkpoint("complete")
        return 0
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        recorder.checkpoint(result.get("stage", "unknown"))
        traceback.print_exc()
        return 1


def _rss_bytes(pid: int) -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            return None
    else:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip()) * 1024
    return None


def _supervise(spec: dict, directory: Path, config: dict) -> dict:
    spec_path = directory / f"{spec['case_id']}.spec.json"
    checkpoint = directory / f"{spec['case_id']}.checkpoint.json"
    log_path = directory / f"{spec['case_id']}.log"
    _write_json(spec_path, spec)
    _write_json(checkpoint, dict(spec, status="starting", stage="startup"))
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    env.setdefault("MPLCONFIGDIR", str(directory / "mplconfig"))
    sampled_peak = 0
    reason = None
    start = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             str(spec_path), str(checkpoint)], stdout=log, stderr=log, env=env,
        )
        try:
            while process.poll() is None:
                rss = _rss_bytes(process.pid)
                if rss is not None:
                    sampled_peak = max(sampled_peak, rss)
                if sampled_peak > config["memory_limit_gib"] * 1024 ** 3:
                    reason = "memory_limit"
                elif time.monotonic() - start > config["timeout_s"]:
                    reason = "timeout"
                if reason:
                    process.kill()
                    break
                time.sleep(config["poll_seconds"])
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    result = json.loads(checkpoint.read_text())
    result.update(worker_wall_s=time.monotonic() - start,
                  sampled_peak_rss_bytes=sampled_peak, exit_code=process.returncode,
                  log_file=str(log_path), finished_utc=_utc_now())
    result["peak_rss_bytes"] = max(sampled_peak, result.get("peak_rss_bytes", 0))
    if reason:
        result["status"] = reason
    elif process.returncode != 0 and result.get("status") != "error":
        result["status"] = "worker_failed"
    elif result.get("status") != "ok":
        result["status"] = "error"
    return result


def build_cases(config: dict) -> list[dict]:
    """Build paired workloads with shuffled modes within ascending sizes.

    Parameters
    ----------
    config : dict
        Validated CLI configuration.

    Returns
    -------
    list of dict
        Deterministic cases with stable seeds, IDs and all scientific settings.
    """
    cases = []
    order_rng = random.Random(config["base_seed"])
    modes = ["diffs"] if config["suite"] == "diff-scaling" else config["modes"]
    for dim in config["dimensions"]:
        sizes = config[f"sizes_{dim}d"] or SCHEDULES[config["suite"]][dim]
        for n_points in sorted(set(sizes)):
            for sampler in config["samplers"]:
                for repeat in range(config["repeats"]):
                    seed = int(_digest([config["base_seed"], dim, n_points,
                                        sampler, repeat])[:8], 16)
                    functionals = (["constant_one", "edge_length" if dim == 2 else "surface_area"]
                                   if config["suite"] == "landscapes" else ["none"])
                    block = []
                    for mode in modes:
                        for functional in functionals:
                            case = dict(suite=config["suite"], dim=dim, n_points=n_points,
                                        sampler=sampler, repeat=repeat, seed=seed,
                                        mode=mode, functional=functional,
                                        max_k=config["max_k"], grid_points=config["grid_points"],
                                        signed=config["signed"], filtration_tol=1e-12,
                                        reduce=True, schema_version=SCHEMA_VERSION)
                            case["case_id"] = _digest(case)[:20]
                            block.append(case)
                    order_rng.shuffle(block)
                    cases.extend(block)
    return cases


def _metadata() -> dict:
    paths = sorted((ROOT / "src/persforest").rglob("*.py"))
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "poetry.lock"]
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths if path.exists()}
    packages = {}
    for name in ("persforest", "numpy", "gudhi", "matplotlib", "seaborn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    from sampling import SAMPLER_PARAMETERS
    try:
        physical_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        physical_memory = None
    return dict(python=sys.version, executable=sys.executable,
                platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), cpu_count=os.cpu_count(),
                physical_memory_bytes=physical_memory,
                packages=packages, source_hashes=files,
                sampler_parameters=SAMPLER_PARAMETERS,
                thread_policy="one thread; PYTHONHASHSEED=0",
                git_revision=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip(),
                git_status=subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip())


def load_results(path: Path) -> list[dict]:
    """Read append-only records, keeping the latest attempt per case ID.

    Parameters
    ----------
    path : Path
        results.jsonl written by the supervisor.

    Returns
    -------
    list of dict
        Latest complete records; an interrupted final line is ignored.
    """
    records = {}
    if path.exists():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise
            records[row["case_id"]] = row
    return list(records.values())


def _export_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return parsed


def main() -> None:
    """Parse arguments and preview or execute an isolated benchmark schedule."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SCHEDULES, default="barcode")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "barcode")
    parser.add_argument("--run", action="store_true", help="Actually launch computations; otherwise preview only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, retry unsuccessful cases")
    parser.add_argument("--dimensions", nargs="+", type=int, choices=[2, 3], default=[2, 3])
    parser.add_argument("--samplers", nargs="+", choices=["uniform", "noisy_sphere", "holes"],
                        default=["uniform", "noisy_sphere", "holes"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--sizes-2d", nargs="+", type=_positive_int)
    parser.add_argument("--sizes-3d", nargs="+", type=_positive_int)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--memory-limit-gib", type=_positive_float, default=32.0)
    parser.add_argument("--timeout-s", type=_positive_float, default=3600.0)
    parser.add_argument("--poll-seconds", type=_positive_float, default=0.25)
    parser.add_argument("--grid-points", type=_positive_int, default=512)
    parser.add_argument("--max-k", type=_positive_int, default=5)
    parser.add_argument("--signed", action="store_true", help="Use signed chains (default: unsigned API behavior)")
    args = parser.parse_args()
    if args.grid_points < 2:
        parser.error("--grid-points must be at least 2")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    for name in ("dimensions", "samplers", "modes"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"--{name} must not contain duplicates")
    config = {key: value for key, value in vars(args).items()
              if key not in ("run", "resume", "retry_failed", "output")}
    cases = build_cases(config)
    print(json.dumps(dict(config=config, cases=len(cases), first_cases=cases[:2]), indent=2))
    if not args.run:
        print("Preview only. Add --run to execute. No benchmark was run.")
        return
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        parser.error("RSS measurement currently supports macOS and Linux")
    directory = args.output.resolve()
    metadata = _metadata()
    manifest = dict(schema_version=SCHEMA_VERSION, config=config, metadata=metadata,
                    case_count=len(cases), created_utc=_utc_now())
    manifest_path = directory / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            parser.error("Cannot resume without manifest.json")
        previous = json.loads(manifest_path.read_text())
        stable_fields = ("python", "executable", "platform", "machine", "packages",
                         "source_hashes", "sampler_parameters", "thread_policy")
        if (previous["config"] != config or previous["schema_version"] != SCHEMA_VERSION
                or any(previous["metadata"][key] != metadata[key] for key in stable_fields)):
            parser.error("Resume requires identical config, sources and environment; use a new output directory")
    else:
        if directory.exists() and any(directory.iterdir()):
            parser.error("Output directory is nonempty; use --resume or a new directory")
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, manifest)
    workers = directory / "workers"
    workers.mkdir(exist_ok=True)
    results_path = directory / "results.jsonl"
    rows = {row["case_id"]: row for row in load_results(results_path)}
    # Repair only a truncated final write before appending on resume.
    if results_path.exists():
        compacted_path = results_path.with_suffix(".tmp")
        with compacted_path.open("w") as stream:
            for row in rows.values():
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        compacted_path.replace(results_path)
    blocked = {}
    if not args.retry_failed:
        for row in rows.values():
            if row["status"] in ("memory_limit", "timeout"):
                key = (row["dim"], row["sampler"], row["mode"], row["functional"])
                blocked[key] = min(blocked.get(key, math.inf), row["n_points"])
    for index, spec in enumerate(cases, 1):
        old = rows.get(spec["case_id"])
        if old and (old["status"] == "ok" or not args.retry_failed):
            continue
        key = (spec["dim"], spec["sampler"], spec["mode"], spec["functional"])
        if spec["n_points"] >= blocked.get(key, math.inf):
            result = dict(spec, status="skipped_after_limit")
        else:
            print(f"[{index}/{len(cases)}] {spec['dim']}D {spec['sampler']} "
                  f"n={spec['n_points']} {spec['mode']} {spec['functional']}", flush=True)
            result = _supervise(spec, workers, config)
            if result["status"] in ("memory_limit", "timeout"):
                blocked[key] = spec["n_points"]
        rows[spec["case_id"]] = result
        with results_path.open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _export_csv(directory / "results.csv", list(rows.values()))
        print(f"  {result['status']}", flush=True)
    print(f"Results: {directory / 'results.csv'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker(Path(sys.argv[2]), Path(sys.argv[3])))
    main()
