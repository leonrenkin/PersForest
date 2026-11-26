"""Serial, isolated runtime/RSS benchmarks. No computations without --run.

See README.md for definitions, watchdog limitations and examples.
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
try:
    import resource
except ImportError:  # Windows: main() rejects --run there before measuring RSS.
    resource = None
import subprocess
import sys
import time
import traceback
from typing import Callable, Any

ROOT = Path(__file__).resolve().parents[2]
# Sibling modules (sampling) also when this file is loaded by path, e.g. in tests.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
SCHEMA_VERSION = 3
KINDS = ("points", "cubical")
# Sizes are point counts for clouds and grid side lengths for cubical inputs
# (side**dim top cells). Landscapes run only up to their caps, in the same
# units; barcodes run at every size.
SCHEDULES = {
    "pilot": {
        ("points", 2): dict(sizes=[10_000, 100_000, 400_000, 1_000_000],
                            landscape_max=100_000),
        ("points", 3): dict(sizes=[5_000, 50_000, 250_000],
                            landscape_max=50_000),
        ("cubical", 2): dict(sizes=[128, 512, 2048], landscape_max=512),
        ("cubical", 3): dict(sizes=[32, 64, 128], landscape_max=64),
    },
    "full": {
        ("points", 2): dict(sizes=[10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
                            landscape_max=100_000),
        ("points", 3): dict(sizes=[5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
                            landscape_max=25_000),
        ("cubical", 2): dict(sizes=[128, 256, 512, 1024, 1448, 2048],
                             landscape_max=512),
        ("cubical", 3): dict(sizes=[24, 32, 48, 64, 96, 128],
                             landscape_max=64),
    },
}
FUNCTIONAL = "boundary_measure"  # boundary length in 2D, boundary area in 3D


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
    for name, stage in {"compute_barcode": "barcode"}.items():
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
        import persforest
        import persforest._forest.arrays as arrays_module
        import persforest.forest as forest_module
        from persforest import PersistenceForest
        from persforest.measurements.basic import signed_chain_volume
        from persforest.measurements.cubical import cubical_chain_volume
        # Load the lazy landscape module outside every scientific timer, with
        # the same imported baseline for barcode and landscape workers.
        import persforest.landscapes.family
        from sampling import sample_grid, sample_points
        from unittest.mock import patch

        result["persforest_module"] = str(Path(persforest.__file__).resolve())
        result["baseline_import_peak_rss_bytes"] = _peak_bytes()
        kind = spec["kind"]
        sampler = sample_points if kind == "points" else sample_grid
        data = recorder.call("sampling", sampler, spec["sampler"], spec["size"],
                             spec["dim"], spec["seed"])
        result["input_sha256"] = hashlib.sha256(data.tobytes()).hexdigest()

        # Time the production preparation step (alpha complex or cubical
        # cells) inside the normal factory, without copying its output.
        prepare_name = ("prepare_point_filtration" if kind == "points"
                        else "prepare_cubical_filtration")
        original_prepare = getattr(forest_module, prepare_name)

        def measured_prepare(*args, **kwargs):
            return recorder.call("filtration", original_prepare, *args, **kwargs)

        # The compiled engine: sweep, reduction and bar extraction.
        original_engine = arrays_module.build_forest_arrays

        def measured_engine(*args, **kwargs):
            return recorder.call("forest_build", original_engine, *args, **kwargs)

        measured = _instrumented_class(PersistenceForest, recorder)
        factory = measured.from_points if kind == "points" else measured.from_cubical_grid
        recorder.checkpoint("constructor")
        overhead_before = recorder.overhead_s
        with patch.object(forest_module, prepare_name, measured_prepare), \
             patch.object(arrays_module, "build_forest_arrays", measured_engine):
            start = time.perf_counter()
            forest = factory(data, reduce=True, compute_barcode=True, print_info=False,
                             filtration_tol=spec["filtration_tol"])
            constructor_s = time.perf_counter() - start
        constructor_s -= recorder.overhead_s - overhead_before
        del data
        filtration_s = result["time_filtration_s"]
        barcode_s = result.get("time_barcode_s", 0.0)
        result.update(
            time_to_barcode_s=constructor_s,
            # Primary metric: forest sweep, reduction and barcode extraction,
            # excluding filtration preparation shared with other software.
            time_persforest_s=constructor_s - filtration_s,
            # The compiled engine and the forest's setup around it.
            time_forest_s=constructor_s - filtration_s - barcode_s,
            peak_rss_to_barcode_bytes=_peak_bytes(),
            n_events=len(forest._cells.event_cells), n_nodes=len(forest._forest.node_value),
            n_bars=len(forest.barcode),
        )
        recorder.checkpoint("barcode_complete")
        # Endpoint summaries are outside timed regions; no representative
        # is reconstructed.
        endpoints = np.array(sorted((bar.birth, bar.death) for bar in forest.barcode))
        result["barcode_endpoints_sha256"] = hashlib.sha256(endpoints.tobytes()).hexdigest()
        del endpoints
        landscape_s = 0.0
        if spec["landscape"]:
            functional = signed_chain_volume if kind == "points" else cubical_chain_volume
            family = recorder.call(
                "landscape", forest.compute_measurement_landscapes,
                functional, FUNCTIONAL, max_k=spec["max_k"],
                num_grid_points=spec["grid_points"], min_bar_length=0.0,
                signed=spec["signed"], cache=False, cache_profiles=False,
            )
            landscape_s = result["time_landscape_s"]
            result["landscape_levels"] = len(family.landscapes)
            result["landscape_grid_points"] = len(family.x_grid)
        result.update(status="ok", time_pipeline_s=constructor_s + landscape_s,
                      time_persforest_and_landscape_s=constructor_s - filtration_s + landscape_s,
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
    """Build paired workloads in ascending size within each input group.

    Parameters
    ----------
    config : dict
        Validated CLI configuration.

    Returns
    -------
    list of dict
        Deterministic cases with stable seeds, IDs and all scientific settings.
        Every repeat of an input family and size has its own seed.
    """
    from sampling import GRID_SAMPLERS, POINT_SAMPLERS
    cases = []
    for kind in config["kinds"]:
        samplers = POINT_SAMPLERS if kind == "points" else GRID_SAMPLERS
        for dim in config["dimensions"]:
            schedule = SCHEDULES[config["schedule"]][(kind, dim)]
            override = config[f"{kind}_sizes_{dim}d"]
            for size in sorted(set(override or schedule["sizes"])):
                for sampler in samplers:
                    if config["samplers"] and sampler not in config["samplers"]:
                        continue
                    for repeat in range(config["repeats"]):
                        seed = int(_digest([config["base_seed"], kind, dim, size,
                                            sampler, repeat])[:8], 16)
                        landscape = size <= schedule["landscape_max"]
                        case = dict(kind=kind, dim=dim, size=size,
                                    n_input=size if kind == "points" else size ** dim,
                                    sampler=sampler, repeat=repeat, seed=seed,
                                    landscape=landscape,
                                    functional=FUNCTIONAL if landscape else "none",
                                    max_k=config["max_k"], grid_points=config["grid_points"],
                                    signed=config["signed"], filtration_tol=1e-12,
                                    reduce=True, schema_version=SCHEMA_VERSION)
                        case["case_id"] = _digest(case)[:20]
                        cases.append(case)
    return cases


def _metadata() -> dict:
    paths = sorted((ROOT / "src/persforest").rglob("*.py"))
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
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


def _limit_group(row: dict) -> tuple:
    """Cases sharing a group stop growing after a memory or time limit."""
    return (row["kind"], row["dim"], row["sampler"], row["landscape"])


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
    parser.add_argument("--schedule", choices=SCHEDULES, default="pilot",
                        help="pilot: sparse sizes for runtime intuition; full: dense overnight schedule")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Actually launch computations; otherwise preview only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, retry unsuccessful cases")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--dimensions", nargs="+", type=int, choices=[2, 3], default=[2, 3])
    parser.add_argument("--samplers", nargs="+",
                        choices=sorted({"uniform", "noisy_sphere", "single_hole", "holes"}),
                        help="Restrict input families (default: all of each kind)")
    for kind in KINDS:
        unit = "point counts" if kind == "points" else "grid side lengths"
        for dim in (2, 3):
            parser.add_argument(f"--{kind}-sizes-{dim}d", nargs="+", type=_positive_int,
                                help=f"Override the schedule's {dim}D {unit}")
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--memory-limit-gib", type=_positive_float, default=32.0)
    parser.add_argument("--timeout-s", type=_positive_float, default=1800.0)
    parser.add_argument("--poll-seconds", type=_positive_float, default=0.25)
    parser.add_argument("--grid-points", type=_positive_int, default=512)
    parser.add_argument("--max-k", type=_positive_int, default=5)
    parser.add_argument("--signed", action="store_true", help="Use signed chains (default: unsigned API behavior)")
    args = parser.parse_args()
    if args.grid_points < 2:
        parser.error("--grid-points must be at least 2")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    for name in ("kinds", "dimensions", "samplers"):
        if getattr(args, name) and len(getattr(args, name)) != len(set(getattr(args, name))):
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
                key = _limit_group(row)
                blocked[key] = min(blocked.get(key, math.inf), row["size"])
    for index, spec in enumerate(cases, 1):
        old = rows.get(spec["case_id"])
        if old and (old["status"] == "ok" or not args.retry_failed):
            continue
        key = _limit_group(spec)
        if spec["size"] >= blocked.get(key, math.inf):
            result = dict(spec, status="skipped_after_limit")
        else:
            print(f"[{index}/{len(cases)}] {spec['kind']} {spec['dim']}D {spec['sampler']} "
                  f"size={spec['size']} {spec['functional']}", flush=True)
            result = _supervise(spec, workers, config)
            if result["status"] in ("memory_limit", "timeout"):
                blocked[key] = spec["size"]
        rows[spec["case_id"]] = result
        with results_path.open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _export_csv(directory / "results.csv", list(rows.values()))
        summary = (f"persforest {result['time_persforest_s']:.1f}s"
                   if "time_persforest_s" in result else result.get("stage", ""))
        if "time_landscape_s" in result:
            summary += f", landscape {result['time_landscape_s']:.1f}s"
        print(f"  {result['status']}: {summary}, peak "
              f"{result.get('peak_rss_bytes', 0) / 1024 ** 3:.1f} GiB", flush=True)
    print(f"Results: {directory / 'results.csv'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker(Path(sys.argv[2]), Path(sys.argv[3])))
    main()
