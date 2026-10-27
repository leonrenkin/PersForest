"""Runtime and memory of cubical-grid forests; previews unless ``--run`` is given.

Each case runs in a fresh interpreter so that peak RSS is per case. Grids are
uniform noise or Gaussian-smoothed noise from seeded local generators.

Usage from the repository root::

    .venv/bin/python benchmarks/performance/cubical.py                      # preview
    .venv/bin/python benchmarks/performance/cubical.py --run --output DIR   # measure
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHAPES = [(256, 256), (512, 512), (1024, 1024), (32, 32, 32), (64, 64, 64), (96, 96, 96)]
# Stored cycles grow superlinearly in memory; keep them to sizes that fit comfortably.
CYCLES_MAX_CELLS = 512 * 512
NOISE = ("uniform", "smoothed")


def build_cases(shapes, repeats: int, base_seed: int) -> list[dict]:
    """Return deterministic cases with seeds that depend on shape, noise and repeat."""
    cases = []
    for shape in shapes:
        cells = math.prod(shape)
        for noise in NOISE:
            for repeat in range(repeats):
                digest = hashlib.sha256(json.dumps([base_seed, shape, noise, repeat]).encode()).hexdigest()
                seed = int(digest[:8], 16)
                for storage in ("diffs", "cycles"):
                    if storage == "cycles" and cells > CYCLES_MAX_CELLS:
                        continue
                    cases.append(dict(shape=list(shape), noise=noise, repeat=repeat, seed=seed, storage=storage))
    return cases


def smoothed(values, sigma: float = 2.0):
    """Separable Gaussian smoothing with reflected borders (NumPy only)."""
    import numpy as np
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    for axis in range(values.ndim):
        width = [(0, 0)] * values.ndim
        width[axis] = (radius, radius)
        padded = np.pad(values, width, mode="reflect")
        values = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), axis, padded)
    return values


def run_case(case: dict) -> dict:
    """Measure one case in the current process."""
    import resource
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from persforest import PersistenceForest
    from persforest.filtrations.cubical import prepare_cubical_filtration

    values = np.random.default_rng(case["seed"]).random(case["shape"])
    if case["noise"] == "smoothed":
        values = smoothed(values)
    start = time.perf_counter()
    _, cells = prepare_cubical_filtration(values)
    prepare_s = time.perf_counter() - start
    n_events = len(cells.events)
    del cells
    start = time.perf_counter()
    forest = PersistenceForest.from_cubical_grid(values, storage=case["storage"])
    total_s = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return dict(case, n_events=n_events, n_bars=len(forest.barcode), time_prepare_s=prepare_s,
                time_total_s=total_s, time_sweep_and_barcode_s=total_s - prepare_s, peak_rss_bytes=peak_bytes)


def metadata() -> dict:
    import gudhi
    import numpy
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                numpy=numpy.__version__, gudhi=gudhi.__version__, cycles_max_cells=CYCLES_MAX_CELLS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="Actually launch computations; otherwise preview only")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "cubical")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--max-cells", type=int, default=None, help="Skip shapes with more top cells")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(run_case(json.loads(args.case))))
        return
    shapes = [s for s in SHAPES if args.max_cells is None or math.prod(s) <= args.max_cells]
    cases = build_cases(shapes, args.repeats, args.base_seed)
    if not args.run:
        print(json.dumps(dict(cases=len(cases), first_cases=cases[:4]), indent=1))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metadata.json").write_text(json.dumps(dict(metadata(), base_seed=args.base_seed,
                                                               repeats=args.repeats), indent=1))
    with open(args.output / "results.jsonl", "w") as results:
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case['shape']} {case['noise']} {case['storage']}", flush=True)
            completed = subprocess.run([sys.executable, "-I", __file__, "--case", json.dumps(case)],
                                       capture_output=True, text=True)
            if completed.returncode != 0:
                record = dict(case, status="error", error=completed.stderr[-2000:])
            else:
                record = dict(json.loads(completed.stdout.strip().splitlines()[-1]), status="ok")
            results.write(json.dumps(record) + "\n")
            results.flush()
    print(f"Results: {args.output / 'results.jsonl'}")


if __name__ == "__main__":
    main()
