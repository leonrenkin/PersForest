# PersistenceForest runtime and memory benchmarks

Serial, isolated measurements of `PersistenceForest` on point clouds (alpha
filtration) and cubical grids (`top_cells` sublevel filtration), in 2D and 3D.
The focus is `storage="diffs"` (the default): time to build the persistence
forest and barcode, and time to compute measurement landscapes of a fixed
boundary functional. `storage="cycles"` is measured at smaller sizes as a
comparison. **A command without `--run` only prints a plan.**

## Inputs

All inputs come from seeded local generators in `sampling.py`; seeds are stable
functions of base seed, input kind, dimension, size, family and repeat, so all
modes of one repeat receive identical inputs.

| Family | Point clouds (`size` = number of points) | Cubical grids (`size` = side length, `side**dim` top cells) |
| --- | --- | --- |
| noise | `uniform`: uniform in the unit cube | `uniform`: i.i.d. uniform pixel values |
| one large cycle | `noisy_sphere`: unit circle/sphere, coordinate noise std 0.05 | `single_hole`: `| |x - c| - 0.3 |` around the center, Gaussian noise std 0.03 |
| fixed holes | `holes`: unit cube minus 30 balls, radii uniform in [0, 0.05] | `holes`: low background with 30 cones `max(0, 1 - |x - c_i| / r_i)` (same ball distribution), noise std 0.03 |

The one-large-cycle families are the expensive case: one long-lived cycle with
a large boundary. Point-cloud families are unchanged from the September 2026
runs, so the same seed gives the same cloud.

## Schedules

Sizes are point counts for clouds and side lengths for grids. Diff-mode
barcodes run at every size. Landscapes and the `cycles` comparison run only up
to their caps (inclusive); above the landscape cap, cases measure the barcode
only.

| Schedule | Input | Sizes | Landscapes up to | `cycles` up to |
| --- | --- | --- | --- | --- |
| `pilot` | points 2D | 10k, 100k, 400k, 1M | 100k | 100k |
| | points 3D | 5k, 50k, 250k | 50k | 50k |
| | cubical 2D | 128², 512², 2048² | 512² | 512² |
| | cubical 3D | 32³, 64³, 128³ | 64³ | 64³ |
| `full` | points 2D | 10k, 25k, 50k, 100k, 250k, 500k, 1M | 100k | 100k |
| | points 3D | 5k, 10k, 25k, 50k, 100k, 250k | 25k | 25k |
| | cubical 2D | 128², 256², 512², 1024², 1448², 2048² | 512² | 512² |
| | cubical 3D | 24³, 32³, 48³, 64³, 96³, 128³ | 64³ | 48³ |

The pilot is a sparse, single-repeat run to get intuition about runtimes; tune
`SCHEDULES` in `run.py` from its results before the overnight `full` run.
`--kinds`, `--dimensions`, `--samplers`, `--modes` and
`--{points,cubical}-sizes-{2,3}d` restrict or override a schedule.

## Run from the repository root

```sh
# Preview (no computation)
.venv/bin/python benchmarks/performance/run.py --schedule pilot \
  --output benchmarks/performance/results/2026-11-pilot

# Pilot: one repeat, 15 min per-case timeout, keep the laptop awake
caffeinate -i .venv/bin/python benchmarks/performance/run.py --schedule pilot \
  --repeats 1 --timeout-s 900 --output benchmarks/performance/results/2026-11-pilot --run

# Overnight
caffeinate -i .venv/bin/python benchmarks/performance/run.py --schedule full \
  --repeats 3 --output benchmarks/performance/results/2026-11-full --run

# Summary table and figures (also for partial runs)
MPLCONFIGDIR=/tmp/persforest-benchmark-mpl .venv/bin/python benchmarks/performance/plot.py \
  benchmarks/performance/results/2026-11-pilot
```

Add `--resume` to the same run command after an interruption (configuration,
sources and environment must be unchanged; `--retry-failed` also retries limit
hits). Use a new output directory after code changes. Keep the machine on mains
power and avoid other heavy work during a run; record anything unusual.

## What is measured

Each case runs in a fresh worker process (one thread, `PYTHONHASHSEED=0`) via
the public factories `PersistenceForest.from_points` and
`PersistenceForest.from_cubical_grid` with `reduce=True` and
`filtration_tol=1e-12`. A benchmark-local subclass times the existing phase
methods; the filtration preparation function is wrapped in place, without
copying its output.

| Result column | Meaning |
| --- | --- |
| `time_persforest_s` | **Primary**: forest sweep, reduction and barcode (with interiors for `cycles`), excluding filtration preparation |
| `time_landscape_s` | One fresh `compute_measurement_landscapes` call (diff mode includes reconstructing representatives) |
| `time_persforest_and_landscape_s` | Sum of the two above |
| `time_filtration_s` | Alpha complex (GUDHI) or cubical cell preparation |
| `time_forest_s`, `time_barcode_with_interior_s` | Split of `time_persforest_s` |
| `time_to_barcode_s`, `time_pipeline_s` | Including filtration preparation; pipeline also includes the landscape |
| `time_sampling_s` | Input generation, never included above |
| `peak_rss_to_barcode_bytes`, `peak_rss_bytes` | Cumulative process peaks through the barcode / the whole worker |
| `n_input`, `n_events`, `n_bars` | Points or top cells; filtration events; bars |

The landscape functional is the boundary measure of each representative:
length in 2D, area in 3D (`signed_chain_volume` for clouds,
`cubical_chain_volume` for grids; the latter measures staircase length along
grid cells). It is unsigned, as in the public API, with all bars, 5 levels and
512 grid points, and with caching disabled. Input, endpoint and package hashes
are recorded as diagnostics; diffs and cycles should give identical barcode
endpoint hashes.

## Limits and memory

The supervisor polls worker RSS (every 0.25 s) and stops a worker above
`--memory-limit-gib` (default 32 GiB, for a 48 GB machine) or after
`--timeout-s` (default 30 min). A limit skips larger sizes of the same input
kind, dimension, family, mode and landscape setting. A landscape that hits a
limit keeps its already completed barcode timings, which the plots use.
This is a sampled stop rule, not a hard allocation limit, and RSS does not see
compressed or swapped memory on macOS. Killed workers only provide lower bounds.

## Data and plots

Each output directory contains `manifest.json` (configuration, sampler
parameters, Python/platform/package versions, source and lockfile hashes, Git
revision and status), `results.jsonl` (canonical raw records), `results.csv`
(flat export) and `workers/` (per-case spec, checkpoint and log).

`plot.py RESULTS_DIR` writes `plots/summary.csv` (median, min and max over
repeats), prints a diff-mode table and saves:

- `runtime`: diff-mode forest + barcode (solid) and landscape (dashed) time
  against size on log-log axes, one panel per input kind and dimension, with a
  slope-one reference line.
- `breakdown`: diff-mode stage times (filtration, forest, barcode, landscape)
  at the largest size with landscapes.
- `modes`: diffs vs cycles, total time and peak memory, where both ran.

Results are local research output and are git-ignored.
`results/historical-2026-09-barcode` is the one kept earlier run, made with the
removed root-level package and the former mode names (`diff_only_mode`
corresponds to `diffs`, `compute_interior` to `cycles`); its schema differs
and `plot.py` does not read it.

## Lightweight verification of the harness

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-benchmark-mpl \
  .venv/bin/python -m unittest discover -s tests/tests-by-agents \
  -p 'test_performance_harness.py' -v
```

These tests use small generated inputs, mocked scientific stages and synthetic
records. They do not run a real forest and do not provide benchmark results.
