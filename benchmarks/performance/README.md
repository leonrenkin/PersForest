# PersistenceForest runtime and memory benchmarks

Serial, isolated measurements of `PersistenceForest` on point clouds (alpha
filtration) and cubical grids (`top_cells`), in 2D and 3D: time to build the
forest and barcode, and time to compute measurement landscapes. **A command
without `--run` only prints the plan.**

## Inputs

Seeded generators in `sampling.py`; each repeat has its own seed. `size` is the point count for clouds and the side length for grids.

| Family | Point clouds | Cubical grids |
| --- | --- | --- |
| noise | `uniform` in the unit cube | `uniform` i.i.d. pixel values |
| one large cycle | `noisy_sphere`: unit circle/sphere, noise std 0.05 | `single_hole`: `\| \|x - c\| - 0.3 \|`, noise std 0.03 |
| fixed holes | `holes`: unit cube minus 30 small balls | `holes`: 30 cones on a low background |

The one-large-cycle families are the expensive case. Schedules (`pilot`: sparse,
one repeat; `full`: dense, overnight) are defined in `SCHEDULES` in `run.py`;
landscapes run only up to per-schedule size caps. `--kinds`,
`--dimensions`, `--samplers` and `--{points,cubical}-sizes-{2,3}d`
restrict or override a schedule.

## Running (from the repository root)

```sh
# Preview, then run (caffeinate keeps a Mac awake)
.venv/bin/python benchmarks/performance/run.py --schedule pilot --output benchmarks/performance/results/2026-11-pilot
caffeinate -i .venv/bin/python benchmarks/performance/run.py --schedule full \
  --repeats 3 --output benchmarks/performance/results/2026-11-full --run

# Summary table and figures (also for partial runs)
MPLCONFIGDIR=/tmp/persforest-benchmark-mpl .venv/bin/python benchmarks/performance/plot.py \
  benchmarks/performance/results/2026-11-full
```

Add `--resume` after an interruption (`--retry-failed` also retries limit
hits); use a new output directory after code changes. Avoid other heavy work
during a run.

## What is measured

Each case runs in a fresh single-threaded worker via `from_points` /
`from_cubical_grid`. The primary column `time_persforest_s` covers forest
sweep, reduction and barcode, excluding filtration preparation
(`time_filtration_s`); it splits into `time_forest_s` (the compiled engine,
`time_forest_build_s`, and the setup around it) and `time_barcode_s` (creating
the bar objects). `time_landscape_s` is one uncached
`compute_measurement_landscapes` call (including representative
reconstruction) of the unsigned boundary measure: length in 2D, area in 3D.
Peak RSS is recorded through the barcode and for the whole worker.

Workers are stopped above `--memory-limit-gib` or after `--timeout-s`, and
larger sizes of the same configuration are skipped. Memory is sampled RSS, not
a hard limit; killed workers give only lower bounds.

Each output directory holds `manifest.json` (configuration, environment,
source hashes, Git state), `results.jsonl`, `results.csv` and per-case
`workers/`. `plot.py` writes `plots/summary.csv` and the `runtime` and
`breakdown` figures. Results are git-ignored; `results/2026-11-pilot` (schema 2,
with storage modes) and `results/historical-2026-09-barcode` predate the current
schema 3 and measure the former Python engine.

## Harness tests

```sh
MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests/tests-by-agents -p 'test_performance_harness.py'
```

These use mocked stages and do not produce benchmark results.
