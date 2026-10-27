# PersistenceForest runtime and memory benchmarks

This is a separate benchmark harness; `benchmarks/benchmark.py` is unchanged.
It uses the existing environment and dependencies. **A command without `--run`
only prints a plan.** No calibration or warm-up benchmark runs implicitly.

## Plan and workloads

| Suite | Modes | 2D sizes | 3D sizes |
| --- | --- | --- | --- |
| `barcode` | all four | 1k, 5k, 10k, 25k, 50k, 100k | 500, 1k, 2.5k, 5k, 10k, 15k, 20k |
| `diff-scaling` | diff-only | 150k, 200k, 300k, 400k | 20k, 25k, 50k, 75k |
| `landscapes` | all four × two functionals | 500, 1k, 2.5k, 5k, 10k, 20k | 250, 500, 1k, 2.5k, 5k |

The old benchmark used maxima of 200k / 50k with 10 repeats. The new common
schedule is smaller to accommodate interiors, especially in 3D. These are
proposed sizes, **not measured memory guarantees**. The extension is a separate,
opt-in experiment. Landscapes start smaller because the current implementation
allocates and sorts arrays proportional to bar count × grid size, in addition to
building measurement profiles and kernels. Override any schedule after seeing
your machine's results.

Modes use the production constructor, with `reduce=True`, radius-unit alpha
filtration values, `filtration_tol=1e-12`, and normal barcode dispatch:

| Name | Constructor flags |
| --- | --- |
| `standard` | defaults |
| `keep_simplex_diff` | `keep_simplex_diff=True` |
| `compute_interior` | `keep_simplex_diff=True, compute_interior=True` |
| `diff_only_mode` | `keep_simplex_diff=True, diff_only_mode=True` |

All suites default to three independently seeded repetitions and three cloud
families per dimension: uniform unit cube, noisy unit circle/sphere (coordinate
noise std 0.05), and uniform cube with 30 excluded balls (radii uniform in
[0, 0.05]). Seeds are stable functions of base seed, dimension, size, sampler,
and repeat. All modes and functionals receive identical clouds for a given
case; mode/functional order is shuffled deterministically within each block.
Sizes increase so a limit can stop subsequent larger cases in the same group.
These repeats measure both sampling variability and runtime variability;
min–max bands are descriptive, not confidence intervals.

`sampling.py` implements the same distribution families using local generators;
it does not reproduce the old clouds bit for bit. Two issues in the old hole
sampler motivated a separate implementation: its benchmark label says radius
0.05 but passes 0.5, and its fill loop counts array batches instead of accepted
points. This benchmark explicitly uses 0.05 and preallocates exactly n points
with bounded rejection batches. The old files are unchanged.

## Run from the repository root

Preview the complete comparison (468 cases by default):

```sh
.venv/bin/python benchmarks/performance/run.py --suite barcode
```

Start with a short, real check **when you are ready to compute**:

```sh
.venv/bin/python benchmarks/performance/run.py --suite barcode \
  --sizes-2d 100 500 --sizes-3d 100 250 --samplers uniform --repeats 1 \
  --output benchmarks/performance/results/pilot --run
```

Then run the experiments separately:

```sh
.venv/bin/python benchmarks/performance/run.py --suite barcode \
  --output benchmarks/performance/results/barcode --run

.venv/bin/python benchmarks/performance/run.py --suite landscapes \
  --output benchmarks/performance/results/landscapes --run

.venv/bin/python benchmarks/performance/run.py --suite diff-scaling \
  --output benchmarks/performance/results/diff-scaling --run
```

For 2D only, add `--dimensions 2`. To extend the four-mode comparison to your
upper requested sizes, use `--sizes-2d 100000 150000 200000 --sizes-3d 20000 25000`
in a **new output directory**. `--modes`, `--samplers`, `--repeats`,
`--base-seed`, `--grid-points`, and `--max-k` are configurable; see `--help`.
The diff-scaling suite always selects diff-only, regardless of `--modes`.

Resume using the **same command and configuration**, adding `--resume`.
Completed and failed/skipped records are retained. Add `--retry-failed` to
retry unsuccessful cases; another limit still blocks that group at that size
and above during the new invocation. A manifest mismatch refuses resume; use a
new directory for changed code, configuration, Python/packages or environment.
Run only one supervisor per output directory. Ctrl-C stops the active worker;
completed rows survive and the interrupted case can run again on resume.

Workers run **serially**, with common native-library thread limits set to one.
Keep power/thermal conditions consistent and close memory-heavy applications.
There is no in-process warm-up: imports and process startup are outside the
reported scientific timers, while alpha/forest/landscape calls are first calls
in that process. `worker_wall_s` also records startup and supervisor overhead.

## Compare the barcode-diff optimization with saved runs

Reuse the existing seeds and run settings, writing to a **new** directory. This
400k noisy-circle command runs three repetitions; omit `--run` to preview:

```sh
.venv/bin/python benchmarks/performance/run.py --suite diff-scaling \
  --dimensions 2 --sizes-2d 400000 --samplers noisy_sphere --repeats 3 \
  --output benchmarks/performance/results/diff-unions-400k --run

.venv/bin/python benchmarks/performance/compare_runs.py \
  benchmarks/performance/results/diff-scaling \
  benchmarks/performance/results/diff-unions-400k
```

Add `--resume` to the same run command after an interruption. Do not resume the
old pre-optimization directory with changed code. The comparison script only
reads results, pairs matching settings/seeds, verifies point-cloud and barcode
endpoint hashes, and reports total/barcode median timings and peak RSS. Hash
agreement checks endpoints, not representatives; regression tests check those.
Historical speedups may also reflect power, load, and thermal differences.

For a shorter comparison, use `--sizes-2d 50000`, a new output directory, and
compare against `benchmarks/performance/results/barcode`. Seeds do not depend on
the suite name. Broader validation can use the original `diff-scaling` schedule
in another new directory.

## Memory measurement and the 48 GB machine

The default watchdog threshold is **32 GiB process RSS**, leaving headroom on a
48 GB MacBook; each case also has a **one-hour timeout**. Change them with
`--memory-limit-gib` and `--timeout-s`. An external supervisor polls RSS every
0.25 seconds (configurable), so it can stop a worker even during native code.
macOS uses `ps`; Linux uses `/proc`. A memory/timeout limit skips subsequent
same-or-larger sizes for that dimension/sampler/mode/functional, including
remaining repetitions. Other modes continue.

This is a sampled stop rule, **not a hard allocation limit**. Allocation spikes
can overshoot it. RSS and `ru_maxrss` measure resident memory, including Python
and native allocations; they do not measure total swapped/compressed backing
storage, GPU memory, or whole-system pressure. In particular, macOS compression
can mask pressure. Lower the threshold if other applications need substantial
memory. There is deliberately no `RLIMIT_AS`: virtual address limits do not
provide a reliable macOS physical-memory budget.

The OS high-water mark (`resource.ru_maxrss`, correctly normalized from bytes
on macOS / KiB on Linux) captures short resident peaks missed by polling in
successful workers. The result retains both that information and the sampled
peak. Reported peaks include imports, sampling, alpha, and all prior stages;
they are **cumulative process peaks**, not Python object sizes, retained heap,
or incremental per-stage allocations. `baseline_import_peak_rss_bytes` gives
context but should not simply be subtracted to infer algorithm allocation.
Failed/killed workers only provide the last checkpoint and sampled peak: these
are lower bounds on what a completed run would require.

## Timing definitions

| Result column | Meaning |
| --- | --- |
| `time_to_barcode_excluding_alpha_s` | Primary package comparison: complete constructor minus alpha build |
| `time_to_barcode_s` | Point cloud → fully computed barcode, including alpha and requested interiors |
| `time_forest_s` | Constructor setup, filtration traversal/materialization, forest build, reduction and activity; excludes alpha, barcode and interiors |
| `time_barcode_s` | Normal cycle/diff barcode method |
| `time_interior_s` | Additional interior reconstruction, when enabled |
| `time_barcode_with_interior_s` | Barcode plus interior reconstruction |
| `time_alpha_s` | GUDHI AlphaComplex initialization + create_simplex_tree |
| `alpha_pct_to_barcode` | 100 × alpha / total time to barcode |
| `time_landscape_s` | Complete fresh measurement-landscape call, including representative iterator, functionals, kernels, evaluation and sorting |
| `time_pipeline_s` | Alpha + forest + barcode/interiors + one landscape functional |
| `landscape_pct_pipeline`, `alpha_pct_pipeline` | Respective share of that full pipeline |
| `landscape_pct_excluding_alpha` | Landscape share with alpha removed from the denominator |
| `time_sampling_s`, `time_from_sampling_s` | Separate data generation time, and generation + pipeline |
| `peak_rss_to_forest_bytes` | Cumulative peak just before barcode computation |
| `peak_rss_to_barcode_bytes` | Cumulative peak through completed barcode and interiors |
| `peak_rss_bytes` | Whole-worker peak, including landscape if requested |

Point cloud → pipeline timings start with an existing array, excluding synthetic
sampling. Timers exclude imports, endpoint/count diagnostics and checkpoint I/O.
A benchmark-local subclass wraps existing phase methods; a worker-local alpha
wrapper times creation and returns the original GUDHI simplex tree. It neither
supplies a manual filtration nor duplicates tree storage. Iterating/extracting
the alpha filtration is assigned to forest setup, matching the constructor's
own separation. Additional raw build/reduction/activity timers are retained.
Small wrapper overhead remains; very small cases are primarily smoke checks.

Landscape functionals are `constant_one_functional` and
`signed_chain_edge_length` in 2D. In 3D, codimension-one representatives are
2-chains, so the geometric comparison uses `signed_chain_volume` (intrinsic
triangle area / boundary surface area, **not enclosed volume**). This keeps two
functionals per dimension. Default measurement is unsigned, as in the public
API; `--signed` selects signed chains in a distinct experiment. Even constant-one
still pays iterator and unsigned-conversion costs. All bars are included
(`min_bar_length=0`), with 5 levels and 512 grid points by default. Both family
and functional caching are disabled. Each functional gets its **own fresh
process and forest**, avoiding cross-functional cache/allocator contamination.
Its full-pipeline denominator contains that functional only, never both.

## Data and plots

Each output directory contains:

- `manifest.json`: schedule, sampler parameters, package/Python/platform details,
  source/lockfile SHA-256 hashes, Git revision/status, and thread policy.
- `results.jsonl`: durable raw records after every attempt; canonical data for
  plotting/resume. On resume it is compacted to complete latest records.
- `results.csv`: a convenient flat export with seconds, bytes, seeds, modes,
  counts, percentages, exit statuses and diagnostic hashes. Missing fields
  remain empty, rather than looking like successful zero-time runs.
- `workers/`: per-case specification, most recent checkpoint and stdout/stderr
  log, including partial progress if a process is killed. Retrying replaces
  that case's worker files while prior JSONL attempts remain until compaction.

Generate plots independently, including for partially finished runs:

```sh
MPLCONFIGDIR=/tmp/persforest-benchmark-mpl .venv/bin/python benchmarks/performance/plot.py \
  benchmarks/performance/results/barcode \
  benchmarks/performance/results/landscapes \
  benchmarks/performance/results/diff-scaling
```

Each run gets a `plots/` directory containing PDF/PNG runtime and memory scaling,
phase times, alpha/landscape percentages, completion fractions, and matched
standard/diff-only ratios (values above one favor diff-only). Landscape plots
also compare functionals directly. `summary.csv` contains median/min/max and
success counts; `paired_ratios.csv` contains ratios for identical successful
cloud/functional/repeat pairs; `failures.csv` lists unsuccessful cases when any
exist. Plots never pool distinct manifests or mix the extended diff-only
schedule into four-mode aggregates. Diff-scaling overlaps the common schedule
at its endpoints to support later comparisons.

Review completion plots and sample counts with timing curves: successful-only
medians at memory limits can be biased. Cloud and exact sorted-barcode-endpoint
hashes are diagnostics, not mathematical proofs or tolerance-aware correctness
tests; investigate mismatches before interpreting mode comparisons. The
benchmark does not assume diff-only is faster or smaller: those are hypotheses
for the measured results.

Generated datasets/logs/plots can be large. Keep `results/` out of version control
(for example using your local Git exclude configuration), or choose an output
directory outside this repository. No generated scientific data is included.

## Lightweight verification of the harness

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-benchmark-mpl \
  .venv/bin/python -m unittest discover -s tests/tests-by-agents \
  -p 'test_performance_harness.py' -v
```

These tests use small sampler arrays, mocked scientific stages, and synthetic
plot records. They do not construct a real alpha complex or PersistenceForest,
and they do not provide empirical benchmark results.
