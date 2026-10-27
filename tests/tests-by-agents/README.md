# AI-generated regression tests

These AI-generated tests accompany the manually written tests in
[`test_persistence_forest_pipeline.py`](../test_persistence_forest_pipeline.py).
They use standard-library `unittest` and the package's existing dependencies.

## Run

From the repository root, with the Python 3.13 environment, run all tests:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-tests-mpl \
  .venv/bin/python tests/tests-by-agents/run_restructured_suite.py --implementation src
```

[`test_agent_suite.py`](../test_agent_suite.py) includes this directory in normal
discovery. The runner asserts all loaded package paths and guards child processes.
The old root-level package and reference mode have been removed. After editable
installation, plain root-level unittest discovery also selects src. For a focused
src run:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-tests-mpl \
  .venv/bin/python tests/tests-by-agents/run_restructured_suite.py \
  --implementation src --pattern 'test_measurement*.py'
```

To run one module:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-tests-mpl \
  .venv/bin/python tests/tests-by-agents/run_restructured_suite.py \
  --implementation src --pattern test_filtration_ties.py
```

Use a writable temporary directory for `MPLCONFIGDIR`. Tests run without a display;
Plotly and GIF export checks skip when their optional dependencies are unavailable.
GIF output is created in a temporary directory and removed automatically.

## Coverage

- The public `PersistenceForest` entry point, canonical class definitions, and
  removal of legacy import shims (`test_package_imports.py`).
- Literal simplicial filtrations, signed chains, cycle progressions, interiors,
  filtration ties, and half-open activity intervals.
- Deferred barcode-diff accumulation, source ownership, and root-reaching bars
  (`test_barcode_diff_accumulation.py`).
- Barcode-diff readiness, rejection of incomplete interior/reconstruction data,
  empty results, failed builds, and internal builder dispatch hooks
  (`test_barcode_readiness.py`).
- Reduced root normalization, simultaneous births, root representatives, and
  preserved signed cycle progressions across storage modes
  (`test_root_normalization.py`).
- Analytic geometric measurements, profiles, landscape kernels, and sampled
  landscape values, using independent algebra and integration checks.
- Real GUDHI alpha complexes, public queries, validation, caching, and storage modes.
- Plot data, animation frames, optional dependencies, and small GIF exports.
- Animation writer settings, ffmpeg diagnostics,
  paired markers, and 3D cycle/camera updates (`test_animation_package.py`).
- Benchmark scheduling, seeds, RSS units/watchdog, resume behavior, mocked phase
  timing, and synthetic summary plots (`test_performance_harness.py`). These
  checks do not execute real PersistenceForest benchmarks.

Fixtures and random seeds are deterministic. The earlier iterator, barcode
orientation and interior-volume regressions are discovered directly in
`test_bar_cycle_iterator.py`, `test_barcode_orientation.py` and
`test_interior_volume.py`. The obsolete `old_test_*` import bridge was removed;
none of those regression classes was deleted.

The [report](../../docs/restructuring-report.md) records the historical
restructuring baseline and installation checks. The src implementation now rejects
interior queries and diff-based reconstruction when the eager builder did not
prepare accumulated barcode diffs. The root reference and its comparison-only
hash, AST, and differential checks have been retired; behavioral regressions and
canonical import identity checks remain active.

`verify_restructured_distribution.py DIST_DIR` compares built wheel/sdist
members byte-for-byte with the src package and rejects reference/private artifacts.
