# AI-generated regression tests

These tests accompany the hand-written tests in the parent directory
([`test_persistence_forest_pipeline.py`](../test_persistence_forest_pipeline.py) and others).
They use standard-library `unittest`.

## Run

From the repository root, run everything (hand-written and agent tests):

```sh
uv sync --all-extras --locked
MPLBACKEND=Agg uv run --locked python -m unittest discover -s tests -t .
```

[`test_agent_suite.py`](../test_agent_suite.py) pulls this directory into normal
discovery. To run a subset, use the runner with a filename pattern:

```sh
MPLBACKEND=Agg uv run --locked python tests/tests-by-agents/run_restructured_suite.py \
  --implementation src --pattern 'test_measurement*.py'
```

Plotly checks are skipped when the extra is missing, and the MP4 test is skipped
when ffmpeg is missing. `test_installation.py` checks a core-only install with all
optional dependencies blocked.

## What is covered

- Public API, package imports, and installation without plotting extras.
- Simplicial and cubical filtrations, signed chains, cycle progressions,
  interiors, filtration ties, and root normalization.
- Construction resources: flat filtration regions and the cyclic garbage
  collector being paused while bars are created and restored afterwards.
- Golden snapshots of small fixtures (`data/engine_golden.json`): bars, their
  forest structure, every representative with its interior, and interior
  activity. Regenerate them with `python tests/tests-by-agents/engine_golden.py`
  only after an intended change.
- The compiled engine against `reference_engine.py`, a frozen copy of the former
  pure-Python engine, on random cubical, alpha, Delaunay–Čech and fixture
  filtrations (`test_compiled_engine.py`). Do not change the reference's logic.
- The engine's linear-time cubical event order against a lexicographic sort of
  all cells, with ties (`test_cubical_event_order.py`).
- The array-based simplicial adapter against a dict-based numbering of the same
  events, for alpha and Delaunay–Čech in 1D–4D and custom input, with its
  error cases (`test_simplicial_adapter.py`); orientations of the compiled
  engine against single simplices and NumPy determinants
  (`test_orientation_numerics.py`).
- Measurements, measurement profiles and landscapes, checked against
  independent analytic computations.
- Plot data, animations, MP4/HTML export, droplet segmentation, and the
  benchmark harness (with mocked timings).

Fixtures and random seeds are deterministic.
