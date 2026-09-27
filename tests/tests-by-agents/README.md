# AI-generated regression tests

These AI-generated tests accompany the manually written tests in
[`test_persistence_forest_pipeline.py`](../test_persistence_forest_pipeline.py).
They use standard-library `unittest` and the package's existing dependencies.

## Run

From the repository root, with the Python 3.13 environment, run all tests:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-tests-mpl \
  .venv/bin/python -m unittest discover -v
```

[`test_agent_suite.py`](../test_agent_suite.py) includes this directory in normal
discovery. To run only this suite:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-tests-mpl \
  .venv/bin/python -m unittest tests.test_agent_suite -v
```

To run one module:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-tests-mpl \
  .venv/bin/python -m unittest discover -s tests/tests-by-agents -p test_filtration_ties.py -v
```

Use a writable temporary directory for `MPLCONFIGDIR`. Tests run without a display;
Plotly and GIF export checks skip when their optional dependencies are unavailable.
GIF output is created in a temporary directory and removed automatically.

## Coverage

- Literal simplicial filtrations, signed chains, cycle progressions, interiors,
  filtration ties, and half-open activity intervals.
- Analytic geometric measurements, profiles, landscape kernels, and sampled
  landscape values, using independent algebra and integration checks.
- Real GUDHI alpha complexes, public queries, validation, caching, and storage modes.
- Plot data, animation frames, optional dependencies, and small GIF exports.

Fixtures and random seeds are deterministic. `test_existing_regressions.py` loads
the `old_test_*.py` cases; retain the default `test*.py` discovery pattern to avoid
running them twice.
