# Persistence Forests and Measurement Landscapes

Implementation accompanying the manuscript on persistent cycle progressions
and measurement landscapes, available at
https://doi.org/10.48550/arXiv.2512.09668.

## What this repo provides
- `PersistenceForest` (primary entry point) builds the forest of optimal cycles for an alpha complex by default, together with barcodes and cycle representatives over the filtration. Custom simplicial filtrations are also supported under the requirements below.
- Plotting and animation methods for cycle representatives, barcodes and persistence forests in codimension 1.
- Measurement landscapes using cycle functionals such as length, enclosed area or volume, and excess curvature.
- Beginner-friendly tutorial notebooks in `examples/tutorials/`.
- Runnable script quickstart in `examples/pers_forest_example.py`.

## Installation
Supports Python 3.11–3.14. The base installation includes NumPy and GUDHI for
computation; visualization dependencies are optional.

```bash
git clone https://github.com/leonrenkin/persforest.git
cd persforest
python -m pip install .
```

The implementation lives in `src/persforest/`. For local development, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
uv sync --all-extras --locked --python 3.13
uv run python examples/pers_forest_example.py
```

uv installs the project editably, so imports resolve to `src/persforest/`.
With pip, the equivalent development installation is
`python -m pip install -e ".[plot,plotly,notebook,examples]"`.
Use `from persforest import PersistenceForest` for the public entry point.
Import helpers from their implementation modules, such as
`from persforest.chains import SignedChain`. Legacy module shims have been removed.

Optional extras:

```bash
# Matplotlib plots and color palettes (needed for the quickstart below)
python -m pip install ".[plot]"

# Plotly-based 2D/3D interactive plotting
python -m pip install ".[plotly]"

# Notebook inline rendering (MIME/widget renderers)
python -m pip install ".[notebook]"

# Extra dependencies for regression/benchmark examples
python -m pip install ".[examples]"
```

The `plotly`, `notebook`, and `examples` extras include the plotting
dependencies they need. Importing the package and computing forests or landscapes
does not load plotting libraries. Missing plotting dependencies produce an
installation hint when a plotting method is called.
The `plot` extra uses Matplotlib directly and does not require Seaborn or pandas.

`uv.lock` records exact development dependencies. The published package declares
broader minimum versions, allowing pip users to resolve compatible dependencies
for their own environment. After changing dependencies, run `uv lock`, verify the
tests, and include the lockfile with the metadata change. Poetry is no longer
required.

## Quickstart
```python
import numpy as np
import matplotlib.pyplot as plt
from persforest import PersistenceForest
from persforest.measurements.basic import signed_chain_edge_length

# 1) Create a point cloud
rng = np.random.default_rng(0)
pts = rng.random((300, 2))

# 2) Build the persistence forest (alpha complex)
forest = PersistenceForest(pts, print_info=True)

# 3) Visualize
forest.plot_barcode(coloring="forest")
forest.plot_at_filtration(0.1)

# 4) Measurement landscapes
grid = np.linspace(0.0, 0.5, 512)
family = forest.compute_measurement_landscapes(
    cycle_func=signed_chain_edge_length,
    max_k=5,
    x_grid=grid,
    label="edge-length",
)
forest.plot_measurement_landscapes(label="edge-length", cmap="viridis")

# Sample the first five landscape levels on the grid
values = family.evaluate_on_grid()
plt.show()
```
Run the script quickstart with:
```bash
python examples/pers_forest_example.py
```

## Filtrations and activity intervals

The default filtration for a point cloud is the alpha filtration with radius values.

To supply a custom filtration, 
use `PersistenceForest(point_cloud, filtration=ordered_pairs)`, 
where each pair is `(simplex_vertex_indices, filtration_value)`. 
Vertex indices refer to rows of `point_cloud`. 
Simplices are stored with increasing vertex order; orientations come from the
point coordinates, so the vertex order of the input does not matter.
Simplices of higher dimension than the ambient space are rejected.
The algorithm requires geometric embeddedness and a terminal
complex with trivial homology groups. 
These geometric and topological conditions are not checked for custom filtrations; 
the constructor emits a warning that results are meaningless if they fail.

## Difference storage and interior activity

Use `keep_simplex_diff=True` to retain simplex additions and removals.
`diff_only_mode=True` stores simplex additions and removals, which drastically reduces memory usage.
Forest-level cycle queries and static filtration and barcode-cycle plots support diff-only mode.
Animation methods raise an error in diff-only mode because reconstructing cycles for every frame can be slow.
With `compute_interior=True`, interior simplices are stored for all cycle representatives.
Both `diff_only_mode=True` and `compute_interior=True` set  `keep_simplex_diff=True`.

With `keep_simplex_diff=True`,
`forest.interior_simplex_activity()` returns the active interior intervals for each full-dimensional simplex. 
Visualize them with
`forest.plot_interior_simplex_activity()` or
`forest.plot_interior_simplex_activity_plotly()`.

## Tutorials
For a guided introduction, start with `examples/tutorials/README.md`.
The tutorial notebooks are intended to be read in this order:

1. `examples/tutorials/01_visualizing_cycle_representatives.ipynb` - plot barcodes and cycle representatives in 2D and 3D.
2. `examples/tutorials/02_extracting_cycle_representatives.ipynb` - extract representatives as simplices, coordinates and planar paths.
3. `examples/tutorials/03_animating_cycle_representatives.ipynb` - create filtration and measurement animations.
4. `examples/tutorials/04_measurement_landscapes.ipynb` - compute, plot and vectorize measurement landscapes.
5. `examples/tutorials/05_plotting_persistence_forests.ipynb` - plot persistence forests and compare them with barcodes.

## Measurement profiles and landscapes

`PersistenceForest` provides both the one-call landscape workflow above and
explicit measurement profiles for reuse:

```python
from persforest.landscapes.family import landscapes_from_profiles
from persforest.landscapes.kernels import landscape_kernel

profiles = forest.compute_measurement_profiles(
    signed_chain_edge_length, "edge length", signed=False,
    measurement_params={"units": "coordinate units"},
)
family = landscapes_from_profiles(profiles, x_grid=grid, max_k=5)
first_kernel = landscape_kernel(profiles, profiles.bar_ids[0])
# Recompute ranks on another grid without measuring or reconstructing cycles.
refined = landscapes_from_profiles(profiles, x_grid=np.linspace(0.0, 0.5, 1024))
```

Measurement profiles own read-only scalar arrays and captured bar endpoints.
`profiles.bars` contains immutable endpoint records; lookup also accepts the
original source bar. Integer bar IDs are local to the result, and selections
preserve their requested order. A missing or duplicate landscape bar selection
raises an error. Results describe the source at measurement time; later source
mutation requires explicit remeasurement if fresh data is wanted. Measurement
callbacks borrow chains and coordinates and must not modify them.
Both the forest method and standalone `compute_measurement_profiles` default
to `signed=False`; use `signed=True` to measure stored signed representatives.

`forest.measurement_profiles[label]` and `forest.landscape_families[label]` are
optional result registries. The one-call method accepts `cache_profiles=True`
to retain profiles alongside a landscape family and publishes them only after
success. Labels can be overwritten without changing previously returned results.
To reuse a result, pass it directly to `landscapes_from_profiles`; there is no
implicit cache-reuse switch or ignored measurement callable. The old
`forest.compute_barcode_functionals(...)` name is a small wrapper around
`forest.compute_measurement_profiles(...)`.

Standalone profiles support signed values, arbitrary baselines and infinite
bar deaths. Landscapes require finite bar/profile endpoints, finite nonnegative
measurements and a zero baseline. Raw signed convolution is available separately
as `convolve_profile(profile, birth, death)`. Invalid modes, malformed intervals,
noninteger levels and unavailable selections raise errors.
Convolution `tol` has coordinate units: it quantizes event coordinates and
ignores intervals narrower than `tol`, but never discards a measurement because
its amplitude is small. Set `tol=0` to disable coordinate quantization and the
narrow-interval cutoff.

Landscape levels are read-only sample arrays: `family.landscapes[k]` contains
values aligned with the single shared `family.x_grid`. Levels are not callable;
only the per-bar kernels remain piecewise-linear callable functions.
`family.evaluate_on_grid()` retrieves all
levels on the stored grid; query coordinates must be exact stored coordinates.
Use `landscapes_from_profiles` to recompute on a new grid. Plot explicit results
with `persforest.plotting.landscapes.plot_landscape_family(family)` or compare a
list of results with `plot_landscape_comparison(families)`. The forest plotting
methods remain convenient label-based entry points.

## Repository guide
- `src/persforest/forest.py` - public facade, construction orchestration and delegates.
- `src/persforest/chains.py`, `barcode.py`, `_forest/graph_model.py` - canonical domain objects.
- `src/persforest/_forest/` - construction, queries, reduction, barcode extraction and representative access.
- `src/persforest/filtrations/` - input adapters behind the internal cell-filtration interface (`_interface.py`); `geometry/` - orientation and geometry.
- `src/persforest/measurements/`, `landscapes/` - measurements and numerical landscapes.
- `src/persforest/plotting/` - rendering, shared scenes, colors and animation.
- `src/persforest/plotting/animation/` - animation dispatch, filtration and measurement renderers, timing, layout and export.
- [Restructuring report](docs/restructuring-report.md) and [definition map](docs/restructuring-map.md) - completed extraction, verification and deferred architecture.
- `examples/tutorials/` - guided tutorial notebooks.
- `examples/pers_forest_example.py` - compact runnable quickstart.
- `examples/animation_tutorial.ipynb` - animation example.
- `examples/point_cloud_sampling.py` - shared synthetic point-cloud samplers.
- `benchmarks/benchmark.py` - legacy runtime benchmark script.
- `benchmarks/performance/` - runtime and memory benchmark harness.
- `research/paper_examples/paper-examples.py` - manuscript figure examples.
- `research/tendril_regression/` - synthetic-data and regression experiments.

## Notes
- MP4 animation export uses the `plot` extra and requires `ffmpeg` on PATH
  (for example, `brew install ffmpeg` on macOS). Video filenames must end in `.mp4`.
- Interactive 3D animations can be exported to Plotly HTML with the `plotly` extra.
- Plotly figures require the `plotly` or `notebook` extra.

## Tests

There are the manually written tests in
[`test_persistence_forest_pipeline.py`](tests/test_persistence_forest_pipeline.py),
accompanied by AI-generated tests in [`tests/tests-by-agents`](tests/tests-by-agents/README.md)

Install all extras and run all tests from the repository root:

```bash
uv sync --all-extras --locked
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-tests-mpl \
  uv run --locked python tests/tests-by-agents/run_restructured_suite.py --implementation src
uv lock --check
```

This runs full unittest discovery and asserts package locations, including in
child processes. After an editable installation, ordinary
`python -m unittest discover` from the repository root also tests the src package.

CI tests Python 3.11–3.14, installs the built wheel outside the checkout, and
checks computation without optional dependencies. A separate Python 3.11 job
checks the declared minimum direct dependency versions.

Build distributions locally with:

```bash
uv build --no-sources
uv run python tests/tests-by-agents/verify_restructured_distribution.py dist
```

This creates a wheel and source distribution for a future PyPI release. Building
does not publish the package.
