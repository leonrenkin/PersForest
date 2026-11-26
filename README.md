# Persistence Forests and Measurement Landscapes

`persforest` computes persistent homology in **codimension one** — loops (H1)
of 2D point clouds and images, voids (H2) of 3D point clouds and volumes —
together with a **cycle representative for every bar at every filtration
value**. Each bar of the barcode comes with the progression of cycles that
bound its hole as the filtration grows, so you can see, extract and measure
the actual loops and surfaces, not just their birth and death. The compiled
engine handles millions of points or pixels in seconds.

## Installation

```bash
pip install persforest            # core: NumPy only
pip install "persforest[plot]"    # adds Matplotlib for plots and animations
```

Requires Python 3.11–3.14. Further options (GUDHI for higher dimensions,
Plotly, building from source, development setup) are under
[Installation details](#installation-details).

## Quickstart: barcode and cycle representatives

```python
import numpy as np
from persforest import PersistenceForest

rng = np.random.default_rng(0)
angles = rng.uniform(0, 2 * np.pi, 200)
points = np.c_[np.cos(angles), np.sin(angles)] + rng.normal(0, 0.05, (200, 2))

forest = PersistenceForest(points)     # alpha filtration, H1 of a 2D point cloud
bars = forest.barcode                  # set of PFBar, each with .birth and .death
bar = forest.max_bar()                 # the circle: born ≈ 0.06, dies ≈ 0.86

# The representative of this bar at filtration value 0.5
cycle = forest.cycle_for_bar_at(bar, 0.5)
cycle.signed_simplices                 # {((15, 32), -1), ((15, 46), 1), ...}: oriented edges
cycle.simplices(signed=False)          # [(164, 180), (108, 171), ...]: edges as point indices
cycle.interior                         # oriented triangles enclosed by the cycle
cycle.active_start, cycle.active_end   # it is the representative on [0.499, 0.522)

# Every representative of the bar, from its death back to its birth
for rep in forest.iter_bar_cycle_reps(bar):
    edges = rep.simplices(signed=False)
```

With the `plot` extra, the same forest can be drawn:

```python
import matplotlib.pyplot as plt

forest.plot_barcode(coloring="forest")
forest.plot_at_filtration(0.5)     # complex and active representatives at 0.5
plt.show()
```

## What is computed

**Codimension one.** For a filtration of a *d*-dimensional complex — a 2D or
3D point cloud with its alpha or Delaunay–Čech filtration, or an image or
volume — persforest computes the barcode of homology in dimension *d* − 1:
loops in 2D, closed surfaces (voids) in 3D. Other homology dimensions are not
computed. In this dimension every class is the boundary of a region that is
not yet filled in (a hole), which makes it possible to choose canonical,
geometrically meaningful representatives.

**Cycle representatives.** A bar is born when a new cycle closes off a hole
and dies when the hole is completely filled. In between, the hole shrinks as
cells are added, and its representative changes with it. persforest stores
this whole progression: for every bar and every value in `[birth, death)`
there is a representative cycle, chosen to enclose the smallest possible
volume, and it changes only at the bar's `bar.critical_values` (the last one
is the death).

A representative is a `SignedChain`:

- `signed_simplices` — the cycle as a set of `(simplex, orientation)` pairs,
  where a simplex is a tuple of point indices (edges in 2D, triangles in 3D);
  for cubical grids, cells are keyed by doubled grid coordinates.
- `interior` — the oriented top-dimensional cells (triangles in 2D, tetrahedra
  in 3D) enclosed by the cycle.
- `active_start`, `active_end` — the half-open interval
  `[active_start, active_end)` on which it is the bar's representative.

Ways to get representatives:

| Call | Returns |
| --- | --- |
| `forest.cycle_for_bar_at(bar, value)` | the representative of `bar` at `value` |
| `forest.iter_bar_cycle_reps(bar)` | all representatives of `bar`, from death back to birth |
| `forest.cycle_reps_at(value)` | the representatives of all bars alive at `value` |
| `forest.barcode_cycle_reps(relative_position=0.1)` | one representative per bar, at a relative position in its lifespan |

`forest.interior_simplex_activity()` returns, for each top-dimensional cell,
when it lies inside a representative; `forest.plot_interior_simplex_activity()`
draws it. [Tutorial 02](examples/tutorials/02_extracting_cycle_representatives.ipynb)
walks through extraction.

**Persistence forest.** When a cell splits a hole in two, the new hole's bar
is born as a child of the old one. The bars therefore form a forest:
`forest.parent_bar(bar)` is the bar it split off at its birth (None for a
root), `forest.child_bars(bar)` the bars that split off it, and `bar.tree_id`
names its tree. Which of the two holes keeps the old bar is decided by the
elder rule. `forest.plot_persistence_forest()` draws the forest.

**Measurements.** Evaluating a geometric quantity — length, enclosed area or
volume, circularity, curvature — on the representatives turns each bar into a
function of the filtration value (a *measurement profile*). *Measurement
landscapes* summarize these profiles as vectors, in the way persistence
landscapes summarize a barcode; see
[Measurement profiles and landscapes](#measurement-profiles-and-landscapes).

The mathematics is described in the accompanying paper (see
[Paper and citation](#paper-and-citation)); none of it is needed to use the
package.

## Filtrations

| Input | Constructor | `filtration_type` |
| --- | --- | --- |
| Point cloud | `PersistenceForest.from_points(points)` | `"alpha"` (default), `"delaunay_cech"` |
| Point cloud + simplices | `PersistenceForest.from_simplicial_filtration(points, filtration)` | `"custom_simplicial"` |
| Grid values | `PersistenceForest.from_cubical_grid(values)` | `"top_cells"` (default), `"vertices"` |

`PersistenceForest(points)` is shorthand for `from_points`.

- **Alpha and Delaunay–Čech filtrations** of 2D and 3D point clouds are
  computed by persforest's compiled builders on
  [geogram](https://github.com/BrunoLevy/geogram)'s Delaunay triangulation,
  with exact predicates, in radius units. Four or more dimensions use GUDHI
  (the `gudhi` extra); 1D point clouds are not supported.
  Degenerate inputs (grids, cocircular points) are triangulated by symbolic
  perturbation. Of identical points only the first is used.
  `forest.filtration` lists the top cells and their facets in filtration order.
- **Custom simplicial filtrations** are `(simplex, value)` pairs with
  nondecreasing values, every face listed before its cofaces. The complex must
  be geometrically embedded and its final complex must be contractible.
  The factory warns that it does not verify these requirements.
- **Cubical grids** follow GUDHI's conventions (`top_cells`: entries are
  pixels/voxels; `vertices`: lower-star filtration). `spacing` and `origin`
  place the grid in space. In 2D, bars enclose regions brighter than their
  surroundings; use the negative image for dark objects.
  [Tutorial 07](examples/tutorials/07_plotting_cubical_grids.ipynb) shows the
  grid plots.

## Performance

The times below run from a NumPy array to the complete persistence forest: the
codimension-one barcode (H1 in 2D, H2 in 3D) together with the cycle
representatives of every bar at every filtration value. Single representatives
are then read from the forest on demand. Peak memory is that of the whole
Python process, input included.
Measured on an Apple M4 Pro (macOS, Python 3.13) in November 2026.

**Point clouds** (alpha filtration, uniform random points; median of 3 runs;
Delaunay–Čech takes about the same time):

| Points | 2D time | 2D memory | 3D time | 3D memory |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.01 s | 0.05 GiB | 0.04 s | 0.07 GiB |
| 100,000 | 0.14 s | 0.14 GiB | 0.36 s | 0.33 GiB |
| 1,000,000 | 1.7 s | 0.96 GiB | 4.8 s | 3.0 GiB |
| 3,000,000 | 6.1 s | 2.7 GiB | 17 s | 8.9 GiB |
| 10,000,000 | 24 s | 9.3 GiB | 63 s¹ | 26 GiB¹ |

¹ One run.

**Cubical grids** (top-cell filtration; one run per size). Uniform noise is the
slowest input; smooth images, closer to real data, are 1.4–3× faster:

| Grid | Cells | Smooth field | Uniform noise | Memory |
| --- | ---: | ---: | ---: | ---: |
| 1024² | 1.0 M | 0.20 s | 0.32 s | 0.4 GiB |
| 2048² | 4.2 M | 0.86 s | 1.7 s | 1.2–1.4 GiB |
| 4096² | 16.8 M | 4.2 s | 10 s | 4.6–5.5 GiB |
| 8192² | 67.1 M | 18 s | 52 s | 18 GiB |
| 128³ | 2.1 M | 0.47 s | 0.67 s | 0.8 GiB |
| 256³ | 16.8 M | 5.7 s | 8.7 s | 5.8–6.6 GiB |

Memory grows linearly with the input size, so it usually limits the input size
before time does. The engine is single-threaded; the 3D point-cloud builder
uses about 1.5 cores. The benchmark harness is in
[`benchmarks/performance/`](benchmarks/performance/README.md).

## Measurement profiles and landscapes

A measurement is a callable `(chain, geometry) -> float`. Ready-made ones are in
`persforest.measurements` (`basic`, `shape`, `loops`, `cubical`). Shape
measurements of 2D cycles can be computed on smoothed or taut loops via
`persforest.measurements.loops` (see
[tutorial 08](examples/tutorials/08_cycle_shapes_from_loops.ipynb)).

`compute_measurement_landscapes` does everything in one call:

```python
from persforest.measurements.basic import signed_chain_edge_length

family = forest.compute_measurement_landscapes(
    signed_chain_edge_length,
    label="edge-length",
    max_k=5,
    x_grid=np.linspace(0.0, 1.0, 512),
)
forest.plot_measurement_landscapes(label="edge-length")
```

To reuse the measurements, compute the profiles once and build landscapes from
them:

```python
from persforest.landscapes import landscapes_from_profiles

profiles = forest.compute_measurement_profiles(signed_chain_edge_length, "edge length")
forest.plot_measurement_profiles(profiles)
family = landscapes_from_profiles(profiles, x_grid=grid, max_k=5)
forest.plot_measurement_landscapes(family)
```

Landscapes are sampled on `family.x_grid`; `family.landscapes[k]` is the k-th
level. Landscapes require nonnegative measurements. Measurements default to
`signed=False` (opposite simplices cancel before measuring) in every entry
point, including `animate_barcode_measurement`.

### Additive measurements

Lengths, areas, volumes and cell counts are sums of fixed per-cell weights.
Such measurements are `AdditiveMeasurement` objects, and profiles and landscapes
evaluate them incrementally from the forest's stored diffs instead of
re-measuring every representative, which is much faster for long-lived cycles
with large boundaries. The built-ins `signed_chain_volume` (boundary length in 2D,
area in 3D), `signed_chain_interior_volume`, `cubical_chain_volume` and
`cubical_chain_interior_volume` are additive. All other measurements, such as
shape and loop functionals, are evaluated on each representative as before.

An `AdditiveMeasurement` is still an ordinary `(chain, geometry) -> float`
callable. To define one, give a vectorized weight function returning one weight
per cell and the support it sums over (`"boundary"` or `"interior"`):

```python
import numpy as np
from persforest.measurements.additive import AdditiveMeasurement

edge_count = AdditiveMeasurement(lambda cells, geometry: np.ones(len(cells)), "boundary")
profiles = forest.compute_measurement_profiles(edge_count, "edge count")
```

Weights must not depend on orientation. Boundary cells are counted once per
stored orientation, so with the default `signed=False` cells present in both
orientations cancel.

## Tutorials

Start with [`examples/tutorials/README.md`](examples/tutorials/README.md):

1. Visualizing cycle representatives
2. Extracting cycle representatives
3. Animating cycle representatives
4. Measurement profiles
5. Measurement landscapes
6. Plotting persistence forests
7. Plotting cubical grids
8. Cycle shapes from loops

A droplet segmentation application for 2D microscopy images is in
[`applications/droplet_segmentation/`](applications/droplet_segmentation/README.md).

## Installation details

The base installation needs only NumPy. Optional extras (install with
`pip install "persforest[extra]"`; the quotes keep shells such as zsh from
expanding the brackets):

| Extra | Adds |
| --- | --- |
| `gudhi` | point clouds in four or more dimensions |
| `plot` | Matplotlib plots and animations |
| `plotly` | interactive 2D/3D Plotly figures |
| `notebook` | inline rendering and widgets in Jupyter |
| `examples` | pandas and scikit-learn for the research scripts |
| `droplets` | the droplet segmentation application |

MP4 export additionally needs `ffmpeg` on `PATH`. Changes between releases,
including how to upgrade from 0.2.0, are listed in [`CHANGELOG.md`](CHANGELOG.md).

Binary wheels are available for Linux (x86_64, aarch64), macOS (Intel and Apple
silicon, macOS 11 or later) and Windows (x86_64). On other platforms, and when
installing from a Git checkout, pip compiles a small C++ extension, which needs
a C++17 compiler (Xcode Command Line Tools on macOS, GCC or Clang on Linux,
Visual Studio Build Tools on Windows); CMake and the other build tools are
fetched automatically.

### Development setup

Use [uv](https://docs.astral.sh/uv/), which installs the package editably
together with all extras:

```bash
uv sync --all-extras --locked
uv run python examples/pers_forest_example.py
```

`uv sync` rebuilds the C++ extension when files in `cpp/` or the build
configuration change.

Notebooks are committed without outputs. Enable the Git filter that strips
them (your working copy keeps its outputs) once per clone:

```bash
git config filter.strip-notebook-outputs.clean "python3 tools/strip_notebook_outputs.py"
```

Run the tests with:

```bash
uv sync --all-extras --locked
MPLBACKEND=Agg uv run --locked python -m unittest discover -s tests -t .
```

CI runs the suite on Python 3.11–3.14 (Linux) and on macOS and Windows, and
builds and tests the wheels.

### Repository layout

- `src/persforest/` – the library (`forest.py` is the public facade;
  `filtrations/`, `measurements/`, `landscapes/`, `plotting/` hold the parts).
- `cpp/` – the compiled engine and the 2D/3D Delaunay filtration builders,
  with geogram's Delaunay module vendored in `cpp/third_party/geogram/`.
- `examples/` – tutorials, a quickstart script, and point-cloud samplers.
- `applications/droplet_segmentation/` – droplet segmentation (`dropseg`).
- `applications/nuclei-boundary-segmentation/` – nuclei from envelope
  stainings (`nucseg`), a separately installed package.
- `research/` – manuscript figures and the tendril regression experiment.
- `benchmarks/` – runtime and memory benchmarks.
- `tests/` – hand-written tests and [agent-written tests](tests/tests-by-agents/README.md).

## Paper and citation

This package accompanies the manuscript on persistent cycle progressions and
measurement landscapes, https://doi.org/10.48550/arXiv.2512.09668. Citation
metadata is in [`CITATION.cff`](CITATION.cff).

Release `v0.1.0` is the version of the code used for the results and figures
in the paper. It is archived on Zenodo; the DOI is shown on the GitHub release
page. The `main` branch has since been restructured and its API differs, so
use `v0.1.0` to reproduce the paper:

```bash
pip install "persforest @ git+https://github.com/leonrenkin/persforest.git@v0.1.0"
```

At `v0.1.0`, the figures are produced by `examples/paper-examples.py` (run from
inside `examples/`, requires LaTeX), and the tendril regression experiment lives
in `examples/tendril_regression/`.
