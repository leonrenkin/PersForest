# Persistence Forests and Measurement Landscapes

Implementation accompanying the manuscript on persistent cycle progressions
and measurement landscapes, available at
https://doi.org/10.48550/arXiv.2512.09668.

## Paper version and citation

Release `v0.1.0` is the version of the code used for the results and figures
in the paper. It is archived on Zenodo; the DOI is shown on the GitHub release
page. The `main` branch has since been restructured and its API differs, so
use `v0.1.0` to reproduce the paper:

```bash
pip install "persforest @ git+https://github.com/leonrenkin/persforest.git@v0.1.0"
```

At `v0.1.0`, the figures are produced by `examples/paper-examples.py` (run from
inside `examples/`, requires LaTeX), and the tendril regression experiment lives
in `examples/tendril_regression/`. Citation metadata is in
[`CITATION.cff`](CITATION.cff).

## What this repo provides

- `PersistenceForest`: the forest of optimal cycles of a filtration, with its
  barcode and cycle representatives. Supports alpha and Delaunay–Čech
  filtrations of point clouds, custom simplicial filtrations, and cubical grids
  (images and volumes).
- Plots and animations of cycle representatives, barcodes and persistence
  forests (Matplotlib and Plotly).
- Measurement profiles and landscapes built from cycle measurements such as
  length, enclosed area or volume, circularity and excess curvature.
- Tutorial notebooks in [`examples/tutorials/`](examples/tutorials/README.md).
- A droplet segmentation application for 2D microscopy images in
  [`applications/droplet_segmentation/`](applications/droplet_segmentation/README.md).

## Installation

Requires Python 3.11–3.14. The base installation needs only NumPy and GUDHI;
plotting dependencies are optional extras.

```bash
git clone https://github.com/leonrenkin/persforest.git
cd persforest
python -m pip install ".[plot]"
```

| Extra | Adds |
| --- | --- |
| `plot` | Matplotlib plots and animations |
| `plotly` | interactive 2D/3D Plotly figures |
| `notebook` | inline rendering and widgets in Jupyter |
| `examples` | pandas and scikit-learn for the research scripts |
| `droplets` | the droplet segmentation application |

MP4 export additionally needs `ffmpeg` on `PATH`.

For development, use [uv](https://docs.astral.sh/uv/), which installs the
package editably together with all extras:

```bash
uv sync --all-extras --locked
uv run python examples/pers_forest_example.py
```

## Quickstart

```python
import numpy as np
import matplotlib.pyplot as plt
from persforest import PersistenceForest
from persforest.measurements.basic import signed_chain_edge_length

rng = np.random.default_rng(0)
pts = rng.random((300, 2))

forest = PersistenceForest(pts)            # alpha filtration
forest.plot_barcode(coloring="forest")
forest.plot_at_filtration(0.1)

family = forest.compute_measurement_landscapes(
    signed_chain_edge_length,
    label="edge-length",
    max_k=5,
    x_grid=np.linspace(0.0, 0.5, 512),
)
forest.plot_measurement_landscapes(label="edge-length")
plt.show()
```

`PersistenceForest` is the public entry point. Other functions are imported
from their modules, e.g. `persforest.measurements.shape` or
`persforest.chains.SignedChain`.

## Filtrations

| Input | Constructor | `filtration_type` |
| --- | --- | --- |
| Point cloud | `PersistenceForest.from_points(points)` | `"alpha"` (default), `"delaunay_cech"` |
| Point cloud + simplices | `PersistenceForest.from_simplicial_filtration(points, filtration)` | `"custom_simplicial"` |
| Grid values | `PersistenceForest.from_cubical_grid(values)` | `"top_cells"` (default), `"vertices"` |

`PersistenceForest(points)` is shorthand for `from_points`. The tracked
homology has dimension `dim - 1`: H1 of a 2D point cloud or image, H2 of a 3D
point cloud or volume.

- **Custom simplicial filtrations** are `(simplex, value)` pairs with
  nondecreasing values, every face listed before its cofaces. The complex must
  be geometrically embedded and its final complex must have trivial homology;
  this is not checked.
- **Cubical grids** follow GUDHI's conventions (`top_cells`: entries are
  pixels/voxels; `vertices`: lower-star filtration). `spacing` and `origin`
  place the grid in space. In 2D, bars enclose regions brighter than their
  surroundings; use the negative image for dark objects.
  [Tutorial 07](examples/tutorials/07_plotting_cubical_grids.ipynb) shows the
  grid plots.

## Representatives and storage

`storage="diffs"` (default) stores only the changes between consecutive
representatives and reconstructs them on demand. It is faster and uses less
memory. `storage="cycles"` stores every representative, which pays off when
querying many single filtration values or rendering long animations.

Get representatives with `forest.iter_bar_cycle_reps(bar)` or
`forest.cycle_for_bar_at(bar, value)`. `forest.interior_simplex_activity()`
returns when each top-dimensional cell lies inside a representative; plot it
with `forest.plot_interior_simplex_activity()`.

## Measurement profiles and landscapes

A measurement is a callable `(chain, geometry) -> float`. Ready-made ones are in
`persforest.measurements` (`basic`, `shape`, `loops`, `cubical`). Shape
measurements of 2D cycles can be computed on smoothed or taut loops via
`persforest.measurements.loops` (see
[tutorial 08](examples/tutorials/08_cycle_shapes_from_loops.ipynb)).

`compute_measurement_landscapes` does everything in one call. To reuse the
measurements, compute the profiles once and build landscapes from them:

```python
from persforest.landscapes.family import landscapes_from_profiles

profiles = forest.compute_measurement_profiles(signed_chain_edge_length, "edge length")
forest.plot_measurement_profiles(profiles)
family = landscapes_from_profiles(profiles, x_grid=grid, max_k=5)
```

Landscapes are sampled on `family.x_grid`; `family.landscapes[k]` is the k-th
level. Landscapes require nonnegative measurements.

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

## Repository layout

- `src/persforest/` – the library (`forest.py` is the public facade;
  `filtrations/`, `measurements/`, `landscapes/`, `plotting/` hold the parts).
- `examples/` – tutorials, a quickstart script, and point-cloud samplers.
- `applications/droplet_segmentation/` – droplet segmentation (`dropseg`).
- `research/` – manuscript figures and the tendril regression experiment.
- `benchmarks/` – runtime and memory benchmarks.
- `tests/` – hand-written tests and [agent-written tests](tests/tests-by-agents/README.md).

## Tests

```bash
uv sync --all-extras --locked
MPLBACKEND=Agg uv run --locked python -m unittest discover -s tests -t .
```

CI runs the suite on Python 3.11–3.14.
