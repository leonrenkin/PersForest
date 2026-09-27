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
Requires Python `>=3.13,<3.14`.

```bash
git clone https://github.com/leonrenkin/persforest.git
cd persforest
pip install .
```

Optional extras:

```bash
# Plotly-based 2D/3D interactive plotting
pip install ".[plotly]"

# Notebook inline rendering (MIME/widget renderers)
pip install ".[notebook]"

# GIF export via Pillow
pip install ".[animation]"

# Extra dependencies for regression/benchmark examples
pip install ".[examples]"
```

## Quickstart
```python
import numpy as np
import matplotlib.pyplot as plt
from persforest import PersistenceForest
from persforest.cycle_rep_vectorisations import signed_chain_edge_length

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
values = family.evaluate_on_grid(grid, levels=5)
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
The algorithm requires geometric embeddedness and a terminal
complex with trivial homology groups. 
These geometric and topological conditions are not checked for custom filtrations; 
the constructor emits a warning about them.

## Difference storage and interior activity

Use `keep_simplex_diff=True` to retain simplex additions and removals.
`diff_only_mode=True` stores only simplex and additions and removals, which drastically reduces memory usage.
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

## Measurement Landscapes
- Define cycle functionals in `persforest/cycle_rep_vectorisations.py` (examples: edge length, enclosed area or volume, connected components, signed/unsigned variants).
- `forest.compute_measurement_landscapes(...)` builds families for one functional; `plot_landscape_comparison_between_functionals` contrasts multiple labels.
- Use `family.evaluate_on_grid(grid, levels=max_k)` to sample landscape values numerically.
- Landscape measurements must be finite and nonnegative

## Repository guide
- `persforest/PersistenceForest.py` - forest construction, barcodes, plotting wrappers and measurement landscapes.
- `persforest/cycle_rep_vectorisations.py` - cycle functionals for measurement landscapes.
- `persforest/forest_landscapes.py` - landscape computation, evaluation and comparison utilities.
- `persforest/forest_plotting.py` - barcode and persistence-forest plotting helpers.
- `persforest/simplicial_filtration_animation.py` - filtration and barcode-panel animation helpers.
- `persforest/simplicial_filtration_plotting.py` - Matplotlib filtration plotting.
- `persforest/simplicial_filtration_plotly.py` - Plotly filtration plotting.
- `persforest/interior_activity_plotting.py` - Matplotlib interior-activity plotting.
- `persforest/interior_activity_plotly.py` - Plotly interior-activity plotting.
- `examples/tutorials/` - guided tutorial notebooks.
- `examples/pers_forest_example.py` - compact runnable quickstart.
- `examples/animation_tutorial.ipynb` - animation example.
- `examples/benchmark.py` - runtime benchmark script.
- `examples/paper-examples.ipy` - manuscript figure examples.
- `examples/point_cloud_sampling.py` - synthetic point-cloud samplers used by examples.

## Notes
- MP4 animation export requires `ffmpeg`; GIF export requires the `animation` extra.
- Plotly figures require the `plotly` or `notebook` extra.

## Tests

There are the manually written tests in
[`test_persistence_forest_pipeline.py`](tests/test_persistence_forest_pipeline.py),
accompanied by AI-generated tests in [`tests/tests-by-agents`](tests/tests-by-agents/README.md)

Run all tests from the repository root with the Python 3.13 environment:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/persforest-tests-mpl .venv/bin/python -m unittest discover -v
```
