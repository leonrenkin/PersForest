# Persistence Forests and Measurement Landscapes

Implementation accompanying the manuscript on persistent cycle progressions
and measurement landscapes, available at
https://doi.org/10.48550/arXiv.2512.09668.

## What this repo provides
- `PersistenceForest` (primary entry point) builds the forest of optimal cycles for an alpha complex, together with barcodes and cycle representatives over the filtration.
- Plotting and animation methods for cycle representatives and barcodes in codimension 1.
- Measurement landscapes using cycle functionals such as length, enclosed area and excess curvature.
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
forest.plot_barcode(min_bar_length=0.01, coloring="forest")
forest.plot_at_filtration(0.1)

# 4) Measurement landscapes
grid = np.linspace(0.0, 0.5, 512)
family = forest.compute_measurement_landscapes(
    cycle_func=signed_chain_edge_length,
    max_k=5,
    x_grid=grid,
    label="edge-length",
)
forest.plot_measurement_landscapes(label="edge-length")

# Sample the first five landscape levels on the grid
values = family.evaluate_on_grid(grid, levels=5)
plt.show()
```
Run the script quickstart with:
```bash
python examples/pers_forest_example.py
```

## Tutorials
For a guided introduction, start with `examples/tutorials/README.md`.
The tutorial notebooks are intended to be read in this order:

1. `examples/tutorials/01_visualizing_cycle_representatives.ipynb` - plot barcodes and cycle representatives in 2D and 3D.
2. `examples/tutorials/02_extracting_cycle_representatives.ipynb` - extract representatives as simplices, coordinates and planar paths.
3. `examples/tutorials/03_animating_cycle_representatives.ipynb` - create filtration and measurement animations.
4. `examples/tutorials/04_measurement_landscapes.ipynb` - compute, plot and vectorize measurement landscapes.

## Measurement Landscapes
- Define cycle functionals in `persforest/cycle_rep_vectorisations.py` (examples: edge length, area, connected components, signed/unsigned variants).
- `forest.compute_measurement_landscapes(...)` builds families for one functional; `plot_landscape_comparison_between_functionals` contrasts multiple labels.
- Use `family.evaluate_on_grid(grid, levels=max_k)` to sample landscape values numerically.

## Repository guide
- `persforest/PersistenceForest.py` - forest construction, barcodes, plotting wrappers and measurement landscapes.
- `persforest/cycle_rep_vectorisations.py` - cycle functionals for measurement landscapes.
- `persforest/forest_landscapes.py` - landscape computation, evaluation and comparison utilities.
- `persforest/forest_plotting.py` - shared barcode, forest and animation plotting helpers.
- `persforest/simplicial_filtration_plotting.py` - Matplotlib filtration plotting.
- `persforest/simplicial_filtration_plotly.py` - Plotly filtration plotting.
- `examples/tutorials/` - guided tutorial notebooks.
- `examples/pers_forest_example.py` - compact runnable quickstart.
- `examples/animation_tutorial.ipynb` - animation example.
- `examples/benchmark.py` - runtime benchmark script.
- `examples/paper-examples.ipy` - manuscript figure examples.
- `examples/point_cloud_sampling.py` - synthetic point-cloud samplers used by examples.

## Notes
- MP4 animation export requires `ffmpeg`; GIF export requires the `animation` extra.
- Plotly figures require the `plotly` or `notebook` extra.
