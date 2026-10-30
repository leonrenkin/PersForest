# Persforest Tutorials

These notebooks are the recommended entry point for learning the package. Run
them in order if you are new to `persforest`:

1. `01_visualizing_cycle_representatives.ipynb` - plot barcodes and cycle representatives in 2D and 3D.
2. `02_extracting_cycle_representatives.ipynb` - extract representatives and convert them to simplices, vertex coordinates, and planar paths.
3. `03_animating_cycle_representatives.ipynb` - create 2D animations, barcode-panel animations, barcode-measurement animations, and optional 3D exports.
4. `04_measurement_profiles.ipynb` - compute, inspect, plot, and animate how a cycle measurement evolves along each bar.
5. `05_measurement_landscapes.ipynb` - compute, plot, compare, and vectorize measurement landscapes on a star-shaped example.
6. `06_plotting_persistence_forests.ipynb` - plot a 2D forest and compare it with a vertical barcode using matching colors.
7. `07_plotting_cubical_grids.ipynb` - plot cubical filtrations of images and volumes: image backgrounds, cycles, interior activity, Plotly views and animations.
8. `08_cycle_shapes_from_loops.ipynb` - turn representatives into raw, midpoint, smoothed, averaged and taut planar loops, measure and plot them, and use them in profiles and landscapes.

Optional extras:

```bash
pip install ".[notebook]"
pip install ".[plotly]"
```

MP4 export requires `ffmpeg` on PATH in addition to the plotting dependencies
included by these extras. Interactive 3D HTML export uses Plotly.

Animation exports write to `examples/example_figures/` only when the notebook variable `SAVE_OUTPUTS` is set to `True`.
