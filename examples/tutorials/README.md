# Persforest Tutorials

These notebooks are the recommended entry point for learning the package. Run
them in order if you are new to `persforest`:

1. `01_visualizing_cycle_representatives.ipynb` - plot barcodes and cycle representatives in 2D and 3D.
2. `02_extracting_cycle_representatives.ipynb` - extract representatives and convert them to simplices, vertex coordinates, and planar paths.
3. `03_animating_cycle_representatives.ipynb` - create 2D animations, barcode-panel animations, barcode-measurement animations, and optional 3D exports.
4. `04_measurement_landscapes.ipynb` - compute, plot, compare, and vectorize measurement landscapes on a star-shaped example.

Optional extras:

```bash
pip install ".[notebook]"
pip install ".[plotly]"
pip install ".[animation]"
```

Animation exports write to `examples/example_figures/` only when the notebook variable `SAVE_OUTPUTS` is set to `True`.
