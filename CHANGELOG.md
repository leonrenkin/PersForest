# Changelog

## 0.3.0 (unreleased)

The forest is now built by a compiled C++ engine, and 2D/3D point-cloud
filtrations come from a compiled builder on geogram's Delaunay triangulation.
Building a forest is several times faster and uses less memory. Binary wheels
are published on PyPI (`pip install persforest`), so no compiler is needed on
supported platforms.

### Upgrading from 0.2.0

- **`storage` option removed.** Forests always keep diffs, and there are no stored
  cycle representatives (`storage="cycles"` is gone). Use
  `forest.iter_bar_cycle_reps(bar)` and `forest.cycle_for_bar_at(bar, value)`.
- **Nodes are internal.** `forest.nodes`, `forest.cycles`, `forest.roots`,
  `forest.levels`, `active_nodes_at`, `leaves_below_node`, `leaf_to_node_path`,
  `node_to_leaf_path` and `get_root` were removed. The forest structure is
  available through bars:
  - `bar.critical_values`: values at which the representative changes, ending at the death
  - `bar.tree_id`: the tree that contains the bar
  - `bar.index`: a deterministic position among the forest's bars
  - `forest.parent_bar(bar)` and `forest.child_bars(bar)`

  `PFBar.cycle_reps`, `cycle_at_filtration_value`, `is_max_tree_bar` and
  `root_id` were removed; use the forest methods above.
- **`simplex_tree` removed.** `forest.filtration` lists the `(simplex, value)`
  pairs of the forest's filtration.
- **GUDHI is optional.** Install it with the `gudhi` extra
  (`pip install "persforest[gudhi]"`); it is only needed for point clouds in four or
  more dimensions. 1D point clouds now raise `ValueError`.
- **Alpha and Delaunay–Čech filtrations in 2D and 3D** come from the compiled
  builder instead of GUDHI:
  - Filtration values are computed exactly where GUDHI's are rounded, so values
    can differ from 0.2.0 in the last digits. Ties are ordered deterministically,
    and the order can differ from GUDHI's.
  - Of identical points, the first is kept.
  - A point cloud without `dim + 1` affinely independent points gives an empty
    filtration.
  - Coordinates that span too many orders of magnitude (below 2^-100 of the
    largest after scaling) raise `ValueError`.
- **Measurement animations default to `signed=False`**, like every other
  measurement entry point.
- **Plots no longer call `plt.show()` by default.** Every plotting method
  takes `show=False`; pass `show=True` (or call `plt.show()`) in scripts.
  `plot_barcode` gained the `show` parameter. Notebooks are unaffected.
- **Unknown `style`, `style_2d` and `style_3d` keys raise `ValueError`**
  for point clouds as they already did for grids, listing the supported keys.

### Other changes

- `forest.barcode` iterates in `bar.index` order. Ties between bars with equal
  endpoints are broken by `bar.index`, so lifespan rankings, profile IDs and
  `max_bar`/`longest_bars` are the same in every run.
- New figures use Matplotlib's `figure.dpi` unless `dpi` is passed, and plots
  of one kind share a figure size (5×5 scenes, 6×4 charts, 10×6 forests).
- `plot_barcode_measurement` titles the plot with the measurement name and the
  bar interval and labels both axes; `coloring` values other than `"forest"`,
  `"bars"`, `"none"` and `"grey"` raise `ValueError` in barcode and profile plots.
- `cycle_for_bar_at` raises `ValueError` for NaN and for values outside
  `[birth, death)`, and no longer copies the bar's progression.
- Cubical grids whose doubled grid exceeds 2^31 - 1 positions (including thin
  grids in many dimensions, such as shape `(1,) * 20`) raise `ValueError`.
- Incremental signed additive measurements no longer overflow to NaN when huge
  weights of facets enter and leave within one step.
- Droplet segmentation: replaying `*_parameters.json` restores channel,
  preprocessing and tiling unless they are given on the command line.
- The source distribution needs a C++17 compiler; CMake and nanobind are
  fetched by the build. The license is `MIT AND BSD-3-Clause` (geogram).

## 0.2.0 (2026-10-30)

Restructured package; cubical grids (images and volumes), Delaunay–Čech and
custom simplicial filtrations, measurement profiles, loop shape measurements,
diff-based storage and the droplet segmentation application. See the GitHub
release notes.

## 0.1.0 (2026-10-30)

Version used for the results and figures of the paper.
