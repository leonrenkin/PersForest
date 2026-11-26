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
- **Plots are no longer shown automatically.** Every plotting method defaults
  to `show=False` (`plot_barcode` gained the parameter); call
  `matplotlib.pyplot.show()` or pass `show=True`. New figures use Matplotlib's
  `figure.dpi` instead of a fixed resolution, and plots of one kind share a
  default size.
- **Unknown style keys raise** for point-cloud plots, as they already did for
  grids.
- **`plot_barcode` shows at most the 100 longest bars by default**, and
  `sort="tree"` groups the bars of each tree.

### Plot appearance

- `coloring="forest"` gives all bars of a tree one color (trees ranked by
  their longest bar); `coloring="bars"` uses tab20 by lifespan rank.
- 2D cycles are drawn as joined lines on a light grey complex.
  `style_2d["cycle_halo"]` adds a white halo, on by default for cubical grids.
- 3D plots draw the complex as a faint shell and active cycles as shaded,
  opaque surfaces. The 3D interior plot shows each bar's region as a surface;
  `style["opacity_by"] = "activity"` restores per-face opacity.
- Barcodes use round-capped bars that fill the rows.
- Landscape, profile and measurement charts have an open frame and a faint
  grid; landscape levels run from dark (k = 1) to light.
- **Plots no longer call `plt.show()` by default.** Every plotting method
  takes `show=False`; pass `show=True` (or call `plt.show()`) in scripts.
  `plot_barcode` gained the `show` parameter. Notebooks are unaffected.
- **New bar colors.** `coloring="forest"` gives every bar of a tree the same
  color from Paul Tol's vibrant palette, trees ranked by their longest bar;
  `coloring="bars"` uses tab20 (dark shades first) by lifespan rank. Both
  repeat once the palette is used up. The color-generation helpers
  (`build_color_scheme`, `distinct_base_colors`, `color_map_for_bars`, …) were
  removed; `set_longest_bar_colors` still overrides colors.
- **New barcode layout.** `plot_barcode` draws at most the 100 longest bars by
  default (`max_bars=0` shows all) in a fixed 6×4 figure, with round-capped
  bars whose width follows the number of rows (`bar_width=None`), only the
  filtration axis, the title on the left (`ax.get_title(loc="left")`) and the
  number of bars shown on the right. `sort="tree"` groups the bars of each
  tree, trees in order of their longest bar.
- **Unknown `style`, `style_2d` and `style_3d` keys raise `ValueError`**
  for point clouds as they already did for grids, listing the supported keys.

### Other changes

- `forest.barcode` iterates in `bar.index` order. Ties between bars with equal
  endpoints are broken by `bar.index`, so lifespan rankings, profile IDs and
  `max_bar`/`longest_bars` are the same in every run.
- 2D cycles are drawn as joined lines, without notches at their corners. 2D
  filtration plots show a light grey complex, smaller dark points and 2.2 pt
  cycles above the points (`vertex_size=5`, `point_zorder=4`);
  `style_2d={"cycle_halo": True}` adds a white halo beneath each cycle, on by
  default for representatives on grayscale images. Representatives default to
  `linewidth_cycle=1.5`. Titles are short (`r = 0.19` for radius
  filtrations, `t = 0.19` otherwise, `representatives at 10% of each bar`),
  and the colorbar label reads downward.
- 3D plots (point clouds and grids) draw the complex as a faint shell of its
  boundary, cycles as opaque shaded surfaces in one depth-sorted collection,
  faint points (`vertex_size=None`: 1.5 in 3D, 5 in 2D) and no axes
  (`style_3d={"remove_axes": False}` restores them). The 3D interior plot
  defaults to `geometry="surface"`: regions of bars at least 25% of the
  longest drawn bar are opaque and shaded, shorter ones a faint tint
  (`style` keys `faint_below`, `opacity_by="activity"` for the former
  per-face opacity).
- New figures use Matplotlib's `figure.dpi` unless `dpi` is passed, and plots
  of one kind share a figure size (5×5 scenes, 6×4 charts, 10×6 forests).
- Charts (landscapes, landscape comparisons, profiles, measurements) have an
  open frame with a faint grid; landscape levels run from dark (λ₁) to light
  along plasma unless `cmap` is given.
- `plot_barcode_measurement` titles the plot with the measurement name and the
  bar interval, labels both axes and draws in the bar's forest color; `coloring` values other than `"forest"`,
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
