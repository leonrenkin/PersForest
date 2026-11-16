# Nuclei from envelope stainings with persistence forests

`nucseg` finds nuclei in fluorescence z-stacks of a staining that also labels
the nuclear envelope, such as an ER marker. In such images a section through a
nucleus is a bright ring around a dark inside, and in 3D the envelope is a
closed bright surface. `nucseg` finds these rings and surfaces as loops and
voids of the image's persistence forest (built with
[`persforest`](https://github.com/leonrenkin/persforest)):

- **2D:** one outline per nucleus, taken from the slice window where its ring is
  cleanest; with area, centroid and shape.
- **3D** (optional): the closed envelope around every 2D outline; with volume,
  centroid, height and z extent.

Every parameter is a length in µm or an intensity relative to the image, so
the same parameters work across pixel sizes and brightness.

This application is developed inside the `persforest` repository but is not
part of the `persforest` package: it has its own `pyproject.toml` and is
installed separately.

## Suited for

The method needs the nucleus to appear as a **closed bright ring (2D) or shell
(3D) around a darker inside**. Stainings where this is expected:

- ER markers, whose signal is continuous with the nuclear envelope
  (e.g. Sec61β, calnexin, KDEL fusions, PDI, concanavalin A as in Cell Painting);
- nuclear envelope and lamina markers (e.g. lamins, LBR, emerin, nucleoporins).

It has so far been developed and checked on confocal z-stacks of U2OS cells
with an ER marker (about 0.065 µm pixels, 0.3 µm slices). On other stainings,
cell types or microscopes, check the result in the overview figures and expect
to adjust parameters (see [Tuning](#tuning)).

**Not suited for** DNA stainings (DAPI, Hoechst), where a nucleus is a filled
bright blob instead of a ring; a threshold- or blob-based method is the better
choice there. 3D results are lower bounds for nuclei that reach the first or
last slice (marked in the table).

## Install

Python 3.11-3.14. From this folder, with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra notebooks     # creates .venv with nucseg, persforest and the notebook dependencies
```

or with pip, in an environment of your choice, from the repository root:

```sh
pip install -e .                                                       # persforest
pip install -e "applications/nuclei-boundary-segmentation[notebooks]"  # nucseg
```

The `notebooks` extra adds pandas, plotly (interactive 3D view) and a Jupyter
kernel; the command line and the Python API need neither.

**Running the notebooks.** The notebook kernel must be the environment
`nucseg` is installed in; otherwise the first cell fails with
`ModuleNotFoundError: No module named 'nucseg'`. In VS Code, open a notebook,
click the kernel name at the top right, choose *Select Another Kernel →
Python Environments* and pick `.venv/bin/python` of this folder (or *Enter
interpreter path* if it is not listed). For JupyterLab, register the
environment once and choose the kernel `nucseg` there:

```sh
.venv/bin/python -m ipykernel install --user --name nucseg --display-name nucseg
```

Notebooks run with this folder as working directory, so `data/` and
`results/` are the folders next to them.

## Quick start

**Command line:**

```sh
nucseg data/ --out results                                   # every TIFF in data/, 2D
nucseg data/ --out results --3d                              # 2D and 3D
nucseg stack.tif --out results --z-first 4 --z-last 14 --border-frame true --3d
nucseg data/ --out results --parameters results/stack_parameters.json   # repeat an earlier run
```

Every parameter is an option (`--min-lifetime 0.1`; 3D ones with a `--3d-`
prefix, e.g. `--3d-z-step-um 0.35`). `nucseg --help` lists them all.
`--channel 1` selects the channel of a multichannel file, and
`--um-per-pixel 0.065` sets the pixel size if the file has none.

**Python:**

```python
import nucseg

stack = nucseg.read_stack("stack.tif")      # voxel size from the ImageJ/OME metadata
parameters = nucseg.Parameters(z_first=4, z_last=14, border_frame=True)
segmentation = nucseg.segment_2d(stack, parameters)
segmentation_3d = nucseg.segment_3d(stack, segmentation)            # optional
nucseg.save_results("results", stack, segmentation, segmentation_3d)

segmentation.table()           # one dict per nucleus
segmentation.labels()          # label image, 0 = background
```

**Notebooks:** run them on the TIFFs in `data/`, or on a small synthetic stack
if `data/` is empty.

| notebook | content |
|---|---|
| [`1_quickstart_2d.ipynb`](1_quickstart_2d.ipynb) | segment your stacks in 2D, save and check the results |
| [`2_method_2d.ipynb`](2_method_2d.ipynb) | how the 2D step works, step by step, and how to tune it |
| [`3_quickstart_3d.ipynb`](3_quickstart_3d.ipynb) | 2D and 3D, save the results, interactive 3D view |
| [`4_method_3d.ipynb`](4_method_3d.ipynb) | how the 3D step works, 2D against 3D, sensitivity to the z step |

## Input

- **TIFF z-stacks**: ImageJ TIFFs (also hyperstacks), OME-TIFFs or plain
  multi-page TIFFs, one stack per file, 8 or 16 bit or float. A single 2D
  image works for the 2D step.
- **One channel.** Files with several channels need `channel=` (`--channel`);
  channels are never mixed into z. Time series must be split into one file per
  time point.
- **Voxel size.** The pixel size (and for 3D the slice distance) is read from
  the ImageJ or OME metadata; set it with `um_per_pixel=` / `--um-per-pixel`
  and `Parameters3D(z_step_um=...)` / `--3d-z-step-um` if it is missing or
  wrong. The stage step often differs from the true slice distance in the
  sample (refractive-index mismatch); volumes scale with it.

## Output

For every stack `<name>`, in the output folder:

| file | content |
|---|---|
| `<name>_labels.tif` | 2D labels at the input resolution (0 = background, 1, 2, ... largest first), with pixel size |
| `<name>_nuclei.csv` | per nucleus: area, centroid, width, solidity, `border`, loop lifetime, z window, ... |
| `<name>_parameters.json` | input, voxel size, all parameters and package versions; `--parameters` reads it back |
| `<name>_overview.png`, `<name>_gallery.png` | outlines on the image, to check the result |
| `<name>_labels_3d.tif` | with 3D: label stack at the input resolution, same labels, with voxel size |
| `<name>_nuclei_3d.csv` | with 3D: volume, centroid, height, z extent, `top_cap`/`bottom_cap`, ... |
| `<name>_sections_3d.png` | with 3D: outlines on every slice |

Outlines are found on binned images (`bin_factor`, by default 4 x 4 pixels), so
label images are blocky at the input resolution. Nuclei with `border = True` are
cut by the image edge (only with `border_frame`): their sizes are those of the
visible part, so leave them out of size statistics.

## Using nucseg on a new dataset

Lengths and areas are in µm, and intensities are relative: each projection is
scaled to its own 99.8th percentile, so brightness does not matter. A constant
camera offset is not removed, though, and lowers the relative contrast of the
rings; subtract a large offset (e.g. 100 or 1000 counts in dark regions)
before segmenting. What to check, in this order:

1. **Voxel size.** Read from the file; check the values printed for every
   stack or written to `_parameters.json`. Volumes scale with the z step.
2. **`window`** (default 3 slices): slices averaged per projection, about
   1 µm. Averaging lowers the noise and closes small gaps in the ring; much
   thicker windows blur nuclei at different depths together.
3. **Size limits** for your cell type: `min_area_um2` (25), `max_area_um2`
   (800), `min_width_um` (5). The defaults are broad for mammalian nuclei.
4. **`border_frame`** (off): also find the visible part of nuclei cut by the
   image edge.
5. **Speed:** `bin_factor` (default 4), the pixels averaged along each axis
   before the forests are built, and `z_first`, `z_last` (default: all
   slices). A higher `bin_factor` is faster, but binning too coarsely blurs
   thin envelope rings and loses nuclei; nuclei outside the z range are
   missed. When in doubt, keep the defaults.

Then check the `_overview.png` and `_gallery.png` of a few stacks.
[`2_method_2d.ipynb`](2_method_2d.ipynb) shows every step and ends with a
table of symptoms and the parameters that fix them. All parameters are
documented in `help(nucseg.Parameters)` and `help(nucseg.Parameters3D)`.

**Runtime.** Time and memory grow with the number of binned pixels. On an
Apple M4 Pro, a 2048 x 2048 x 20 stack with the default parameters takes
35-40 s in 2D (about half that with `z_first=4, z_last=14`), and the 3D step
adds 20-30 s; memory stays below about 3 GB. `bin_factor=2` makes the 2D step
about 4 times slower. There is no tiling, so very large images (e.g. 8k x 8k)
need much memory.

## Data

Microscopy data is not stored in this repository: a single 2048 x 2048 x 20
stack is about 160 MB, beyond what belongs in a Git repository. Put your stacks
in `data/` (ignored by Git). `nucseg.write_synthetic_stack(path)` writes a small
synthetic stack for trying the pipeline; the notebooks use it when `data/` is
empty.

Clear notebook outputs before committing: figures of your stacks are stored in
the notebook files.

## How it works

In the superlevel filtration of an image (thresholds from bright to dark), the
ring around a nucleus closes into a loop when the threshold falls below its
weakest point, and the loop is filled when the threshold reaches the brightest
point inside. Its **lifetime** (ring minus inside) measures how clean the ring
is. `nucseg` builds a cubical persistence forest of several projections of
the stack, keeps long-lived loops whose inside has the size and shape of a
nucleus, picks each nucleus from the projection where its ring is cleanest,
and merges pieces of a nucleus split by an envelope fold. In 3D the same
holds one dimension up: each nucleus is a void, found in a crop of the stack
around its 2D outline. The module docstrings of `src/nucseg/segment2d.py` and
`segment3d.py` and the method notebooks describe every step.

## Tests

```sh
uv run --extra notebooks python -m unittest discover tests
```

The tests run the whole pipeline, file reading and writing, and the command
line on the synthetic stack (about 1 min).
