# Droplet segmentation with persistence forests

`dropseg` segments bright, roughly round objects in 2D grayscale images:
lipid droplets, puncta, vesicles, condensates. It finds every droplet as a
peak of the image's persistence forest (built with `persforest`), takes one
level-set outline per droplet, splits 8-shaped outlines of touching droplets
at their waist, and grows the outlines into segments. Every parameter is an
intensity in [0, 1] or a length in pixels of the original image, so the same
parameters work at the original resolution and upsampled.

**Suited for:** bright objects on a darker background, isolated or touching,
from a few pixels to tens of pixels across, in 8- or 16-bit images.
**Not suited for:** dark objects on a bright background (invert the image
first), objects with a dark interior or textured inside (e.g. whole cells),
and 3D stacks (segment each plane separately).

## Install

From the root of the repository:

```sh
pip install -e ".[droplets]"        # or: uv sync --extra droplets
```

This installs `persforest` with matplotlib, scipy, pillow, pandas and
ipywidgets. For the notebooks you also need Jupyter (or VS Code with the
Jupyter extension). `dropseg` itself is not installed: run the command line,
the notebooks and your own scripts from this folder, or add it to
`PYTHONPATH`.

## Quick start

**Command line**, from this folder:

```sh
python -m dropseg example_data/simulated_puncta.png --out results
python -m dropseg my_images/ --upsample 4 --smooth-radius 0.3 --min-area 4 --workers 4 --out results
python -m dropseg my_images/ --parameters results/simulated_puncta_parameters.json --out results
```

Every parameter below is an option (`--min-bar-length 0.15`), and
`--parameters` reads a JSON file of parameters, such as the
`_parameters.json` of an earlier run. `--workers 4` segments the tiles of
upsampled or large images on 4 processes (default 1); the results don't
depend on it. `python -m dropseg --help` lists all options.

**Python**:

```python
import dropseg

raw, full_scale = dropseg.load_image("my_image.tif")      # channel=... for color images
image = dropseg.preprocess(raw, full_scale)                # scale to [0, 1]
parameters = dropseg.Parameters(upsample=4, smooth_radius=0.3, min_area=4)
segmentation = dropseg.segment_image(image, parameters, workers=4)
segmentation.labels                                        # 0 = background, 1, 2, ... = droplets
segmentation.original_labels                               # the same on the original pixels
dropseg.save_results("results", "my_image", image, segmentation, parameters)
```

`workers` defaults to 1. More workers are faster for tiled images and give
the same labels; in a script, put the call under `if __name__ == "__main__":`
when `workers > 1` (parallel processes re-import the script).

**Notebooks:**

- [`segmenting_droplets.ipynb`](segmenting_droplets.ipynb): how to use it,
  from loading an image and the choice of resolution to tuning every step
  with sliders and saving the results. Start here, and then run it on your own
  image.
- [`example_large_image.ipynb`](example_large_image.ipynb): automatic
  detection on a simulated 1000 × 1000 px image with 300 droplets.

## Input

- A **2D grayscale** image: PNG, TIFF or JPEG, 8 or 16 bit. For color or
  multichannel images, pass the channel (`load_image(path, channel=1)`,
  `--channel 1`).
- **Intensities in [0, 1].** `preprocess` divides by the largest value of
  the image type (`normalize="full scale"`, 255 for 8 bit). For images that
  use only part of their range, e.g. 12-bit data saved as 16 bit, use
  `Preprocessing(normalize="percentile")` (`--normalize percentile`), which
  maps the 1st and 99.9th percentile to 0 and 1. `segment_image` refuses
  values outside [0, 1].
- `Preprocessing(crop=..., blur=...)` crops the image and smooths noise
  (Gaussian, in pixels) before the segmentation.
- **Droplets touching the image border get no segment:** the forest treats
  everything beyond the border as brighter than the image.

## Output

`segment_image` returns a `Segmentation`:

| Attribute | Meaning |
|---|---|
| `labels` | Integer label image on the grid that was segmented (4× finer with `upsample=4`); 0 is background. |
| `original_labels` | The labels on the original pixels: the most frequent label of each block of fine pixels. |
| `n_segments` | Number of segments. |
| `image` | The image that was segmented, upsampled if `factor > 1`. |
| `factor` | The upsampling factor. |
| `groups` | Per pixel, the outline a segment came from. Touching segments of one group were split at a waist (step 4), and of different groups by the forest (step 2). |

`save_results` (and the command line) write, for an image `name`:

| File | Contents |
|---|---|
| `name_labels.tif` | Labels on the original pixels, 16 bit (32 bit beyond 65535 segments). Opens in Fiji or napari as a label image. |
| `name_labels_4x.tif` | When upsampled: the labels on the fine grid. |
| `name_segments.csv` | Per segment: `area` and equivalent `radius` in original pixels, centroid `row` and `column` (pixel index), `mean` and `max` intensity in [0, 1]. |
| `name_boundaries.png` | The image with the boundaries, each original pixel drawn as `scale` × `scale` pixels. Yellow edges border the background, red edges separate neighbors cut apart by the forest, blue edges lobes of one 8-shaped outline. |
| `name_parameters.json` | Parameters, preprocessing and segment count, readable by `--parameters`. |

## The steps and their parameters

| Step | What it does | Parameter (default) |
|---|---|---|
| 1. peaks | Keeps the peaks that rise at least `min_bar_length` above the saddle to a brighter neighbor or to the background. | `min_bar_length` (0.2) |
| 2. branches | Two peaks separating below `split_level` are two droplets; above it, the dimmer one is merged into the brighter. | `split_level` (0.3) |
| 3. outlines | Takes each droplet's outline at `min_level`, or after its split from a neighbor if that is higher. | `min_level` (0.1); `convexity_tol` (off) |
| 4. lobes | Splits 8-shaped outlines at their waist. A lobe whose brightest pixel does not rise `min_bar_length` above the outline's birth is removed. | `split_lobes` (on), `lobe_waist` (0.75) |
| 5. flooding | Grows each segment from bright to dark down to `min_level`. Touching segments meet in the valley between them. A lobe still smaller than `min_area` joins the neighboring lobe of its outline with the brightest shared boundary. | none (uses `min_level`, `min_area`) |
| 6. smoothing | Alpha opening of each segment: removes protrusions narrower than about 2 × `smooth_radius`. | `smooth_radius` (off) |
| 7. filter | Drops pieces smaller than `min_area` pixels. | `min_area` (2 px) |

All parameters of `dropseg.Parameters`:

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `upsample` | 1 | factor | Segment a grid this many times finer: 1, 2 or 4. |
| `kernel_width` | 0.35 | original px | Width of the Gaussian upsampling kernel. Wider smooths more but blurs touching droplets together. |
| `min_bar_length` | 0.2 | intensity | Minimal rise of a peak above its saddle (step 1) and of a lobe (step 4). |
| `split_level` | 0.3 | intensity | Peaks separating below it are separate droplets (step 2). Must exceed `min_level`. |
| `min_level` | 0.1 | intensity | Level of the outlines (step 3) and floor of the flooding (step 5). Must lie above the background. |
| `convexity_tol` | None | deficit | If set (0.4 to 0.5), step 3 climbs to the first outline whose convex-hull perimeter deficit is at most this. Against messy outlines on a noisy background. |
| `split_lobes` | True | | Split 8-shaped outlines (step 4). |
| `smooth_radius` | None | original px | Alpha opening radius (step 6). Useful only when upsampled, 0.3 to 0.5. |
| `min_area` | 2 | original px | Smallest segment kept (step 7). |
| `birth_offset` | 0.1 | × `min_bar_length` | Rarely changed. Outlines lie at least this far above the bar's birth, past its first wiggly outlines. |
| `lobe_waist` | 0.75 | ratio | Split if the waist is at most this fraction of the lobe's width: s/ρ, where s is half the waist width and ρ the lobe radius, read from the alpha filtration of the outline. Independent of size and resolution. |

The defaults suit small dim puncta (radius 1 to 3 px) on a dark background
in 8-bit confocal images. For other data, check `min_level` against the
background first.

## Choosing the parameters

A procedure that works for people and for agents:

1. **Background.** Look at the intensity histogram of the preprocessed image
   (section 2 of the notebook). `min_level` must lie clearly above the
   background (including its noise and any uneven illumination), and
   `split_level` above `min_level`. A background of 0.15 needs a `min_level`
   of about 0.3, for example.
2. **Size.** If the droplets are less than about 6 px across, use
   `upsample=4` with `smooth_radius=0.3` and `min_area=4`. Droplets of 10 px
   and more need no upsampling.
3. **Run and look.** Segment with these values and open
   `_boundaries.png`, or use `plot_segmentation` in the notebook. Judge the
   boundaries by eye on a few typical regions, including touching droplets.
4. **Fix what is wrong**, one parameter at a time, with the table below, or
   step by step with `dropseg.Tuner` (section 5 of the notebook).
5. **Record the parameters**: the `_parameters.json` reruns the same
   segmentation with `--parameters`.

| Symptom | Change |
|---|---|
| Noise or dim haze becomes droplets | raise `min_bar_length` |
| Faint droplets are missed | lower `min_bar_length` (stay above the noise) |
| Segments spread into the background | raise `min_level` |
| Segments miss the dim rim of the droplets | lower `min_level` (stay above the background) |
| Everything is merged into large blobs | raise `min_level`; the background is above it |
| Two touching droplets with a dim valley between them are one segment | raise `split_level` |
| A single droplet with two maxima near its top is cut in two | lower `split_level` |
| Touching droplets without a valley (an 8-shape) are one segment | raise `lobe_waist` (0.85), or upsample |
| A single elongated droplet is cut in two | lower `lobe_waist` (0.6) |
| Outlines are staircases of a few pixels, sub-pixel gaps not resolved | `upsample=4` |
| Ragged boundaries at 4× | `smooth_radius` 0.3 to 0.5 |
| Warning: "smoothing cut … segment(s) into several pieces" | lower `smooth_radius` |
| Tiny fragments remain | raise `min_area` |
| Messy outlines wandering through a noisy background | `convexity_tol=0.4` |

## Resolution and upsampling

`upsample=4` segments a 4× finer image. Each fine pixel is a Gaussian-weighted
average of the original pixels (`kernel_width`). Its level sets are smooth
curves, so outlines of small droplets become smooth and waists narrower than
a pixel show. Since lengths are in original pixels, the parameters don't
change with the resolution. The total number of droplets found changes
little: the segments fit better, and touching droplets are cut along the
valley between them.

## Speed and memory

Upsampled images, and images of more than 600 × 600 px at the original
resolution, are segmented in tiles of `tile_size` (64) original pixels, on `workers`
processes (default 1). Each tile has a margin of `tile_margin` (12)
pixels, which should exceed the largest droplet with its bright
surroundings. A tile that cuts a droplet it owns is redone with the margin
doubled, up to twice. For droplets larger than about 10 px, pass a larger
margin, e.g. `tile_size=128, tile_margin=32` (`--tile-size`, `--tile-margin`).

Typical times on an Apple M4 Pro: a 256 × 256 px image takes about 2 s at 1×
(one forest, `workers` doesn't matter), and about 30 s at 4× on one process or
9 s on 4. A 1000 × 1000 px image with large droplets takes about
25 s at 1× on one process or 7 s on 8. Memory stays below about 1 GB per
process. Results are identical in every run and for any number of workers.

## Files

| File | Contents |
|---|---|
| `dropseg/__init__.py` | The public interface. |
| `dropseg/__main__.py` | The command line. |
| `dropseg/parameters.py` | `Parameters` and the steps that read them (`STEPS`). |
| `dropseg/whole_image.py` | `segment_image` and `Segmentation`. |
| `dropseg/images.py` | Loading, preprocessing, upsampling. |
| `dropseg/results.py` | `save_results`, `segment_table`, `boundary_overlay`. |
| `dropseg/tuning.py`, `dropseg/stages.py` | The `Tuner` and its cached step-by-step runner. |
| `dropseg/plots.py` | Figures used in the notebooks. |
| `dropseg/pipeline.py` | The steps as functions, on one grid. |
| `dropseg/peaks.py`, `outlines.py`, `lobes.py`, `segments.py` | Steps 1–2, 3, 4 and 5–7. |
| `dropseg/tiles.py` | Tiled segmentation. |
| `dropseg/simulate.py` | Simulated droplet images with known labels, for the examples and tests. |
| `example_data/` | The simulated example image and the script that makes it. |

The tests are in `tests/tests-by-agents/test_droplet_segmentation.py` at the
root of the repository.
