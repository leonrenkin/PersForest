# Droplet segmentation

Segments bright, roughly round objects (droplets, puncta) in 2D grayscale
images with the persistence forest of the image: every peak is a bar, and
the representatives of a bar are the nested level-set outlines around its
peak. The pipeline picks one outline per droplet, splits 8-shaped ones, and
grows the outlines into segments.

## Quick start

```python
from droplets import Parameters, segment_image
from real_images import Preprocessing, load_image, preprocess

raw, full_scale = load_image("data_lucy/golgi_dog_16somite_rep3_head_z51.png")
image = preprocess(raw, full_scale, Preprocessing())        # intensities in [0, 1]

segmentation = segment_image(image, Parameters())            # or Parameters(upsample=4), workers=8
segmentation.labels                                          # 0 = background, 1, 2, ... = segments
```

To choose the parameters, open `data_lucy_segmentation.ipynb`: it goes through
the steps one by one on a window of the image, with sliders for each step's
parameters and comparisons of several values side by side.

## The steps and their parameters

Intensities are in [0, 1] and lengths in pixels of the original image, at
every resolution. Defaults are tuned on the `data_lucy` Golgi image.

| Step | What it does | Parameter (default) | What to look for |
|---|---|---|---|
| 1. peaks | Keeps the peaks that rise at least `min_bar_length` above the saddle to a brighter neighbor or to the background (the bar length). | `min_bar_length` (0.2) | Noise peaks are dropped (red crosses), faint puncta are kept (green). |
| 2. branches | Two peaks separating below `split_level` are two droplets. Peaks separating above it are merged into one droplet. | `split_level` (0.3) | Touching puncta are separate; a blob with two maxima near its top stays one. |
| 3. outlines | Takes each droplet's outline at `min_level`, or after its split from a neighbor if that is higher. The outline is always at least `birth_offset` × `min_bar_length` above the bar's birth. | `min_level` (0.1); optionally `convexity_tol` (off) | Outlines hug the punctum without reaching into neighbors. |
| 4. lobes | Splits 8-shaped outlines at their waist, found in the alpha filtration of the outline. A lobe whose brightest pixel doesn't rise `min_bar_length` above the birth of the outline's bar (step 1 for the lobes) is cut off too, keeps the other lobes out of its territory during flooding, and is then removed. | `split_lobes` (on), `lobe_waist` (0.75) | The gallery shows each split; real pairs, not single elongated blobs. Dim lobes are red. |
| 5. flooding | Grows each segment from bright to dark down to `min_level`, the level of the outlines, and no further. Touching segments meet in the valley between them. | none (uses `min_level`) | Segments cover the punctum's flanks; to include more of them, lower `min_level`. |
| 6. smoothing | Alpha opening of each segment's boundary. | `smooth_radius` (off) | Only useful when upsampled; at 1×, radii near 1 px erase puncta. A warning means it cut a segment into several pieces of at least `min_area`: the radius is too large. |
| 7. filter | Drops pieces smaller than `min_area`: whole segments, or parts of a segment cut apart by the smoothing. Brightness is not checked again: step 1 checks every punctum, step 4 every lobe. | `min_area` (2 px) | Red outlines: fragments, not real puncta. |

**Resolution.** `upsample=4` segments a 4× finer image. Each fine pixel is a
Gaussian-weighted average of the original pixels; `kernel_width` (0.35
original px) is the width of that Gaussian. Outlines of small puncta become
smooth, and waists narrower than a pixel show. Upsampled images are
segmented in tiles, which takes about 15 s on 8 workers instead of 5 s for
a 512² image. A tile that cuts a segment it owns (a droplet whose
bright surroundings reach past the 12 px margin) is redone with a doubled
margin. The parameters don't change with the resolution.

**Derived parameters.** These rarely need changing:

| Parameter | Default | Meaning |
|---|---|---|
| `birth_offset` | 0.1 | Outlines lie at least 0.1 × `min_bar_length` above the bar's birth, past the wiggly first outlines. |
| `lobe_waist` | 0.75 | Split an outline if the waist is at most 0.75 × as wide as the split-off lobe. |

The lobe splitting is purely geometric. In the alpha filtration of the
outline's vertices (a point cloud, in pixels), the outline's hole splits at
the radius s, half the width of the waist; the split-off lobe dies at its
radius ρ. The ratio s/ρ measures how much of an 8 the outline is (≈ 1: no
constriction, → 0: two lobes touching at a point), independent of size and
resolution. Lobes must also have a radius of at least 1.2 grid pixels,
because smaller alpha lobes of a pixel staircase are its corners. Pieces
split off too small are removed by `min_area` in step 7. The cut itself
follows the intensity valley: the pixels between the two lobe polygons are
flooded from bright to dark inside the outline.

**Convexity (optional, off).** `convexity_tol` guards against messy
outlines. On a background that is noisy rather than flat 0, the first
outlines after a bar's birth can wander through the noise. With a tolerance
of about 0.4 to 0.5, step 3 climbs from its start to the first outline whose
convex-hull perimeter deficit is at most the tolerance, but not past
`split_level`. Lower tolerances cut into irregular puncta: a deficit of 0.3
is still a reasonable outline. It isn't needed for `data_lucy`; to try it,
use `tuner.step("outlines", extra=("convexity_tol",))`.

## Tuning workflow

```python
from tuning import Tuner

tuner = Tuner(image, window=(slice(230, 290), slice(190, 250)), parameters=Parameters(), vmax=0.5)
tuner.step("peaks")                                          # sliders + view; "branches", "outlines", ...
tuner.compare("min_level", [0.06, 0.08, 0.1, 0.12])          # one parameter at several values
tuner.step("outlines", extra=("birth_offset",))             # also show a derived parameter
tuner.parameters                                             # the values chosen so far
```

Each step caches its result on the parameters it reads. A slider therefore
recomputes only its step and the later ones, which takes well under a
second on a 60 × 60 window, at 4× too. Work through the steps in order:
later steps start from the values chosen in earlier ones.

The sliders change only `tuner.parameters`. The notebook segments the whole
image with the parameters written in its *Parameters* cell, unless
`USE_TUNED = True`. To keep tuned values, copy them into that cell.

## Files

| File | Contents |
|---|---|
| `droplets.py` | Public interface: `Parameters`, `segment_image`, and `Stages`, the cached step-by-step runner. |
| `tuning.py` | Interactive views and sliders for each step (`Tuner`). |
| `pipeline.py` | The steps as functions (`find_peaks`, `split_forest`, `find_outlines`, `find_lobes`, `relabel`) and the full internal `Settings`, which `Parameters` derives. |
| `candidates.py`, `selection.py`, `splitting.py`, `segmentation.py` | Steps 1–2, step 3 (outline selection rules), step 4 (alpha lobes), steps 5–7 (label images, flooding, opening, filter). |
| `tiles.py` | Tiled segmentation of upsampled images. |
| `real_images.py` | Loading, normalizing and upsampling images; look-alike test scenes. |
| `simulate.py`, `evaluation.py`, `benchmark.py` | Simulated droplets and scoring against a known truth. |
| `plots.py` | Shared figure helpers. |
| `data_lucy_segmentation.ipynb` | Step-by-step tuning on the `data_lucy` images, whole-image run, saving, and tests on look-alike scenes. |
| `droplet_segmentation.ipynb` | The method, explained on simulated droplets with the full `Settings`, including the stable-plateau outline rule. |

Results of the notebook are saved in `outputs/data_lucy`: a boundary overlay
PNG, the label array, a per-segment CSV and the parameters as JSON.
