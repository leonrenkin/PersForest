"""Nuclei in 3D: voids of the superlevel filtration around the 2D outlines.

In a z-stack the nuclear envelope is a closed bright surface around a dark
inside, so every nucleus is a void (a 2-cycle) of the superlevel filtration of
the stack, the 3D counterpart of the loops in 2D. Searching the whole stack is
unnecessary: the 2D segmentation already says where the nuclei are.

Pipeline
--------
1. The stack is binned like the 2D projections, smoothed with a Gaussian of
   ``smooth_um`` along all three axes (in µm, so the z step matters), and
   scaled so its ``normalize_percentile`` is 1.
2. Per 2D nucleus (the *seed*), a crop of its bounding box plus
   ``margin_um`` through all slices is taken, and a cap as bright as the
   brightest crop voxel is added below the first and above the last slice.
   A stack is often only about as deep as a nucleus, and the top and bottom
   of the envelope lie in the focal plane where the axial blur makes them
   faint, so without caps most nuclei would not close. A nucleus closed by a
   cap is marked; its z extent is then a lower bound. If the seed reaches the
   image edge (2D ``border_frame``), the crop also gets a bright wall on that
   side, so that nuclei cut by the edge close too; they are marked ``border``.
3. A cubical persistence forest is built on the negated crop. For every bar
   with lifetime at least ``min_lifetime``, the representative at
   ``cycle_position`` of the bar is reconstructed and its inside filled. The
   nucleus is the void covering most of the seed, the 2D region placed at the
   centre slice of its projection window. The longest bar is the wrong
   choice: it is usually a small pocket next to a cap.
4. Optionally the inside is smoothed: a Gaussian of ``mask_smooth_um`` on
   the binary mask, thresholded at one half.

The 3D step only searches near the 2D nuclei, so it can lose a nucleus but
not find a new one. Labels are those of the 2D table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from persforest import PersistenceForest

from ._grid import bin_xy
from ._grid import normalize
from ._grid import unbin_xy
from .io import Stack
from .parameters import Parameters3D
from .segment2d import Region
from .segment2d import Segmentation


@dataclass
class Nucleus3D:
    """One nucleus in 3D.

    Masks live on the crop: slices ``0 .. n_slices - 1`` (no caps) and binned
    rows ``rows[0]:rows[1]``, columns ``cols[0]:cols[1]``.

    Attributes
    ----------
    label : int
        Label of the seed in ``Segmentation.table``.
    seed : Region
        The 2D nucleus.
    rows, cols : tuple of int
        Crop in binned pixels (stop exclusive).
    mask : ndarray of bool or None
        Inside of the void; ``None`` if no void covers enough of the seed.
    smooth_mask : ndarray of bool or None
        ``mask`` after smoothing (equal to ``mask`` without smoothing).
    birth, death, lifetime : float
        Bar of the void, in negated normalized intensity.
    seed_overlap : float
        Fraction of the seed covered by the void.
    border : bool
        The seed reaches the image edge, so the crop got a bright wall there
        (the 3D counterpart of the 2D ``border_frame``). The volume is that of
        the visible part.
    """

    label: int
    seed: Region
    rows: tuple[int, int]
    cols: tuple[int, int]
    mask: NDArray[np.bool_] | None
    smooth_mask: NDArray[np.bool_] | None
    birth: float = np.nan
    death: float = np.nan
    lifetime: float = np.nan
    seed_overlap: float = 0.0
    border: bool = False

    @property
    def found(self) -> bool:
        return self.mask is not None

    @property
    def capped(self) -> bool:
        """Whether the void reaches the first or last slice, so a cap closes it."""
        return self.found and bool(self.mask[0].any() or self.mask[-1].any())

    def full_mask(self, shape: tuple[int, int, int], smooth: bool = True) -> NDArray[np.bool_]:
        """The mask placed in the binned stack of ``shape``."""
        out = np.zeros(shape, dtype=bool)
        if self.found:
            out[:, self.rows[0]:self.rows[1], self.cols[0]:self.cols[1]] = self.smooth_mask if smooth else self.mask
        return out


@dataclass
class Segmentation3D:
    """Result of ``segment_3d``.

    Attributes
    ----------
    parameters : Parameters3D
    z_step_um, um_per_bin : float
        Voxel size along z and along y, x.
    image : ndarray
        The binned, smoothed and normalized stack ``(slices, rows, cols)``.
    nuclei : list of Nucleus3D
        One per seed, ordered by label.
    bin_factor : int
        Binning in x and y relative to the original stack.
    shape : tuple of int
        Shape of the original stack.
    """

    parameters: Parameters3D
    z_step_um: float
    um_per_bin: float
    image: NDArray[np.float64]
    nuclei: list[Nucleus3D]
    bin_factor: int
    shape: tuple[int, int, int]

    @property
    def spacing(self) -> tuple[float, float, float]:
        return (self.z_step_um, self.um_per_bin, self.um_per_bin)

    def labels(self, smooth: bool = True, full_resolution: bool = False) -> NDArray[np.int32]:
        """Label stack, 0 for background; labels as in the 2D table.

        Binned by default; with ``full_resolution`` every binned voxel becomes
        a ``bin_factor`` x ``bin_factor`` block in the original stack's shape.
        Where the smoothed masks of two nuclei overlap, the larger label wins.
        """
        out = np.zeros(self.image.shape, dtype=np.int32)
        for nucleus in self.nuclei:
            out[nucleus.full_mask(self.image.shape, smooth)] = nucleus.label
        return unbin_xy(out, self.bin_factor, self.shape) if full_resolution else out

    def table(self, smooth: bool = True) -> list[dict]:
        """One row per nucleus: void bar, volume, centroid, z extent and whether a cap closes it."""
        voxel = float(np.prod(self.spacing))
        area = self.um_per_bin**2
        rows = []
        for n in self.nuclei:
            row = {"label": n.label, "found": n.found, "border": n.border, "area_2d_um2": round(float(n.seed.area_um2), 2),
                   "z_window": f"{n.seed.window[0]}-{n.seed.window[1] - 1}"}
            if n.found:
                mask = n.smooth_mask if smooth else n.mask
                sections = mask.sum(axis=(1, 2))
                zs = np.flatnonzero(sections)
                cz, cr, cc = ndimage.center_of_mass(mask)
                row.update({
                    "volume_um3": round(float(mask.sum() * voxel), 1),
                    "volume_raw_um3": round(float(n.mask.sum() * voxel), 1),
                    "centroid_x_um": round(float((n.cols[0] + cc + 0.5) * self.um_per_bin), 2),
                    "centroid_y_um": round(float((n.rows[0] + cr + 0.5) * self.um_per_bin), 2),
                    "centroid_z_um": round(float((cz + 0.5) * self.z_step_um), 2),
                    "z_first": int(zs[0]),
                    "z_last": int(zs[-1]),
                    "height_um": round(float(len(zs) * self.z_step_um), 2),
                    "z_largest": int(np.argmax(sections)),
                    "largest_section_um2": round(float(sections.max() * area), 2),
                    "bottom_cap": bool(n.mask[0].any()),
                    "top_cap": bool(n.mask[-1].any()),
                    "lifetime_3d": round(float(n.lifetime), 4),
                    "seed_overlap": round(float(n.seed_overlap), 3),
                })
            rows.append(row)
        return rows


def prepare_stack(data: NDArray, um_per_pixel: float, bin_factor: int, z_step_um: float,
                  parameters: Parameters3D) -> NDArray[np.float64]:
    """Bin every slice, smooth in µm along all axes, and scale the ``normalize_percentile`` to 1."""
    binned = bin_xy(data, bin_factor)
    if parameters.smooth_um > 0:
        um_per_bin = um_per_pixel * bin_factor
        sigma = (parameters.smooth_um / z_step_um, parameters.smooth_um / um_per_bin, parameters.smooth_um / um_per_bin)
        binned = ndimage.gaussian_filter(binned, sigma)
    return normalize(binned, parameters.normalize_percentile)


# ----- voids of one crop -----

def void_inside(chain, shape: tuple[int, int, int]) -> NDArray[np.bool_]:
    """Voxels enclosed by a cubical 2-cycle, holes filled.

    A square with doubled coordinates ``(2z + 1, 2r + 1, 2c)`` separates the
    voxels ``(z, r, c - 1)`` and ``(z, r, c)``; along every row of voxels, one
    is inside when an odd number of the cycle's squares lie before it.
    """
    toggles = np.zeros((shape[0], shape[1], shape[2] + 1), dtype=np.int8)
    for (a0, a1, a2), _ in chain.unsigned().signed_simplices:
        if a0 & 1 and a1 & 1 and not a2 & 1:
            toggles[a0 // 2, a1 // 2, a2 // 2] ^= 1
    inside = np.bitwise_xor.accumulate(toggles[..., :-1], axis=2).astype(bool)
    return ndimage.binary_fill_holes(inside)


def find_void(crop: NDArray[np.float64], seed: NDArray[np.bool_], spacing: tuple[float, float, float],
              parameters: Parameters3D, walls: tuple[bool, bool, bool, bool] = (False, False, False, False)):
    """The void of a capped crop covering most of ``seed``.

    Parameters
    ----------
    crop : ndarray, shape (slices, rows, cols)
        Normalized intensities, without caps.
    seed : ndarray of bool, same shape
        The 2D region on its centre slice.
    walls : tuple of bool
        Add a one-voxel wall as bright as the caps on the first row, last row,
        first column, last column side: the 3D counterpart of the 2D
        ``border_frame``, so that a nucleus cut by the image edge closes.

    Returns
    -------
    (mask, bar, overlap) or None
        The inside (without caps and walls), the bar, and the fraction of the
        seed it covers; ``None`` if no void covers ``min_seed_overlap`` of the seed.
    """
    r0, r1, c0, c1 = (int(w) for w in walls)
    capped = np.pad(crop, ((1, 1), (r0, r1), (c0, c1)), constant_values=crop.max())
    inner = (slice(1, -1), slice(r0, capped.shape[1] - r1), slice(c0, capped.shape[2] - c1))
    forest = PersistenceForest.from_cubical_grid(-capped, spacing=spacing)
    seed_size = seed.sum()
    best = None
    for bar in forest.barcode:
        lifetime = bar.death - bar.birth
        if lifetime < parameters.min_lifetime:
            continue
        chain = forest.cycle_for_bar_at(bar, bar.birth + parameters.cycle_position * lifetime)
        inside = void_inside(chain, capped.shape)[inner]
        overlap = (inside & seed).sum() / seed_size
        # Nested voids cover the seed equally; prefer the longer-lived one.
        if best is None or (overlap, lifetime) > (best[2], best[1].death - best[1].birth):
            best = (inside, bar, overlap)
    if best is None or best[2] < parameters.min_seed_overlap:
        return None
    return best


def smooth_mask(mask: NDArray[np.bool_], spacing: tuple[float, float, float], sigma_um: float) -> NDArray[np.bool_]:
    """Gaussian of ``sigma_um`` on a binary mask, thresholded at one half, holes filled.

    Thresholding at one half roughly keeps the volume of a smooth body; thin
    protrusions are removed and narrow notches filled.
    """
    if sigma_um <= 0:
        return mask.copy()
    field = ndimage.gaussian_filter(mask.astype(np.float64), [sigma_um / s for s in spacing], mode="constant")
    return ndimage.binary_fill_holes(field >= 0.5)


# ----- whole pipeline -----

def segment_3d(stack: Stack, segmentation: Segmentation,
               parameters: Parameters3D = Parameters3D()) -> Segmentation3D:
    """Find the 3D nucleus of every 2D nucleus of ``segmentation``.

    Parameters
    ----------
    stack : Stack
        The stack the 2D segmentation was computed on. Its ``z_step_um`` is
        used unless ``parameters.z_step_um`` is set.
    segmentation : Segmentation
        Supplies the seeds and the binning.
    parameters : Parameters3D

    Raises
    ------
    ValueError
        Without a z step, for a single slice, or if the stack does not match
        the segmentation.
    """
    z_step = parameters.z_step_um if parameters.z_step_um is not None else stack.z_step_um
    if z_step is None or not z_step > 0:
        raise ValueError("the 3D step needs a z step: none in the file metadata, set Parameters3D(z_step_um=...)")
    if len(stack.data) < 2:
        raise ValueError("the 3D step needs a stack of several slices")
    if tuple(stack.data.shape[1:]) != tuple(segmentation.shape) or stack.um_per_pixel != segmentation.um_per_pixel:
        raise ValueError("the stack is not the one the 2D segmentation was computed on")
    bin_factor = segmentation.parameters.bin_factor
    um_per_bin = segmentation.um_per_bin
    image = prepare_stack(stack.data, stack.um_per_pixel, bin_factor, z_step, parameters)
    spacing = (z_step, um_per_bin, um_per_bin)
    _, h, w = image.shape
    margin = int(np.ceil(parameters.margin_um / um_per_bin))
    nuclei = []
    for label, seed in enumerate(segmentation.nuclei, start=1):
        rr, cc = np.unravel_index(seed.pixels, (h, w))
        rows = (max(int(rr.min()) - margin, 0), min(int(rr.max()) + margin + 1, h))
        cols = (max(int(cc.min()) - margin, 0), min(int(cc.max()) + margin + 1, w))
        crop = image[:, rows[0]:rows[1], cols[0]:cols[1]]
        seed_mask = np.zeros(crop.shape, dtype=bool)
        seed_mask[seed.centre_slice] = seed.mask((h, w))[rows[0]:rows[1], cols[0]:cols[1]]
        # Walls only where the seed itself reaches the image edge; elsewhere a
        # wall could close dark pockets of the neighbourhood into false voids.
        walls = (bool(rr.min() == 0), bool(rr.max() == h - 1), bool(cc.min() == 0), bool(cc.max() == w - 1))
        found = find_void(crop, seed_mask, spacing, parameters, walls)
        if found is None:
            nuclei.append(Nucleus3D(label, seed, rows, cols, None, None, border=any(walls)))
            continue
        mask, bar, overlap = found
        nuclei.append(Nucleus3D(label, seed, rows, cols, mask, smooth_mask(mask, spacing, parameters.mask_smooth_um),
                                bar.birth, bar.death, bar.death - bar.birth, float(overlap), any(walls)))
    return Segmentation3D(parameters, z_step, um_per_bin, image, nuclei, bin_factor, tuple(stack.data.shape))
