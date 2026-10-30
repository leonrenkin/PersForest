"""The parameters of the segmentation: one or two per step, the same at every resolution.

Lengths are in pixels of the original image and intensities in [0, 1], so the
same parameters segment the image at its own resolution (``upsample=1``) or
upsampled (``upsample=4``). ``Parameters.settings`` converts them to the
units of the grid the steps run on.
"""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
import math

from .pipeline import Settings

# The parameters each step reads; a step also depends on those of all earlier steps.
STEPS = {
    "peaks": ("upsample", "kernel_width", "min_bar_length"),
    "branches": ("split_level",),
    "outlines": ("min_level", "convexity_tol", "birth_offset"),
    "lobes": ("split_lobes", "lobe_waist"),
    "flooding": (),
    "smoothing": ("smooth_radius",),
    "filter": ("min_area",),
}


@dataclass(frozen=True)
class Parameters:
    """Parameters of the segmentation; lengths in original pixels, intensities in [0, 1].

    The first block is meant to be tuned; the second has defaults relative to
    the first and rarely needs changing.

    Attributes
    ----------
    upsample : int
        Segment on a grid this many times finer (1, 2 or 4). Use 4 when the
        droplets are only a few pixels across.
    kernel_width : float
        Width of the Gaussian upsampling kernel, original pixels; wider
        smooths more and lowers small peaks.
    min_bar_length : float
        Step 1 (peaks): minimal rise of a peak above the saddle to its
        neighbor or the background. Also the minimal rise of a lobe in step 4.
    split_level : float
        Step 2 (branches): peaks separating from their neighbor below this
        level are droplets of their own; above it, they are part of the neighbor.
    min_level : float
        Step 3 (outlines): outlines are taken at this level, or after the
        droplet's last split below ``split_level`` if that is higher.
        Step 5 (flooding): segments grow down to this level. Must lie below
        ``split_level`` and above the background.
    convexity_tol : float or None
        Step 3: if set, the outline search climbs to the first outline whose
        convex-hull perimeter deficit is at most this (but not past
        ``split_level``); None takes the first outline. Meant against messy
        outlines on a noisy background: about 0.4 to 0.5.
    split_lobes : bool
        Step 4 (lobes): split 8-shaped outlines at their waist.
    smooth_radius : float or None
        Step 6 (smoothing): alpha opening radius, original pixels; None to
        skip. Useful when upsampled, about 0.3 to 0.5.
    min_area : float
        Step 7 (filter): smallest segment kept, original pixels.
    birth_offset : float
        Step 3: outlines lie at least this fraction of ``min_bar_length``
        above the bar's birth, past the wiggly outlines just after it.
    lobe_waist : float
        Step 4: in the alpha filtration of the outline's vertices, the bar of
        a split-off lobe is born at the split radius s (half the waist width)
        and dies at the lobe radius rho. The outline is split if
        s / rho <= ``lobe_waist``: the waist is at most that fraction of the
        lobe's width. Independent of the size of the droplet, the resolution
        and the intensities.
    """

    upsample: int = 1
    kernel_width: float = 0.35
    min_bar_length: float = 0.2
    split_level: float = 0.3
    min_level: float = 0.1
    convexity_tol: float | None = None
    split_lobes: bool = True
    smooth_radius: float | None = None
    min_area: float = 2.0

    birth_offset: float = 0.1
    lobe_waist: float = 0.75

    def __post_init__(self):
        if self.min_level >= self.split_level:
            raise ValueError(f"min_level {self.min_level} must lie below split_level {self.split_level}")
        if int(self.upsample) != self.upsample or self.upsample < 1:
            raise ValueError(f"upsample must be a positive integer, got {self.upsample}")
        if self.min_bar_length <= 0:
            raise ValueError(f"min_bar_length must be positive, got {self.min_bar_length}")
        if self.kernel_width <= 0:
            raise ValueError(f"kernel_width must be positive, got {self.kernel_width}")

    def settings(self, storage: str = "cycles") -> Settings:
        """The settings of the steps, in pixels of the upsampled grid.

        ``storage="diffs"`` reconstructs representatives instead of storing
        them, for large grids; the segments are the same.
        """
        factor = self.upsample
        return Settings(
            storage=storage,
            min_bar_length=self.min_bar_length,
            split_level=self.split_level,
            min_level=self.min_level,
            birth_offset=self.birth_offset * self.min_bar_length,
            max_defect=math.inf if self.convexity_tol is None else self.convexity_tol,
            split_lobes=self.split_lobes,
            lobe_waist=self.lobe_waist,
            smooth_radius=None if not self.smooth_radius else self.smooth_radius * factor,
            min_area=self.min_area * factor**2,
        )

    def record(self) -> dict:
        """The parameters as a plain dict, for saving."""
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> "Parameters":
        """Parameters from a dict as written by ``record``; unknown keys raise an error."""
        names = {field.name for field in fields(cls)}
        unknown = set(record) - names
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}; known are {sorted(names)}")
        return cls(**record)
