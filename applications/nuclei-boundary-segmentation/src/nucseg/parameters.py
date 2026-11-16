"""Parameters of the 2D and 3D steps.

Lengths are in micrometers, areas in square micrometers, and intensities in
units of the ``normalize_percentile`` intensity of the prepared image, so the
parameters depend neither on the pixel size nor on the brightness of a stack.
"""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from typing import Any


class _ParameterMixin:
    def to_dict(self) -> dict[str, Any]:
        """The parameters as a plain dict (for JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]):
        """Parameters from a dict; unknown keys are an error, missing ones keep their default."""
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class Parameters(_ParameterMixin):
    """Parameters of the 2D segmentation.

    Attributes
    ----------
    z_first, z_last : int or None
        Slices (inclusive) used for the projections; ``None`` means the first
        or last slice of the stack. Restrict them to the slices through the
        nuclei: slices far above or below add work and out-of-focus loops.
    window : int
        Slices averaged per projection (at most the number of slices used).
        Averaging lowers noise and closes small gaps in the envelope ring;
        choose it to cover about 1 µm (3 slices at 0.3 µm).
    window_step : int
        Distance between consecutive windows.
    bin_factor : int
        Pixels averaged along each axis before building the forest, for
        speed. Binning too coarsely blurs thin envelope rings and loses
        nuclei. Halving it makes the forests about 4-5 times slower; at full
        resolution of a 2048 x 2048 image a forest needs about 12 GB.
    smooth_um : float
        Standard deviation of the Gaussian smoothing after binning.
    normalize_percentile : float
        Percentile of each projection that is scaled to 1.
    cycle_position : float
        Where in its bar the region of a loop is taken, as a fraction of the
        lifetime after birth. Small values give the outline just inside the
        envelope; the region shrinks towards the brightest inside point later.
    min_lifetime : float
        Smallest lifetime of a nucleus loop.
    min_area_um2, max_area_um2 : float
        Area range of a nucleus, or of a piece to be merged.
    min_width_um : float
        Smallest width of a candidate: twice the radius of the largest disk
        inside it.
    min_solidity : float
        Smallest ratio of area to convex hull area of a candidate.
    border_frame : bool
        Surround each projection with a one-pixel frame as bright as its
        brightest pixel, so that nuclei cut by the image edge become loops
        too. Their regions end at the edge; their area is that of the visible
        part.
    max_overlap : float
        Largest fraction of the smaller region that may overlap a kept one.
    merge : bool
        Merge pieces of a nucleus cut by an envelope fold.
    wall_um : float
        Widest gap between two pieces that counts as a shared wall.
    notch_depth : float
        Depth of a pinch, relative to the equivalent radius of the smaller
        piece, that counts as one end of a waist.
    clean_outlines : bool
        Remove spurs from the outlines. Builds a second forest per projection.
    clean_smooth_um : float
        Smoothing of the projections on which the clean outlines are found.
    clean_reach_um : float
        How far the clean outline may grow back towards the original one.
    """

    z_first: int | None = None
    z_last: int | None = None
    window: int = 3
    window_step: int = 2
    bin_factor: int = 4
    smooth_um: float = 0.26
    normalize_percentile: float = 99.8
    cycle_position: float = 0.05
    min_lifetime: float = 0.08
    min_area_um2: float = 25.0
    max_area_um2: float = 800.0
    min_width_um: float = 5.0
    min_solidity: float = 0.75
    border_frame: bool = False
    max_overlap: float = 0.2
    merge: bool = True
    wall_um: float = 1.0
    notch_depth: float = 0.15
    clean_outlines: bool = False
    clean_smooth_um: float = 0.8
    clean_reach_um: float = 0.5

    def z_range(self, n_slices: int) -> tuple[int, int]:
        """First and last slice (inclusive) used in a stack of ``n_slices``."""
        first = 0 if self.z_first is None else self.z_first
        last = n_slices - 1 if self.z_last is None else self.z_last
        if not 0 <= first <= last < n_slices:
            raise ValueError(f"z range {first}..{last} outside the slices 0..{n_slices - 1} of the stack")
        return first, last

    def windows(self, n_slices: int) -> list[tuple[int, int]]:
        """Slice ranges ``(start, stop)`` (stop exclusive) of the projections.

        Windows of ``window`` slices start at ``z_first`` and every
        ``window_step`` slices after it, as long as they end by ``z_last``. A
        z range shorter than ``window`` gives one window over the whole range,
        so a single 2D image is one projection.
        """
        if self.window < 1 or self.window_step < 1:
            raise ValueError("window and window_step must be at least 1")
        first, last = self.z_range(n_slices)
        size = min(self.window, last - first + 1)
        return [(start, start + size) for start in range(first, last - size + 2, self.window_step)]


@dataclass(frozen=True)
class Parameters3D(_ParameterMixin):
    """Parameters of the 3D step.

    Attributes
    ----------
    z_step_um : float or None
        Distance between slices; ``None`` uses the one read from the file.
        Real samples are often stretched or squashed along z by the
        refractive-index mismatch, so this is a free parameter.
    smooth_um : float
        Standard deviation of the Gaussian smoothing of the binned stack.
    normalize_percentile : float
        Percentile of the binned, smoothed stack that is scaled to 1.
    margin_um : float
        Margin around the seed's bounding box in the crop.
    cycle_position : float
        Where in its bar a void's inside is taken, as a fraction of the
        lifetime after birth (as in 2D).
    min_lifetime : float
        Bars shorter than this are not considered.
    min_seed_overlap : float
        Smallest fraction of the seed that the chosen void must cover.
    mask_smooth_um : float
        Standard deviation of the Gaussian smoothing of the inside; 0 switches
        smoothing off.
    """

    z_step_um: float | None = None
    smooth_um: float = 0.3
    normalize_percentile: float = 99.8
    margin_um: float = 3.0
    cycle_position: float = 0.05
    min_lifetime: float = 0.02
    min_seed_overlap: float = 0.3
    mask_smooth_um: float = 0.5
