"""Intensity parameters inferred from the image: thresholds in units of its background noise.

The three intensity parameters are absolute levels in [0, 1], so fixed values
only suit images with a known background and brightness. ``infer_parameters``
derives the ones left unset from the background level ``b`` and the noise
``sigma`` of the image:

- ``min_level = b + LEVEL_NOISE * sigma``: outlines and flooding stay this far
  above the background;
- ``min_bar_length = BAR_NOISE * sigma``: a peak must rise this far above its
  saddle;
- ``split_level = min_level + min_bar_length``: two peaks are
  separate droplets if their saddle lies less than one bar length
  above the outline level.

Thresholds inferred this way move with any rescaling or offset of the
intensities, so the segmentation does not depend on how the image was
normalized (apart from clipping). The factors were calibrated on simulated
images (``simulate``) with known droplets: large and small droplets, dark and
bright backgrounds, uneven backgrounds and several noise levels. They are a
starting point, not a guarantee; check the boundaries on your images.

The levels are tied to the pixel noise. Real images often hold dim structure
above it (haze, out-of-focus light) whose peaks are no droplets; there the
inferred ``min_bar_length`` is too permissive. Inferred levels are therefore
announced with an ``InferredLevelsWarning`` recommending to tune them with
``Tuner`` and pass them explicitly. If the noise cannot be measured (a
background clipped at its lowest value, common after a black-level or
background subtraction, or most dark pixels sharing one intensity) nothing
is inferred and ``estimate_background`` raises: a level from a noise of zero
would split every faint ripple into droplets.
"""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import warnings

import numpy as np
from numpy.typing import NDArray

from .parameters import Parameters

#: ``min_level`` lies this many noise standard deviations above the background.
LEVEL_NOISE = 5.0
#: ``min_bar_length`` is this many noise standard deviations.
BAR_NOISE = 8.0

_TUNE = "tune the levels on a window of the image with dropseg.Tuner and pass them explicitly"
# For errors, which the Tuner raises as well: it then needs starting values.
_GIVE = ("give min_level and min_bar_length, e.g. tuned with dropseg.Tuner(image, window, "
         "Parameters(min_level=..., min_bar_length=...)) from starting values read off the intensity histogram")


class InferredLevelsWarning(UserWarning):
    """Intensity levels were inferred from the image instead of given."""


@dataclass(frozen=True)
class Background:
    """Background level and noise of an image, intensities in [0, 1].

    Attributes
    ----------
    level : float
        Median of the image.
    noise : float
        Robust standard deviation of the pixels at or below the median.
    """

    level: float
    noise: float

    def record(self) -> dict:
        """The estimate as a plain dict, for saving."""
        return asdict(self)


def estimate_background(image: NDArray[np.float64]) -> Background:
    """Estimate the background level and noise of an image of bright droplets.

    The background is the median; the noise is 1.4826 times the median
    distance of the pixels below the median from it, the standard deviation
    of Gaussian noise. Only the dark half of the pixels enters, so droplets
    covering less than about half of the image do not inflate the estimate;
    uneven illumination does, which keeps the thresholds above it.

    Raises
    ------
    ValueError
        If the noise cannot be measured: the median distance is below one
        intensity step of the image, because the background is clipped at
        the lowest intensity or most dark pixels share one value (also for a
        constant image). Levels inferred from such an estimate would be
        arbitrary; give ``min_level`` and ``min_bar_length`` instead.
    """
    image = np.asarray(image, dtype=float)
    level = float(np.median(image))
    values = np.unique(image)
    step = float(np.diff(values).min()) if values.size > 1 else 0.0
    spread = float(np.median(level - image[image <= level]))
    if values.size < 2 or spread < step:
        lowest = float(values[0])
        if level <= lowest:
            share = float((image <= lowest).mean())
            cause = (f"{share:.0%} of the pixels lie at the lowest intensity {lowest:g}, so the background is "
                     "clipped")
        else:
            cause = (f"most pixels below the median {level:g} share its value (intensity step {step:g}), so the "
                     "background is too flat or too coarsely quantized")
        raise ValueError(f"the background noise cannot be estimated: {cause}, and levels inferred from it would "
                         f"be arbitrary; {_GIVE}")
    return Background(level, 1.4826 * spread)


def infer_parameters(image: NDArray[np.float64], parameters: Parameters | None = None,
                     background: Background | None = None) -> Parameters:
    """Fill in the intensity parameters of ``parameters`` that are None from ``image``.

    Parameters
    ----------
    image : ndarray, shape (rows, columns)
        The whole image to be segmented, original pixels, intensities in
        [0, 1] (the input of ``segment_image``). Infer from the whole image,
        not from a tile or window, so that every part is segmented alike.
    parameters : Parameters, optional
        Values given are kept; only ``min_bar_length``, ``split_level`` and
        ``min_level`` left at None are inferred. A ``split_level`` is derived
        from the final ``min_level`` and ``min_bar_length``, given or inferred.
        Default: infer all three.
    background : Background, optional
        Estimate to use instead of ``estimate_background(image)``.

    Returns
    -------
    Parameters
        With all intensity parameters set.

    Warns
    -----
    InferredLevelsWarning
        Whenever a level is inferred, naming the values and recommending
        ``Tuner``.

    Raises
    ------
    ValueError
        If a given ``split_level`` does not lie above the inferred
        ``min_level``, if the inferred ``min_level`` lies above the brightest
        pixel, or if the background must be estimated and cannot be
        (``estimate_background``). With ``min_level`` and ``min_bar_length``
        given, the background is not needed.
    """
    parameters = Parameters() if parameters is None else parameters
    if parameters.resolved:
        return parameters
    if background is None and needs_background(parameters):
        background = estimate_background(image)
    min_level = parameters.min_level
    if min_level is None:
        min_level = background.level + LEVEL_NOISE * background.noise
    min_bar_length = parameters.min_bar_length
    if min_bar_length is None:
        min_bar_length = BAR_NOISE * background.noise
    split_level = parameters.split_level
    if split_level is None:
        split_level = min_level + min_bar_length
    try:
        inferred = replace(parameters, min_bar_length=min_bar_length, split_level=split_level, min_level=min_level)
    except ValueError as error:
        raise ValueError(f"{error}: min_level was inferred from a background of {background.level:.3g} with "
                         f"noise {background.noise:.3g}; set min_level, or a higher split_level") from None
    brightest = float(np.max(image))
    if parameters.min_level is None and min_level >= brightest:
        raise ValueError(f"the inferred min_level {min_level:.3g} (background {background.level:.3g} + "
                         f"{LEVEL_NOISE:g} x noise {background.noise:.3g}) lies above the brightest pixel "
                         f"{brightest:.3g}, so nothing would be segmented: the noise is too strong for the "
                         f"inferred levels; {_GIVE}")
    warnings.warn(describe_inferred(inferred, parameters, background), InferredLevelsWarning, stacklevel=2)
    return inferred


def describe_inferred(inferred: Parameters, given: Parameters, background: Background | None) -> str:
    """What ``infer_parameters`` filled in, and the advice to tune it."""
    levels = ", ".join(f"{name} = {getattr(inferred, name):.3g}" for name in given.unset)
    source = ("" if background is None else
              f" from the background {background.level:.3g} and its noise {background.noise:.3g}")
    return (f"intensity levels inferred{source}: {levels}. Automatic levels are only a starting point "
            f"(dim haze above the pixel noise passes them); {_TUNE}")


def needs_background(parameters: Parameters) -> bool:
    """Whether inferring the missing levels of ``parameters`` needs the background estimate."""
    return parameters.min_level is None or parameters.min_bar_length is None
