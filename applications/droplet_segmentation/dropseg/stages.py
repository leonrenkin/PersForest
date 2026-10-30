"""The steps of the segmentation on a window of an image, each cached on the parameters it reads.

``tuning.Tuner`` builds its interactive views on ``Stages``: changing a
parameter recomputes only its step and the later ones.
"""

from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

from persforest import PersistenceForest

from .parameters import STEPS
from .parameters import Parameters
from .pipeline import Result
from .pipeline import Settings
from .pipeline import find_lobes
from .pipeline import find_outlines
from .pipeline import find_peaks
from .pipeline import outlines_result
from .pipeline import relabel
from .pipeline import split_forest
from .whole_image import upsampler


class Stages:
    """The steps of the segmentation on a window of an image, each cached on the parameters it reads.

    The window is segmented with a margin around it, so that droplets at its
    edge are complete. Coordinates of the products are fine pixels of the
    upsampled window with margin; ``view`` is the window in those coordinates.

    Parameters
    ----------
    image : ndarray
        The whole image, original pixels.
    window : tuple of slice
        ``(rows, columns)`` in original pixels.
    margin : int
        Original pixels added on every side.
    cache_size : int
        Products kept per step, so that going back to earlier values is fast.
    """

    def __init__(self, image: NDArray[np.float64], window: tuple[slice, slice], margin: int = 12,
                 cache_size: int = 6):
        self.window = window
        self.outer = tuple(slice(max(s.start - margin, 0), min(s.stop + margin, n)) for s, n in zip(window, image.shape))
        self.original = image[self.outer]
        self.cache_size = cache_size
        self._cache = {step: OrderedDict() for step in ("image", "forest", *STEPS)}

    def _cached(self, step: str, key: tuple, compute):
        cache = self._cache[step]
        if key not in cache:
            cache[key] = compute()
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    @staticmethod
    def _key(parameters: Parameters, step: str) -> tuple:
        # A step depends on its own parameters and those of every earlier step.
        names = []
        for name, fields in STEPS.items():
            names += fields
            if name == step:
                break
        return tuple(getattr(parameters, name) for name in names)

    def view(self, parameters: Parameters) -> tuple[slice, slice]:
        """The window in fine pixels of the segmented image."""
        factor = parameters.upsample
        return tuple(slice((s.start - o.start) * factor, (s.stop - o.start) * factor)
                     for s, o in zip(self.window, self.outer))

    def image(self, parameters: Parameters) -> NDArray[np.float64]:
        """The window with margin, upsampled."""
        key = (parameters.upsample, parameters.kernel_width)
        return self._cached("image", key, lambda: upsampler(parameters)(self.original))

    def settings(self, parameters: Parameters) -> Settings:
        return parameters.settings()

    def forest(self, parameters: Parameters) -> PersistenceForest:
        key = (parameters.upsample, parameters.kernel_width)
        return self._cached("forest", key, lambda: PersistenceForest.from_cubical_grid(
            self.image(parameters), storage=self.settings(parameters).storage))

    def peaks(self, parameters: Parameters) -> list:
        """Step 1: the candidate bars."""
        return self._cached("peaks", self._key(parameters, "peaks"),
                            lambda: find_peaks(self.forest(parameters), self.settings(parameters)))

    def branches(self, parameters: Parameters) -> tuple:
        """Step 2: ``(branches, outlined bars, late bars)``."""
        return self._cached("branches", self._key(parameters, "branches"), lambda: split_forest(
            self.forest(parameters), self.peaks(parameters), self.settings(parameters)))

    def outlines(self, parameters: Parameters) -> dict:
        """Step 3: the selection of every outlined bar."""
        def compute():
            branches, outlined, _ = self.branches(parameters)
            return find_outlines(self.forest(parameters), branches, outlined, self.settings(parameters))
        return self._cached("outlines", self._key(parameters, "outlines"), compute)

    def lobes(self, parameters: Parameters) -> dict:
        """Step 4: the splits of the examined outlines."""
        return self._cached("lobes", self._key(parameters, "lobes"), lambda: find_lobes(
            self.forest(parameters), self.outlines(parameters), self.image(parameters), self.settings(parameters)))

    def result(self, parameters: Parameters) -> Result:
        """Steps 5 to 7: the segments, with every intermediate product."""
        def compute():
            branches, _, late = self.branches(parameters)
            image = self.image(parameters)
            partial_result = outlines_result(self.forest(parameters), self.peaks(parameters), branches,
                                             self.outlines(parameters), late, self.lobes(parameters), image.shape)
            return relabel(partial_result, image, self.settings(parameters))
        return self._cached("filter", self._key(parameters, "filter"), compute)
