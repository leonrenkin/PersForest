"""Random scenes for an honest evaluation of the segmentation.

A scene is drawn from fixed sampling parameters and a seed; its truth is the
droplets' footprints (see ``simulate``), fixed before the segmentation runs.
Scenes are paired: for one seed and layout, the droplets, the noise and the
background field are the same whatever the overlap mode or background
variation, so that the settings can be compared on the same droplets.
"""

from dataclasses import dataclass
from dataclasses import replace
import itertools
import zlib

import numpy as np
from numpy.typing import NDArray

from evaluation import evaluate
from pipeline import relabel
from pipeline import Settings
from pipeline import segment
from simulate import Droplet
from simulate import droplet_image
from simulate import sample_droplets
from simulate import true_labels


@dataclass(frozen=True)
class Scene:
    layout: str
    overlap: str
    background_variation: float
    seed: int
    droplets: list[Droplet]
    image: NDArray[np.float64]
    truth: NDArray[np.int_]

    @property
    def name(self) -> str:
        background = f"background ±{self.background_variation:g}" if self.background_variation else "flat background"
        return f"{self.layout}, {self.overlap}, {background}, seed {self.seed}"


def make_scene(layout: str, overlap: str, background_variation: float, seed: int,
               shape: tuple[int, int] = (192, 192), **sampling) -> Scene:
    """Draw a scene; ``sampling`` is passed to ``sample_droplets``."""
    # The droplets depend on the layout and seed only, not on the other options.
    rng = np.random.default_rng([seed, zlib.crc32(layout.encode())])
    droplets = sample_droplets(rng, shape, layout, **sampling)
    image = droplet_image(droplets, shape, seed=seed, overlap=overlap, background_variation=background_variation)
    return Scene(layout, overlap, background_variation, seed, droplets, image, true_labels(droplets, shape))


def scenes(layouts, overlaps, background_variations, seeds, **kwargs):
    """All combinations of the options, in a fixed order."""
    for layout, seed, overlap, variation in itertools.product(layouts, seeds, overlaps, background_variations):
        yield make_scene(layout, overlap, variation, seed, **kwargs)


def run(scene: Scene, settings: Settings, segment_variants: dict[str, dict] | None = None):
    """Segment a scene and score the result.

    ``segment_variants`` maps names to changes of the segment options
    (``seed_late_peaks``, ``fill_in``, ``boundary_radius``), which are scored on
    the same outlines.
    Returns the result for ``settings`` and the scores of ``settings`` (key
    ``"settings"``) and of every variant.
    """
    result = segment(scene.image, settings)
    scores = {"settings": evaluate(result.labels, scene.truth)}
    for name, changes in (segment_variants or {}).items():
        variant = relabel(result, scene.image, replace(settings, **changes))
        scores[name] = evaluate(variant.labels, scene.truth)
    return result, scores
