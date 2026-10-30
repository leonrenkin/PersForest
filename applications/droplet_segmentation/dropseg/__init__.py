"""Segmentation of bright droplets in 2D images with persistence forests.

Quick start::

    import dropseg

    raw, full_scale = dropseg.load_image("image.tif")
    image = dropseg.preprocess(raw, full_scale)              # intensities in [0, 1]
    segmentation = dropseg.segment_image(image, dropseg.Parameters(upsample=4), workers=4)
    segmentation.labels                                      # 0 = background, 1, 2, ... = droplets
    dropseg.save_results("results", "image", image, segmentation, dropseg.Parameters(upsample=4))

``Tuner`` (needs ipywidgets) shows every step on a window of the image with
sliders for its parameters. ``python -m dropseg --help`` runs the
segmentation from the command line. See ``README.md`` for the parameters.
"""

from .images import Preprocessing
from .images import downsample_labels
from .images import gaussian_upsample
from .images import image_files
from .images import load_image
from .images import preprocess
from .parameters import STEPS
from .parameters import Parameters
from .results import boundary_overlay
from .results import save_results
from .results import segment_table
from .whole_image import Segmentation
from .whole_image import segment_image

__all__ = [
    "Parameters",
    "Preprocessing",
    "STEPS",
    "Segmentation",
    "Tuner",
    "boundary_overlay",
    "downsample_labels",
    "gaussian_upsample",
    "image_files",
    "load_image",
    "preprocess",
    "save_results",
    "segment_image",
    "segment_table",
]


def __getattr__(name):
    # The tuner needs ipywidgets, which the command line does not.
    if name == "Tuner":
        from .tuning import Tuner
        return Tuner
    raise AttributeError(f"module 'dropseg' has no attribute {name!r}")
