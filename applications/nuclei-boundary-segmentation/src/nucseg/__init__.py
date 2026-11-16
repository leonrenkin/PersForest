"""Nuclear outlines (2D) and volumes (3D) from envelope-like stainings with persistence forests.

Typical use::

    import nucseg

    stack = nucseg.read_stack("stack.tif")                      # voxel size from the metadata
    segmentation = nucseg.segment_2d(stack, nucseg.Parameters(z_first=4, z_last=14))
    segmentation_3d = nucseg.segment_3d(stack, segmentation)    # optional
    nucseg.save_results("results", stack, segmentation, segmentation_3d)

or, from a shell, ``nucseg data/ --out results --3d``.
"""

from .io import Stack
from .io import find_stacks
from .io import read_parameters
from .io import read_stack
from .io import save_results
from .parameters import Parameters
from .parameters import Parameters3D
from .segment2d import Region
from .segment2d import Segmentation
from .segment2d import postprocess
from .segment2d import segment_2d
from .segment3d import Nucleus3D
from .segment3d import Segmentation3D
from .segment3d import segment_3d
from .synthetic import synthetic_stack
from .synthetic import write_synthetic_stack

__all__ = [
    "Nucleus3D",
    "Parameters",
    "Parameters3D",
    "Region",
    "Segmentation",
    "Segmentation3D",
    "Stack",
    "find_stacks",
    "postprocess",
    "read_parameters",
    "read_stack",
    "save_results",
    "segment_2d",
    "segment_3d",
    "synthetic_stack",
    "write_synthetic_stack",
]
