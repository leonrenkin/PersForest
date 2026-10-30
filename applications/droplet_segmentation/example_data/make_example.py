"""Regenerate ``simulated_puncta.png``: small, dense droplets rendered like an 8-bit camera image.

Run from ``applications/droplet_segmentation``: ``python example_data/make_example.py``.
"""

from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dropseg.simulate import camera_scene  # noqa: E402

SHAPE, N_DROPLETS, SEED = (256, 256), 720, 0


def main():
    scene = camera_scene(SEED, SHAPE, N_DROPLETS)
    folder = Path(__file__).parent
    Image.fromarray(np.round(scene.image * 255).astype(np.uint8)).save(folder / "simulated_puncta.png")
    print(f"{folder / 'simulated_puncta.png'}: {SHAPE[0]} x {SHAPE[1]} px, {N_DROPLETS} droplets")


if __name__ == "__main__":
    main()
