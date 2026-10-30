"""Command line: ``python -m dropseg IMAGE [IMAGE ...] --out FOLDER [parameters]``.

Segments every image and writes the files of ``results.save_results`` for
each. Every field of ``Parameters`` is an option (``--min-bar-length 0.15``);
``--parameters FILE`` reads them from a JSON file, either a plain
``{"upsample": 4, ...}`` dict or the ``_parameters.json`` a previous run
wrote, and options given on the command line override it.
"""

from dataclasses import fields
from pathlib import Path
import argparse
import json
import sys
import time

from .images import IMAGE_SUFFIXES
from .images import Preprocessing
from .images import load_image
from .images import preprocess
from .parameters import Parameters
from .results import save_results
from .whole_image import segment_image

_HELP = {
    "upsample": "segment on a grid this many times finer; 4 for droplets only a few pixels across",
    "kernel_width": "width of the upsampling kernel, original pixels",
    "min_bar_length": "step 1: minimal rise of a peak above its saddle, intensity",
    "split_level": "step 2: peaks separating below this level are separate droplets",
    "min_level": "steps 3 and 5: outlines are taken and segments grown down to this level",
    "convexity_tol": "step 3: climb to the first outline with at most this convex-hull deficit; 'none' for off",
    "split_lobes": "step 4: split 8-shaped outlines (true/false)",
    "smooth_radius": "step 6: alpha opening radius, original pixels; 'none' for off",
    "min_area": "step 7: smallest segment kept, original pixels",
    "birth_offset": "step 3: outlines lie at least this fraction of min_bar_length above the bar's birth",
    "lobe_waist": "step 4: split if the waist is at most this fraction of the lobe's width",
}


def _optional_float(text: str) -> float | None:
    return None if text.lower() in ("none", "off") else float(text)


def _bool(text: str) -> bool:
    if text.lower() in ("true", "yes", "1", "on"):
        return True
    if text.lower() in ("false", "no", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def _type(field):
    if field.type in (bool, "bool"):
        return _bool
    if field.type in (int, "int"):
        return int
    if "None" in str(field.type):
        return _optional_float
    return float


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dropseg",
        description="Segment bright droplets in 2D grayscale images with persistence forests.",
        epilog="Example: python -m dropseg example_data/simulated_puncta.png --upsample 4 --out results")
    parser.add_argument("images", nargs="+", type=Path,
                        help=f"image files or folders of images ({', '.join(IMAGE_SUFFIXES)})")
    parser.add_argument("--out", type=Path, required=True, help="folder for the results")
    parser.add_argument("--parameters", type=Path, help="JSON file with parameters: a plain dict, or the _parameters.json of an earlier run")
    group = parser.add_argument_group("parameters (defaults: dropseg.Parameters)")
    for field in fields(Parameters):
        group.add_argument(f"--{field.name.replace('_', '-')}", dest=field.name, type=_type(field), default=None,
                           metavar=field.type.__name__.upper() if isinstance(field.type, type) else "VALUE",
                           help=f"{_HELP[field.name]} (default {field.default})")
    images = parser.add_argument_group("images")
    images.add_argument("--channel", type=int, help="channel of a color image")
    images.add_argument("--normalize", choices=("full-scale", "percentile"), default="full-scale",
                        help="divide by the largest value of the image type, or stretch two percentiles to 0 and 1")
    images.add_argument("--percentiles", type=float, nargs=2, default=(1.0, 99.9), metavar=("LOW", "HIGH"),
                        help="percentiles mapped to 0 and 1 with --normalize percentile")
    images.add_argument("--blur", type=float, default=0.0, help="Gaussian blur against noise, pixels")
    run = parser.add_argument_group("run")
    run.add_argument("--workers", type=int, default=1,
                     help="processes segmenting tiles in parallel, e.g. 4; the results do not depend on it")
    run.add_argument("--tile-size", type=int, default=64, help="tile size for upsampled images, original pixels")
    run.add_argument("--tile-margin", type=int, default=12,
                     help="tile margin, original pixels; must exceed the largest droplet")
    run.add_argument("--overlay-scale", type=int, default=8,
                     help="output pixels per original pixel in the boundary overlay")
    run.add_argument("--vmax", type=float, help="brightness shown as white in the overlay")
    return parser


def read_parameters(arguments) -> Parameters:
    """``Parameters`` from the JSON file, overridden by the options given."""
    values = {}
    if arguments.parameters is not None:
        record = json.loads(arguments.parameters.read_text())
        values = record.get("parameters", record)
    for field in fields(Parameters):
        given = getattr(arguments, field.name)
        if given is not None:
            values[field.name] = given
    return Parameters.from_record(values)


def image_paths(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if path.is_dir():
            found += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.exists():
            found.append(path)
        else:
            raise SystemExit(f"error: {path} does not exist")
    return found


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    try:
        parameters = read_parameters(arguments)
    except (ValueError, TypeError) as error:
        raise SystemExit(f"error: {error}")
    preprocessing = Preprocessing("full scale" if arguments.normalize == "full-scale" else "percentile",
                                  *arguments.percentiles, blur=arguments.blur)
    print(f"parameters: {parameters}")
    for path in image_paths(arguments.images):
        clock = time.perf_counter()
        raw, full_scale = load_image(path, arguments.channel)
        image = preprocess(raw, full_scale, preprocessing)
        segmentation = segment_image(image, parameters, arguments.workers, arguments.tile_size, arguments.tile_margin)
        written = save_results(arguments.out, path.stem, image, segmentation, parameters, preprocessing,
                               arguments.overlay_scale, arguments.vmax,
                               extra={"file": str(path), "tile_size": arguments.tile_size,
                                      "tile_margin": arguments.tile_margin})
        print(f"{path.name}: {segmentation.n_segments} segments in {time.perf_counter() - clock:.1f} s -> "
              f"{', '.join(p.name for p in written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
