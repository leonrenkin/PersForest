"""Command line: ``nucseg STACK [STACK ...] --out FOLDER [--3d] [parameters]``.

Also available as ``python -m nucseg``. Segments every TIFF stack (folders are
expanded) and writes the files of ``save_results`` for each. Every field of
``Parameters`` is an option (``--min-lifetime 0.1``), every field of
``Parameters3D`` an option with the prefix ``--3d-`` (``--3d-margin-um 4``).
``--parameters FILE`` reads the ``_parameters.json`` of an earlier run (or a
plain JSON dict of 2D parameters); options on the command line override it.
"""

from __future__ import annotations

from dataclasses import fields
from dataclasses import replace
from pathlib import Path
import argparse
import sys
import time

from .io import find_stacks
from .io import read_parameters
from .io import read_stack
from .io import save_results
from .parameters import Parameters
from .parameters import Parameters3D

_HELP_2D = {
    "z_first": "first slice used (0-based; default: first of the stack)",
    "z_last": "last slice used, inclusive (default: last of the stack)",
    "window": "slices averaged per projection",
    "window_step": "distance between consecutive projection windows",
    "bin_factor": "pixels averaged per axis before building the forests; higher is faster",
    "smooth_um": "Gaussian smoothing of the projections, µm",
    "normalize_percentile": "percentile of each projection scaled to 1",
    "cycle_position": "where in its bar the outline is taken, fraction of the lifetime",
    "min_lifetime": "smallest loop lifetime (ring minus inside, normalized intensity)",
    "min_area_um2": "smallest nucleus area, µm²",
    "max_area_um2": "largest nucleus area, µm²",
    "min_width_um": "smallest nucleus width, µm",
    "min_solidity": "smallest area / convex hull area",
    "border_frame": "also find nuclei cut by the image edge (true/false)",
    "max_overlap": "largest overlap with a kept nucleus, fraction of the smaller one",
    "merge": "merge pieces of a nucleus cut by an envelope fold (true/false)",
    "wall_um": "widest wall between pieces that are merged, µm",
    "notch_depth": "pinch depth that marks a waist, fraction of the radius",
    "clean_outlines": "cut spurs off the outlines; builds a second forest (true/false)",
    "clean_smooth_um": "smoothing for the clean outlines, µm",
    "clean_reach_um": "how far clean outlines grow back, µm",
}
_HELP_3D = {
    "z_step_um": "slice distance, µm (default: from the file)",
    "smooth_um": "Gaussian smoothing of the stack, µm",
    "normalize_percentile": "percentile of the stack scaled to 1",
    "margin_um": "margin around each 2D nucleus, µm",
    "cycle_position": "where in its bar the inside is taken, fraction of the lifetime",
    "min_lifetime": "smallest void lifetime",
    "min_seed_overlap": "smallest fraction of the 2D nucleus the void must cover",
    "mask_smooth_um": "smoothing of the 3D masks, µm; 0 for off",
}


def _bool(text: str) -> bool:
    if text.lower() in ("true", "yes", "1", "on"):
        return True
    if text.lower() in ("false", "no", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def _converter(annotation):
    # A type (bool, int, float), a union such as ``int | None``, or a string.
    annotation = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    if annotation == "bool":
        return _bool
    if annotation.startswith("int"):
        return int
    return float


def _add_fields(group, cls, helps: dict[str, str], prefix: str) -> None:
    for field in fields(cls):
        group.add_argument(f"--{prefix}{field.name.replace('_', '-')}", dest=f"{prefix}{field.name}",
                           type=_converter(field.type), default=None, metavar="VALUE",
                           help=f"{helps[field.name]} (default {field.default})")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nucseg",
        description="Segment nuclei in z-stacks of an envelope-like staining (e.g. an ER marker) "
                    "with persistence forests: 2D outlines, and with --3d volumes.",
        epilog="Example: nucseg data/ --out results --z-first 4 --z-last 14 --border-frame true --3d")
    parser.add_argument("stacks", nargs="+", type=Path, help="TIFF stacks or folders of them")
    parser.add_argument("--out", type=Path, required=True, help="folder for the results")
    parser.add_argument("--3d", dest="three_d", action="store_true", help="also segment the nuclei in 3D")
    parser.add_argument("--parameters", type=Path,
                        help="the _parameters.json of an earlier run, or a JSON dict of 2D parameters")
    parser.add_argument("--channel", type=int, help="channel of a multichannel file (0-based)")
    parser.add_argument("--um-per-pixel", type=float, help="pixel size in µm (default: from the file)")
    parser.add_argument("--no-figures", dest="figures", action="store_false", help="skip the PNG figures")
    _add_fields(parser.add_argument_group("2D parameters (defaults: nucseg.Parameters)"), Parameters, _HELP_2D, "")
    _add_fields(parser.add_argument_group("3D parameters (with --3d; defaults: nucseg.Parameters3D)"),
                Parameters3D, _HELP_3D, "3d-")
    return parser


def _parameters(args: argparse.Namespace) -> tuple[Parameters, Parameters3D | None]:
    parameters, parameters_3d = read_parameters(args.parameters) if args.parameters else (Parameters(), None)
    values = vars(args)
    parameters = replace(parameters, **{f.name: values[f.name] for f in fields(Parameters)
                                        if values[f.name] is not None})
    changes_3d = {f.name: values[f"3d-{f.name}"] for f in fields(Parameters3D) if values[f"3d-{f.name}"] is not None}
    if changes_3d and not args.three_d:
        raise SystemExit("3D parameters given without --3d")
    if not args.three_d:
        return parameters, None
    return parameters, replace(parameters_3d or Parameters3D(), **changes_3d)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    parameters, parameters_3d = _parameters(args)
    import matplotlib

    matplotlib.use("Agg")
    from .segment2d import segment_2d
    from .segment3d import segment_3d

    paths = find_stacks(args.stacks)
    if not paths:
        print("no TIFF stacks found", file=sys.stderr)
        return 1
    failed = 0
    for path in paths:
        start = time.perf_counter()
        try:
            stack = read_stack(path, channel=args.channel, um_per_pixel=args.um_per_pixel)
            segmentation = segment_2d(stack, parameters)
            segmentation_3d = None if parameters_3d is None else segment_3d(stack, segmentation, parameters_3d)
            save_results(args.out, stack, segmentation, segmentation_3d, figures=args.figures)
        except Exception as error:  # report and go on with the other stacks
            failed += 1
            print(f"{path}: failed: {error}", file=sys.stderr)
            continue
        summary = f"{len(segmentation.nuclei)} nuclei"
        if segmentation_3d is not None:
            summary += f", {sum(n.found for n in segmentation_3d.nuclei)} in 3D"
        print(f"{path.name}: {summary} ({time.perf_counter() - start:.0f} s)")
    print(f"results in {args.out.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
