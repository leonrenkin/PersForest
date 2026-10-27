"""Figure layout, camera settings, and radius labels for animations."""

from .._dependencies import require_plotting

require_plotting()

from matplotlib.offsetbox import AnchoredText
from numbers import Real
from typing import Any, Optional

_RADIUS_BOX_CORNERS = ("upper right", "upper left", "lower right", "lower left")

def _validate_radius_box_corner(corner):
    if corner not in _RADIUS_BOX_CORNERS:
        raise ValueError(f"radius_box_corner must be one of {_RADIUS_BOX_CORNERS}.")

def _add_radius_box(ax, text, corner="upper right"):
    """Place the radius label eight points inside the selected axes corner."""
    _validate_radius_box_corner(corner)
    fontsize = 11
    box = AnchoredText(
        text, loc=corner, prop={"size": fontsize},
        pad=0.3, borderpad=8 / fontsize, frameon=True,
        bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes,
    )
    box.patch.set_boxstyle("round,pad=0,rounding_size=0.15")
    box.patch.set_facecolor((1, 1, 1, 0.9))
    box.patch.set_edgecolor((0.8, 0.8, 0.8, 0.8))
    box.patch.set_linewidth(0.6)
    box.set_zorder(1000)
    ax.add_artist(box)
    return box

def _resolve_matplotlib_figsize(
    *,
    with_barcode: bool,
    dpi: int,
    figsize: Optional[tuple[float, float]] = None,
    pixel_size: Optional[tuple[int, int]] = None,
    # Deprecated aliases
    width: Optional[int] = None,
    height: Optional[int] = None,
    cloud_figsize: Optional[tuple[float, float]] = None,
    total_figsize: Optional[tuple[float, float]] = None,
) -> tuple[tuple[float, float], tuple[int, int]]:
    """
    Resolve a deterministic even-pixel canvas and matching figure size.

    Parameters
    ----------
    with_barcode : bool
        Whether the figure contains a barcode panel.
    dpi : int
        Dots per inch used to convert between inches and pixels.
    figsize : tuple[float, float] | None
        Preferred figure size in inches.
    pixel_size : tuple[int, int] | None
        Preferred canvas size in pixels. Takes precedence over ``figsize``.
    width, height : int | None
        Deprecated pixel-size aliases; both must be supplied together.
    cloud_figsize, total_figsize : tuple[float, float] | None
        Deprecated figure-size aliases used when ``figsize`` is omitted.

    Returns
    -------
    tuple
        ``(figsize_inches, pixel_size_even)``.
    """
    if int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer.")

    def _even(v: int) -> int:
        return v if (v % 2 == 0) else (v + 1)

    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("width and height must be provided together.")
        if pixel_size is None:
            pixel_size = (int(width), int(height))

    if pixel_size is not None:
        if len(pixel_size) != 2:
            raise ValueError("pixel_size must be a 2-tuple (width, height).")
        px_w = _even(max(2, int(round(float(pixel_size[0])))))
        px_h = _even(max(2, int(round(float(pixel_size[1])))))
        return (px_w / float(dpi), px_h / float(dpi)), (px_w, px_h)

    if figsize is not None:
        base_w, base_h = float(figsize[0]), float(figsize[1])
    else:
        if with_barcode:
            if total_figsize is None:
                base_w, base_h = 10.0, 5.0
            else:
                base_w, base_h = float(total_figsize[0]), float(total_figsize[1])
        else:
            if cloud_figsize is None:
                base_w, base_h = 6.0, 6.0
            else:
                base_w, base_h = float(cloud_figsize[0]), float(cloud_figsize[1])

    px_w = _even(max(2, int(round(base_w * float(dpi)))))
    px_h = _even(max(2, int(round(base_h * float(dpi)))))
    return (px_w / float(dpi), px_h / float(dpi)), (px_w, px_h)

def _resolve_figure_margins(
    *,
    with_barcode: bool,
    figure_margins: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Resolve validated Matplotlib subplot margins.

    Parameters
    ----------
    with_barcode : bool
        Selects defaults for one-panel or two-panel layouts.
    figure_margins : dict[str, float] | None
        Optional dict with exactly ``left``, ``right``, ``bottom`` and ``top``.

    Returns
    -------
    dict[str, float]
        Margins suitable for ``fig.subplots_adjust``.
    """
    if figure_margins is None:
        if with_barcode:
            return {"left": 0.02, "right": 0.98, "bottom": 0.10, "top": 0.90}
        return {"left": 0.02, "right": 0.98, "bottom": 0.10, "top": 0.92}

    if not isinstance(figure_margins, dict):
        raise ValueError(
            "figure_margins must be a dict with keys {'left','right','bottom','top'}."
        )
    required_keys = {"left", "right", "bottom", "top"}
    provided_keys = set(figure_margins.keys())
    if provided_keys != required_keys:
        raise ValueError(
            "figure_margins must contain exactly the keys "
            "{'left','right','bottom','top'}."
        )

    margins: dict[str, float] = {}
    for key in ("left", "right", "bottom", "top"):
        value = figure_margins[key]
        if not isinstance(value, Real):
            raise ValueError(f"figure_margins['{key}'] must be numeric.")
        margins[key] = float(value)

    if not (0.0 <= margins["left"] < margins["right"] <= 1.0):
        raise ValueError("figure_margins must satisfy 0 <= left < right <= 1.")
    if not (0.0 <= margins["bottom"] < margins["top"] <= 1.0):
        raise ValueError("figure_margins must satisfy 0 <= bottom < top <= 1.")

    return margins

def _parse_camera_eye(camera_eye: Optional[Any]) -> tuple[float, float]:
    """
    Convert a camera specification to (elev, azim) for matplotlib 3D.
    """
    elev, azim = 22.0, -55.0
    if camera_eye is None:
        return elev, azim
    if isinstance(camera_eye, dict):
        elev = float(camera_eye.get("elev", elev))
        azim = float(camera_eye.get("azim", azim))
        return elev, azim
    if isinstance(camera_eye, (tuple, list)) and len(camera_eye) >= 2:
        return float(camera_eye[0]), float(camera_eye[1])
    raise ValueError(
        "camera_eye must be None, a dict with keys {'elev','azim'}, "
        "or a tuple/list (elev, azim)."
    )
