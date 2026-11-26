"""Shared defaults of the static plots: figure sizes and style-dict merging.

Renderers take their default sizes from here, so plots of one kind match.
Figure resolution is left to Matplotlib's ``figure.dpi`` unless a caller
passes ``dpi``.
"""

from __future__ import annotations

from typing import Any
from typing import Optional

# Square panels for geometric scenes (complexes, cycles, interiors, loops).
SCENE_FIGSIZE = (5.0, 5.0)
# Charts with a filtration axis (profiles, landscapes, measurements).
CHART_FIGSIZE = (6.0, 4.0)
# Persistence forests need width for their trees.
FOREST_FIGSIZE = (10.0, 6.0)


def resolve_style(style: Optional[dict[str, Any]], defaults: dict[str, Any], name: str) -> dict[str, Any]:
    """Merge style overrides into ``defaults``, rejecting unknown keys.

    Parameters
    ----------
    style : dict or None
        User overrides.
    defaults : dict
        Every supported key with its default value.
    name : str
        Parameter name used in the error message, e.g. ``"style_2d"``.

    Returns
    -------
    dict
        A new dictionary with the overrides applied.

    Raises
    ------
    ValueError
        If ``style`` contains a key that ``defaults`` does not.
    """
    resolved = dict(defaults)
    if style:
        unknown = sorted(set(style) - set(defaults))
        if unknown:
            raise ValueError(
                f"Unknown {name} keys {unknown}; supported keys are {sorted(defaults)}.")
        resolved.update(style)
    return resolved
