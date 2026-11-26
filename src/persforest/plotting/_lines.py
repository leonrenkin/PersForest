"""Cycles in 2D as continuous polylines, with an optional halo.

A cycle arrives as separate edge segments. Drawn one by one, every turn ends
two strokes and leaves a notch on its outer side. Chaining the edges into
polylines lets Matplotlib join them instead; closed loops start halfway along
an edge, so their seam is a straight butt joint.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Sequence

import numpy as np
from matplotlib.collections import LineCollection

# Halo collections carry this gid, so callers can tell them from cycles.
HALO_GID = "persforest-cycle-halo"


def _key(point) -> tuple[float, float]:
    # Endpoints of one vertex come from the same coordinates; rounding only
    # guards against representation noise from coordinate transforms.
    return (round(float(point[0]), 9), round(float(point[1]), 9))


def cycle_polylines(segments: Sequence) -> list[np.ndarray]:
    """Chain edge segments into maximal polylines.

    Chains run through vertices met by exactly two segments and stop at any
    other vertex (an end, a branch or a pinch). Closed loops of degree-two
    vertices start and end at the midpoint of one of their edges. Segments
    keep their direction where the chain allows it, so a consistently oriented
    cycle is traced along its orientation.

    Parameters
    ----------
    segments : sequence of (2, 2) array-likes
        Edge segments ``[[x0, y0], [x1, y1]]``.

    Returns
    -------
    list of numpy.ndarray
        Polylines of shape ``(k, 2)``, each using every segment once.
    """
    segs = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float)) for a, b in segments]
    segs = [(a, b) for a, b in segs if _key(a) != _key(b)]
    ends = [(_key(a), _key(b)) for a, b in segs]
    incident = defaultdict(list)
    for index, (ka, kb) in enumerate(ends):
        incident[ka].append(index)
        incident[kb].append(index)
    used = [False] * len(segs)

    def step(index, key):
        """Point and key at the far end of segment ``index`` seen from ``key``."""
        (a, b), (ka, kb) = segs[index], ends[index]
        return (b, kb) if key == ka else (a, ka)

    def walk(points, key):
        while len(incident[key]) == 2:
            following = [i for i in incident[key] if not used[i]]
            if not following:
                break
            # Prefer the edge leaving this vertex, so a consistently oriented
            # (signed) cycle is traced in its own direction.
            following.sort(key=lambda i: ends[i][0] != key)
            used[following[0]] = True
            point, key = step(following[0], key)
            points.append(point)
        return points

    lines = []
    for index, (ka, kb) in enumerate(ends):
        for key, point in ((ka, segs[index][0]), (kb, segs[index][1])):
            if used[index] or len(incident[key]) == 2:
                continue
            used[index] = True
            far, far_key = step(index, key)
            lines.append(np.array(walk([point, far], far_key)))
    for index in range(len(segs)):
        if used[index]:
            continue
        used[index] = True
        (a, b), (_, kb) = segs[index], ends[index]
        middle = (a + b) / 2
        points = walk([middle, b], kb)
        points.append(middle)
        lines.append(np.array(points))
    # An open chain may have been walked from its head; flip lines that run
    # mostly against their segments.
    directed = set(ends)
    for index, line in enumerate(lines):
        keys = [_key(point) for point in line]
        pairs = list(zip(keys[:-1], keys[1:]))
        forward = sum(pair in directed for pair in pairs)
        backward = sum(pair[::-1] in directed for pair in pairs)
        if backward > forward:
            lines[index] = line[::-1].copy()
    return lines


def add_cycle_lines(ax, segments: Sequence, color: Any, linewidth: float, zorder: float, *,
                    square: bool = False, halo: bool = False, halo_width: float = 2.4,
                    halo_color: Any = "white", clip_on: bool = True) -> LineCollection | None:
    """Draw one cycle as joined polylines, optionally on a wider halo.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    segments : sequence of (2, 2) array-likes
        Edge segments of the cycle.
    color : color
        Line color.
    linewidth : float
        Line width in points.
    zorder : float
        Z-order of the line; the halo sits just beneath it.
    square : bool
        Miter joins and square caps, for axis-aligned grid edges. Otherwise
        joins and caps are round.
    halo, halo_width, halo_color :
        Whether to draw a ``halo_width`` wider line of ``halo_color`` beneath.
    clip_on : bool
        Whether to clip to the axes.

    Returns
    -------
    matplotlib.collections.LineCollection or None
        The cycle's line collection, or None for an empty cycle.
    """
    lines = cycle_polylines(segments)
    if not lines:
        return None
    joins = dict(capstyle="projecting", joinstyle="miter") if square else dict(capstyle="round", joinstyle="round")
    if halo:
        halo_lines = LineCollection(lines, linewidths=linewidth + halo_width, colors=[halo_color],
                                    zorder=zorder - 0.001, clip_on=clip_on, **joins)
        halo_lines.set_gid(HALO_GID)
        ax.add_collection(halo_lines)
    collection = LineCollection(lines, linewidths=linewidth, colors=[color], zorder=zorder,
                                clip_on=clip_on, **joins)
    ax.add_collection(collection)
    return collection
