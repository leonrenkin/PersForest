"""Read drawn 2D cycles back as oriented edges.

Cycles are drawn as joined polylines (``plotting._lines``); closed loops start
and end at the midpoint of one edge. These helpers undo that, so tests can
assert on edges and their orientation.
"""
import numpy as np
from matplotlib.collections import LineCollection

from persforest.plotting._lines import HALO_GID


def cycle_collections(ax):
    """Line collections of an axes, without cycle halos."""
    return [c for c in ax.collections if isinstance(c, LineCollection) and c.get_gid() != HALO_GID]


def drawn_edges(collection):
    """Oriented edges ``(start, end)`` of the polylines in ``collection``."""
    edges = []
    for line in collection.get_segments():
        line = np.asarray(line, dtype=float)
        closed = (len(line) >= 4 and np.allclose(line[0], line[-1])
                  and np.allclose(line[0], (line[1] + line[-2]) / 2))
        if closed:
            edges += list(zip(line[1:-2], line[2:-1])) + [(line[-2], line[1])]
        else:
            edges += list(zip(line[:-1], line[1:]))
    return edges
