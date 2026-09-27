"""Analytic polygons and signed graphs G1–G8, independent of forest output."""
import math
import numpy as np
from persforest.PersistenceForest import SignedChain


def polygon_terms(indices):
    return {(tuple(sorted((a, b))), 1 if a < b else -1)
            for a, b in zip(indices, indices[1:] + indices[:1])}


def polygon(points):
    points = np.array(points, dtype=float)
    return points, SignedChain(polygon_terms(list(range(len(points)))))


def square():
    return polygon([(0, 0), (2, 0), (2, 2), (0, 2)])


def ell():
    return polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])


def ring():
    points = np.array([(0, 0), (4, 0), (4, 4), (0, 4),
                       (1, 1), (3, 1), (3, 3), (1, 3)], float)
    return points, SignedChain(polygon_terms([0, 1, 2, 3]) | polygon_terms([4, 7, 6, 5]))


def disjoint_squares():
    points = np.array([(0, 0), (1, 0), (1, 1), (0, 1),
                       (3, 0), (4, 0), (4, 1), (3, 1)], float)
    return points, SignedChain(polygon_terms([0, 1, 2, 3]) | polygon_terms([4, 5, 6, 7]))


def touching_triangles():
    points = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)], float)
    return points, SignedChain(polygon_terms([0, 1, 2]) | polygon_terms([0, 3, 4]))


def doubled(kind='edge'):
    if kind == 'edge':
        points, edges = [(0, 0), (1, 0)], [(0, 1)]
    elif kind == 'path':
        points, edges = [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 2)]
    elif kind == 'y':
        points = [(0, 0), (1, 0), (-.5, math.sqrt(3) / 2), (-.5, -math.sqrt(3) / 2)]
        edges = [(0, 1), (0, 2), (0, 3)]
    else:
        raise ValueError(kind)
    return np.array(points, float), SignedChain({(edge, sign) for edge in edges for sign in (-1, 1)})
