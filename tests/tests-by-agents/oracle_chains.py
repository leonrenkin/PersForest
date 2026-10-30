"""Small exact algebra oracles independent of persforest's forest algorithm."""
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import math
import numpy as np


def canonical(simplex):
    sign = (-1) ** sum(a > b for i, a in enumerate(simplex) for b in simplex[i + 1:])
    return tuple(sorted(simplex)), sign


def coefficients(signed_terms):
    result = defaultdict(int)
    for simplex, value in signed_terms:
        key, sign = canonical(simplex)
        result[key] += sign * value
    return {s: c for s, c in result.items() if c}


def boundary(chain):
    result = defaultdict(int)
    for simplex, value in chain.items():
        for index in range(len(simplex)):
            face = simplex[:index] + simplex[index + 1:]
            if face:
                result[face] += (-1) ** index * value
    return {s: c for s, c in result.items() if c}


def rank(columns, rows):
    """Exact column rank with rational elimination; no tolerance or SVD."""
    matrix = [[Fraction(column.get(row, 0)) for column in columns] for row in rows]
    pivot_row = 0
    for col in range(len(columns)):
        pivot = next((r for r in range(pivot_row, len(rows)) if matrix[r][col]), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        value = matrix[pivot_row][col]
        matrix[pivot_row] = [x / value for x in matrix[pivot_row]]
        for r in range(pivot_row + 1, len(rows)):
            factor = matrix[r][col]
            matrix[r] = [x - factor * y for x, y in zip(matrix[r], matrix[pivot_row])]
        pivot_row += 1
    return pivot_row


def homology_data(values, ambient_dim, time):
    cycles_dim = ambient_dim - 1
    present = [s for s, value in values.items() if value <= time]
    rows = [s for s in present if len(s) == ambient_dim]
    lower = [s for s in present if len(s) == cycles_dim]
    boundaries = [boundary({s: 1}) for s in present if len(s) == ambient_dim + 1]
    cycle_boundary = [boundary({s: 1}) for s in rows]
    betti = len(rows) - rank(cycle_boundary, lower) - rank(boundaries, rows)
    return rows, boundaries, betti


def simplex_volume(points, simplex):
    """Gram formula handles simplices below the ambient dimension."""
    vertices = np.asarray(points)[list(simplex)]
    edges = vertices[1:] - vertices[0]
    return math.sqrt(max(0., float(np.linalg.det(edges @ edges.T)))) / math.factorial(len(simplex) - 1)


def sample_times(values):
    levels = sorted(set(values.values()))
    return sorted(set([levels[0] - 1, levels[-1] + 1] + levels
                      + [(a + b) / 2 for a, b in zip(levels, levels[1:])]))


def weighted_minimum_basis(values, points, time, weights):
    """Exhaust all 0/1 geometric fillings of the tiny planar fixtures.

    Independence of their ordinary boundaries is checked over the rationals.
    This is a finite candidate-basis check, not a proof over all chain coefficients.
    Each selected triangle has coefficient +1 in its geometric orientation.
    """
    dim = points.shape[1]
    top = [s for s in values if len(s) == dim + 1]
    rows, boundaries, betti = homology_data(values, dim, time)
    candidates = []
    for mask in range(1, 1 << len(top)):
        chosen = [s for i, s in enumerate(top) if mask & (1 << i)]
        interior = {s: int(np.sign(np.linalg.det(points[list(s)][1:] - points[s[0]]))) for s in chosen}
        cycle = boundary(interior)
        if not set(cycle).issubset(rows):
            continue
        candidates.append((cycle, sum(weights[s] for s in chosen)))
    if not betti:
        return 0
    return min(sum(weight for _, weight in choice)
               for choice in combinations(candidates, betti)
               if rank(boundaries + [c for c, _ in choice], rows) == rank(boundaries, rows) + betti)
