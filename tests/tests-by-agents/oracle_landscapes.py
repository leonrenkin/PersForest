"""Direct interval-intersection integration, independent of slope events."""
import numpy as np


def kernel(t, birth, death, records, *, raw=False):
    x = t if raw else 2 * t
    value = sum(v * max(0., min(end, x - birth) - max(start, x - death))
                for start, end, v in records)
    return value if raw else value / 2


def landscape(grid, specs, levels, *, raw=False):
    values = np.array([[kernel(t, b, d, records, raw=raw) for t in grid]
                       for b, d, records in specs], dtype=float).reshape(len(specs), len(grid))
    result = np.zeros((levels, len(grid)))
    if len(specs):
        result[:min(levels, len(specs))] = np.sort(values, axis=0)[::-1][:levels]
    return result


def event_grid(birth, death, records):
    knots = sorted({(x + y) / 2 for start, end, _ in records
                    for x in (start, end) for y in (birth, death)})
    return np.array(sorted(set(knots + [(a + b) / 2 for a, b in zip(knots, knots[1:])]
                               + [birth - 1, death + 1])))
