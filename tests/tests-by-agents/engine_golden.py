"""Golden snapshots of construction output, used to guard engine refactors.

Records bars, the bar-level forest structure, every representative with its
interior, and interior activity, independent of node IDs.

Run ``python tests/tests-by-agents/engine_golden.py`` from the repository root to
regenerate ``data/engine_golden.json``. Regenerate only from an engine whose
output is trusted; the snapshot records behavior, not correctness.
"""
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fixtures_small_filtrations import (  # noqa: E402
    annulus, build, disconnected, fan, reps, split, tetrahedra, triangle,
)
from persforest import PersistenceForest  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / 'data' / 'engine_golden.json'


def _alpha(points):
    return lambda **options: PersistenceForest(points, **options)


def _fixture(factory):
    return lambda **options: build(factory(), **options)


def _lattice_2d():
    x, y = np.meshgrid(np.arange(4.0), np.arange(4.0))
    return np.column_stack((x.ravel(), y.ravel()))


def _cubical(values, filtration_type='top_cells'):
    return lambda **options: PersistenceForest.from_cubical_grid(
        values, filtration_type=filtration_type, **options)


def _nested_plateaus_2d():
    # Flat-valued holes: a value-3 annulus around a value-1 block whose
    # value-2 centre is a second hole. Reduction collapses long equal-valued chains here.
    values = np.zeros((12, 12))
    values[2:10, 2:10] = 3.0
    values[4:8, 4:8] = 1.0
    values[5:7, 5:7] = 2.0
    return values


def _near_ties_2d():
    # Values within the default filtration_tol of an integer level.
    rng = np.random.default_rng(6)
    return rng.integers(0, 3, (10, 10)).astype(float) + 1e-13 * rng.random((10, 10))


CASES = {
    'triangle': _fixture(triangle),
    'split': _fixture(split),
    'fan': _fixture(fan),
    'disconnected': _fixture(disconnected),
    'tetrahedra': _fixture(tetrahedra),
    'annulus': _fixture(annulus),
    'alpha_uniform_2d_seed0_n40': _alpha(np.random.default_rng(0).random((40, 2))),
    'alpha_lattice_2d_4x4': _alpha(_lattice_2d()),
    'alpha_uniform_3d_seed1_n25': _alpha(np.random.default_rng(1).random((25, 3))),
    'cubical_uniform_2d_seed2_12x12': _cubical(np.random.default_rng(2).random((12, 12))),
    'cubical_ties_2d_seed3_12x12': _cubical(np.random.default_rng(3).integers(0, 4, (12, 12)).astype(float)),
    'cubical_nested_plateaus_2d_12x12': _cubical(_nested_plateaus_2d()),
    'cubical_near_ties_2d_seed6_10x10': _cubical(_near_ties_2d()),
    'cubical_ties_3d_seed4_5x5x5': _cubical(np.random.default_rng(4).integers(0, 3, (5, 5, 5)).astype(float)),
    'cubical_vertices_ties_2d_seed5_9x9': _cubical(
        np.random.default_rng(5).integers(0, 4, (9, 9)).astype(float), 'vertices'),
}


def _signed(terms):
    return sorted([list(simplex), int(sign)] for simplex, sign in terms)


def bar_structure(forest) -> dict:
    """Map each bar to ``(critical_values, parent_bar)``, independent of node IDs.

    ``critical_values`` are the ascending values at which the bar's
    representative changes, ending at its death; ``parent_bar`` is the bar it
    splits off at its birth, or None.
    """
    owner = {node_id: bar for bar in forest.barcode for node_id in bar._node_progression}
    structure = {}
    for bar in forest.barcode:
        values = tuple(forest.nodes[node_id].filt_val for node_id in bar._node_progression)
        structure[bar] = (values, owner.get(forest.nodes[bar._node_progression[0]].parent))
    return structure


def snapshot(forest) -> dict:
    """Return a JSON-compatible record of bars, forest structure and representatives.

    The record does not depend on node IDs or iteration order. A parent bar is
    identified by its own birth, death and critical values.
    """
    structure = bar_structure(forest)
    bars = []
    for bar in forest.barcode:
        values, parent = structure[bar]
        cycle_reps = [[r.active_start, r.active_end, _signed(r.signed_simplices),
                       _signed(r.interior) if r.interior_available else None]
                      for r in reps(forest, bar)]
        parent_key = None if parent is None else [parent.birth, parent.death, list(structure[parent][0])]
        bars.append([bar.birth, bar.death, list(values), parent_key, cycle_reps])
    bars.sort(key=lambda item: json.dumps(item))
    record = {'bars': bars}
    activity = forest.interior_simplex_activity()
    record['interior_activity'] = sorted(
        [list(simplex), sorted([b.birth, b.death, start, end] for b, start, end in rows)]
        for simplex, rows in activity.items())
    return record


def digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def summary(record: dict) -> dict:
    """Readable part stored next to the digest for diagnosing mismatches."""
    return {'endpoints': sorted([bar[0], bar[1]] for bar in record['bars']), 'digest': digest(record)}


MODES = {'reduce=False': {'reduce': False}, 'reduce=True': {'reduce': True}}


def compute_all() -> dict:
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for name, make in CASES.items():
            for mode, options in MODES.items():
                result[f'{name}/{mode}'] = summary(snapshot(make(**options)))
    return result


if __name__ == '__main__':
    records = compute_all()
    lines = [f'{json.dumps(case)}: {json.dumps(records[case], sort_keys=True)}' for case in sorted(records)]
    GOLDEN_PATH.write_text('{\n' + ',\n'.join(lines) + '\n}\n')
    print(f'wrote {GOLDEN_PATH}')
