"""Golden snapshots of construction output, used to guard engine refactors.

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
    annulus, build, disconnected, fan, mode_options, reps, split, tetrahedra, triangle,
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
}


def _signed(terms):
    return sorted([list(simplex), int(sign)] for simplex, sign in terms)


def snapshot(forest) -> dict:
    """Return a JSON-compatible, iteration-order-independent construction record."""
    bars = []
    for bar in forest.barcode:
        cycle_reps = [[r.active_start, r.active_end, _signed(r.signed_simplices),
                       _signed(r.interior) if r.interior_available else None]
                      for r in reps(forest, bar)]
        bars.append([bar.birth, bar.death, cycle_reps])
    bars.sort(key=lambda item: json.dumps(item))
    record = {'bars': bars, 'levels': list(forest.levels), 'node_count': len(forest.nodes)}
    if forest._barcode_diffs_ready:
        activity = forest.interior_simplex_activity()
        record['interior_activity'] = sorted(
            [list(simplex), sorted([b.birth, b.death, start, end] for b, start, end in rows)]
            for simplex, rows in activity.items())
    return record


def digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def summary(record: dict) -> dict:
    """Readable part stored next to the digest for diagnosing mismatches."""
    return {'endpoints': sorted([b, d] for b, d, _ in record['bars']),
            'node_count': record['node_count'], 'digest': digest(record)}


def compute_all() -> dict:
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for name, make in CASES.items():
            for mode, options in mode_options():
                result[f'{name}/{mode}'] = summary(snapshot(make(**options)))
    return result


if __name__ == '__main__':
    records = compute_all()
    lines = [f'{json.dumps(case)}: {json.dumps(records[case], sort_keys=True)}' for case in sorted(records)]
    GOLDEN_PATH.write_text('{\n' + ',\n'.join(lines) + '\n}\n')
    print(f'wrote {GOLDEN_PATH}')
