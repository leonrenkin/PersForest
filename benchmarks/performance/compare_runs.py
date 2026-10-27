"""Compare saved before/after runs on matching inputs; never run benchmarks.

Usage: .venv/bin/python benchmarks/performance/compare_runs.py BEFORE_DIR AFTER_DIR
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


PAIR_FIELDS = ('dim', 'sampler', 'n_points', 'repeat', 'seed', 'mode',
               'functional', 'signed', 'filtration_tol', 'reduce', 'max_k', 'grid_points')


def matched_rows(before: list[dict], after: list[dict]) -> list[tuple[dict, dict]]:
    """Pair successful cases with identical settings, clouds and endpoints.

    Parameters
    ----------
    before, after : list of dict
        Rows read from the benchmark CSV exports.

    Returns
    -------
    list of tuple of dict
        Before/after pairs. Unmatched or unsuccessful rows are omitted.

    Raises
    ------
    ValueError
        If case keys are duplicated or a paired hash is missing or differs.
    """
    def index(rows: list[dict]) -> dict:
        result = {}
        for row in rows:
            if row['status'] != 'ok':
                continue
            key = tuple(row[field] for field in PAIR_FIELDS)
            if key in result:
                raise ValueError(f'Duplicate successful case: {key}')
            result[key] = row
        return result

    old, new = index(before), index(after)
    pairs = []
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]
        for field in ('point_cloud_sha256', 'barcode_endpoints_sha256'):
            if not a.get(field) or a[field] != b.get(field):
                raise ValueError(f'{field} missing or different for {key}')
        pairs.append((a, b))
    return pairs


def main() -> None:
    """Print paired median runtimes, speedups and cumulative peak RSS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    args = parser.parse_args()
    rows, metadata = [], []
    for directory in (args.before, args.after):
        with (directory / 'results.csv').open(newline='') as stream:
            rows.append(list(csv.DictReader(stream)))
        metadata.append(json.loads((directory / 'manifest.json').read_text())['metadata'])
    for field in ('python', 'platform', 'machine', 'packages', 'thread_policy', 'sampler_parameters'):
        if metadata[0].get(field) != metadata[1].get(field):
            print(f'NOTE: {field} differs between runs; inspect manifests.')
    pairs = matched_rows(*rows)
    if not pairs:
        parser.error('No matching successful cases')
    print(f'{len(pairs)} paired cases; point-cloud and endpoint hashes all match.')
    print('Unmatched/unsuccessful cases are omitted. Timings are historical comparisons.')
    groups = {}
    for a, b in pairs:
        key = (int(b['dim']), b['sampler'], int(b['n_points']), b['mode'], b['functional'])
        groups.setdefault(key, []).append((a, b))
    for key, group in sorted(groups.items()):
        print(f'\n{key[0]}D {key[1]} n={key[2]} {key[3]} {key[4]} ({len(group)} repeats)')
        for field, label in (('time_to_barcode_s', 'Total'), ('time_barcode_s', 'Barcode')):
            old = [float(a[field]) for a, _ in group]
            new = [float(b[field]) for _, b in group]
            print(f'  {label}: {median(old):.3f} -> {median(new):.3f} s; '
                  f'{median(old) / median(new):.2f}x ratio of medians; '
                  f'after range {min(new):.3f}–{max(new):.3f} s')
        old_rss = median(float(a['peak_rss_bytes']) for a, _ in group) / 2**30
        new_rss = median(float(b['peak_rss_bytes']) for _, b in group) / 2**30
        print(f'  Peak RSS: {old_rss:.3f} -> {new_rss:.3f} GiB (median cumulative peak)')


if __name__ == '__main__':
    main()
