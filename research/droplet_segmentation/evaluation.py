"""Scores of a label image against the true droplets.

Every pair of a true droplet and a segment has an intersection over union
(IoU). A pair with IoU above 0.5 is a *match*; since two segments cannot both
cover more than half of the union with one droplet, matches are one-to-one and
need no assignment step. From the matches:

- precision: matched segments / segments,
- recall: matched droplets / droplets,
- F1: their harmonic mean,
- panoptic quality (PQ): summed IoU of the matches / (matches + unmatched
  segments / 2 + unmatched droplets / 2), which combines detection and
  outline quality,
- mean best IoU: the IoU of each droplet with its best segment, averaged,
- merges: segments holding more than half of two or more droplets.

Droplets that the truth assigns no pixel (hidden under brighter ones) are not
counted.
"""

from dataclasses import asdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def iou_matrix(labels: NDArray[np.int_], truth: NDArray[np.int_]):
    """IoU of every true droplet with every segment.

    Returns ``(true_ids, segment_ids, iou)`` with ``iou[i, j]`` the IoU of
    droplet ``true_ids[i]`` and segment ``segment_ids[j]``.
    """
    true_ids = np.unique(truth[truth > 0])
    segment_ids = np.unique(labels[labels > 0])
    true_index = np.searchsorted(true_ids, truth)
    segment_index = np.searchsorted(segment_ids, labels)
    both = (truth > 0) & (labels > 0)
    intersection = np.zeros((len(true_ids), len(segment_ids)))
    np.add.at(intersection, (true_index[both], segment_index[both]), 1)
    true_sizes = np.array([(truth == k).sum() for k in true_ids], dtype=float)
    segment_sizes = np.array([(labels == k).sum() for k in segment_ids], dtype=float)
    union = true_sizes[:, None] + segment_sizes[None, :] - intersection
    return true_ids, segment_ids, intersection / np.maximum(union, 1)


@dataclass(frozen=True)
class Scores:
    droplets: int
    segments: int
    matches: int
    precision: float
    recall: float
    f1: float
    pq: float
    mean_best_iou: float
    merges: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(labels: NDArray[np.int_], truth: NDArray[np.int_]) -> Scores:
    """Score a label image against the true labels."""
    true_ids, segment_ids, iou = iou_matrix(labels, truth)
    n_true, n_segments = len(true_ids), len(segment_ids)
    matched = iou > 0.5
    matches = int(matched.sum())
    precision = matches / n_segments if n_segments else 0.0
    recall = matches / n_true if n_true else 0.0
    f1 = 2 * matches / (n_true + n_segments) if n_true + n_segments else 0.0
    denominator = matches + (n_segments - matches) / 2 + (n_true - matches) / 2
    pq = float(iou[matched].sum() / denominator) if denominator else 0.0
    mean_best = float(iou.max(axis=1).mean()) if n_segments and n_true else 0.0
    true_sizes = np.array([(truth == k).sum() for k in true_ids])
    covered = np.zeros((n_true, n_segments))
    if n_true and n_segments:
        both = (truth > 0) & (labels > 0)
        np.add.at(covered, (np.searchsorted(true_ids, truth[both]), np.searchsorted(segment_ids, labels[both])), 1)
    merges = int(((covered > true_sizes[:, None] / 2).sum(axis=0) >= 2).sum())
    return Scores(n_true, n_segments, matches, precision, recall, f1, pq, mean_best, merges)


def best_matches(labels: NDArray[np.int_], truth: NDArray[np.int_]) -> list[tuple[int, int, float]]:
    """For each true droplet, the segment overlapping it best and their IoU.

    Returns ``(true_label, segment_label, iou)`` triples; the segment label is
    0 if no segment overlaps the droplet.
    """
    true_ids, segment_ids, iou = iou_matrix(labels, truth)
    result = []
    for index, true_label in enumerate(true_ids):
        if len(segment_ids) == 0 or iou[index].max() == 0:
            result.append((int(true_label), 0, 0.0))
        else:
            best = int(iou[index].argmax())
            result.append((int(true_label), int(segment_ids[best]), float(iou[index, best])))
    return result
