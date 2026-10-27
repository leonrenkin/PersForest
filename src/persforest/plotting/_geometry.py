"""plotting. geometry: mechanically extracted reference implementation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Literal
from typing import Optional
import numpy as np
import warnings

def _camera_from_eye(camera_eye: Optional[Any]) -> tuple[float, float]:
    """
    Parse a camera specification into Matplotlib ``(elev, azim)``.

    ``camera_eye`` may be ``None``, a dict with ``"elev"`` and ``"azim"``
    keys, or a two-element tuple/list.
    """
    elev, azim = 22.0, -55.0
    if camera_eye is None:
        return elev, azim
    if isinstance(camera_eye, dict):
        return float(camera_eye.get("elev", elev)), float(camera_eye.get("azim", azim))
    if isinstance(camera_eye, (tuple, list)) and len(camera_eye) >= 2:
        return float(camera_eye[0]), float(camera_eye[1])
    raise ValueError(
        "camera_eye must be None, a dict with keys {'elev','azim'}, "
        "or a tuple/list (elev, azim)."
    )

def _validated_activity_tetrahedra(
    point_cloud: np.ndarray,
    rows: list[tuple[tuple[int, ...], Any, float]],
) -> list[tuple[tuple[int, ...], Any, float]]:
    """Validate tetrahedral activity rows and omit degenerate cells."""
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError("point_cloud must be an (n_points, 3) array-like.")

    if len(point_cloud) == 0:
        return []

    valid_rows: list[tuple[tuple[int, ...], Any, float]] = []
    degenerate_count = 0

    for simplex_key, bar, activity_length in rows:
        if len(simplex_key) != 4:
            raise ValueError(
                "3D interior activity must be indexed by tetrahedra with four vertices; "
                f"got {simplex_key!r}."
            )
        if len(set(simplex_key)) != 4:
            degenerate_count += 1
            continue
        if min(simplex_key) < 0 or max(simplex_key) >= len(point_cloud):
            raise ValueError(
                f"Activity simplex {simplex_key!r} contains a point index outside "
                f"[0, {len(point_cloud) - 1}]."
            )

        tetrahedron = point_cloud[list(simplex_key)]
        edge_vectors = tetrahedron[1:] - tetrahedron[0]
        edge_scale = float(np.max(np.linalg.norm(edge_vectors, axis=1)))
        if not np.isfinite(edge_scale) or edge_scale == 0.0:
            degenerate_count += 1
            continue
        normalized_volume_six = abs(float(np.linalg.det(edge_vectors / edge_scale)))
        if (
            not np.isfinite(normalized_volume_six)
            or normalized_volume_six <= 128.0 * np.finfo(float).eps
        ):
            degenerate_count += 1
            continue
        valid_rows.append((simplex_key, bar, activity_length))

    if degenerate_count:
        warnings.warn(
            f"Skipped {degenerate_count} degenerate activity tetrahedron"
            f"{'s' if degenerate_count != 1 else ''}.",
            RuntimeWarning,
            stacklevel=3,
        )

    return valid_rows

def _shrink_tetrahedron(tetrahedron: np.ndarray, factor: float) -> np.ndarray:
    """Shrink a tetrahedron towards its barycenter by ``factor``."""
    if not 0.0 < factor <= 1.0:
        raise ValueError("cell_shrink must be greater than 0 and at most 1.")
    if factor == 1.0:
        return tetrahedron
    center = np.mean(tetrahedron, axis=0)
    return center + factor * (tetrahedron - center)

def _oriented_tetrahedron_faces(
    simplex_key: tuple[int, ...],
    tetrahedron: np.ndarray,
) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    """Return outward-oriented triangular faces and canonical face keys."""
    faces: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for omitted_index in range(4):
        local_face = [index for index in range(4) if index != omitted_index]
        face_points = tetrahedron[local_face].copy()
        omitted_point = tetrahedron[omitted_index]
        normal = np.cross(
            face_points[1] - face_points[0],
            face_points[2] - face_points[0],
        )
        if float(np.dot(normal, omitted_point - face_points[0])) > 0.0:
            face_points[[1, 2]] = face_points[[2, 1]]
            local_face[1], local_face[2] = local_face[2], local_face[1]
        face_key = tuple(sorted(simplex_key[index] for index in local_face))
        faces.append((face_key, face_points))
    return faces

def _activity_face_records(
    point_cloud: np.ndarray,
    rows: list[tuple[tuple[int, ...], Any, float]],
    geometry: Literal["cells", "surface"],
    cell_shrink: float,
) -> list[tuple[np.ndarray, Any, float, tuple[int, ...]]]:
    """Convert activity tetrahedra into backend-neutral triangular faces."""
    if geometry not in ("cells", "surface"):
        raise ValueError("geometry must be 'cells' or 'surface'.")

    if geometry == "cells":
        records: list[tuple[np.ndarray, Any, float, tuple[int, ...]]] = []
        for simplex_key, bar, activity_length in rows:
            tetrahedron = _shrink_tetrahedron(
                point_cloud[list(simplex_key)],
                factor=cell_shrink,
            )
            records.extend(
                (face_points, bar, activity_length, simplex_key)
                for _face_key, face_points in _oriented_tetrahedron_faces(
                    simplex_key,
                    tetrahedron,
                )
            )
        return records

    # A surface is computed separately for every bar. This retains interfaces
    # between differently colored regions while cancelling faces internal to a
    # single bar's region. Repeated rows for the same bar/cell are unified.
    cells_by_bar: dict[Any, dict[tuple[int, ...], float]] = defaultdict(dict)
    for simplex_key, bar, activity_length in rows:
        existing_length = cells_by_bar[bar].get(simplex_key)
        if existing_length is None or activity_length > existing_length:
            cells_by_bar[bar][simplex_key] = activity_length

    records = []
    for bar, cells in cells_by_bar.items():
        exposed_faces: dict[
            tuple[int, int, int],
            tuple[np.ndarray, float, tuple[int, ...]],
        ] = {}
        for simplex_key, activity_length in cells.items():
            tetrahedron = point_cloud[list(simplex_key)]
            for face_key, face_points in _oriented_tetrahedron_faces(
                simplex_key,
                tetrahedron,
            ):
                if face_key in exposed_faces:
                    exposed_faces.pop(face_key)
                else:
                    exposed_faces[face_key] = (
                        face_points,
                        activity_length,
                        simplex_key,
                    )
        records.extend(
            (face_points, bar, activity_length, simplex_key)
            for face_points, activity_length, simplex_key in exposed_faces.values()
        )
    return records

def _camera_angles(camera_eye: Any) -> tuple[float, float]:
    """Resolve a Matplotlib camera specification to elevation and azimuth."""
    default = (22.0, -55.0)
    if camera_eye is None:
        return default
    if isinstance(camera_eye, dict):
        return (
            float(camera_eye.get("elev", default[0])),
            float(camera_eye.get("azim", default[1])),
        )
    if isinstance(camera_eye, (tuple, list)) and len(camera_eye) >= 2:
        return float(camera_eye[0]), float(camera_eye[1])
    raise ValueError(
        "camera_eye must be None, a dict with keys {'elev','azim'}, "
        "or a tuple/list (elev, azim)."
    )
