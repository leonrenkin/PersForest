"""plotting. scene: mechanically extracted reference implementation."""

from __future__ import annotations

from persforest.geometry.simplicial import boundary_faces_from_tetrahedra
from typing import Any
from typing import Dict
from typing import Literal
import itertools
import numpy as np

def _activity_rows_for_plotting(
    forest,
    overlap: Literal["longest", "layer"],
    min_activity_length: float,
) -> list[tuple[tuple[int, ...], Any, float]]:
    """Collect activity rows using the established overlap semantics."""
    if overlap not in ("longest", "layer"):
        raise ValueError("overlap must be 'longest' or 'layer'.")

    rows: list[tuple[tuple[int, ...], Any, float]] = []
    for simplex_key, simplex_activity in forest.interior_simplex_activity().items():
        for bar, active_start, active_end in simplex_activity:
            activity_length = float(active_end - active_start)
            if activity_length >= min_activity_length:
                rows.append((tuple(simplex_key), bar, activity_length))

    if overlap == "longest":
        longest_by_simplex: dict[tuple[int, ...], tuple[Any, float]] = {}
        for simplex_key, bar, activity_length in rows:
            current = longest_by_simplex.get(simplex_key)
            if current is None or activity_length > current[1]:
                longest_by_simplex[simplex_key] = (bar, activity_length)
        return [
            (simplex_key, bar, activity_length)
            for simplex_key, (bar, activity_length) in longest_by_simplex.items()
        ]

    rows.sort(key=lambda row: row[2])
    return rows

def _complex_edge_segments_3d(
    forest,
    point_cloud: np.ndarray,
    complex_max_filtration: float | None,
) -> list[np.ndarray]:
    """Build unique complex edges from stored 3D faces and tetrahedra."""
    edge_keys: set[tuple[int, int]] = set()
    for simplex, filtration_value in forest.filtration:
        if (
            complex_max_filtration is not None
            and filtration_value > complex_max_filtration
        ):
            continue
        simplex_key = tuple(simplex)
        if len(simplex_key) < 2:
            continue
        for edge in itertools.combinations(simplex_key, 2):
            edge_keys.add(tuple(sorted(edge)))
    return [point_cloud[list(edge)] for edge in sorted(edge_keys)]

def _simplices_present_at_filtration(self, filt_val: float) -> Dict[str, Any]:
    """
        Return simplices present at a filtration value.

        For ambient dimension 2:
            returns edges, triangles
        For ambient dimension 3:
            returns edges, triangles, tetrahedra

        Parameters
        ----------
        filt_val : float
            Filtration threshold.

        Returns
        -------
        dict
            Dictionary containing present simplices.
        """
    if self.dim not in (2, 3):
        raise ValueError("Plotting is only implemented for ambient dimensions 2 and 3.")

    out: Dict[str, Any] = {
        "edges": [],
        "triangles": [],
        "tetrahedra": [],
    }

    for simplex, f in self.filtration:
        if f > filt_val:
            break

        simplex_t = tuple(simplex)

        if self.dim == 2:
            if len(simplex_t) == 2:
                out["edges"].append(simplex_t)
            elif len(simplex_t) == 3:
                out["triangles"].append(simplex_t)

        elif self.dim == 3:
            if len(simplex_t) == 2:
                out["edges"].append(simplex_t)
            elif len(simplex_t) == 3:
                out["triangles"].append(simplex_t)
            elif len(simplex_t) == 4:
                out["tetrahedra"].append(simplex_t)

    return out

def _complex_snapshot_at_filtration(self, filt_val: float) -> Dict[str, Any]:
    """
        Build a backend-agnostic snapshot of the complex and active cycles at
        a given filtration value.

        Parameters
        ----------
        filt_val : float
            Filtration threshold.

        Returns
        -------
        dict
            Snapshot containing points and present simplices.
        """
    pts = np.asarray(self.point_cloud, dtype=float)
    simplices = self._simplices_present_at_filtration(filt_val)

    snapshot: Dict[str, Any] = {
        "dimension": self.dim,
        "filt_val": float(filt_val),
        "points": pts,
    }

    if self.dim == 2:
        snapshot["edges"] = simplices["edges"]
        snapshot["triangles"] = simplices["triangles"]

    elif self.dim == 3:
        tetrahedra = simplices["tetrahedra"]
        standalone_triangles = simplices["triangles"]
        edges = simplices["edges"]

        boundary_from_tets = boundary_faces_from_tetrahedra(
            tetrahedra=tetrahedra,
            point_cloud=pts,
        )
        tet_face_keys = {tuple(sorted(face)) for face in boundary_from_tets}
        standalone_keys = {tuple(sorted(face)) for face in standalone_triangles}

        # Include all boundary faces from tetrahedra, plus triangles not covered by them.
        boundary_faces = list(tet_face_keys | standalone_keys)

        snapshot["triangles"] = boundary_faces
        snapshot["edges"] = edges
        snapshot["tetrahedra"] = tetrahedra
        snapshot["standalone_triangles"] = standalone_triangles

    return snapshot
