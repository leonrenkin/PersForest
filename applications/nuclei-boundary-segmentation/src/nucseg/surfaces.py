"""Surfaces of 3D nuclei: voxel-boundary meshes, Taubin smoothing, and an interactive Plotly view."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .plots import outline_color
from .segment3d import Nucleus3D
from .segment3d import Segmentation3D


def surface_mesh(mask: NDArray[np.bool_], spacing: tuple[float, float, float],
                 origin: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Triangulated boundary of a voxel mask, the squares between inside and outside voxels.

    These squares are the mod-2 boundary of the mask, a cubical 2-cycle. Every
    square is split into two triangles, ordered so that normals point outwards.

    Returns
    -------
    vertices : ndarray, shape (n, 3)
        Positions ``(z, y, x)`` in µm of the grid corners used.
    triangles : ndarray of int, shape (m, 3)
    """
    padded = np.pad(mask, 1).astype(np.int8)
    corner_shape = tuple(n + 1 for n in padded.shape)
    quads = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        step = np.diff(padded, axis=a)  # +1: outside -> inside along +a, -1: inside -> outside
        for sign in (1, -1):
            idx = np.argwhere(step == sign)
            if not len(idx):
                continue
            base = idx.copy()
            base[:, a] += 1  # the square lies on the corner plane between the two voxels
            corners = []
            # (b, c) offsets counterclockwise around +a (e_b x e_c = e_a).
            for db, dc in ((0, 0), (1, 0), (1, 1), (0, 1)):
                corner = base.copy()
                corner[:, b] += db
                corner[:, c] += dc
                corners.append(np.ravel_multi_index(corner.T, corner_shape))
            quad = np.stack(corners, axis=1)
            # The outward normal is -a where the inside lies ahead (sign +1).
            quads.append(quad[:, ::-1] if sign == 1 else quad)
    if not quads:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    quads = np.concatenate(quads)
    used, inverse = np.unique(quads, return_inverse=True)
    quads = inverse.reshape(quads.shape)
    triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    # Corner k of the padded grid sits at voxel boundary k - 1 of the mask.
    vertices = (np.stack(np.unravel_index(used, corner_shape), axis=1) - 1) * np.asarray(spacing)
    return vertices + np.asarray(origin), triangles


def taubin_smooth(vertices: NDArray[np.float64], triangles: NDArray[np.int64], iterations: int = 10,
                  lam: float = 0.5, mu: float = -0.53) -> NDArray[np.float64]:
    """Taubin smoothing of a mesh: alternating Laplacian steps that remove the voxel steps
    without the shrinking of plain Laplacian smoothing."""
    from scipy import sparse

    if iterations <= 0 or not len(triangles):
        return vertices.copy()
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    n = len(vertices)
    adjacency = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0  # duplicate edges count once
    mean = sparse.diags(1.0 / np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency
    out = vertices.copy()
    for _ in range(iterations):
        out = out + lam * (mean @ out - out)
        out = out + mu * (mean @ out - out)
    return out


def nucleus_mesh(result: Segmentation3D, nucleus: Nucleus3D, *, smooth: bool = True, mesh_iterations: int = 10):
    """Surface of a nucleus in µm, ``(vertices (z, y, x), triangles)``.

    With ``smooth`` the smoothed mask is used. ``mesh_iterations`` Taubin steps
    then remove the voxel staircase; 0 keeps the raw voxel faces.
    """
    if not nucleus.found:
        raise ValueError(f"nucleus {nucleus.label} has no 3D void")
    mask = nucleus.smooth_mask if smooth else nucleus.mask
    origin = (0.0, nucleus.rows[0] * result.um_per_bin, nucleus.cols[0] * result.um_per_bin)
    vertices, triangles = surface_mesh(mask, result.spacing, origin)
    return taubin_smooth(vertices, triangles, mesh_iterations), triangles


def plot_3d(result: Segmentation3D, labels: list[int] | None = None, *, smooth: bool = True,
            mesh_iterations: int = 10, opacity: float = 1.0, z_exaggeration: float = 1.0):
    """Surfaces of the nuclei in an interactive Plotly figure, in µm.

    Each surface is the voxel boundary of the (smoothed) mask, smoothed as a
    mesh by ``nucleus_mesh``. The dashed outline at the bottom shows the
    extent of the stack. Needs ``plotly`` (the ``notebooks`` extra).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError("plot_3d needs plotly: pip install plotly (or the 'notebooks' extra)") from error

    fig = go.Figure()
    for nucleus in result.nuclei:
        if not nucleus.found or (labels is not None and nucleus.label not in labels):
            continue
        vertices, triangles = nucleus_mesh(result, nucleus, smooth=smooth, mesh_iterations=mesh_iterations)
        z, y, x = vertices.T
        # Listing (x, y, z) instead of (z, y, x) mirrors the mesh; swap two
        # corners so the normals still point outwards for the lighting.
        i, k, j = triangles.T
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=outline_color(nucleus.label), opacity=opacity,
                                name=f"nucleus {nucleus.label}", showlegend=True,
                                hovertemplate=f"nucleus {nucleus.label}<extra></extra>",
                                lighting=dict(ambient=0.45, diffuse=0.7, specular=0.15, roughness=0.6)))
    nz, h, w = result.image.shape
    s, dz = result.um_per_bin, result.z_step_um
    fig.add_trace(go.Scatter3d(x=[0, w * s, w * s, 0, 0], y=[0, 0, h * s, h * s, 0], z=[0] * 5, mode="lines",
                               line=dict(color="gray", dash="dash"), name="stack", showlegend=False,
                               hoverinfo="skip"))
    extents = np.array([w * s, h * s, nz * dz * z_exaggeration])
    extents = extents / extents.max()
    fig.update_layout(scene=dict(xaxis_title="x (µm)", yaxis_title="y (µm)", zaxis_title="z (µm)",
                                 yaxis_autorange="reversed", aspectmode="manual",
                                 aspectratio=dict(x=extents[0], y=extents[1], z=extents[2])),
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig
