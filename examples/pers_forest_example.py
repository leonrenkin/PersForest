"""Runnable quickstart for the persforest package.

For a guided walkthrough, use the notebooks in examples/tutorials/.
This script is intentionally compact: it builds one forest, shows the main
plots, extracts cycle representatives, and computes measurement landscapes.
"""

import matplotlib.pyplot as plt
import numpy as np

from persforest import PersistenceForest
from persforest.cycle_rep_vectorisations import (
    signed_chain_edge_length,
    signed_chain_excess_curvature,
)


def sample_noisy_star(
    points_per_edge: int = 14,
    inner_radius: float = 0.48,
    outer_radius: float = 1.2,
    noise: float = 0.02,
    seed: int = 11,
) -> np.ndarray:
    """Sample points around a noisy star-shaped loop."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2.0 * np.pi, 10, endpoint=False)
    radii = np.where(np.arange(10) % 2 == 0, outer_radius, inner_radius)
    vertices = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

    edge_points = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        t = np.linspace(0.0, 1.0, points_per_edge, endpoint=False)[:, None]
        edge_points.append((1.0 - t) * start + t * end)

    points = np.vstack(edge_points)
    return points + rng.normal(scale=noise, size=points.shape)


def main() -> None:
    points = sample_noisy_star()
    forest = PersistenceForest(points)

    print(f"Built a {forest.dim}D PersistenceForest with {len(forest.barcode)} bars.")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(points[:, 0], points[:, 1], s=9, color="black")
    ax.set_aspect("equal")
    ax.set_title("Input point cloud")

    fig, ax = plt.subplots(figsize=(6, 3))
    forest.plot_barcode(
        ax=ax,
        min_bar_length=0.01,
        coloring="forest",
        bar_width=3,
        title="Barcode",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, filt_val in zip(axes, [0.35, 0.75]):
        forest.plot_at_filtration(
            filt_val,
            ax=ax,
            min_bar_length=0.05,
            coloring="bars",
            vertex_size=7,
            show=False,
            title=f"filtration value {filt_val:g}",
        )
    plt.tight_layout()

    forest.plot_barcode_cycle_reps(
        relative_position=0.1,
        min_bar_length=0.05,
        coloring="bars",
        linewidth_cycle=2.0,
        vertex_size=7,
        style_2d={"point_color": "0.15", "point_alpha": 0.75},
        show=False,
    )

    cycle_reps = forest.barcode_cycle_reps(relative_position=0.1, min_bar_length=0.05)
    print(f"Extracted {len(cycle_reps)} representative cycle(s).")
    if cycle_reps:
        first_rep_coords = cycle_reps[0].vertex_coordinates(
            forest.point_cloud,
            signed=False,
        )
        print(f"First representative touches {first_rep_coords.shape[0]} vertices.")

    grid = np.linspace(0.0, 1.4, 200)
    landscape_specs = [
        ("edge length", signed_chain_edge_length),
        ("excess curvature", signed_chain_excess_curvature),
    ]

    for label, cycle_func in landscape_specs:
        forest.compute_measurement_landscapes(
            cycle_func=cycle_func,
            label=label,
            max_k=3,
            x_grid=grid,
            min_bar_length=0.05,
            cache=True,
        )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    forest.plot_measurement_landscapes(
        label="edge length",
        ks=[1, 2, 3],
        ax=axes[0],
        title="Edge-length landscapes",
        linewidth=2.0,
        show=False,
    )
    forest.plot_measurement_landscapes(
        label="excess curvature",
        ks=[1, 2, 3],
        ax=axes[1],
        title="Excess-curvature landscapes",
        linewidth=2.0,
        show=False,
    )
    plt.tight_layout()

    length_values = forest.landscape_families["edge length"].evaluate_on_grid(
        grid,
        levels=3,
    )
    print(f"Edge-length landscape feature array shape: {length_values.shape}")

    plt.show()


if __name__ == "__main__":
    main()
