"""Render candidate arc-length measurement-landscape construction figures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from construction_point_clouds import (
    sample_circle_rounded_rectangle_intersection,
    sample_circle_star_pair,
    sample_compartmented_oval,
    sample_mixed_bouquet,
    sample_mixed_loops,
    sample_tendril_triplet,
    sample_three_cavities,
)
from persforest import PersistenceForest
from persforest.cycle_rep_vectorisations import (
    signed_chain_circularity_complement,
    signed_chain_circularity_complement_squared,
    signed_chain_convex_hull_area_deficit,
    signed_chain_convex_hull_perimeter_deficit,
    signed_chain_edge_length,
    signed_chain_excess_curvature,
)


PAPER_WIDTH = 448.13095 / 72.27
OUTPUT_DIRECTORY = Path(__file__).with_name("paper_figures")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.titlesize": 10,
        "figure.dpi": 300,
        "axes.grid": False,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _style_barcode_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", width=0.6, length=2.5, pad=1.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.35)
    ax.set_facecolor("none")


def plot_measurement_landscape_construction(
    forest: PersistenceForest,
    filtration_values: Sequence[float],
    *,
    cycle_func,
    measurement_label: str,
    n_bars: int,
    n_landscapes: int,
    signed: bool = False,
    zorder_by_birth: bool = True,
) -> Figure:
    """Plot one measurement-landscape construction.

    Earlier-born bars are drawn in front by default. Set
    ``zorder_by_birth=False`` to draw longer bars in front instead.
    """
    filtration_values = np.asarray(filtration_values, dtype=float)
    if (
        filtration_values.ndim != 1
        or len(filtration_values) == 0
        or not np.all(np.isfinite(filtration_values))
        or np.any(filtration_values < 0.0)
        or np.any(np.diff(filtration_values) <= 0.0)
    ):
        raise ValueError(
            "filtration_values must be finite, nonnegative, and strictly increasing"
        )

    bars = forest.longest_bars(n_bars)
    if len(bars) != n_bars:
        raise ValueError(f"forest has fewer than {n_bars} bars")

    forest.set_longest_bar_colors(
        coloring="bars",
        colors=[f"C{index}" for index in range(n_bars)],
    )
    color_map = forest._get_color_map(coloring="bars")
    min_bar_length = bars[-1].lifespan()
    x_max = max(filtration_values[-1], max(bar.death for bar in bars)) + 0.05
    x_range = (0.0, x_max)

    landscapes = forest.compute_measurement_landscapes(
        cycle_func,
        label=measurement_label,
        max_k=n_landscapes,
        min_bar_length=min_bar_length,
        x_grid=np.linspace(*x_range, 2000),
        signed=signed,
        cache_functionals=True,
    )
    profiles = forest.barcode_functionals[measurement_label]
    if len(profiles.bars) != n_bars:
        raise ValueError("the selected persistence cutoff does not retain exactly n_bars")
    if zorder_by_birth:
        birth_order = sorted(bars, key=lambda bar: bar.birth)
        bar_priority = {
            bar: n_bars - index for index, bar in enumerate(birth_order)
        }
    else:
        bar_priority = {bar: bar.lifespan() for bar in bars}

    fig = plt.figure(
        figsize=(PAPER_WIDTH, 0.97 * PAPER_WIDTH),
    )
    construction_grid = fig.add_gridspec(
        nrows=6,
        ncols=1,
        height_ratios=[0.18, 0.95, 0.62, 1.16, 1.45, 1.45],
        left=0.08,
        right=0.99,
        bottom=0.065,
        top=0.975,
        hspace=0.38,
    )

    ax_header = fig.add_subplot(construction_grid[0])
    snapshot_grid = construction_grid[1].subgridspec(
        1,
        len(filtration_values),
        wspace=0.04,
    )
    snapshot_axes = [
        fig.add_subplot(snapshot_grid[0, index])
        for index in range(len(filtration_values))
    ]
    ax_barcode = fig.add_subplot(construction_grid[2])
    ax_profiles = fig.add_subplot(construction_grid[3], sharex=ax_barcode)
    ax_convolutions = fig.add_subplot(construction_grid[4], sharex=ax_barcode)
    ax_landscapes = fig.add_subplot(construction_grid[5], sharex=ax_barcode)

    ax_header.set_axis_off()
    ax_header.text(
        0.0,
        0.5,
        r"\textbf{(a)} Cycle progression barcode",
        transform=ax_header.transAxes,
        fontsize=plt.rcParams["axes.titlesize"],
        ha="left",
        va="center",
    )

    points = forest.point_cloud
    point_padding = 0.04 * np.ptp(points, axis=0)
    point_x_range = (
        points[:, 0].min() - point_padding[0],
        points[:, 0].max() + point_padding[0],
    )
    point_y_range = (
        points[:, 1].min() - point_padding[1],
        points[:, 1].max() + point_padding[1],
    )
    cycle_style = {
        "complex_face_alpha": 0.10,
        "complex_edge_width": 0.18,
        "cycle_edge_width": 1.0,
    }

    for time_index, (snapshot_ax, filtration_value) in enumerate(
        zip(snapshot_axes, filtration_values)
    ):
        forest.plot_at_filtration(
            filt_val=filtration_value,
            ax=snapshot_ax,
            title="",
            show=False,
            vertex_size=1.1,
            coloring="bars",
            min_bar_length=min_bar_length,
            point_zorder=4.5,
            style_2d=cycle_style,
        )
        active_bars = [
            bar
            for bar in forest.barcode
            if (
                bar.birth <= filtration_value < bar.death
                and bar.lifespan() >= min_bar_length
            )
        ]
        n_cycle_collections = len(active_bars)
        if len(points) > 500:
            complex_collections = (
                snapshot_ax.collections[:-n_cycle_collections]
                if n_cycle_collections
                else snapshot_ax.collections
            )
            for collection in complex_collections:
                collection.set_rasterized(True)
        if active_bars:
            cycle_collections = snapshot_ax.collections[-n_cycle_collections:]
            for cycle_collection, bar in zip(cycle_collections, active_bars):
                cycle_collection.set_zorder(5.0 + bar_priority[bar])
        snapshot_ax.set(
            xlim=point_x_range,
            ylim=point_y_range,
            aspect="equal",
            title=rf"$t_{{{time_index}}}={filtration_value:.1f}$",
        )
        snapshot_ax.title.set_fontsize(6)
        snapshot_ax.set_axis_off()

    forest.plot_barcode(
        ax=ax_barcode,
        max_bars=n_bars,
        sort="length",
        coloring="bars",
        title="",
        xlabel="",
        bar_width=1.5,
        descending=True,
        tight_layout=False,
    )
    _style_barcode_axis(ax_barcode)
    for bar_index, bar in enumerate(bars):
        ax_barcode.text(
            bar.death + 0.012,
            bar_index,
            rf"$I_{{{bar_index + 1}}}$",
            color=color_map[bar],
            fontsize=6,
            ha="left",
            va="center",
            clip_on=False,
        )

    profile_max = 0.0
    for bar in bars:
        step_function = profiles[bar]
        step_function.plot(
            ax=ax_profiles,
            x_range=x_range,
            color=color_map[bar],
            linewidth=1.3,
            solid_capstyle="butt",
            zorder=2.0 + bar_priority[bar],
        )
        profile_max = max(profile_max, float(np.max(step_function.vals)))

    for convolution_index, bar in enumerate(profiles.bars):
        convolution = landscapes.bar_kernels[convolution_index]
        convolution_xs = np.concatenate(
            ([x_range[0]], convolution.xs, [x_range[1]])
        )
        convolution_ys = np.concatenate(
            ([0.0], convolution.ys, [0.0])
        )
        ax_convolutions.plot(
            convolution_xs,
            convolution_ys,
            color=color_map[bar],
            linewidth=1.5,
            alpha=0.9,
            zorder=2.0 + bar_priority[bar],
        )

    landscape_colors = plt.get_cmap("viridis")(
        np.linspace(0.1, 0.9, n_landscapes)
    )
    for landscape_index, landscape in landscapes.landscapes.items():
        ax_landscapes.plot(
            landscape.xs,
            landscape.ys,
            label=rf"$\lambda_{{{landscape_index}}}$",
            color=landscape_colors[landscape_index - 1],
            linewidth=1.4,
            zorder=10 + landscape_index,
        )
    ax_landscapes.legend(
        loc="upper right",
        ncols=n_landscapes,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.0,
        borderaxespad=1.1,
    )

    landscape_max = max(
        float(np.max(landscape.ys))
        for landscape in landscapes.landscapes.values()
    )
    ax_barcode.set_xlim(x_range)
    ax_profiles.set_ylim(0.0, 1.06 * profile_max)
    ax_convolutions.set_ylim(0.0, 1.08 * landscape_max)
    ax_landscapes.set_ylim(0.0, 1.08 * landscape_max)

    measurement_title = measurement_label.capitalize()
    ax_profiles.set_ylabel(measurement_label)
    ax_profiles.set_title(
        rf"\textbf{{(b)}} {measurement_title} measurement profiles",
        loc="left",
        pad=3.0,
    )
    ax_convolutions.set_ylabel("convolution value")
    ax_convolutions.set_title(
        r"\textbf{(c)} Per-interval rescaled convolutions",
        loc="left",
        pad=2.0,
    )
    ax_landscapes.set(
        xlabel="filtration value",
        ylabel="landscape value",
    )
    ax_landscapes.set_title(
        rf"\textbf{{(d)}} {measurement_title} measurement landscapes",
        loc="left",
        pad=2.0,
    )

    for filtration_value in filtration_values:
        ax_barcode.axvline(
            filtration_value,
            color="0.55",
            linewidth=0.6,
            linestyle=(0, (2, 2)),
            zorder=0,
        )
        ax_barcode.scatter(
            filtration_value,
            1.0,
            s=8,
            facecolor="white",
            edgecolor="0.4",
            linewidth=0.55,
            transform=ax_barcode.get_xaxis_transform(),
            clip_on=False,
            zorder=4,
        )
        ax_profiles.axvline(
            filtration_value,
            color="0.55",
            linewidth=0.6,
            linestyle=(0, (2, 2)),
            zorder=0,
        )

    for ax in (ax_barcode, ax_profiles, ax_convolutions, ax_landscapes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(
            axis="x",
            color="0.82",
            linewidth=0.45,
            linestyle=":",
            alpha=0.8,
        )
    for ax in (ax_barcode, ax_profiles, ax_convolutions):
        ax.tick_params(axis="x", which="both", labelbottom=False)
    ax_landscapes.spines["bottom"].set_position(("data", 0))
    fig.align_ylabels((ax_profiles, ax_convolutions, ax_landscapes))

    for snapshot_ax, filtration_value in zip(snapshot_axes, filtration_values):
        leader = ConnectionPatch(
            xyA=(0.5, -0.02),
            coordsA=snapshot_ax.transAxes,
            xyB=(filtration_value, 1.0),
            coordsB=ax_barcode.get_xaxis_transform(),
            arrowstyle="-",
            color="0.55",
            linewidth=0.5,
            clip_on=False,
            zorder=1,
        )
        leader.set_in_layout(False)
        fig.add_artist(leader)

    return fig


if __name__ == "__main__":
    scale = 10.0
    cases = [
        (
            "mixed_loops",
            sample_mixed_loops(),
            [0.00, 0.08, 0.18, 0.20, 0.55, 0.67, 0.73, 1.10],
            3,
            3,
        ),
        (
            "compartmented_oval",
            sample_compartmented_oval(),
            [0.00, 0.07, 0.10, 0.21, 0.40, 0.64, 0.66, 0.80],
            3,
            3,
        ),
        (
            "tendril_triplet",
            sample_tendril_triplet(),
            [0.00, 0.1, 0.30, 0.4, 0.5,  0.6, 0.7, 1],
            3,
            3,
        ),
        (
            "three_cavities",
            sample_three_cavities(),
            [0.00, 0.10, 0.12, 0.25, 0.45, 0.48, 0.60, 0.74],
            3,
            3,
        ),
        (
            "circle_star_pair",
            sample_circle_star_pair(),
            [0.00, 0.14, 0.22, 0.32, 0.415, 0.52, 0.70],
            3,
            3,
        ),
        (
            "circle_rounded_rectangle_intersection",
            sample_circle_rounded_rectangle_intersection(),
            [0.00, 0.085, 0.18, 0.40, 0.69, 0.85],
            3,
            3,
        ),
        (
            "mixed_bouquet",
            sample_mixed_bouquet(),
            [0.00, 0.10, 0.17, 0.28, 0.39, 0.52, 0.70, 1.05],
            4,
            4,
        ),
    ]
    measurements = [
        ("", signed_chain_edge_length, "arc length", False),
        (
            "_non_circularity",
            signed_chain_circularity_complement,
            "non-circularity",
            False,
        ),
        (
            "_circularity_complement_squared",
            signed_chain_circularity_complement_squared,
            r"$(1-\mathrm{circularity})^2$",
            False,
        ),
        (
            "_excess_curvature",
            signed_chain_excess_curvature,
            "excess curvature",
            False,
        ),
        (
            "_convex_hull_area_deficit",
            signed_chain_convex_hull_area_deficit,
            "convex hull area deficit",
            True,
        ),
        (
            "_convex_hull_perimeter_deficit",
            signed_chain_convex_hull_perimeter_deficit,
            "convex hull perimeter deficit",
            True,
        ),
    ]

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for case_name, points, times, n_bars, n_landscapes in cases:
        forest = PersistenceForest(scale * points)
        times = scale * np.asarray(times)
        print(case_name)
        for bar in forest.longest_bars(n_bars + 1):
            print(
                f"  birth={bar.birth:.4f}, death={bar.death:.4f}, "
                f"lifespan={bar.lifespan():.4f}"
            )
        for filename_suffix, cycle_func, measurement_label, signed in measurements:
            print(f"  {measurement_label}")
            figure = plot_measurement_landscape_construction(
                forest,
                times,
                cycle_func=cycle_func,
                measurement_label=measurement_label,
                n_bars=n_bars,
                n_landscapes=n_landscapes,
                signed=signed,
            )
            figure.savefig(
                OUTPUT_DIRECTORY
                / (
                    "measurement_landscape_construction_"
                    f"{case_name}{filename_suffix}.pdf"
                ),
                dpi=300,
                transparent=True,
            )
            plt.close(figure)
