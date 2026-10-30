#%%
import numpy as np
import matplotlib.pyplot as plt
from persforest import PersistenceForest
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
from matplotlib.transforms import Bbox

plt.rcParams.update({
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
    "grid.color": "#d3d3d3",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

latex_textwidth = 448.13095
latex_linewidth = 448.13095

inches_per_pt = 1/72.27
width = latex_linewidth * inches_per_pt

def savefig_trim_vertical_preserve_width(fig, path, pad_inches=0.02, **kwargs):
    """Trim top/bottom whitespace while preserving the figure's full width."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bbox = fig.get_tightbbox(renderer)
    vertical_bbox = Bbox.from_extents(
        0,
        tight_bbox.y0 - pad_inches,
        fig.get_figwidth(),
        tight_bbox.y1 + pad_inches,
    )
    fig.savefig(path, bbox_inches=vertical_bbox, **kwargs)

def add_right_margin_matching_left_overhang(fig, left_ax):
    """Add right margin matching labels/ticks protruding left of the leftmost axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = left_ax.get_window_extent(renderer)
    tight_bbox = left_ax.get_tightbbox(renderer)
    left_overhang = max(0, axes_bbox.x0 - tight_bbox.x0) / fig.bbox.width
    fig.subplots_adjust(right=fig.subplotpars.right - left_overhang)

def style_paper_barcode(ax, *, show_x_arrow=True):
    """Use a light L-shaped axis frame for compact paper barcode panels."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["left"].set_visible(True)
    #ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("0")
    ax.spines["bottom"].set_visible(True)
    #ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("0")

    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", width=0.6, length=2.5, pad=1.5)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.4, alpha=0.35)
    ax.set_facecolor("none")
    if show_x_arrow:
        ax.annotate(
            "",
            xy=(1.025, 0),
            xytext=(1.0, 0),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "->",
                "color": "0",
                "linewidth": 0.6,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            clip_on=False,
        )

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

Path("paper_figures").mkdir(parents=True, exist_ok=True)


# %% Example: Point clouds with barcodes

from point_cloud_sampling import sample_noisy_circle, sample_noisy_star,  sample_noisy_ellipse,  sample_noisy_circle_with_tendril, sample_noisy_star_pointy
from persforest.geometry.planar import polygon_area, polygon_length, polygon_length_squared_area_ratio_normalized, curvature_excess

n_points = 1000

circle = sample_noisy_circle(n_points, noise_std=0.02, seed=4, radius = 0.38)*10
star = sample_noisy_star(n_points, spikes=5, amplitude=0.4,radius=0.62, noise_std=0.02)*10
star_pointy = sample_noisy_star_pointy(n_points, spikes=7, r_inner=0.4, r_outer=1, noise_std=0.02, seed=42)*10
ellipse = sample_noisy_ellipse(n_points, a=0.9,b=0.37, noise_std = 0.02)*10
circle_line = sample_noisy_circle_with_tendril(n_points, radius=0.51, tendril_length=0.3, tendril_fraction=0.05, noise_std=0.02, tendril_width_deg=2, seed=1)*10
circle_forest = PersistenceForest(circle)
star_forest = PersistenceForest(star)
pointy_star_forest = PersistenceForest(star_pointy)
ellipse_forest = PersistenceForest(ellipse)
circle_line_forest = PersistenceForest(circle_line)
color_palette = list(plt.get_cmap("tab10").colors)

name_forest_list = [("Circle", circle_forest), ("Ellipse", ellipse_forest), ("Spoke Circle", circle_line_forest), ("Star", star_forest)]

fig, axes = plt.subplots(nrows=2,ncols=4, figsize=(0.95*width, 0.95*width*2.2/5), gridspec_kw={"height_ratios": [3, 1]}, layout='constrained', sharey='row')
for i, (name,forest) in enumerate(name_forest_list):
    point_cloud = forest.point_cloud

    #point cloud
    axes[0,i].scatter(point_cloud[:,0], point_cloud[:,1], s=.55, color= color_palette[i], ec='none')
    axes[0,i].set_box_aspect(1)

    #barcode
    forest.plot_barcode(ax=axes[1,i],
                        max_bars=12, 
                        sort="death", 
                        coloring="grey", 
                        title="", 
                        bar_width = 1)
    axes[1,i].set_xlim((0,4))
    axes[1,i].grid(False)
    axes[1,i].set_xticks(np.linspace(0,4,3))
    axes[1,i].set_xlabel(None)
axes[1,0].set_ylabel("barcode $H_1$", fontsize = 8)

for i in range (4):
    axes[0,i].set_xlim((-10,10))
    axes[0,i].set_ylim((-10,10))
fig.savefig('paper_figures/point_clouds_with_barcodes.pdf', transparent=True, dpi=300)
plt.show()

# %% Example: Measurement comparisons
from persforest.measurements.basic import signed_chain_area, signed_chain_edge_length
from persforest.measurements.shape import signed_chain_excess_curvature, signed_chain_circularity_complement, signed_chain_convex_hull_area_deficit, signed_chain_convex_hull_perimeter_deficit
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(0.92*width, 0.85 * 1.5/4 *width), sharex='col', layout='constrained')

cycle_funcs_list = [
    ("$f$ = arc length",signed_chain_edge_length),
    ("$f$ = enclosed area",signed_chain_area), 
    #("$f$ = excess curvature",signed_chain_excess_curvature)
    ("$f$ = hull perimeter deficit", signed_chain_convex_hull_perimeter_deficit),
    #("$f$ = hull area deficit", signed_chain_convex_hull_area_deficit),
]

use_signed = False


from persforest.plotting.landscapes import plot_landscape_comparison

forests = [forest for _, forest in name_forest_list]
forest_labels = [name for name, _ in name_forest_list]


for i, (func_name, cycle_func) in enumerate(cycle_funcs_list):
    for forest in forests:
        forest.compute_measurement_landscapes(
            cycle_func=cycle_func,
            max_k=6,
            x_grid=np.linspace(0,4,20000),
            label=func_name,
            signed=use_signed,
            cache_profiles=True)

    for j, (name, forest) in enumerate(name_forest_list):
        forest.plot_measurement_landscapes(
            label=func_name,
            ax=axes[1, i],
            show=False,
            ks=[1],
            show_legend=False,
            linewidth=1.5
        )
    axes[1,i].set_autoscaley_on(True)
    axes[1,i].relim()
    axes[1,i].autoscale_view(scalex=False, scaley=True)

    _, ymax = axes[1,i].get_ylim()
    axes[1,i].set_ylim(0, ymax)


    axes[1, i].set_title(None)
    axes[1, i].set_xlim((0,4))
    axes[1, i].set_xticks(np.linspace(0,4,3))
    axes[1, i].set_xlabel(None)
    axes[1, i].set_ylabel(None) 
    axes[1, i].grid(False)

for i, (func_name, cycle_func) in enumerate(cycle_funcs_list):
    for j,(name, forest) in enumerate(name_forest_list):
        forest.plot_barcode_measurement(cycle_func=cycle_func, 
                                        ax = axes[0, i], 
                                        signed=use_signed, 
                                        label = name, 
                                        show_baseline = False, 
                                        x_range = (0,4),
                                        color = color_palette[j],
                                        linewidth=1.5)

        
    axes[0, i].set_title(func_name)
    axes[0, i].set_xlim((0,4))
    axes[0, i].set_xticks(np.linspace(0,4,3))
    axes[0, i].set_ylim(bottom=0) 
    axes[0, i].grid(False)


axes[0,0].set_ylabel(r"$f \circ \gamma_I$")
axes[1,0].set_ylabel(r"$\lambda_1(-, f)$")
fig.align_ylabels()
fig.savefig('paper_figures/point_clouds_landscape_comparisons.pdf', transparent=True, dpi=300)
plt.show()

# %% Mixed bouquet non-circularity measurement-landscape construction
from point_cloud_sampling import sample_mixed_bouquet
from measurement_landscape_construction import (
    OUTPUT_DIRECTORY,
    plot_measurement_landscape_construction,
)
from persforest.measurements.shape import signed_chain_circularity_complement

# Match the selected construction figure, including its seeded point cloud,
# filtration scale, snapshot times, and unsigned measurement.
construction_scale = 10.0
forest = PersistenceForest(construction_scale * sample_mixed_bouquet(seed=16))
snapshot_times = construction_scale * np.array(
    [0.00, 0.10, 0.17, 0.28, 0.39, 0.52, 0.70, 1.05]
)
fig_bouquet_construction = plot_measurement_landscape_construction(
    forest,
    snapshot_times,
    cycle_func=signed_chain_circularity_complement,
    measurement_label="non-circularity",
    n_bars=4,
    n_landscapes=4,
    signed=False,
)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
fig_bouquet_construction.savefig(
    OUTPUT_DIRECTORY / "measurement_landscape_construction_mixed_bouquet_non_circularity.pdf",
    dpi=300,
    transparent=True,
)
plt.show()


#%% Non-circularity landscapes
cmap = "viridis"
higher_layers_on_top = True
x_max = 15.5
seed =5
from point_cloud_sampling import sample_points_without_balls
points_with_6_holes = sample_points_without_balls(3000, dim=2, num_discs=6, radius_range=[0.09,0.15], seed=seed) * 100
forest_6_holes = PersistenceForest(points_with_6_holes)

ratios = [1,1.2,1.2]

fig, axes = plt.subplots(ncols=3,figsize=(width,width*0.36), gridspec_kw={"width_ratios": ratios, 'wspace':0.25})
forest_6_holes.plot_at_filtration(0,ax=axes[0], coloring="forest", vertex_size=1.3, show=False)
#ax.set_axis_off()
axes[0].set_title("Point Cloud")
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)

from persforest.measurements.shape import signed_chain_circularity_complement
from persforest.measurements.basic import constant_one_functional


forest_6_holes.compute_measurement_landscapes(cycle_func=constant_one_functional,
                                                    label = "standard", 
                                                    signed=False,
                                                    max_k=7,
                                                    x_grid= np.linspace(0,x_max,5000))
forest_6_holes.plot_measurement_landscapes(ax=axes[1], 
                                           label="standard", 
                                           show = False,
                                           linewidth=1.3,
                                           cmap = cmap,
                                           higher_layers_on_top=higher_layers_on_top)
axes[1].set_title("Persistence Landscapes", fontsize = 10)
axes[1].legend(
    frameon=False,
    fontsize=6,
    handlelength=0.8,   # length of colored line segment
    #handletextpad=0.3,  # gap between line and text
    borderpad=0.2,      # padding inside legend box
    labelspacing=0.3,   # vertical space between entries
    columnspacing=0.5,
)
axes[1].grid(False)
axes[1].set_xlim(0,x_max)
axes[1].set_ylim(bottom=0)
axes[1].set_xlabel("")
axes[1].set_ylabel("")


forest_6_holes.compute_measurement_landscapes(cycle_func=signed_chain_circularity_complement,
                                                    label = "circularity_complement", 
                                                    signed=False,
                                                    max_k=7,
                                                    x_grid= np.linspace(0,x_max,5000))
forest_6_holes.plot_measurement_landscapes(ax=axes[2], 
                                           label="circularity_complement", 
                                           show = False, 
                                           linewidth=1.3,
                                           cmap = cmap,
                                           higher_layers_on_top=higher_layers_on_top)
axes[2].set_title("Non-Circularity Landscapes")
axes[2].legend(
    frameon=False,
    fontsize=6,
    handlelength=0.8,   # length of colored line segment
    #handletextpad=0.3,  # gap between line and text
    borderpad=0.2,      # padding inside legend box
    labelspacing=0.3,   # vertical space between entries
    columnspacing=0.5,
)
axes[2].grid(False)
axes[2].set_xlim(0,x_max)
axes[2].set_ylim(bottom=0)
axes[2].set_xlabel("")
axes[2].set_ylabel("")

axes[0].set_box_aspect(1)
axes[1].set_box_aspect(ratios[0] / ratios[1])
axes[2].set_box_aspect(ratios[0] / ratios[2])
fig.subplots_adjust(left=0.04, right=0.995, bottom=0.14, top=0.86)

fig.savefig(f"paper_figures/circle_6holes_random_points_non-circularity-landscapes_seed{seed}_cmap-{cmap}_zorder-{higher_layers_on_top}.pdf",dpi=300, transparent=True)
plt.show()

#%% Non-convexity landscapes
import persforest
cmap = "viridis"
higher_layers_on_top = True
x_max = 15.5
seed =5
from point_cloud_sampling import sample_points_without_balls
points_with_6_holes = sample_points_without_balls(3000, dim=2, num_discs=6, radius_range=[0.09,0.15], seed=seed) * 100
forest_6_holes = PersistenceForest(points_with_6_holes)

ratios = [1,1.2,1.2]

fig, axes = plt.subplots(ncols=3,figsize=(width,width*0.36), gridspec_kw={"width_ratios": ratios, 'wspace':0.25})
forest_6_holes.plot_at_filtration(0,ax=axes[0], coloring="forest", vertex_size=1.3, show=False)
#ax.set_axis_off()
axes[0].set_title("Point Cloud")
axes[0].set_xlim(0,100)
axes[0].set_ylim(0,100)

from persforest.measurements.shape import signed_chain_convex_hull_perimeter_deficit, signed_chain_convex_hull_area_deficit
from persforest.measurements.basic import constant_one_functional


forest_6_holes.compute_measurement_landscapes(cycle_func=constant_one_functional,
                                                    label = "standard", 
                                                    signed=False,
                                                    max_k=7,
                                                    x_grid= np.linspace(0,x_max,5000))
forest_6_holes.plot_measurement_landscapes(ax=axes[1], 
                                           label="standard", 
                                           show = False,
                                           linewidth=1.3,
                                           cmap = cmap,
                                           higher_layers_on_top=higher_layers_on_top)
axes[1].set_title("Persistence Landscapes", fontsize = 10)
axes[1].legend(
    frameon=False,
    fontsize=6,
    handlelength=0.8,   # length of colored line segment
    #handletextpad=0.3,  # gap between line and text
    borderpad=0.2,      # padding inside legend box
    labelspacing=0.3,   # vertical space between entries
    columnspacing=0.5,
)
axes[1].grid(False)
axes[1].set_xlim(0,x_max)
axes[1].set_ylim(bottom=0)
axes[1].set_xlabel("")
axes[1].set_ylabel("")


forest_6_holes.compute_measurement_landscapes(cycle_func=signed_chain_convex_hull_area_deficit,
                                                    label = "hull_area_deficit", 
                                                    signed=False,
                                                    max_k=7,
                                                    x_grid= np.linspace(0,x_max,5000))
forest_6_holes.plot_measurement_landscapes(ax=axes[2], 
                                           label="hull_area_deficit", 
                                           show = False, 
                                           linewidth=1.3,
                                           cmap = cmap,
                                           higher_layers_on_top=higher_layers_on_top)
axes[2].set_title("Hull Area Deficit Landscapes")
axes[2].legend(
    frameon=False,
    fontsize=6,
    handlelength=0.8,   # length of colored line segment
    #handletextpad=0.3,  # gap between line and text
    borderpad=0.2,      # padding inside legend box
    labelspacing=0.3,   # vertical space between entries
    columnspacing=0.5,
)
axes[2].grid(False)
axes[2].set_xlim(0,x_max)
axes[2].set_ylim(bottom=0)
axes[2].set_xlabel("")
axes[2].set_ylabel("")

axes[0].set_box_aspect(1)
axes[1].set_box_aspect(ratios[0] / ratios[1])
axes[2].set_box_aspect(ratios[0] / ratios[2])
fig.subplots_adjust(left=0.04, right=0.995, bottom=0.14, top=0.86)

fig.savefig(f"paper_figures/circle_6holes_random_points_non-convexity-landscapes_seed{seed}_cmap-{cmap}_zorder-{higher_layers_on_top}.pdf",dpi=300, transparent=True)
plt.show()

# %% cycle rep showcase
points_with_6_holes = sample_points_without_balls(3000, dim=2, num_discs=6, radius_range=[0.05,0.15], seed=5) * 100
forest_6_holes = PersistenceForest(points_with_6_holes)

fig, axes = plt.subplots(ncols=3, figsize=(width,width*0.39), dpi=300, gridspec_kw={"width_ratios": [3,3,2.5]})
cycle_rep_style_2d = {"point_color": "0.2", "point_alpha": 0.9}

forest_6_holes.plot_barcode_cycle_reps(relative_position=0, min_bar_length=5, coloring="bars", linewidth_cycle=1.2, ax=axes[0], show=False, remove_double_edges=True, vertex_size=1, style_2d=cycle_rep_style_2d)
axes[0].set_title("Cycle reps at birth")
axes[0].set_xlim((0,100))
axes[0].set_ylim((0,100))


forest_6_holes.plot_barcode_cycle_reps(relative_position=0.4, min_bar_length=5, coloring="bars", linewidth_cycle=1.2, ax=axes[1], show=False, remove_double_edges=True, vertex_size=1, style_2d=cycle_rep_style_2d)
axes[1].set_title(r"Cycle reps at rel. pos. 0.4")
axes[1].set_xlim((0,100))
axes[1].set_ylim((0,100))

forest_6_holes.plot_barcode(coloring="bars", sort = "length", min_bar_length=1,max_bars= 30 , ax=axes[2], bar_width = 1.0, descending=False)
axes[2].set_title(r"$H_1$ Barcode")
axes[2].set_xlabel(None)
style_paper_barcode(axes[2])

add_right_margin_matching_left_overhang(fig, axes[0])

savefig_trim_vertical_preserve_width(
    fig,
    f"paper_figures/circle_6holes_random_points_cycle_reps_seed{5}.pdf",
    dpi=300,
    transparent=True,
)
