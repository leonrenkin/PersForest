#%%
import numpy as np
import matplotlib.pyplot as plt
from persforest.PersistenceForest import PersistenceForest
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
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

def style_paper_barcode(ax):
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

%load_ext autoreload
%autoreload 3
!mkdir -p paper_figures


# %% Example: Point clouds with barcodes

from point_cloud_sampling import sample_noisy_circle, sample_noisy_star,  sample_noisy_ellipse,  sample_noisy_circle_with_tendril, sample_noisy_star_pointy
from persforest.cycle_rep_vectorisations import polygon_area, polygon_length, polygon_length_squared_area_ratio_normalized, curvature_excess

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
color_palette = sns.color_palette("tab10")

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
    axes[1,i].set_xlim((0,8))
    axes[1,i].grid(False)
    axes[1,i].set_xticks(np.linspace(0,8,3))
    axes[1,i].set_xlabel(None)
axes[1,0].set_ylabel("barcode $H_1$", fontsize = 8)

for i in range (4):
    axes[0,i].set_xlim((-10,10))
    axes[0,i].set_ylim((-10,10))
fig.savefig('paper_figures/point_clouds_with_barcodes.pdf', transparent=True, dpi=300)
plt.show()

# %% Example: Measurement comparisons
from persforest.cycle_rep_vectorisations import signed_chain_area, signed_chain_edge_length, signed_chain_excess_curvature, signed_chain_circularity_complement
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(0.85*width, 0.85 * 1.5/4 *width), sharex='col', layout='constrained')

cycle_funcs_list = [
    ("$f$ = length",signed_chain_edge_length),
    ("$f$ = area",signed_chain_area), 
    ("$f$ = excess curvature",signed_chain_excess_curvature)
]

use_signed = False


from persforest.forest_landscapes import plot_landscape_comparison

forests = [forest for _, forest in name_forest_list]
forest_labels = [name for name, _ in name_forest_list]


for i, (func_name, cycle_func) in enumerate(cycle_funcs_list):
    for forest in forests:
        forest.compute_measurement_landscapes(
            cycle_func=cycle_func,
            max_k=6,
            x_grid=np.linspace(0,8,20000),
            label=func_name,
            signed=use_signed,
            cache_functionals=True) 

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
    axes[1, i].set_xlim((0,8)) 
    axes[1, i].set_xticks(np.linspace(0,8,3))
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
                                        x_range = (0,8),
                                        color = color_palette[j],
                                        linewidth=1.5)

        
    axes[0, i].set_title(func_name)
    axes[0, i].set_xlim((0,8)) 
    axes[0, i].set_xticks(np.linspace(0,8,3))
    axes[0, i].set_ylim(bottom=0) 
    axes[0, i].grid(False)


axes[0,0].set_ylabel(r"$f \circ \gamma_I$")
axes[1,0].set_ylabel(r"$\lambda_1(-, f)$")
fig.align_ylabels()
fig.savefig('paper_figures/point_clouds_landscape_comparisons.pdf', transparent=True, dpi=300)
plt.show()


# %% four leaf clover
def sample_four_leaf_clover(n, r=1.0, noise=0.0, R = (0, 2*np.pi)):
    t = np.random.uniform(R[0], R[1], n)
    p = (r * np.sin(2*t))[:, None] * np.column_stack((np.cos(t), np.sin(t)))
    return p + np.random.normal(0, noise, p.shape)
np.random.seed(6)
pts = np.vstack([
    sample_four_leaf_clover(30, noise=0.01, R = (0, .5*np.pi)),
    .7*sample_four_leaf_clover(40, noise=0.01, R = (.5*np.pi, np.pi)),
    1.5*sample_four_leaf_clover(60, noise=0.01, R = (1.5*np.pi, 2*np.pi)),
    2.5*sample_four_leaf_clover(50, noise=0.01, R = (np.pi, 1.5*np.pi)),
])
pts = pts[np.linalg.norm(pts, axis=1) > 0.25]
forest = PersistenceForest(pts)
forest.set_longest_bar_colors(coloring = "forest", colors = [ "#0ec801",  "#a454f8",  "#e7298a",  "#17becf",  "#C74A06",  "#bcbd22",  "#7570b3", "#4a7814"])
landscapes = forest.compute_measurement_landscapes(
    signed_chain_circularity_complement,
    x_grid=np.linspace(-1.5/20, 1.5+1.5/20, 200),
    label = "circularity_complement"
)
fig1 = plt.figure(figsize=(1,1))
ax1 = fig1.gca()
ax1.set_aspect('equal')
ax1.scatter(*pts.T, s=0.5, color='black')
ax1.set_axis_off()
fig1.tight_layout(pad=0.0)
fig1.savefig("paper_figures/four_leaf_clover_landscapes.pdf", dpi=300, transparent=True)

fig2, axs2 = plt.subplots(nrows=2, figsize=(4,.8))
forest.plot_barcode(ax=axs2[0], max_bars=10, sort="length", coloring="forest", title="", bar_width=1.3, descending = True)
axs2[0].set_xlabel("")
axs2[0].set_yticks([])
axs2[0].set_xticks(np.linspace(0, 1.5, 10))
axs2[0].set_xlim((-1.5/20,1.5+1.5/20))
axs2[0].set_xticklabels([])
axs2[0].tick_params(axis='x', which='both', length=0)
axs2[0].grid()
for (k, l) in enumerate(list(landscapes.landscapes.values())[:3]):
    axs2[1].plot(l.xs, l.ys, label=f"{k+1}", lw=1, clip_on=False, zorder=10-k)
axs2[1].set_yticks([])
axs2[1].sharex(axs2[0])
axs2[1].tick_params(axis='x', which='both', length=0)
axs2[1].set_xlabel(None)
axs2[1].legend(loc='upper left', frameon=False)
axs2[1].grid()
axs2[1].set_ylim((0,0.07))

fig2.tight_layout(h_pad=0, w_pad=0.0)
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig2.savefig("paper_figures/four_leaf_clover_landscapes2.pdf", dpi=300, transparent=True)

N = axs2[0].get_xticks().shape[0]
fig3, axs3 = plt.subplots(ncols=N, figsize=(4,.4), sharex=True, sharey=True)
style_dict_2d = {'complex_edge_width':0.2, 'cycle_edge_width':1.2}
for (ax, t) in zip(axs3, axs2[0].get_xticks()):
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    forest.plot_at_filtration(filt_val=t, ax=ax, title="", show=False, vertex_size=1, style_2d = style_dict_2d, point_zorder=4.5)
fig3.tight_layout(pad=.5)
fig3.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig3.savefig("paper_figures/four_leaf_clover_landscapes3.pdf", dpi=300, transparent=True)


# %% four leaf clover with length
from persforest.cycle_rep_vectorisations import signed_chain_edge_length

def sample_four_leaf_clover(n, r=1.0, noise=0.0, R = (0, 2*np.pi)):
    t = np.random.uniform(R[0], R[1], n)
    p = (r * np.sin(2*t))[:, None] * np.column_stack((np.cos(t), np.sin(t)))
    return p + np.random.normal(0, noise, p.shape)
np.random.seed(6)
pts = np.vstack([
    sample_four_leaf_clover(30, noise=0.01, R = (0, .5*np.pi)),
    .7*sample_four_leaf_clover(40, noise=0.01, R = (.5*np.pi, np.pi)),
    1.5*sample_four_leaf_clover(60, noise=0.01, R = (1.5*np.pi, 2*np.pi)),
    2.5*sample_four_leaf_clover(50, noise=0.01, R = (np.pi, 1.5*np.pi)),
])
pts = pts[np.linalg.norm(pts, axis=1) > 0.25]
forest = PersistenceForest(pts)
forest.set_longest_bar_colors(coloring = "forest", colors = [ "#0ec801",  "#a454f8",  "#e7298a",  "#17becf",  "#C74A06",  "#bcbd22",  "#7570b3", "#4a7814"])
landscapes = forest.compute_measurement_landscapes(
    signed_chain_edge_length,
    x_grid=np.linspace(-1.5/20, 1.5+1.5/20, 2000),
    label = "length"
)
fig1 = plt.figure(figsize=(1,1))
ax1 = fig1.gca()
ax1.set_aspect('equal')
ax1.scatter(*pts.T, s=0.5, color='black')
ax1.set_axis_off()
fig1.tight_layout(pad=0.0)
fig1.savefig("paper_figures/four_leaf_clover_length_landscapes.pdf", dpi=300, transparent=True)

fig2, axs2 = plt.subplots(nrows=2, figsize=(4,.8))
forest.plot_barcode(ax=axs2[0], max_bars=10, sort="length", coloring="forest", title="", bar_width=1.3, descending = True)
axs2[0].set_xlabel("")
axs2[0].set_yticks([])
axs2[0].set_xticks(np.linspace(0, 1.5, 10))
axs2[0].set_xlim((-1.5/20,1.5+1.5/20))
axs2[0].set_xticklabels([])
axs2[0].tick_params(axis='x', which='both', length=0)
axs2[0].grid()
for (k, l) in enumerate(list(landscapes.landscapes.values())[:3]):
    axs2[1].plot(l.xs, l.ys, label=f"{k+1}", lw=1, clip_on=False, zorder=10-k)
axs2[1].set_yticks([])
axs2[1].sharex(axs2[0])
axs2[1].tick_params(axis='x', which='both', length=0)
axs2[1].set_xlabel(None)
axs2[1].legend(loc='upper left', frameon=False)
axs2[1].grid()
axs2[1].set_ylim((0,3))

fig2.tight_layout(h_pad=0, w_pad=0.0)
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig2.savefig("paper_figures/four_leaf_clover_length_landscapes2.pdf", dpi=300, transparent=True)

N = axs2[0].get_xticks().shape[0]
fig3, axs3 = plt.subplots(ncols=N, figsize=(4,.4), sharex=True, sharey=True)
style_dict_2d = {'complex_edge_width':0.2, 'cycle_edge_width':1.2}
for (ax, t) in zip(axs3, axs2[0].get_xticks()):
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    forest.plot_at_filtration(filt_val=t, ax=ax, title="", show=False, vertex_size=1, style_2d = style_dict_2d, point_zorder=4.5)
fig3.tight_layout(pad=.5)
fig3.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig3.savefig("paper_figures/four_leaf_clover_length_landscapes3.pdf", dpi=300, transparent=True)

# %% Four leaf clover measurement profiles
from persforest.cycle_rep_vectorisations import signed_chain_edge_length

profile_bars = forest.longest_bars(5)
profile_color_map = forest._get_color_map(coloring="forest")
profile_min_bar_length = profile_bars[-1].lifespan()
profile_x_range = (
    min(bar.birth for bar in profile_bars) - 0.05,
    max(bar.death for bar in profile_bars) + 0.05,
)

# These global filtration values show the birth and evolution of the four
# dominant cycles without spending panels on the many very short bars.
profile_snapshot_times = np.array([0.33,0.38, 0.44, 0.53, 0.75, 1.00, 1.30])

fig_profiles = plt.figure(
    figsize=(width, 0.50 * width),
    layout="constrained",
)
profile_grid = fig_profiles.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=[1.05, 3.8],
    height_ratios=[2.0, 1.0],
)
ax_profile_cloud = fig_profiles.add_subplot(profile_grid[0, 0])
profile_plot_grid = profile_grid[0, 1].subgridspec(
    nrows=2,
    ncols=1,
    height_ratios=[0.65, 1.35],
    hspace=0.08,
)
ax_profile_barcode = fig_profiles.add_subplot(profile_plot_grid[0])
ax_profiles = fig_profiles.add_subplot(
    profile_plot_grid[1],
    sharex=ax_profile_barcode,
)
snapshot_grid = profile_grid[1, :].subgridspec(
    nrows=1,
    ncols=len(profile_snapshot_times),
    wspace=0.04,
)
ax_profile_snapshots = [
    fig_profiles.add_subplot(snapshot_grid[i])
    for i in range(len(profile_snapshot_times))
]

# (a) Point cloud.
point_padding = 0.04 * np.ptp(pts, axis=0)
point_x_range = (
    pts[:, 0].min() - point_padding[0],
    pts[:, 0].max() + point_padding[0],
)
point_y_range = (
    pts[:, 1].min() - point_padding[1],
    pts[:, 1].max() + point_padding[1],
)
ax_profile_cloud.scatter(*pts.T, s=1.7, color="black", edgecolors="none")
ax_profile_cloud.set(
    xlim=point_x_range,
    ylim=point_y_range,
    aspect="equal",
    title="Point cloud",
)
ax_profile_cloud.set_axis_off()
ax_profile_cloud.text(
    -0.08,
    1.04,
    r"\textbf{(a)}",
    transform=ax_profile_cloud.transAxes,
    ha="left",
    va="bottom",
)

# (b) Barcode and the corresponding arc-length measurement profiles.
forest.plot_barcode(
    ax=ax_profile_barcode,
    max_bars=len(profile_bars),
    sort="length",
    coloring="forest",
    title=r"$H_1$ barcode",
    xlabel="",
    bar_width=1.5,
    descending=True,
    tight_layout=False,
)
ax_profile_barcode.set_xlim(profile_x_range)
ax_profile_barcode.tick_params(axis="x", labelbottom=False)
style_paper_barcode(ax_profile_barcode)
ax_profile_barcode.text(
    -0.065,
    1.15,
    r"\textbf{(b)}",
    transform=ax_profile_barcode.transAxes,
    ha="left",
    va="bottom",
)

for bar_index, bar in enumerate(profile_bars):
    ax_profile_barcode.text(
        bar.death + 0.012,
        bar_index,
        rf"$I_{{{bar_index + 1}}}$",
        color=profile_color_map[bar],
        fontsize=6,
        ha="left",
        va="center",
        clip_on=False,
    )
    forest.plot_barcode_measurement(
        cycle_func=signed_chain_edge_length,
        signed=False,
        bar=bar,
        ax=ax_profiles,
        x_range=(bar.birth, bar.death),
        title="",
        show_baseline=False,
        show=False,
        color=profile_color_map[bar],
        linewidth=1.3,
        solid_capstyle="butt",
    )

ax_profiles.set(
    xlim=profile_x_range,
    ylim=(0, None),
    xlabel="filtration value",
    ylabel=r"arc length",
)
ax_profiles.spines[["top", "right"]].set_visible(False)
ax_profiles.grid(False)

for time_index, filtration_value in enumerate(profile_snapshot_times, start=1):
    for ax in (ax_profile_barcode, ax_profiles):
        ax.axvline(
            filtration_value,
            color="0.55",
            linewidth=0.6,
            linestyle=(0, (2, 2)),
            zorder=0,
        )
    ax_profiles.text(
        filtration_value,
        1.02,
        rf"$t_{{{time_index}}}$",
        transform=ax_profiles.get_xaxis_transform(),
        color="0.35",
        fontsize=6,
        ha="center",
        va="bottom",
    )

# (c) Cycle representatives at the marked global filtration values.
profile_cycle_style = {
    "complex_face_alpha": 0.10,
    "complex_edge_width": 0.18,
    "cycle_edge_width": 1.25,
}
for time_index, (ax, filtration_value) in enumerate(
    zip(ax_profile_snapshots, profile_snapshot_times),
    start=1,
):
    forest.plot_at_filtration(
        filt_val=filtration_value,
        ax=ax,
        title="",
        show=False,
        vertex_size=0.8,
        coloring="forest",
        min_bar_length=profile_min_bar_length,
        point_zorder=4.5,
        style_2d=profile_cycle_style,
    )
    ax.set(
        xlim=point_x_range,
        ylim=point_y_range,
        aspect="equal",
        title=rf"$t_{{{time_index}}}={filtration_value:.2f}$",
    )
    ax.title.set_fontsize(6)
    ax.set_axis_off()

ax_profile_snapshots[0].text(
    -0.20,
    1.08,
    r"\textbf{(c)}",
    transform=ax_profile_snapshots[0].transAxes,
    ha="left",
    va="bottom",
)

fig_profiles.savefig(
    "paper_figures/four_leaf_clover_measurement_profiles.pdf",
    dpi=300,
    transparent=True,
)
plt.show()


#%% Non-circularity landscapes
x_max = 31
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

from persforest.cycle_rep_vectorisations import signed_chain_circularity_complement, constant_one_functional


forest_6_holes.compute_measurement_landscapes(cycle_func=constant_one_functional,
                                                    label = "standard", 
                                                    signed=False,
                                                    max_k=7,
                                                    x_grid= np.linspace(0,x_max,500))
forest_6_holes.plot_measurement_landscapes(ax=axes[1], label="standard", show = False,linewidth=1.3)
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
                                                    x_grid= np.linspace(0,x_max,500))
forest_6_holes.plot_measurement_landscapes(ax=axes[2], label="circularity_complement", show = False, linewidth=1.3)
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

fig.savefig(f"paper_figures/circle_6holes_random_points_non-circularity-landscapes_seed{seed}.pdf",dpi=300, transparent=True)
plt.show()


# %%
#cycle rep showcase
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

# %%
