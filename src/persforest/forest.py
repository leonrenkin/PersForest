

#Should I rename class to PersForest?
#
# Set up the following infrastructure for computing the persistence forest
#forest = PersistenceForest.from_points(points) #computes alpha filtration
#forest = PersistenceForest.from_filtration(filtration) 
#forest = PersistenceForest.from_cubical_grid(grid_vales, corner_convention = "8")
#      
class PersistenceForest:
    """
    Compute and store cycle progressions in forest format.

    For an ambient ``dim``-dimensional point cloud, the forest tracks
    codimension-one cycle representatives and their barcode. The default
    filtration is the alpha complex; an ordered filtration can also be supplied.
    """

    def __init__(self, 
                 point_cloud,
                 reduce: bool = True,
                 compute_barcode: bool = True,
                 print_info: bool = False,
                 keep_simplex_diff: bool = False,
                 compute_interior: bool = False,
                 diff_only_mode: bool = False,
                 filtration_tol: float = 1e-12,
                 filtration: Iterable[tuple[Sequence[int], float]] | None = None,
                 ):
        """
        Build a PersistenceForest from a point cloud and a filtration.

        Parameters
        ----------
        point_cloud : array-like, shape (n_points, dim)
            Coordinates of the input point set.
        reduce : bool
            If True, collapse parent-child pairs whose filtration values differ
            by at most ``filtration_tol``, then split branching roots into
            separate trees.
        compute_barcode : bool
            If True, compute and store the barcode after building the forest.
        print_info : bool
            If True, print timing information during construction.
        keep_simplex_diff : bool
            If True, store simplex additions/removals on nodes so cycle
            representatives and interiors can be reconstructed from diffs.
        compute_interior : bool
            If True, reconstruct cycle representatives with their interior
            simplices. Requires ``keep_simplex_diff=True``.
        diff_only_mode : bool
            If True, represent cycle progressions using stored simplex
            additions and removals. This enables ``keep_simplex_diff``
            automatically and is incompatible with ``compute_interior=True``.
        filtration_tol : float
            Absolute numerical resolution for parent-child reduction. 
            Differences at or below this tolerance are ignored and short bars may disappear.
        filtration : iterable of (sequence of int, float), optional
            Ordered simplex-filtration pairs. 
            If None, construct the alpha filtration from ``point_cloud``. 
            Geometric embeddedness and contractibility of the terminal complex are required by the algorithm 
            but are not verified for a supplied filtration.
        """
        self.point_cloud = validate_point_cloud(point_cloud)
        self.filtration_tol = float(filtration_tol)
        if self.filtration_tol < 0 or not math.isfinite(self.filtration_tol):
            raise ValueError("filtration_tol must be a finite non-negative number")

        #check if point cloud has correct shape
        self.dim = self.point_cloud.shape[1]
        if print_info:
            print(f'dimension = {self.dim}')

        self._node_id = itertools.count(1)         #used to assign unique id to each node in the forest
        self._cycle_id = itertools.count(1)
        self.nodes: Dict[int, PFNode] = {}
        self.cycles: Dict[int, SignedChain] = {}

        self._active_node_ids: set[int] = set()               #used for bookkeeping of active nodes in forest computation algorithm
        self.roots: set[int]  = set()                 #List of roots of trees in the forest

        self.levels: list[float] = []               #Critical filtration values, might give duplicates in current implementation

        if filtration is None:
            start = time.perf_counter()
            from filtrations.alpha import alpha_filtration
            self.Filtration = alpha_filtration(point_cloud=point_cloud)
            self.Filtration.remove_low_dim_simplices(dim=self.dim)
            alpha_complex_time = time.perf_counter()-start
            if print_info:
                print(f"Alpha filtration generated in {alpha_complex_time}")
        else:
            start = time.perf_counter()
            from filtrations.simplicial import SimplicialFiltration
            filtration = _validate_filtration(filtration, len(self.point_cloud))
            self.Filtration = SimplicialFiltration(filtration_list=filtration)
            self.Filtration.remove_low_dim_simplices(dim=self.dim)
            filtration_verification_time = time.perf_counter()-start
            if print_info:
                print(f"Filtration order verified in {filtration}")
            warnings.warn(
                "The persistence-forest algorithm requires an embedded and contractible terminal complex;"
                "these conditions are not verified for a manual filtration.",
                UserWarning,
                stacklevel=2,
            )

        self.barcode: set[PFBar] = set()
        self._barcode_computed = False

        self.landscape_families: Dict[str, Any] = {}
        self.barcode_functionals: Dict[str, Any] = {}

        if diff_only_mode and compute_interior:
            raise ValueError("Cannot set diff_only_mode=True and compute_interior=True at the same time.")
        if diff_only_mode and not keep_simplex_diff:
            print("Setting keep_simplex_diff=True since diff_only_mode=True ")
            keep_simplex_diff=True
        if compute_interior and not keep_simplex_diff:
            print("Setting keep_simplex_diff=True since compute_interior=True ")
            keep_simplex_diff=True
        self.keep_simplex_diff = keep_simplex_diff
        self.diff_only_mode = diff_only_mode
        self.compute_interior = compute_interior

        self._build_forest(print_info = print_info)

        self.reduced = reduce

        if reduce:
            from _core.reduction import reduce_forest
            reduce_forest(forest=self,print_info = print_info)
   
        # Compute where each cycle is active.
        if not self.diff_only_mode:
            from _core.representatives import compute_loop_activity
            compute_loop_activity(forest=self)

        if compute_barcode:
            self.compute_barcode(print_info=print_info)

        if compute_interior: #we should probably remove that feature, the interior can be very quickly collected from the barcode interior diff sequence
            from _core.representatives import add_interior_to_barcode
            add_interior_to_barcode(forest=self,print_info = print_info)


        return

    def _build_forest(self, print_info: bool = False):
        """ 
        Compute the persistence forest from the alpha-complex filtration.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.
        """

        loop_forest_start = time.perf_counter()

        from _core.build import build_forest
        build_forest(forest=self)

        loop_forest_time = time.perf_counter() - loop_forest_start
        if print_info:
            print(f"Forest succesfully computed in {loop_forest_time} sec")

        return



    def compute_barcode(self,  print_info: bool = False):

        if self.keep_simplex_diff:
            from _core.compute_barcode import compute_barcode_diff
            compute_barcode_diff(forest=self, print_info=print_info)
        else:
            from _core.compute_barcode import compute_barcode_cycles
            compute_barcode_cycles(forest=self, print_info=print_info)

        return

    def iter_bar_cycle_reps(self, bar: PFBar) -> Iterator[SignedChain]:
        """Yield iterator of representatives of a bar in descending filtration order.

        Normal mode yields stored representatives; 
        diff-only mode reconstructs boundaries and interiors with independent simplex sets.

        Raises
        ------
        ValueError
            On iteration, if the bar is foreign or required diffs are missing.
        """
        from _core.representatives import iter_bar_cycle_reps
        return iter_bar_cycle_reps(forest=self,bar=bar)

    def cycle_for_bar_at(self, bar: PFBar, filt_val: float) -> SignedChain:
        """Return a bar's representative active at a filtration value.

        Stored representatives are used in normal mode. In diff-only mode,
        replay the bar's simplex diffs from death toward birth and return the
        first representative active on ``[active_start, active_end)``.

        Parameters
        ----------
        bar : PFBar
            Bar belonging to this forest.
        filt_val : float
            Filtration value in ``[bar.birth, bar.death)``.

        Returns
        -------
        SignedChain
            Representative active at ``filt_val``.
        """
        from _core.representatives import cycle_for_bar_at
        return cycle_for_bar_at(forest=self, bar=bar, filt_val=filt_val)



    # ----- Useful barcode functions --------

    def max_bar(self)-> PFBar:
        """Return the bar with the longest lifespan."""
        return max(self.barcode, key=lambda bar: bar.lifespan())
    
    def longest_bars(self,k:int) -> List[PFBar]:
        """Return the ``k`` longest bars in descending lifespan order."""
        return sorted(self.barcode, key=lambda bar: bar.lifespan(), reverse=True)[0:k]
    
    def active_bars_at(self, filt_val:float):
        """Return bars whose lifespan contains ``filt_val``."""
        return [bar for bar in self.barcode if (bar.birth<=filt_val and bar.death>filt_val)]

    # ------ extract cycle representatives ---------

    def cycle_reps_at(self, filt_val: float, min_bar_length:float = 0) -> List[SignedChain]:
        """Return cycle representatives active at ``filt_val``."""
        active_bars = self.active_bars_at(filt_val=filt_val)
        cycles = [self.cycle_for_bar_at(bar, filt_val) for bar in active_bars if bar.lifespan()>=min_bar_length]
        return cycles

    def barcode_cycle_reps(self, relative_position=0.1, min_bar_length: float = 0) -> List[SignedChain]:
        """
        Return the list of cycle representatives for all bars in the barcode.

        Parameters
        ----------
        relative_position : float
            Relative position in the barcode (between 0 and 1).
        min_bar_length : float
            Minimum lifespan of bars to consider.

        Returns
        -------
        list[SignedChain]
            Cycle representatives for all bars in the barcode.
        """
        from _core.representatives import barcode_cycle_reps
        return barcode_cycle_reps(forest=forest, relative_position=relative_position, min_bar_length=min_bar_length)

    # ------ generate color scheme  ---------

    def _get_color_map(self, coloring: Literal["forest", "bars"] = "forest"):
        """Return the requested bar color map, building it on first use."""
        from plotting.colors import _get_color_map
        return _get_color_map(forest=self, coloring=coloring)

    def set_longest_bar_colors(
        self,
        colors: Sequence[Any],
        coloring: Literal["forest", "bars"] = "forest",
    ) -> None:
        """Overwrite colors for the longest bars in the selected color map."""
        from plotting.colors import set_longest_bar_colors
        set_longest_bar_colors(forest=self,colors=colors, coloring=coloring)
        return

    # ----- interior activity plotting -------

    def plot_interior_simplex_activity(
        self,
        ax=None,
        show: bool = True,
        figsize: tuple[float, float] = (5, 5),
        dpi: int = 300,
        coloring: Literal["forest", "bars"] = "forest",
        show_complex: bool = False,
        complex_max_filtration: float | None = None,
        overlap: Literal["longest", "layer"] = "longest",
        vertex_size: float = 2,
        min_activity_length: float = 0.0,
        style: dict[str, Any] | None = None,
        geometry: Literal["cells", "surface"] = "cells",
    ):
        """Plot 2D or 3D interior-simplex activity using Matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to draw on. For a 3D forest, an explicitly supplied axes must
            use ``projection="3d"``.
        show : bool
            If True, call ``matplotlib.pyplot.show`` after plotting.
        figsize : tuple[float, float]
            Figure size used when creating an axes.
        dpi : int
            Figure resolution used when creating an axes.
        coloring : {"forest", "bars"}
            Bar color map to use.
        show_complex : bool
            If True, overlay complex edges.
        complex_max_filtration : float or None
            Maximum filtration value included in the optional complex overlay.
        overlap : {"longest", "layer"}
            Whether a multiply active simplex uses its longest interval or is
            drawn once for every activity interval.
        vertex_size : float
            Point-cloud marker size.
        min_activity_length : float
            Ignore activity intervals shorter than this value.
        style : dict or None
            Renderer-specific style overrides.
        geometry : {"cells", "surface"}
            For 3D only, draw tetrahedral cells or per-bar region boundaries.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the activity plot.
        """
        from .interior_activity_plotting import plot_interior_simplex_activity

        return plot_interior_simplex_activity(
            self,
            ax=ax,
            show=show,
            figsize=figsize,
            dpi=dpi,
            coloring=coloring,
            show_complex=show_complex,
            complex_max_filtration=complex_max_filtration,
            overlap=overlap,
            vertex_size=vertex_size,
            min_activity_length=min_activity_length,
            style=style,
            geometry=geometry,
        )

    def plot_interior_simplex_activity_gradient(
        self,
        ax=None,
        show: bool = True,
        figsize: tuple[float, float] = (5, 5),
        dpi: int = 300,
        coloring: Literal["forest", "bars"] = "forest",
        vertex_size: float = 2,
        title: str | None = None,
        min_activity_length: float = 0.0,
        style: dict[str, Any] | None = None,
    ):
        """Plot the smooth 2D interior-activity field.

        This representation is defined only for 2D forests. Parameters are
        forwarded to ``interior_activity_plotting`` without modification.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the activity gradient.
        """
        from .interior_activity_plotting import (
            plot_interior_simplex_activity_gradient,
        )

        return plot_interior_simplex_activity_gradient(
            self,
            ax=ax,
            show=show,
            figsize=figsize,
            dpi=dpi,
            coloring=coloring,
            vertex_size=vertex_size,
            title=title,
            min_activity_length=min_activity_length,
            style=style,
        )

    def plot_interior_simplex_activity_plotly(
        self,
        coloring: Literal["forest", "bars"] = "forest",
        show_complex: bool = False,
        complex_max_filtration: float | None = None,
        overlap: Literal["longest", "layer"] = "longest",
        vertex_size: float = 2.0,
        min_activity_length: float = 0.0,
        geometry: Literal["cells", "surface"] = "surface",
        cell_shrink: float = 1.0,
        opacity: float | None = None,
        show: bool = True,
        renderer: str | None = None,
        width: int | None = None,
        height: int | None = None,
        style: dict[str, Any] | None = None,
        edge_mode: Literal["none", "surface", "all"] = "none",
    ):
        """Interactively plot 3D interior activity using Plotly.

        The default clean view draws the boundary of each per-bar activity
        region with no explicit mesh edges. Face opacity scales with activity
        duration over ``(0.35, 0.95)`` by default. Activity meshes are grouped
        by persistence bar for legend toggling and include simplex,
        activity-length and barcode data in hover labels. Plotly is imported
        only when this method is called.

        Parameters
        ----------
        coloring : {"forest", "bars"}
            Bar color map to use.
        show_complex : bool
            If True, overlay a subdued wireframe of the filtered complex.
        complex_max_filtration : float or None
            Maximum filtration value included in the complex wireframe.
        overlap : {"longest", "layer"}
            Whether a multiply active tetrahedron uses its longest interval or
            is included once for every activity interval.
        vertex_size : float
            Plotly point-cloud marker size.
        min_activity_length : float
            Ignore activity intervals shorter than this value.
        geometry : {"surface", "cells"}
            ``"surface"`` cancels shared faces within each bar and is the clean
            presentation default. ``"cells"`` draws every activity
            tetrahedron for structural inspection.
        cell_shrink : float
            Cell-mode shrink factor in ``(0, 1]``. Keep the default ``1`` to
            avoid artificial gaps; smaller values produce an exploded view.
        opacity : float or None
            Optional fixed face opacity. The default ``None`` scales opacity
            linearly with activity duration, relative to the maximum displayed
            duration, through ``style["activity_alpha_range"]``. The default
            range is ``(0.35, 0.95)``.
        show : bool
            If True, display the figure.
        renderer : str or None
            Plotly renderer passed to ``Figure.show``.
        width, height : int or None
            Figure dimensions in pixels.
        style : dict or None
            Plotly style overrides. The most important are
            ``activity_alpha_range``, ``flatshading``, ``lighting``,
            ``lightposition``, ``camera_eye`` and ``remove_axes``.
            ``show_activity_edges`` remains as a compatibility alias; prefer
            ``edge_mode`` in new code.
        edge_mode : {"none", "surface", "all"}
            Activity-edge policy. ``"none"`` is the clean default.
            ``"surface"`` outlines only faces exposed on each per-bar region
            boundary and is the recommended opt-in mode. ``"all"`` outlines
            every tetrahedral cell, including internal and rear-facing edges.
            Surface edges use a darkened bar color unless
            ``style["activity_edge_color"]`` overrides it.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive 3D figure.
        """
        from .interior_activity_plotly import (
            plot_interior_simplex_activity_plotly,
        )

        return plot_interior_simplex_activity_plotly(
            self,
            coloring=coloring,
            show_complex=show_complex,
            complex_max_filtration=complex_max_filtration,
            overlap=overlap,
            vertex_size=vertex_size,
            min_activity_length=min_activity_length,
            geometry=geometry,
            cell_shrink=cell_shrink,
            opacity=opacity,
            show=show,
            renderer=renderer,
            width=width,
            height=height,
            style=style,
            edge_mode=edge_mode,
        )
    
    # ----- filtration plotting helpers -------
    
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
    

    # ----- filtration plotting tools -------

    def plot_at_filtration(
        self,
        filt_val: float,
        ax=None,
        signed: bool = False,
        show_cycles: bool = True,
        show: bool = True,
        show_complex: bool = True,
        figsize: tuple[float, float] = (5, 5), 
        vertex_size: float = 7,
        coloring: Literal['forest','bars'] = "forest",
        title: Optional[str] = None,
        min_bar_length: float = 0,
        point_zorder: float = 6,
        cycle_zorder: float = 5,
        dpi: int = 100,
        style_2d: Optional[dict[str, Any]] = None,
        style_3d: Optional[dict[str, Any]] = None,
    ):
        """
        Plot the simplicial filtration at a fixed filtration value.

        Parameters
        ----------
        filt_val : float
            Filtration threshold.
        ax : matplotlib.axes.Axes or None
            Axes to draw on; if None, a new figure+axes are created.
        signed : bool
            Orientation policy for cycle chains in both 2D and 3D:
            - False: cancel opposite-oriented duplicates.
            - True: preserve orientation duplicates.
        show_cycles : bool
            If True, overlay the active cycles at the filtration value.
        show : bool
            If True, calls plt.show() when done.
        show_complex : bool
            Whether to render complex geometry (edges/surfaces).
        figsize : tuple[float, float]
            Figure size used when ``ax`` is None.
        vertex_size : float
            Marker size for point cloud.
        point_zorder : float
            Matplotlib z-order for point markers in 2D plots.
            For example, use ``point_zorder=4, cycle_zorder=6`` to draw
            cycles above points.
        cycle_zorder : float
            Matplotlib z-order for cycle edges in 2D plots.
        coloring : {"forest","bars"}
            Color scheme; builds the map on first use.
        title : str | None
            Title for the axes. Defaults to a filtration summary.
        min_bar_length : float
            Only draw active cycles for bars with lifespan at least this value.
        style_2d : dict | None
            Optional 2D style overrides. Supported keys include:
            ``show_orientation_arrows``,
            ``point_color``, ``point_alpha``, ``complex_face_color``,
            ``complex_face_alpha``, ``complex_edge_color``,
            ``complex_edge_width``, ``complex_edge_alpha``,
            ``cycle_edge_width``,
            ``arrow_linewidth``, ``arrow_scale``.
        style_3d : dict | None
            Optional 3D style overrides. Supported keys include:
            ``camera_eye``, ``remove_axes``,
            ``point_color``, ``point_alpha``, ``depthshade_points``,
            ``complex_color``, ``complex_face_alpha``, ``cycle_face_alpha``,
            ``complex_edge_color``, ``cycle_edge_color``,
            ``complex_edge_width``, ``cycle_edge_width``,
            ``complex_edge_alpha``, ``cycle_edge_alpha``,
            ``antialiased``, ``zsort``, ``desaturate_complex``.
        dpi : int
            DPI used when a new Matplotlib figure is created.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .simplicial_filtration_plotting import _plot_at_filtration_generic

        return _plot_at_filtration_generic(
            self,
            filt_val=filt_val,
            ax=ax,
            show=show,
            show_complex=show_complex,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            title=title,
            show_cycles=show_cycles,
            signed=signed,
            min_bar_length=min_bar_length,
            point_zorder=point_zorder,
            cycle_zorder=cycle_zorder,
            dpi=dpi,
            style_2d=style_2d,
            style_3d=style_3d,
        )

    def plot_at_filtration_with_dual(
        self,
        filt_val: float,
        ax=None,
        fill_triangles: bool = True,
        figsize: tuple[float, float] = (7, 7),
        vertex_size: float = 12,
        coloring: Literal['forest','bars'] = "forest",
        dual_vertex_size: float = 12,
        show_cycles: bool = True,
        linewidth_filt: float = 0.6,
        linewidth_cycle: float = 1.8,
        linewidth_dual_edge: float = 0.4,
        show_dual: bool = True,
        show: bool = True,
    ):
        """
        Plot primal and dual edges of the alpha complex at a filtration value.

        Parameters
        ----------
        filt_val : float
            Filtration threshold for displaying simplices.
        ax : matplotlib.axes.Axes or None
            Axes to draw on; a new figure is created if None.
        fill_triangles : bool
            If True, fill triangles that are present at or before filt_val.
        figsize : tuple[float, float]
            Size of the figure if ax is None.
        vertex_size : float
            Marker size for input points.
        coloring : {"forest","bars"}
            Color scheme to use for active cycles.
        dual_vertex_size : float
            Marker size for dual vertices.
        show_cycles : bool
            If True, overlay the active cycles at the filtration value.
        linewidth_filt : float
            Line width for primal edges.
        linewidth_cycle : float
            Line width for cycle edges.
        linewidth_dual_edge : float
            Line width for dual edges.
        show_dual: bool
            If True, overlay the dual edges and dual vertices.  
        show : bool
            If True, calls plt.show() when done.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .simplicial_filtration_plotting import _plot_at_filtration_with_dual_2d

        return _plot_at_filtration_with_dual_2d(
            self,
            filt_val=filt_val,
            ax=ax,
            fill_triangles=fill_triangles,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            dual_vertex_size=dual_vertex_size,
            show_cycles=show_cycles,
            linewidth_filt=linewidth_filt,
            linewidth_cycle=linewidth_cycle,
            linewidth_dual_edge=linewidth_dual_edge,
            show_dual=show_dual,
            show=show,
        )

    def plot_barcode_cycle_reps(
        self,
        min_bar_length: float = 0,
        relative_position: float = 0.1,
        ax=None,
        show: bool = True,
        figsize: tuple[float, float] = (7, 7), 
        vertex_size: float = 3,
        point_zorder: float = 3,
        coloring: Literal['forest','bars'] = "forest",
        title: Optional[str] = None,
        show_orientation_arrows: bool = False,
        remove_double_edges: bool = False,
        linewidth_cycle: float = 0.8,
        style_2d: Optional[dict[str, Any]] = None,
    ):
        """
        Plot one representative cycle for each sufficiently long barcode bar.

        The representative for each selected bar is sampled at
        ``birth + lifespan * relative_position``. This method is implemented
        only for 2D point clouds.

        Parameters
        ----------
        min_bar_length : float
            Minimum lifespan of bars to consider.
        relative_position : float
            Relative position in each bar, between 0 and 1.
        ax : matplotlib.axes.Axes or None
            Axes to draw on; if None, a new figure+axes are created.
        show : bool
            If True, calls plt.show() when done.
        figsize : tuple[float, float]
            Figure size used when ``ax`` is None.
        vertex_size : float
            Marker size for point cloud.
        point_zorder : float
            Matplotlib z-order for the point cloud markers.
        coloring : {"forest","bars"}
            Color scheme; builds the map on first use.
        title : str | None
            Title for the axes. Defaults to a relative-position summary.
        show_orientation_arrows : bool
            If True, draw small arrows along each cycle edge to indicate
            the orientation of the cycle representatives.
        remove_double_edges : bool
            If True, cancel edges appearing with opposite orientations before
            plotting.
        linewidth_cycle : float
            Line width for cycle edges.
        style_2d : dict | None
            2D style overrides. Supports the same keys as
            ``plot_at_filtration``. For vertex styling, use
            ``point_color`` and ``point_alpha``. ``cycle_edge_width`` is used
            when supplied; otherwise ``linewidth_cycle`` is kept for backward
            compatibility.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .simplicial_filtration_plotting import plot_barcode_cycle_reps

        return plot_barcode_cycle_reps(
            self,
            min_bar_length=min_bar_length,
            relative_position=relative_position,
            ax=ax,
            show=show,
            figsize=figsize,
            vertex_size=vertex_size,
            point_zorder=point_zorder,
            coloring=coloring,
            title=title,
            show_orientation_arrows=show_orientation_arrows,
            remove_double_edges=remove_double_edges,
            linewidth_cycle=linewidth_cycle,
            style_2d=style_2d,
        )

    def plot_at_filtration_plotly(
        self,
        filt_val: float,
        coloring: Literal["forest", "bars"] = "forest",
        show_cycles: bool = True,
        fill_triangles: bool = True,
        signed: bool = False,
        min_bar_length: float = 0.0,
        show_complex: bool = True,
        complex_opacity: float = 0.20,
        cycle_opacity: float = 0.55,
        show: bool = True,
        renderer: Optional[str] = None,
        vertex_size: float = 3.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Plot the complex and active cycles at a single filtration value using Plotly.

        Supports ambient dimensions 2 and 3.

        Parameters
        ----------
        filt_val : float
            Filtration threshold.
        coloring : {"forest", "bars"}
            Which color map to use for active bars.
        show_cycles : bool
            If True, overlay cycle representatives.
        fill_triangles : bool
            For 2D only: if True, fill present triangles.
        signed : bool
            If False, cancel opposite-oriented duplicate edges before rendering
            2D cycle representatives or duplicate triangular faces before
            rendering 3D cycle representatives.
        min_bar_length : float
            Only show bars with lifespan >= min_bar_length.
        show_complex : bool
            For 3D only: whether to show the complex boundary mesh.
        complex_opacity : float
            Opacity of the 3D complex mesh.
        cycle_opacity : float
            Opacity of 3D cycle meshes.
        show : bool
            If True, call fig.show().
        renderer : str | None
            Plotly renderer used by ``fig.show()``.
        vertex_size : float
            Plotly marker size for point-cloud vertices.
        width, height : int | None
            Figure dimensions in pixels, forwarded to Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from .simplicial_filtration_plotly import plot_at_filtration_plotly

        return plot_at_filtration_plotly(
            self,
            filt_val=filt_val,
            coloring=coloring,
            show_cycles=show_cycles,
            fill_triangles=fill_triangles,
            signed=signed,
            min_bar_length=min_bar_length,
            show_complex=show_complex,
            complex_opacity=complex_opacity,
            cycle_opacity=cycle_opacity,
            show=show,
            renderer=renderer,
            vertex_size=vertex_size,
            width=width,
            height=height,
        )

    def plot_filtration_interactive(
        self,
        coloring: Literal["forest", "bars"] = "forest",
        show_cycles: bool = True,
        signed: bool = False,
        filt_max: Optional[float] = None,
        min_bar_length: float = 0.0,
        show_complex: bool = True,
        complex_opacity: float = 0.20,
        cycle_opacity: float = 0.55,
        resolution: int =100,
        show: bool = True,
        renderer: Optional[str] = None,
        fill_triangles: bool = True,
        vertex_size: float = 3.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Create a Plotly slider that moves through filtration values.

        Supports ambient dimensions 2 and 3.

        Parameters
        ----------
        coloring : {"forest", "bars"}
            Which bar color map to use.
        show_cycles : bool
            If True, overlay active cycle representatives.
        signed : bool
            If False, removes simplices which appear with both orientations.
        filt_max : float | None
            Maximum filtration value for the slider. If None, uses
            ``1.03 * max(bar.death for bar in self.barcode)``.
        min_bar_length : float
            Only show bars with lifespan >= min_bar_length.
        show_complex : bool
            For 3D only: whether to show the complex boundary mesh.
        complex_opacity : float
            Opacity of 3D complex mesh.
        cycle_opacity : float
            Opacity of 3D cycle meshes.
        resolution : int
            Number of slider intervals. The figure contains
            ``resolution + 1`` equally spaced frames from 0 to ``filt_max``.
        show : bool
            If True, call fig.show().
        renderer : str | None
            Plotly renderer used by ``fig.show()``.
        fill_triangles : bool
            For 2D only: whether to fill present triangles.
        vertex_size : float
            Plotly marker size for point-cloud vertices.
        width, height : int | None
            Figure dimensions in pixels, forwarded to Plotly layout.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if self.diff_only_mode:
            raise ValueError(
                "Animations are unavailable with diff_only_mode=True because "
                "reconstructing cycles for every frame can be slow. "
                "Build the forest with diff_only_mode=False to animate it."
            )
        from .simplicial_filtration_plotly import plot_filtration_interactive

        return plot_filtration_interactive(
            self,
            coloring=coloring,
            show_cycles=show_cycles,
            signed=signed,
            filt_max=filt_max,
            min_bar_length=min_bar_length,
            show_complex=show_complex,
            complex_opacity=complex_opacity,
            cycle_opacity=cycle_opacity,
            resolution=resolution,
            show=show,
            renderer=renderer,
            fill_triangles=fill_triangles,
            vertex_size=vertex_size,
            width=width,
            height=height,
        )

    # ------ forest plotting tools ------------

    def plot_persistence_forest(self, **kwargs):
        """Plot persistence trees with barcode colors and a dominant trunk.

        Supports ``min_tree_span``, ``min_bar_length``, ``min_branch_span``,
        ``nodes="none"|"critical"|"all"``, ``collapse_degree2``, ``coloring``,
        ``sort="lifespan"|"birth"|"death"``, ``descending=True|False``,
        ``orientation="vertical"|"horizontal"``,
        ``edge_style="curved"|"straight"|"routed"``, ``curvature``,
        ``branch_angle``, ``min_clearance``, ``node_size``, ``node_alpha``,
        and Matplotlib styling. Filters affect only the display. See
        :func:`persforest.forest_plotting._plot_persistence_forest_generic`
        for full parameter descriptions and optional layout metadata.
        """
        from .forest_plotting import _plot_persistence_forest_generic
        return _plot_persistence_forest_generic(self, **kwargs)

    def plot_barcode(self, *args, **kwargs):
        """
        Plot a 1D barcode from self.barcode (a set[Bar]).

        Intervals run from birth to death, horizontally by default. Use
        orientation="vertical" to share a filtration axis with a forest plot.
        Infinite intervals end in an arrow.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            If given, draw on this axes. Otherwise a new figure/axes is created.
        sort : {"length","birth","death",None}
            Sort bars before plotting (None preserves current order).
            Default is "birth".
        title : str | None
            Plot title. Defaults to "$H_d$ Barcode", where d is the ambient
            dimension of the point cloud minus one.
        orientation : {"horizontal", "vertical"}
            Direction of barcode intervals (default: horizontal).
        xlabel : str
            Filtration-axis label, also used vertically unless ylabel is given.
        ylabel : str or None
            Override the filtration-axis label for vertical barcodes.
        coloring : {"forest","bars","none","grey"}
            Which color scheme to use:
            - "forest": use self.color_map_forest (tree-structured colors).
            - "bars":   use self.color_map_bars (ignores tree structure).
            - "none":   use Matplotlib defaults.
            - "grey":   draw all bars in black.
            If the chosen color map does not exist yet, it is built as in
            `plot_at_filtration`.
        max_bars : int
            If > 0, display at most this many bars, keeping the longest ones
            (by lifespan). 0 means show all bars.
        min_bar_length : float
            Filter out bars with lifespan < min_bar_length before plotting.
        bar_width : float
            Line width used to draw each barcode interval.
        descending : bool
            If True, reverse the selected sort order.
        tight_layout : bool
            If True, call ``fig.tight_layout()`` after drawing.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the barcode was drawn on.
        """
        from .forest_plotting import _plot_barcode_generic

        return _plot_barcode_generic(self, *args,**kwargs)

    # --------- animation -------------

    def animate_filtration(
        self,
        filename: Optional[str] = None,
        format: Optional[Literal["mp4", "html"]] = None,
        with_barcode: bool = False,
        fps: int = 20,
        frames: int = 200,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        coloring: Literal["forest", "bars"] = "forest",
        signed: bool = False,
        show_cycles: bool = True,
        show_complex: bool = True,
        dpi: int = 200,
        figsize: Optional[tuple[float, float]] = None,
        pixel_size: Optional[tuple[int, int]] = None,
        panel_width_ratios: tuple[float, float] = (1.0, 1.0),
        panel_spacing: float = 0.08,
        figure_margins: Optional[dict[str, float]] = None,
        filtration_kwargs: Optional[dict] = None,
        barcode_kwargs: Optional[dict] = None,
        camera_mode: Literal["fixed", "orbit"] = "fixed",
        # Deprecated aliases (kept for backward compatibility)
        cloud_figsize: Optional[tuple[float, float]] = None,
        total_figsize: Optional[tuple[float, float]] = None,
        plot_kwargs: Optional[dict] = None,
        alpha_digits: Optional[int] = None,
        radius_box_corner: Literal["upper right", "upper left", "lower right", "lower left"] = "upper right",
    ):
        """
        Animate the filtration with automatic 2D/3D backend dispatch.

        Dispatch rules:
        - 2D: uses the existing matplotlib animation pipeline.
        - 3D + MP4: uses a matplotlib frame renderer and ffmpeg assembly.
        - 3D + HTML: uses the existing Plotly interactive pipeline.

        Parameters
        ----------
        filename : str | None
            Output path. If omitted, a default filename is chosen for exports.
        format : {"mp4", "html"} | None
            Output format. If None, inferred from ``filename``; defaults to
            ``"mp4"`` when not inferable.
        with_barcode : bool
            If True, show a barcode panel with a moving filtration marker.
        fps : int
            Frames per second for MP4 output.
        frames : int
            Number of sampled filtration values.
        t_min, t_max : float | None
            Optional filtration interval.
        coloring : {"forest", "bars"}
            Color map used for cycle overlays and barcode.
        signed : bool
            If False, cancel opposite-oriented duplicate simplices first.
        show_cycles : bool
            Whether to display active cycle representatives.
        show_complex : bool
            Whether to render complex geometry.
        dpi : int
            DPI used for matplotlib frame rendering and video export.
        figsize : tuple[float, float] | None
            Matplotlib figure size in inches.
            Defaults to (6, 6) without barcode and (10, 5) with barcode.
        pixel_size : tuple[int, int] | None
            Figure size in pixels. If provided, overrides ``figsize``.
        panel_width_ratios : tuple[float, float]
            Relative widths of cloud/barcode panels when ``with_barcode=True``.
        panel_spacing : float
            Horizontal spacing between cloud and barcode panels.
        figure_margins : dict[str, float] | None
            Outer figure margins with keys ``left``, ``right``, ``bottom``,
            and ``top`` passed to ``subplots_adjust``.
        filtration_kwargs : dict | None
            Extra kwargs forwarded to ``plot_at_filtration`` (except ``ax``,
            ``show`` and ``filt_val``). For rendering controls, use top-level
            ``show_cycles``/``show_complex``; keep this dict for values like
            ``vertex_size`` and 3D style values such as
            ``style_3d={'camera_eye': ..., 'complex_face_alpha': ..., 'cycle_face_alpha': ...}``.
            Backward-compatible keys are translated before forwarding:
            ``fill_triangles`` to ``show_complex``, ``remove_double_edges`` to
            ``signed``, ``linewidth_filt`` and ``linewidth_cycle`` to 2D style
            widths, ``show_orientation_arrows`` to 2D style, and
            ``camera_eye``/``remove_axes`` to 3D style.
        barcode_kwargs : dict | None
            Extra kwargs forwarded to barcode plotting for matplotlib paths.
            For animations with a barcode panel, the generic helper manages
            ``ax`` and uses the top-level ``coloring`` for both panels.
        camera_mode : {"fixed", "orbit"}
            3D camera mode.
        cloud_figsize, total_figsize : tuple[float, float] | None
            Deprecated aliases for ``figsize``. ``total_figsize`` applies when
            ``with_barcode=True``; ``cloud_figsize`` applies otherwise.
        plot_kwargs : dict | None
            Deprecated alias for ``filtration_kwargs``.
        radius_box_corner : {"upper right", "upper left", "lower right", "lower left"}
            Radius label corner for Matplotlib animations (default: "upper right").
            Uses an 8 pt inset. Does not affect the interactive 3D HTML output.
        alpha_digits : int | None
            Number of digits shown in the filtration value overlay (matplotlib paths).

        Returns
        -------
        object
            - 2D path: ``(anim, fig)`` from matplotlib.
            - 3D MP4 path: output filename string.
            - 3D HTML path: Plotly figure.
        """
        if self.diff_only_mode:
            raise ValueError(
                "Animations are unavailable with diff_only_mode=True because "
                "reconstructing cycles for every frame can be slow. "
                "Build the forest with diff_only_mode=False to animate it."
            )
        from pathlib import Path
        from .simplicial_filtration_animation import (
            _animate_filtration_generic, _validate_radius_box_corner,
        )

        _validate_radius_box_corner(radius_box_corner)

        out_format = format
        if out_format is None and filename is not None:
            suffix = Path(filename).suffix.lower()
            if suffix in {".mp4", ".html"}:
                out_format = suffix[1:]
        if out_format is None:
            out_format = "mp4"
        if out_format not in {"mp4", "html"}:
            raise ValueError(f"Unsupported animation format: {out_format!r}. Use 'mp4' or 'html'.")

        if plot_kwargs is not None:
            warnings.warn(
                "`plot_kwargs` is deprecated; use `filtration_kwargs`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filtration_kwargs is None:
                filtration_kwargs = dict(plot_kwargs)
            else:
                merged = dict(plot_kwargs)
                merged.update(filtration_kwargs)
                filtration_kwargs = merged

        if cloud_figsize is not None or total_figsize is not None:
            warnings.warn(
                "`cloud_figsize` and `total_figsize` are deprecated; use `figsize=(w, h)`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if figsize is None:
                if with_barcode and total_figsize is not None:
                    figsize = total_figsize
                elif (not with_barcode) and cloud_figsize is not None:
                    figsize = cloud_figsize

        base_filtration_kwargs: dict[str, Any] = {}
        if filtration_kwargs is not None:
            base_filtration_kwargs.update(filtration_kwargs)
    
        style_2d = dict(base_filtration_kwargs.pop("style_2d", {}) or {})
        style_3d = dict(base_filtration_kwargs.pop("style_3d", {}) or {})

        if "fill_triangles" in base_filtration_kwargs:
            base_filtration_kwargs.setdefault(
                "show_complex",
                bool(base_filtration_kwargs.pop("fill_triangles")),
            )

        if "remove_double_edges" in base_filtration_kwargs:
            base_filtration_kwargs.setdefault(
                "signed",
                not bool(base_filtration_kwargs.pop("remove_double_edges")),
            )

        if "linewidth_filt" in base_filtration_kwargs:
            style_2d.setdefault(
                "complex_edge_width",
                float(base_filtration_kwargs.pop("linewidth_filt")),
            )
        if "linewidth_cycle" in base_filtration_kwargs:
            style_2d.setdefault(
                "cycle_edge_width",
                float(base_filtration_kwargs.pop("linewidth_cycle")),
            )

        if "show_orientation_arrows" in base_filtration_kwargs:
            style_2d.setdefault(
                "show_orientation_arrows",
                bool(base_filtration_kwargs.pop("show_orientation_arrows")),
            )

        if "camera_eye" in base_filtration_kwargs:
            style_3d.setdefault("camera_eye", base_filtration_kwargs.pop("camera_eye"))
        if "remove_axes" in base_filtration_kwargs:
            style_3d.setdefault(
                "remove_axes",
                bool(base_filtration_kwargs.pop("remove_axes")),
            )

        base_filtration_kwargs["signed"] = bool(signed)
        base_filtration_kwargs["show_cycles"] = bool(show_cycles)
        base_filtration_kwargs["show_complex"] = bool(show_complex)

        if style_2d:
            base_filtration_kwargs["style_2d"] = style_2d
        if style_3d:
            base_filtration_kwargs["style_3d"] = style_3d

        if self.dim == 2:
            if out_format == "html":
                raise ValueError("HTML animation export is only implemented for ambient dimension 3.")

            resolved_filename = filename
            if resolved_filename is not None and Path(resolved_filename).suffix.lower() == "":
                resolved_filename = f"{resolved_filename}.mp4"

            filtration_panel_kwargs = dict(base_filtration_kwargs)
            barcode_panel_kwargs = {}
            if barcode_kwargs is not None:
                barcode_panel_kwargs.update(barcode_kwargs)
            t_max_2d = t_max
            if t_max_2d is None:
                finite_deaths = [float(bar.death) for bar in self.barcode if np.isfinite(float(bar.death))]
                if finite_deaths:
                    t_max_2d = max(finite_deaths)
                else:
                    finite_filtration = [float(f) for _, f in self.filtration if np.isfinite(float(f))]
                    if finite_filtration:
                        t_max_2d = max(finite_filtration)

            return _animate_filtration_generic(
                self,
                filename=resolved_filename,
                fps=fps,
                frames=frames,
                coloring=coloring,
                with_barcode=with_barcode,
                t_min=t_min,
                t_max=t_max_2d,
                dpi=dpi,
                figsize=figsize,
                pixel_size=pixel_size,
                panel_width_ratios=panel_width_ratios,
                panel_spacing=panel_spacing,
                figure_margins=figure_margins,
                filtration_kwargs=filtration_panel_kwargs,
                barcode_kwargs=barcode_panel_kwargs,
                alpha_digits=alpha_digits,
                radius_box_corner=radius_box_corner,
            )

        if self.dim == 3:
            if out_format == "html":
                html_width = None
                html_height = None
                if pixel_size is not None:
                    if len(pixel_size) != 2:
                        raise ValueError("pixel_size must be a 2-tuple (width, height).")
                    html_width = int(pixel_size[0])
                    html_height = int(pixel_size[1])
                elif figsize is not None:
                    html_width = int(round(float(figsize[0]) * float(dpi)))
                    html_height = int(round(float(figsize[1]) * float(dpi)))

                filtration_panel_kwargs = dict(base_filtration_kwargs)
                style_3d = dict(filtration_panel_kwargs.get("style_3d", {}) or {})

                fig = self.plot_filtration_interactive(
                    coloring=coloring,
                    show_cycles=filtration_panel_kwargs.get("show_cycles", True),
                    signed=filtration_panel_kwargs["signed"],
                    filt_max=t_max,
                    show_complex=filtration_panel_kwargs.get("show_complex", True),
                    complex_opacity=float(style_3d.get("complex_face_alpha", 0.20)),
                    cycle_opacity=float(style_3d.get("cycle_face_alpha", 0.55)),
                    resolution=frames,
                    show=False,
                    vertex_size=float(filtration_panel_kwargs.get("vertex_size", 3.0)),
                    width=html_width,
                    height=html_height,
                )
                resolved_filename = filename if filename is not None else "filtration_animation_3d.html"
                if Path(resolved_filename).suffix.lower() == "":
                    resolved_filename = f"{resolved_filename}.html"
                fig.write_html(str(resolved_filename))
                return fig

            resolved_filename = filename if filename is not None else "filtration_animation_3d.mp4"
            if Path(resolved_filename).suffix.lower() == "":
                resolved_filename = f"{resolved_filename}.mp4"

            filtration_panel_kwargs = {"camera_mode": camera_mode, **base_filtration_kwargs}

            barcode_panel_kwargs = {}
            if barcode_kwargs is not None:
                barcode_panel_kwargs.update(barcode_kwargs)

            t_max_3d = t_max
            if t_max_3d is None:
                finite_deaths = [float(bar.death) for bar in self.barcode if np.isfinite(float(bar.death))]
                if finite_deaths:
                    t_max_3d = max(finite_deaths)
                else:
                    finite_filtration = [float(f) for _, f in self.filtration if np.isfinite(float(f))]
                    if finite_filtration:
                        t_max_3d = max(finite_filtration)

            return _animate_filtration_generic(
                self,
                filename=resolved_filename,
                with_barcode=with_barcode,
                fps=fps,
                frames=frames,
                t_min=t_min,
                t_max=t_max_3d,
                coloring=coloring,
                dpi=dpi,
                figsize=figsize,
                pixel_size=pixel_size,
                panel_width_ratios=panel_width_ratios,
                panel_spacing=panel_spacing,
                figure_margins=figure_margins,
                filtration_kwargs=filtration_panel_kwargs,
                barcode_kwargs=barcode_panel_kwargs,
                alpha_digits=alpha_digits,
                radius_box_corner=radius_box_corner,
            )

        raise ValueError("animate_filtration is only implemented for ambient dimensions 2 and 3.")

    def animate_barcode_measurement(
                self,
                cycle_func,
                signed: bool = True,
                bar = None,
                *args,
                **kwargs,
        ):
        """
        Animate the filtration together with a barcode measurement.

        This is the animated analogue of :meth:`plot_barcode_measurement`.
        The left panel shows :meth:`plot_at_filtration` at a moving
        filtration value, while the right panel shows the associated
        barcode measurement with a vertical line indicating the current
        filtration value.

        Parameters
        ----------
        cycle_func :
            Callable ``(chain, point_cloud) -> float`` that assigns a scalar
            to each cycle representative associated to the chosen bar.
            See :meth:`plot_barcode_measurement` for details.
        signed : bool, optional
            If ``True`` (default), ``cycle_func`` is called on the signed
            cycle representatives. If ``False``, the chains are first
            passed through ``chain.unsigned()``.
        bar :
            Bar to use for the measurement. If ``None``, the method uses
            ``self.max_bar()`` inside the generic helper.
        *args, **kwargs :
            Forwarded to
            :func:`forest_landscapes.animate_barcode_measurement_generic`.
            Typical keyword arguments include ``filename``, ``fps``,
            ``frames``, ``t_min``, ``t_max``, ``dpi``, ``total_figsize``,
            ``filtration_kwargs`` and ``measurement_kwargs``.
            ``filtration_kwargs`` is copied and normalized before forwarding:
            ``fill_triangles``, ``remove_double_edges``, ``linewidth_filt``,
            ``linewidth_cycle``, ``show_orientation_arrows``, ``camera_eye``
            and ``remove_axes`` are translated to the current
            ``plot_at_filtration`` style keys. ``plot_kwargs`` is no longer
            accepted.

        Returns
        -------
        anim, fig :
            A pair ``(anim, fig)`` where ``anim`` is a
            :class:`matplotlib.animation.FuncAnimation` and ``fig`` is the
            underlying figure.
        """
        if self.diff_only_mode:
            raise ValueError(
                "Animations are unavailable with diff_only_mode=True because "
                "reconstructing cycles for every frame can be slow. "
                "Build the forest with diff_only_mode=False to animate it."
            )
        from .forest_landscapes import animate_barcode_measurement_generic
        from copy import deepcopy

        kwargs = dict(kwargs)
        if "plot_kwargs" in kwargs:
            raise TypeError(
                "`plot_kwargs` is no longer supported; use `filtration_kwargs`."
            )
        raw_filtration_kwargs = kwargs.get("filtration_kwargs", None)
        if raw_filtration_kwargs is not None:
            if not isinstance(raw_filtration_kwargs, dict):
                raise TypeError("filtration_kwargs must be a dict when provided.")
            filtration_kwargs = deepcopy(raw_filtration_kwargs)
        else:
            filtration_kwargs = {}

        style_2d = dict(filtration_kwargs.pop("style_2d", {}) or {})
        style_3d = dict(filtration_kwargs.pop("style_3d", {}) or {})

        if "fill_triangles" in filtration_kwargs:
            filtration_kwargs["show_complex"] = bool(filtration_kwargs.pop("fill_triangles"))

        if "remove_double_edges" in filtration_kwargs:
            filtration_kwargs["signed"] = not bool(filtration_kwargs.pop("remove_double_edges"))

        if "linewidth_filt" in filtration_kwargs:
            style_2d["complex_edge_width"] = float(filtration_kwargs.pop("linewidth_filt"))
        if "linewidth_cycle" in filtration_kwargs:
            style_2d["cycle_edge_width"] = float(filtration_kwargs.pop("linewidth_cycle"))

        if "show_orientation_arrows" in filtration_kwargs:
            style_2d["show_orientation_arrows"] = bool(
                filtration_kwargs.pop("show_orientation_arrows")
            )

        if "camera_eye" in filtration_kwargs:
            style_3d["camera_eye"] = filtration_kwargs.pop("camera_eye")
        if "remove_axes" in filtration_kwargs:
            style_3d["remove_axes"] = bool(filtration_kwargs.pop("remove_axes"))

        # `alpha_digits` is not a `plot_at_filtration` kwarg in this animation path.
        filtration_kwargs.pop("alpha_digits", None)

        if style_2d:
            filtration_kwargs["style_2d"] = style_2d
        if style_3d:
            filtration_kwargs["style_3d"] = style_3d

        kwargs["filtration_kwargs"] = filtration_kwargs

        if signed:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain, point_cloud))
        else:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain.unsigned(), point_cloud))

        return animate_barcode_measurement_generic(
            forest=self,
            cycle_func=_cycle_value,
            bar=bar,
            *args,
            **kwargs,
        )

    #------- measurement landscape ----------------

    def plot_barcode_measurement(
        self,
        cycle_func,
        signed: bool = False,
        bar = None,
        ax=None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        label: Optional[str] = None,
        show_baseline: bool = True,
        show: bool = False,
        *args,
        **kwargs,
    ):
        """
        Plot a scalar measurement of cycles along each bar.

        Parameters
        ----------
        cycle_func : callable
            Function returning a scalar for a SignedChain and point cloud.
        signed : bool
            If True, evaluate ``cycle_func`` on the stored signed chain. If
            False, evaluate it after cancelling opposing simplices.
        bar : PFBar | None
            If provided, restrict the plot to a single bar.
        ax : matplotlib.axes.Axes | None
            Axes to draw on; if None, a new figure is created.
        x_range, y_range : (float, float) | None
            Manual axis limits for the step plot.
        title : str | None
            Custom plot title.
        label : str | None
            Legend label for the curve.
        show_baseline : bool
            If True, draw the baseline of the step function.
        show : bool
            If True, call ``plt.show()`` after plotting.
        *args :
            Forwarded to ``forest_landscapes.plot_barcode_measurement_generic``.
        **kwargs :
            Forwarded to ``StepFunctionData.plot`` (e.g., ``baseline_kwargs``,
            line color/linewidth, etc.).

        Returns
        -------
        tuple[matplotlib.axes.Axes, StepFunctionData]
            The axes and the piecewise-constant measurement data.
        """
        from .forest_landscapes import plot_barcode_measurement_generic

        if signed:
            def _cycle_func(chain, point_cloud):
                return float(cycle_func(chain, point_cloud))
        else:
            def _cycle_func(chain, point_cloud):
                return float(cycle_func(chain.unsigned(), point_cloud))

        return plot_barcode_measurement_generic(
            forest=self,
            cycle_func=_cycle_func,
            bar=bar,
            ax=ax,
            x_range=x_range,
            y_range=y_range,
            title=title,
            label=label,
            show_baseline=show_baseline,
            show=show,
            *args,
            **kwargs,
        )

    def compute_measurement_landscapes(
        self,
        cycle_func,
        label: str,
        *,
        max_k: int = 5,
        num_grid_points: int = 512,
        min_bar_length: float = 0.0,
        x_grid: Optional[NDArray[np.float64]] = None,
        cache: bool = True,
        signed: bool = False,
        cache_functionals: bool = False,
        functionals_label: Optional[str] = None,
        compute_functionals: bool = True,
    ):
        """
        Compute measurement landscapes for this PersistenceForest.

        Parameters
        ----------
        cycle_func : callable
            A function

                cycle_func(chain, point_cloud) -> float

            where ``chain`` is a SignedChain. Measurements must be finite and
            nonnegative; negative, NaN, and infinite results raise
            ``ValueError``. Examples include total length and mass.
        label : str
            Key used to store the family in ``self.landscape_families`` when
            ``cache=True``.
        max_k : int
            Number of landscape levels λ_1..λ_max_k.
        num_grid_points : int, optional
            Number of x-grid samples (if x_grid is None).
        min_bar_length : float, optional
            Ignore bars with lifespan < min_bar_length.
        x_grid : np.ndarray | None, optional
            Optional common grid to evaluate all landscapes on.
        cache : bool
            If True, store the resulting family in ``self.landscape_families``
            under ``label``.
        signed : bool
            If True, evaluate ``cycle_func`` on stored signed chains. If False,
            evaluate it after cancelling opposing simplices.
        cache_functionals : bool
            If True, store the per-bar ``BarcodeFunctionals`` in
            ``self.barcode_functionals``.
        functionals_label : str | None
            Label used when caching or retrieving per-bar functionals. If None,
            the generic helper uses ``label``.
        compute_functionals : bool
            If True, compute per-bar functionals before building landscapes. If
            False, reuse cached functionals with ``functionals_label``.

        Returns
        -------
        MeasurementLandscapeFamily
            Family of landscapes evaluated for each bar.

        Raises
        ------
        RuntimeError
            If the barcode has not been computed yet.
        """

        if not self._barcode_computed:
            raise RuntimeError("Barcode has not been computed; call compute_barcode() first")
        if signed:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain, point_cloud))
        else:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain.unsigned(), point_cloud))
    
        from .forest_landscapes import compute_measurement_landscape_family

        return compute_measurement_landscape_family(
            self,
            _cycle_value,
            max_k=max_k,
            num_grid_points=num_grid_points,
            mode='pyramid',
            label=label,
            min_bar_length=min_bar_length,
            x_grid=x_grid,
            cache = cache,
            cache_functionals=cache_functionals,
            functionals_label=functionals_label,
            compute_functionals=compute_functionals,
        )

    def compute_generalized_landscape_family(
        self,
        cycle_func,
        label: str,
        *,
        max_k: int = 5,
        num_grid_points: int = 512,
        min_bar_length: float = 0.0,
        x_grid: Optional[NDArray[np.float64]] = None,
        cache: bool = True,
        signed: bool = False,
        cache_functionals: bool = False,
        functionals_label: Optional[str] = None,
        compute_functionals: bool = True,
    ):
        """Legacy compatibility wrapper for compute_measurement_landscapes."""
        return self.compute_measurement_landscapes(
            cycle_func=cycle_func,
            label=label,
            max_k=max_k,
            num_grid_points=num_grid_points,
            min_bar_length=min_bar_length,
            x_grid=x_grid,
            cache=cache,
            signed=signed,
            cache_functionals=cache_functionals,
            functionals_label=functionals_label,
            compute_functionals=compute_functionals,
        )

    def plot_measurement_landscapes(
        self,
        label: str,
        ks: Optional[list[int]] = None,
        ax=None,
        title: Optional[str] = None,
        *args,
        show_legend: Optional[bool] = None,
        linewidth: Optional[float] = None,
        cmap: Optional[Union[str, Colormap]] = None,
        higher_layers_on_top: bool = False,
        **kwargs,
    ):
        """
        Plot previously computed measurement landscapes.

        Parameters
        ----------
        label : str
            Identifier used when the family was computed.
        ks : list[int] | None
            Which landscape levels to plot. Defaults to all available.
        ax : matplotlib.axes.Axes | None
            Axes to draw on; if None, a new figure is created.
        title : str | None
            Custom plot title.
        show : bool
            Forwarded through ``**kwargs`` to the generic plotting helper; if
            True, call ``plt.show()`` after plotting.
        show_legend : bool | None
            Whether to show the legend. Defaults to showing it for fewer than
            10 plotted landscapes and hiding it otherwise.
        linewidth : float | None
            Line width for the landscape plots. If omitted, Matplotlib's
            default line width is used.
        cmap : str or matplotlib.colors.Colormap | None
            Sequential colormap used across the selected landscape levels.
            Colors follow increasing ``k``.
            Named reversed colormaps such as ``"viridis_r"`` are supported.
            If omitted, Matplotlib's default color cycle is used.
        higher_layers_on_top : bool
            If True, landscape levels with larger ``k`` are drawn in front of
            levels with smaller ``k``. The default keeps lower levels,
            including the first landscape, in front.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the landscape plot.
        """
        from .forest_landscapes import plot_landscape_family
        return plot_landscape_family(
            self,
            label,
            ks,
            ax,
            title,
            *args,
            show_legend=show_legend,
            linewidth=linewidth,
            cmap=cmap,
            higher_layers_on_top=higher_layers_on_top,
            **kwargs,
        )

    def plot_landscape_family(
        self,
        label: str,
        ks: Optional[list[int]] = None,
        ax=None,
        title: Optional[str] = None,
        *args,
        show_legend: Optional[bool] = None,
        linewidth: Optional[float] = None,
        cmap: Optional[Union[str, Colormap]] = None,
        higher_layers_on_top: bool = False,
        **kwargs,
    ):
        """Legacy compatibility wrapper for plot_measurement_landscapes."""
        return self.plot_measurement_landscapes(
            label,
            ks,
            ax,
            title,
            *args,
            show_legend=show_legend,
            linewidth=linewidth,
            cmap=cmap,
            higher_layers_on_top=higher_layers_on_top,
            **kwargs,
        )

    def plot_landscape_comparison_between_functionals(
        self,
        labels: list[str],
        k: int = 1,
        ax=None,
        title: Optional[str] = None,
        *args, 
        **kwargs
    ):
        """
        Compare multiple measurement landscape families on the same axes.

        Parameters
        ----------
        labels : list[str]
            Labels of the landscape families to compare.
        k : int
            Which landscape level to compare.
        ax : matplotlib.axes.Axes | None
            Axes to draw on; if None, a new figure is created.
        title : str | None
            Custom plot title.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the comparison plot.
        """
        from .forest_landscapes import plot_landscape_comparison_between_functionals
        return plot_landscape_comparison_between_functionals(self, labels=labels, k=k, ax=ax, title=title, *args, **kwargs)
