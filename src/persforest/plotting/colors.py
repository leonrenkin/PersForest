def build_color_map_forest(forest, seed: Optional[int] = 39, start_color: Optional[str] = "#ff7f0e",):
    """
    Computes a color map which assign a color to each bar in the barcode. 
    Bars in same tree will have similiar colors. 
    Saved as a dictionary {bar: "#RRGGBB"} in forest.color_map_forest
    Based on json

    Parameters
    ----------
    seed : int | None
        Random seed to make the palette repeatable.
    start_color : str | None
        Preferred color to start the palette with (hex).
    """

    from .color_scheme import color_map_for_bars

    ordered_bars = sorted(list(forest.barcode), key= lambda bar: bar.lifespan(), reverse=True)

    return color_map_for_bars(
        ordered_bars,
        seed =seed,
        by_id=False,
        prefer_start=start_color
    )

def build_color_map_bars(forest):
    """
    Computes a color map which assign a color to each bar in the barcode. 
    Ignores tree structure and cycles through 20 colors from longest to shortest bar 
    Saved as a dictionary {bar: "#RRGGBB"} in forest.color_map_bars
    """
    bars_sorted = sorted(list(forest.barcode), key = lambda bar:bar.lifespan(), reverse=True )
    colors = sns.color_palette("tab20", len(bars_sorted))
    
    return {bars_sorted[i]: colors[i] for i in range(len(bars_sorted))}

def _get_color_map(forest, coloring: Literal["forest", "bars"] = "forest"):
    """Return the requested bar color map, building it on first use."""
    if coloring == "forest":
        if not hasattr(forest, "color_map_forest"):
            forest.color_map_forest = build_color_map_forest(forest=forest)
        return forest.color_map_forest
    elif coloring == "bars":
        if not hasattr(forest, "color_map_bars"):
            forest.color_map_bars = build_color_map_bars(forest=forest)
            forest.forest.set_longest_bar_colors(coloring = "bars", colors = [   "#1f78b4",   "#e31a1c",  "#33a02c",  "#ff7f00",  "#6a3d9a",  "#e7298a",  "#17becf",   "#bcbd22",   "#b15928",  "#fdb462",  "#7570b3", "#66a61e"])
        return forest.color_map_bars
    else:
        raise ValueError("coloring must be 'forest' or 'bars'")

def set_longest_bar_colors(
    forest,
    colors: Sequence[Any],
    coloring: Literal["forest", "bars"] = "forest",
) -> None:
    """Overwrite colors for the longest bars in the selected color map."""
    from .color_scheme import _to_hex

    color_map = forest._get_color_map(coloring=coloring)
    bars_sorted = sorted(list(forest.barcode), key=lambda bar: bar.lifespan(), reverse=True)

    for bar, color in zip(bars_sorted, colors):
        color_map[bar] = _to_hex(color)

    return
