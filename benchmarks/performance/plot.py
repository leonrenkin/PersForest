"""Summarize and plot completed benchmark records without running computations.

Usage from the repository root::

    .venv/bin/python benchmarks/performance/plot.py RESULTS_DIR [--output DIR]

Writes ``summary.csv`` (medians with min/max over repeats), prints a compact
runtime table, and saves three figures (PNG and PDF):

``runtime``     diff-mode persistence forest + barcode (solid) and boundary
                landscape (dashed) time against input size, log-log.
``breakdown``   diff-mode stage times at the largest size with landscapes.
``modes``       diffs vs cycles: time to barcode + landscape and peak memory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from run import load_results

PANELS = [("points", 2), ("points", 3), ("cubical", 2), ("cubical", 3)]
SAMPLER_COLORS = {"uniform": "#4477AA", "noisy_sphere": "#EE6677", "single_hole": "#EE6677",
                  "holes": "#228833"}
SIZE_LABEL = {"points": "number of points", "cubical": "number of top cells"}
STAGES = [("time_filtration_s", "filtration preparation"),
          ("time_forest_s", "forest sweep + reduction"),
          ("time_barcode_with_interior_s", "barcode"),
          ("time_landscape_s", "landscape")]
METRICS = ("time_filtration_s", "time_forest_s", "time_barcode_with_interior_s",
           "time_persforest_s", "time_landscape_s", "time_persforest_and_landscape_s",
           "peak_rss_to_barcode_bytes", "peak_rss_bytes", "n_bars", "n_events")


def _has(row: dict, metric: str) -> bool:
    """Barcode metrics survive a later landscape limit; others need success."""
    if metric not in row:
        return False
    return row["status"] == "ok" or metric in ("time_persforest_s", "time_filtration_s",
                                               "time_forest_s", "time_barcode_with_interior_s",
                                               "n_bars", "n_events")


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate repeats by input, mode and landscape setting.

    Parameters
    ----------
    rows : list of dict
        Latest result records from one manifest.

    Returns
    -------
    list of dict
        Median, minimum and maximum of each metric over repeats, plus counts
        and the statuses of incomplete attempts.
    """
    groups = {}
    fields = ("kind", "dim", "sampler", "size", "n_input", "mode", "landscape")
    for row in rows:
        groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    output = []
    for key in sorted(groups, key=lambda key: tuple(map(str, key[:3])) + key[3:5] + (key[5],)):
        group = groups[key]
        result = dict(zip(fields, key))
        result.update(n_attempts=len(group), n_ok=sum(row["status"] == "ok" for row in group),
                      statuses=";".join(sorted({row["status"] for row in group})))
        for metric in METRICS:
            values = [row[metric] for row in group if _has(row, metric)]
            if values:
                result[metric + "_median"] = statistics.median(values)
                result[metric + "_min"] = min(values)
                result[metric + "_max"] = max(values)
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _series(summary, kind, dim, sampler, mode, metric):
    points = sorted((row["n_input"], row[metric + "_median"], row[metric + "_min"], row[metric + "_max"])
                    for row in summary
                    if (row["kind"], row["dim"], row["sampler"], row["mode"]) == (kind, dim, sampler, mode)
                    and metric + "_median" in row)
    return [list(column) for column in zip(*points)] if points else None


def _samplers(summary, kind):
    order = ["uniform", "noisy_sphere", "single_hole", "holes"]
    present = {row["sampler"] for row in summary if row["kind"] == kind}
    return [sampler for sampler in order if sampler in present]


def print_table(summary: list[dict]) -> None:
    """Print diff-mode medians: persforest, landscape, peak memory."""
    print(f"{'input':<26}{'size':>10}{'persforest s':>14}{'landscape s':>13}{'peak GiB':>10}{'bars':>10}  status")
    for row in summary:
        if row["mode"] != "diffs":
            continue
        label = f"{row['kind']} {row['dim']}D {row['sampler']}"
        def cell(metric, scale=1.0, width=13):
            value = row.get(metric + "_median")
            return f"{value / scale:>{width}.2f}" if value is not None else f"{'-':>{width}}"
        print(f"{label:<26}{row['n_input']:>10}{cell('time_persforest_s', width=14)}"
              f"{cell('time_landscape_s')}{cell('peak_rss_bytes', 1024 ** 3, 10)}"
              f"{int(row.get('n_bars_median', 0)):>10}  {row['statuses']}")


def make_plots(summary: list[dict], output: Path, title: str) -> list[Path]:
    """Write the three summary figures; no benchmark is launched.

    Parameters
    ----------
    summary : list of dict
        Output of ``summarize``.
    output : Path
        Destination directory for PNG and PDF files.
    title : str
        Figure title prefix, e.g. the results directory name.

    Returns
    -------
    list of Path
        Generated figure paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    panels = [panel for panel in PANELS if any((row["kind"], row["dim"]) == panel for row in summary)]

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            artifacts.append(path)
        plt.close(fig)

    def panel_axes(rows=1):
        fig, axes = plt.subplots(rows, len(panels), figsize=(4.2 * len(panels), 3.6 * rows),
                                 squeeze=False)
        return fig, axes

    # 1. Diff-mode runtime against size.
    fig, axes = panel_axes()
    for ax, (kind, dim) in zip(axes[0], panels):
        for sampler in _samplers(summary, kind):
            color = SAMPLER_COLORS[sampler]
            for metric, style in (("time_persforest_s", "-"), ("time_landscape_s", "--")):
                series = _series(summary, kind, dim, sampler, "diffs", metric)
                if series:
                    x, median, low, high = series
                    ax.plot(x, median, style, marker="o", ms=3, color=color)
                    ax.fill_between(x, low, high, color=color, alpha=0.15, lw=0)
        _slope_guide(ax)
        ax.set(xscale="log", yscale="log", xlabel=SIZE_LABEL[kind], ylabel="time [s]",
               title=f"{kind}, {dim}D")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(handles=[Line2D([], [], color=SAMPLER_COLORS[s], label=s) for s in _samplers(summary, kind)]
                  + [Line2D([], [], color="k", ls="-", label="forest + barcode"),
                     Line2D([], [], color="k", ls="--", label="landscape"),
                     Line2D([], [], color="0.6", ls=":", label="slope 1")], fontsize=7)
    fig.suptitle(f"{title}: storage='diffs' (median, band = min–max over repeats)")
    save(fig, "runtime")

    # 2. Diff-mode stage breakdown at the largest size with a landscape.
    fig, axes = panel_axes()
    for ax, (kind, dim) in zip(axes[0], panels):
        rows = [row for row in summary if (row["kind"], row["dim"], row["mode"]) == (kind, dim, "diffs")
                and row["landscape"] and "time_landscape_s_median" in row]
        if not rows:
            ax.set_visible(False)
            continue
        size = max(row["n_input"] for row in rows)
        rows = sorted((row for row in rows if row["n_input"] == size),
                      key=lambda row: _samplers(summary, kind).index(row["sampler"]))
        labels = [row["sampler"] for row in rows]
        left = [0.0] * len(rows)
        for (metric, label), color in zip(STAGES, ["0.75", "#4477AA", "#66CCEE", "#CCBB44"]):
            values = [row.get(metric + "_median", 0.0) for row in rows]
            ax.barh(labels, values, left=left, color=color, label=label)
            left = [a + b for a, b in zip(left, values)]
        ax.invert_yaxis()
        ax.set(xlabel="time [s]", title=f"{kind}, {dim}D, n = {size:,}")
        ax.legend(fontsize=7)
    fig.suptitle(f"{title}: where the time goes (storage='diffs', medians)")
    save(fig, "breakdown")

    # 3. Diffs vs cycles at sizes where both ran.
    if any(row["mode"] == "cycles" for row in summary):
        fig, axes = panel_axes(rows=2)
        for column, (kind, dim) in enumerate(panels):
            for row_index, (metric, ylabel, scale) in enumerate(
                    (("time_persforest_and_landscape_s", "forest + barcode + landscape [s]", 1.0),
                     ("peak_rss_bytes", "peak memory [GiB]", 1024 ** 3))):
                ax = axes[row_index][column]
                for sampler in _samplers(summary, kind):
                    cycles = _series(summary, kind, dim, sampler, "cycles", metric)
                    if not cycles:
                        continue
                    diffs = _series(summary, kind, dim, sampler, "diffs", metric)
                    for series, style in ((diffs, "-"), (cycles, "--")):
                        x, median = series[0], [value / scale for value in series[1]]
                        keep = [i for i, value in enumerate(x) if value <= max(cycles[0])]
                        ax.plot([x[i] for i in keep], [median[i] for i in keep], style, marker="o",
                                ms=3, color=SAMPLER_COLORS[sampler])
                ax.set(xscale="log", yscale="log", xlabel=SIZE_LABEL[kind], ylabel=ylabel)
                ax.grid(True, which="major", alpha=0.3)
                if row_index == 0:
                    ax.set_title(f"{kind}, {dim}D")
                    ax.legend(handles=[Line2D([], [], color=SAMPLER_COLORS[s], label=s)
                                       for s in _samplers(summary, kind)]
                              + [Line2D([], [], color="k", ls="-", label="diffs"),
                                 Line2D([], [], color="k", ls="--", label="cycles")], fontsize=7)
        fig.suptitle(f"{title}: storage modes (medians; landscapes only up to their size cap)")
        save(fig, "modes")
    return artifacts


def _slope_guide(ax) -> None:
    """Draw a faint linear-scaling reference through the panel's lowest-left data."""
    lines = [line for line in ax.get_lines() if len(line.get_xdata())]
    if not lines:
        return
    x0 = min(min(line.get_xdata()) for line in lines)
    x1 = max(max(line.get_xdata()) for line in lines)
    y0 = min(line.get_ydata()[0] for line in lines)
    ax.plot([x0, x1], [y0, y0 * x1 / x0], ":", color="0.6", zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="Directory containing results.jsonl")
    parser.add_argument("--output", type=Path, help="Default: RESULTS/plots")
    args = parser.parse_args()
    rows = load_results(args.results / "results.jsonl")
    if not rows:
        parser.error("No results found")
    output = args.output or args.results / "plots"
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    _write_csv(output / "summary.csv", summary)
    print_table(summary)
    for path in make_plots(summary, output, args.results.name):
        print(path)


if __name__ == "__main__":
    main()
