"""Plot completed benchmark records without running scientific computations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from run import MODES, load_results

COLORS = dict(zip(MODES, ["#4477AA", "#EE6677", "#228833", "#AA3377"]))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        # A resumed run may resolve every failure; do not leave a stale table.
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate successful repeats while preserving attempted/failed counts.

    Parameters
    ----------
    rows : list of dict
        Latest result records from one manifest.

    Returns
    -------
    list of dict
        Median, minimum and maximum of each metric, and completion counts.
    """
    groups = {}
    fields = ("suite", "dim", "sampler", "n_points", "mode", "functional")
    for row in rows:
        groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    output = []
    for key, group in groups.items():
        successful = [row for row in group if row["status"] == "ok"]
        result = dict(zip(fields, key))
        result.update(n_attempts=len(group), n_ok=len(successful),
                      n_incomplete=len(group) - len(successful))
        metrics = set().union(*(row.keys() for row in successful))
        for metric in metrics:
            if not (metric.startswith(("time_", "peak_rss_")) or "_pct_" in metric):
                continue
            values = [row[metric] for row in successful if metric in row]
            if values:
                result[metric + "_median"] = statistics.median(values)
                result[metric + "_min"] = min(values)
                result[metric + "_max"] = max(values)
        output.append(result)
    return output


def paired_ratios(rows: list[dict]) -> list[dict]:
    """Compare diff-only with standard on the same cloud and functional.

    Parameters
    ----------
    rows : list of dict
        Results from a single manifest; failed pairs are omitted.

    Returns
    -------
    list of dict
        Per-repeat speedup and memory ratios; values above one favor diff-only.
    """
    groups = {}
    for row in rows:
        if row["status"] == "ok":
            key = tuple(row[field] for field in
                        ("dim", "sampler", "n_points", "repeat", "seed", "functional"))
            groups.setdefault(key, {})[row["mode"]] = row
    output = []
    for group in groups.values():
        if "standard" not in group or "diff_only_mode" not in group:
            continue
        standard, diff = group["standard"], group["diff_only_mode"]
        row = {field: diff[field] for field in
               ("dim", "sampler", "n_points", "repeat", "seed", "functional")}
        for metric in ("time_to_barcode_excluding_alpha_s", "time_forest_s",
                       "time_barcode_with_interior_s", "time_landscape_s",
                       "time_pipeline_s", "peak_rss_to_barcode_bytes", "peak_rss_bytes"):
            if standard.get(metric, 0) > 0 and diff.get(metric, 0) > 0:
                row[metric + "_standard_over_diff"] = standard[metric] / diff[metric]
        row["same_point_cloud"] = standard.get("point_cloud_sha256") == diff.get("point_cloud_sha256")
        row["same_barcode_endpoints"] = standard.get("barcode_endpoints_sha256") == diff.get("barcode_endpoints_sha256")
        output.append(row)
    return output


def make_plots(rows: list[dict], output: Path, memory_limit_gib: float) -> list[Path]:
    """Write summary tables and plots, including incomplete configurations.

    Parameters
    ----------
    rows : list of dict
        Latest records from one benchmark manifest.
    output : Path
        Destination for CSV, PDF and PNG files.
    memory_limit_gib : float
        Configured watchdog threshold, shown on memory panels.

    Returns
    -------
    list of Path
        Generated figure paths. No benchmark is launched.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    paired = paired_ratios(rows)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "paired_ratios.csv", paired)
    _write_csv(output / "failures.csv", [row for row in rows if row["status"] != "ok"])
    artifacts = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            artifacts.append(path)
        plt.close(fig)

    for dim, sampler in sorted({(row["dim"], row["sampler"]) for row in rows}):
        subset = [row for row in summary if (row["dim"], row["sampler"]) == (dim, sampler)]
        functionals = sorted({row["functional"] for row in subset})
        for functional in functionals:
            selected = [row for row in subset if row["functional"] == functional]
            landscapes = functional != "none"
            panels = [
                ("time_to_barcode_excluding_alpha_s", "Time to barcode excluding alpha", "seconds", 1, True),
                ("time_to_barcode_s", "Point cloud → barcode (including alpha)", "seconds", 1, True),
                ("peak_rss_to_barcode_bytes", "Peak RSS through barcode", "GiB", 1 / 1024**3, True),
                ("time_forest_s", "Forest including setup, reduction, activity", "seconds", 1, True),
                ("time_barcode_with_interior_s", "Barcode including requested interiors", "seconds", 1, True),
                ("alpha_pct_to_barcode", "Alpha share of time to barcode", "percent", 1, False),
            ]
            if landscapes:
                panels += [
                    ("time_landscape_s", "Landscape construction", "seconds", 1, True),
                    ("landscape_pct_pipeline", "Landscape share of full pipeline", "percent", 1, False),
                    ("peak_rss_bytes", "Peak RSS of full pipeline", "GiB", 1 / 1024**3, True),
                    ("time_pipeline_s", "Point cloud → landscapes (including alpha)", "seconds", 1, True),
                    ("alpha_pct_pipeline", "Alpha share of full pipeline", "percent", 1, False),
                    ("landscape_pct_excluding_alpha", "Landscape share excluding alpha", "percent", 1, False),
                ]
            fig, axes = plt.subplots(len(panels) // 3, 3, figsize=(15, 4 * (len(panels) // 3)), squeeze=False)
            for axis, (metric, title, unit, scale, log) in zip(axes.flat, panels):
                for mode in MODES:
                    available = sorted([row for row in selected
                                        if row["mode"] == mode and metric + "_median" in row],
                                       key=lambda row: row["n_points"])
                    if not available:
                        continue
                    xs = [row["n_points"] for row in available]
                    ys = [scale * row[metric + "_median"] for row in available]
                    axis.plot(xs, ys, "o-", color=COLORS[mode], label=mode, markersize=4)
                    axis.fill_between(xs, [scale * row[metric + "_min"] for row in available],
                                      [scale * row[metric + "_max"] for row in available],
                                      color=COLORS[mode], alpha=0.12)
                axis.set(xscale="log", xlabel="number of points", ylabel=unit, title=title)
                if log and axis.lines:
                    axis.set_yscale("log")
                elif not log:
                    axis.set_ylim(0, 100)
                if unit == "GiB":
                    axis.axhline(memory_limit_gib, color="gray", linestyle=":", label="RSS stop threshold")
                axis.grid(alpha=0.2)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
            fig.suptitle(f"{dim}D · {sampler} · {functional}\nMedian and min–max of successful repeats; see completion plot for censored runs")
            fig.tight_layout(rect=(0, 0.04, 1, 0.94))
            save(fig, f"{dim}d_{sampler}_{functional}")

        # All attempted cases appear here, so survival-biased timings cannot
        # silently look like complete comparisons at large n.
        fig, axis = plt.subplots(figsize=(10, 4))
        for mode in MODES:
            for functional in functionals:
                selected = sorted([row for row in subset if row["mode"] == mode and
                                   row["functional"] == functional], key=lambda row: row["n_points"])
                if selected:
                    axis.plot([row["n_points"] for row in selected],
                              [row["n_ok"] / row["n_attempts"] for row in selected],
                              marker="o", color=COLORS[mode],
                              linestyle="--" if functional == "constant_one" else "-",
                              label=f"{mode} / {functional}")
        axis.set(xscale="log", ylim=(-0.05, 1.05), xlabel="number of points",
                 ylabel="successful / recorded cases", title=f"Completion: {dim}D · {sampler}")
        axis.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        save(fig, f"{dim}d_{sampler}_completion")

        if any(functional != "none" for functional in functionals):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
            for axis, mode in zip(axes.flat, MODES):
                for functional in functionals:
                    metric = "time_landscape_s"
                    selected = sorted([row for row in subset if row["mode"] == mode and
                                       row["functional"] == functional and metric + "_median" in row],
                                      key=lambda row: row["n_points"])
                    if selected:
                        xs = [row["n_points"] for row in selected]
                        axis.plot(xs, [row[metric + "_median"] for row in selected], "o-", label=functional)
                        axis.fill_between(xs, [row[metric + "_min"] for row in selected],
                                          [row[metric + "_max"] for row in selected], alpha=0.12)
                axis.set(xscale="log", xlabel="number of points", ylabel="landscape seconds", title=mode)
                if axis.lines:
                    axis.set_yscale("log")
                    axis.legend()
            fig.suptitle(f"Functional comparison: {dim}D · {sampler} (fresh process per functional)")
            fig.tight_layout()
            save(fig, f"{dim}d_{sampler}_functionals")

        ratios = [row for row in paired if (row["dim"], row["sampler"]) == (dim, sampler)]
        if ratios:
            metrics = [
                ("time_to_barcode_excluding_alpha_s", "Time to barcode excluding alpha"),
                ("time_forest_s", "Forest time"),
                ("time_barcode_with_interior_s", "Barcode time"),
                ("peak_rss_to_barcode_bytes", "Peak RSS through barcode"),
            ]
            if any("time_landscape_s_standard_over_diff" in row for row in ratios):
                metrics += [("time_landscape_s", "Landscape time"),
                            ("peak_rss_bytes", "Full pipeline peak RSS")]
            fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
            for axis, (metric, title) in zip(axes.flat, metrics):
                for functional in functionals:
                    grouped = {}
                    field = metric + "_standard_over_diff"
                    for row in ratios:
                        if row["functional"] == functional and field in row:
                            grouped.setdefault(row["n_points"], []).append(row[field])
                    if grouped:
                        xs = sorted(grouped)
                        axis.plot(xs, [statistics.median(grouped[x]) for x in xs], "o-", label=functional)
                        axis.fill_between(xs, [min(grouped[x]) for x in xs],
                                          [max(grouped[x]) for x in xs], alpha=0.12)
                axis.axhline(1, color="gray", linestyle=":")
                axis.set(xscale="log", yscale="log", xlabel="number of points", title=title,
                         ylabel="standard / diff-only (higher favors diff-only)")
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f"Matched repeats: {dim}D · {sampler}")
            fig.tight_layout()
            save(fig, f"{dim}d_{sampler}_paired")
    return artifacts


def main() -> None:
    """Render each supplied run separately, preserving its configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Directories containing manifest.json and results.jsonl")
    args = parser.parse_args()
    for directory in args.runs:
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = load_results(directory / "results.jsonl")
        if not rows:
            print(f"No completed attempt records in {directory}")
            continue
        figures = make_plots(rows, directory / "plots", manifest["config"]["memory_limit_gib"])
        print(f"Wrote {len(figures)} figures and summary CSVs to {directory / 'plots'}")


if __name__ == "__main__":
    main()
