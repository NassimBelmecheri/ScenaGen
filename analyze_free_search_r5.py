#!/usr/bin/env python3
"""Analyze solving time results from results_generated_free_search_r5/.

Produces tables (printed + LaTeX) and plots showing:
1. Solve time vs. instance size (objects x frames), split by refinement level
2. Heatmaps of solve time for (objects, frames) grid per refinement
3. Cost of constraint density: ratio rN/r0
4. Warm-start benefit: later refinements (r2-r4) faster than r1
5. First-solution time vs total solve time
6. Time normalized per decision variable
"""

import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results_generated_free_search_r5"
OUTPUT_DIR = Path(__file__).parent / "analysis_free_search_r5"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
    "font.family": "serif",
})

REFINEMENT_COLORS = {
    0: "#2c7bb6",
    1: "#d7191c",
    2: "#fdae61",
    3: "#abdda4",
    4: "#1a9641",
}
REFINEMENT_LABELS = {
    0: r"$r{=}0$ (d$\approx$3--5%)",
    1: r"$r{=}1$ (d=25%)",
    2: r"$r{=}2$ (d=50%)",
    3: r"$r{=}3$ (d=75%)",
    4: r"$r{=}4$ (d=100%)",
}
REFINEMENT_LABELS_SHORT = {
    0: "r=0",
    1: "r=1",
    2: "r=2",
    3: "r=3",
    4: "r=4",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_stats(results_dir: Path) -> pd.DataFrame:
    """Load all *_stats.csv files into a single DataFrame."""
    rows = []
    for f in sorted(results_dir.glob("*_stats.csv")):
        m = re.match(r"scenario_result_o(\d+)_f(\d+)_s(\d+)_", f.name)
        if not m:
            continue
        n_obj, n_frames, scenario = int(m.group(1)), int(m.group(2)), int(m.group(3))
        df = pd.read_csv(f)
        df["file_objects"] = n_obj
        df["file_frames"] = n_frames
        df["scenario"] = scenario
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    # Use file-level object/frame counts (they include ego)
    data["objects"] = data["file_objects"]
    data["frames"] = data["file_frames"]
    data["instance_size"] = data["objects"] * data["frames"]
    return data


def load_all_intermediates(results_dir: Path) -> pd.DataFrame:
    """Load all *_intermediate.csv files."""
    rows = []
    for f in sorted(results_dir.glob("*_intermediate.csv")):
        m = re.match(r"scenario_result_o(\d+)_f(\d+)_s(\d+)_", f.name)
        if not m:
            continue
        df = pd.read_csv(f)
        df["objects"] = int(m.group(1))
        df["frames"] = int(m.group(2))
        df["scenario"] = int(m.group(3))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def make_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    """Mean solve time per (objects, frames, refinement), averaged over scenarios."""
    solved = data[data["status"] == "SOLVED"].copy()
    tbl = (
        solved.groupby(["objects", "frames", "refinement"])
        .agg(
            mean_solve_s=("solve_time_seconds", "mean"),
            median_solve_s=("solve_time_seconds", "median"),
            mean_first_sol_s=("first_solution_seconds", "mean"),
            n_scenarios=("scenario", "nunique"),
            n_solved=("status", "count"),
        )
        .reset_index()
    )
    tbl["instance_size"] = tbl["objects"] * tbl["frames"]
    return tbl.sort_values(["objects", "frames", "refinement"])


def make_pivot_table(summary: pd.DataFrame, value_col: str = "mean_solve_s") -> pd.DataFrame:
    """Pivot: rows=(objects,frames), columns=refinement."""
    pvt = summary.pivot_table(
        index=["objects", "frames"],
        columns="refinement",
        values=value_col,
    )
    pvt.columns = [f"r{int(c)}" for c in pvt.columns]
    pvt = pvt.reset_index()
    pvt["instance_size"] = pvt["objects"] * pvt["frames"]
    return pvt.sort_values("instance_size")


def format_time(seconds):
    """Format seconds for display."""
    if pd.isna(seconds):
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def print_tables(summary: pd.DataFrame, pvt: pd.DataFrame):
    """Print tables to stdout."""
    print("\n" + "=" * 90)
    print("MEAN SOLVE TIME (seconds) by (Objects, Frames) and Refinement Level")
    print("=" * 90)

    pvt_display = pvt.copy()
    for col in [c for c in pvt_display.columns if c.startswith("r")]:
        pvt_display[col] = pvt_display[col].apply(format_time)

    print(pvt_display.to_string(index=False))

    # Also print success rate
    print("\n" + "=" * 90)
    print("SCENARIO COUNTS (solved / total attempts per refinement)")
    print("=" * 90)
    count_tbl = summary.pivot_table(
        index=["objects", "frames"],
        columns="refinement",
        values="n_solved",
        fill_value=0,
    )
    count_tbl.columns = [f"r{int(c)}" for c in count_tbl.columns]
    print(count_tbl.to_string())


def write_latex_tables(summary: pd.DataFrame, pvt: pd.DataFrame, outdir: Path):
    """Write LaTeX table files."""
    # Main solve time table
    pvt_latex = pvt.copy()
    rcols = [c for c in pvt_latex.columns if c.startswith("r")]
    for col in rcols:
        pvt_latex[col] = pvt_latex[col].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) and x < 100
            else (f"{x:.0f}" if pd.notna(x) else "—")
        )

    with open(outdir / "table_solve_times.tex", "w") as f:
        f.write("% Auto-generated by analyze_free_search_r5.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Mean solve time (seconds) by instance configuration and refinement level.}\n")
        f.write("\\label{tab:solve_times}\n")
        ncols = 2 + len(rcols) + 1  # objects, frames, r0..r4, size
        col_fmt = "rr" + "r" * len(rcols) + "r"
        f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
        f.write("\\toprule\n")
        header = "Objects & Frames & " + " & ".join(
            [f"$r={i}$" for i in range(len(rcols))]
        ) + " & Size \\\\\n"
        f.write(header)
        f.write("\\midrule\n")
        for _, row in pvt_latex.iterrows():
            vals = [str(int(row["objects"])), str(int(row["frames"]))]
            vals += [str(row[c]) for c in rcols]
            vals += [str(int(row["instance_size"]))]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    # Speedup table (r1..r4 relative to r0)
    pvt_speed = pvt[["objects", "frames", "instance_size"] + rcols].copy()
    if "r0" in pvt_speed.columns:
        for c in rcols[1:]:
            pvt_speed[f"speedup_{c}"] = pvt_speed["r0"] / pvt_speed[c]

        with open(outdir / "table_speedup.tex", "w") as f:
            f.write("% Auto-generated by analyze_free_search_r5.py\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Speedup of refinement steps relative to initial solve ($r=0$).}\n")
            f.write("\\label{tab:speedup}\n")
            scols = [c for c in pvt_speed.columns if c.startswith("speedup_")]
            col_fmt = "rr" + "r" * len(scols)
            f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
            f.write("\\toprule\n")
            header = "Objects & Frames & " + " & ".join(
                [f"$r_0/r_{i}$" for i in range(1, len(scols) + 1)]
            ) + " \\\\\n"
            f.write(header)
            f.write("\\midrule\n")
            for _, row in pvt_speed.iterrows():
                vals = [str(int(row["objects"])), str(int(row["frames"]))]
                for c in scols:
                    v = row[c]
                    vals.append(f"{v:.2f}" if pd.notna(v) else "—")
                f.write(" & ".join(vals) + " \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_low_density(summary: pd.DataFrame, outdir: Path):
    """Line plot: solve time for r=0 (low density), x=frames, one line per #objects."""
    sub = summary[summary["refinement"] == 0].copy()
    if sub.empty:
        print("  -> skipping low_density plot (no r=0 data)")
        return

    obj_vals = sorted(sub["objects"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for oi, n_obj in enumerate(obj_vals):
        s = sub[sub["objects"] == n_obj].sort_values("frames")
        ax.plot(
            s["frames"], s["mean_solve_s"],
            f"{markers[oi % len(markers)]}-",
            color=plt.cm.tab10(oi / max(len(obj_vals) - 1, 1)),
            label=f"{n_obj} objects",
            markersize=5, alpha=0.8,
        )

    ax.set_xlabel("Frames")
    ax.set_ylabel("Mean solve time (s)")
    ax.set_yscale("log")
    # ax.set_title("Low constraint density (d ≈ 3–5 %)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "solve_time_low_density.pdf")
    fig.savefig(outdir / "solve_time_low_density.png")
    plt.close(fig)
    print(f"  -> solve_time_low_density.{{pdf,png}}")


def plot_high_density(summary: pd.DataFrame, outdir: Path):
    """Line plot: solve time for r=4 (d=100%), x=frames, one line per #objects."""
    sub = summary[summary["refinement"] == 4].copy()
    if sub.empty:
        print("  -> skipping high_density plot (no r=4 data)")
        return

    obj_vals = sorted(sub["objects"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for oi, n_obj in enumerate(obj_vals):
        s = sub[sub["objects"] == n_obj].sort_values("frames")
        ax.plot(
            s["frames"], s["mean_solve_s"],
            f"{markers[oi % len(markers)]}-",
            color=plt.cm.tab10(oi / max(len(obj_vals) - 1, 1)),
            label=f"{n_obj} objects",
            markersize=5, alpha=0.8,
        )

    ax.set_xlabel("Frames")
    ax.set_ylabel("Mean solve time (s)")
    ax.set_yscale("log")
    # ax.set_title("Full constraint density (d = 100 %)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "solve_time_high_density.pdf")
    fig.savefig(outdir / "solve_time_high_density.png")
    plt.close(fig)
    print(f"  -> solve_time_high_density.{{pdf,png}}")


def plot_solve_time_vs_size(summary: pd.DataFrame, outdir: Path):
    """Line plot: solve time vs instance size, one line per refinement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: linear scale
    # Right: log scale
    for ax, yscale, title_suffix in [
        (axes[0], "linear", "(linear)"),
        (axes[1], "log", "(log scale)"),
    ]:
        for r in sorted(summary["refinement"].unique()):
            sub = summary[summary["refinement"] == r].sort_values("instance_size")
            ax.plot(
                sub["instance_size"],
                sub["mean_solve_s"],
                "o-",
                color=REFINEMENT_COLORS.get(r, "gray"),
                label=REFINEMENT_LABELS.get(r, f"r={r}"),
                markersize=4,
                alpha=0.8,
            )
        ax.set_xlabel("Instance size (objects × frames)")
        ax.set_ylabel("Mean solve time (s)")
        ax.set_title(f"Solve time vs. instance size {title_suffix}")
        ax.set_yscale(yscale)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "solve_time_vs_size.pdf")
    fig.savefig(outdir / "solve_time_vs_size.png")
    plt.close(fig)
    print(f"  -> solve_time_vs_size.{{pdf,png}}")


def plot_solve_time_by_objects(summary: pd.DataFrame, outdir: Path):
    """Grouped bar/line chart: one subplot per #objects, x=frames, lines per refinement."""
    obj_vals = sorted(summary["objects"].unique())
    n = len(obj_vals)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, n_obj in enumerate(obj_vals):
        ax = axes[idx // ncols][idx % ncols]
        sub_obj = summary[summary["objects"] == n_obj]
        for r in sorted(sub_obj["refinement"].unique()):
            sub = sub_obj[sub_obj["refinement"] == r].sort_values("frames")
            ax.plot(
                sub["frames"],
                sub["mean_first_sol_s"],
                "o-",
                color=REFINEMENT_COLORS.get(r, "gray"),
                label=REFINEMENT_LABELS.get(r, f"r={r}"),
                markersize=4,
            )
        ax.set_title(f"{n_obj} objects")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Solve time (s)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Solve time by number of objects (log scale)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "solve_time_by_objects.pdf")
    fig.savefig(outdir / "solve_time_by_objects.png")
    plt.close(fig)
    print(f"  -> solve_time_by_objects.{{pdf,png}}")


def plot_solve_time_by_frames(summary: pd.DataFrame, outdir: Path):
    """One subplot per #frames, x=objects, lines per refinement."""
    frame_vals = sorted(summary["frames"].unique())
    n = len(frame_vals)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, n_fr in enumerate(frame_vals):
        ax = axes[idx // ncols][idx % ncols]
        sub_fr = summary[summary["frames"] == n_fr]
        for r in sorted(sub_fr["refinement"].unique()):
            sub = sub_fr[sub_fr["refinement"] == r].sort_values("objects")
            ax.plot(
                sub["objects"],
                sub["mean_first_sol_s"],
                "o-",
                color=REFINEMENT_COLORS.get(r, "gray"),
                label=REFINEMENT_LABELS.get(r, f"r={r}"),
                markersize=4,
            )
        ax.set_title(f"{n_fr} frames")
        ax.set_xlabel("Objects")
        ax.set_ylabel("Solve time (s)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Solve time by number of frames (log scale)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "solve_time_by_frames.pdf")
    fig.savefig(outdir / "solve_time_by_frames.png")
    plt.close(fig)
    print(f"  -> solve_time_by_frames.{{pdf,png}}")


def plot_heatmaps(summary: pd.DataFrame, outdir: Path):
    """Heatmap of mean solve time: rows=objects, cols=frames, one per refinement."""
    refinements = sorted(summary["refinement"].unique())
    n = len(refinements)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    obj_vals = sorted(summary["objects"].unique())
    frame_vals = sorted(summary["frames"].unique())

    # Find global min/max for consistent color scale
    vmin = summary["mean_first_sol_s"].min()
    vmax = summary["mean_first_sol_s"].max()
    vmin = max(vmin, 0.01)  # avoid log(0)

    for idx, r in enumerate(refinements):
        ax = axes[0][idx]
        sub = summary[summary["refinement"] == r]
        pvt = sub.pivot_table(index="objects", columns="frames", values="mean_first_sol_s")
        # Reindex to have complete grid
        pvt = pvt.reindex(index=obj_vals, columns=frame_vals)

        im = ax.imshow(
            pvt.values,
            aspect="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="YlOrRd",
            origin="lower",
        )
        ax.set_xticks(range(len(frame_vals)))
        ax.set_xticklabels(frame_vals, rotation=45, ha="right")
        ax.set_yticks(range(len(obj_vals)))
        ax.set_yticklabels(obj_vals)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Objects")
        ax.set_title(REFINEMENT_LABELS.get(r, f"r={r}"))

        # Annotate cells
        for i in range(len(obj_vals)):
            for j in range(len(frame_vals)):
                val = pvt.values[i, j]
                if not np.isnan(val):
                    txt = format_time(val)
                    color = "white" if val > (vmax * 0.3) else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=axes[0].tolist(), label="Solve time (s)", shrink=0.8)
    fig.suptitle("Mean solve time heatmaps by refinement level", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "heatmaps_solve_time.pdf")
    fig.savefig(outdir / "heatmaps_solve_time.png")
    plt.close(fig)
    print(f"  -> heatmaps_solve_time.{{pdf,png}}")


def plot_refinement_speedup(summary: pd.DataFrame, outdir: Path):
    """Plot: ratio of r0 time to rN time (speedup from warm-starting)."""
    pvt = make_pivot_table(summary, "mean_solve_s")
    rcols = [c for c in pvt.columns if c.startswith("r") and c != "r0"]

    if "r0" not in pvt.columns or not rcols:
        print("  -> skipping speedup plot (no r0 or refinements)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in rcols:
        ratio = pvt["r0"] / pvt[c]
        ax.plot(
            pvt["instance_size"],
            ratio,
            "o-",
            label=f"r0 / {c}",
            markersize=4,
            alpha=0.8,
        )

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="No speedup")
    ax.set_xlabel("Instance size (objects × frames)")
    ax.set_ylabel("Speedup factor (r0 / rN)")
    ax.set_title("Refinement speedup relative to initial solve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "refinement_speedup.pdf")
    fig.savefig(outdir / "refinement_speedup.png")
    plt.close(fig)
    print(f"  -> refinement_speedup.{{pdf,png}}")


def plot_first_solution_vs_total(summary: pd.DataFrame, outdir: Path):
    """Scatter: first solution time vs total solve time per refinement."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in sorted(summary["refinement"].unique()):
        sub = summary[summary["refinement"] == r]
        sub = sub.dropna(subset=["mean_first_sol_s", "mean_solve_s"])
        ax.scatter(
            sub["mean_first_sol_s"],
            sub["mean_solve_s"],
            color=REFINEMENT_COLORS.get(r, "gray"),
            label=REFINEMENT_LABELS.get(r, f"r={r}"),
            alpha=0.7,
            s=30,
        )

    # Diagonal reference
    lims = [0.1, max(summary["mean_solve_s"].max(), summary["mean_first_sol_s"].max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.3, label="x = y")
    ax.set_xlabel("Mean first solution time (s)")
    ax.set_ylabel("Mean total solve time (s)")
    ax.set_title("First solution time vs. total solve time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "first_vs_total_solve.pdf")
    fig.savefig(outdir / "first_vs_total_solve.png")
    plt.close(fig)
    print(f"  -> first_vs_total_solve.{{pdf,png}}")


def plot_scaling_3d(summary: pd.DataFrame, outdir: Path):
    """3D surface plot: solve time as function of (objects, frames) for r=0 and r=1."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    for r in [0, 1]:
        sub = summary[summary["refinement"] == r]
        if sub.empty:
            continue
        pvt = sub.pivot_table(index="objects", columns="frames", values="mean_solve_s")
        X, Y = np.meshgrid(pvt.columns.values, pvt.index.values)
        Z = pvt.values

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        # Use log of Z for better visualization
        Z_plot = np.log10(np.where(np.isnan(Z), np.nan, np.maximum(Z, 0.01)))
        surf = ax.plot_surface(X, Y, Z_plot, cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.3)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Objects")
        ax.set_zlabel("log₁₀(solve time / s)")
        ax.set_title(f"Solve time scaling — {REFINEMENT_LABELS.get(r, f'r={r}')}")
        fig.colorbar(surf, ax=ax, shrink=0.5, label="log₁₀(s)")
        fig.tight_layout()
        fig.savefig(outdir / f"scaling_3d_r{r}.pdf")
        fig.savefig(outdir / f"scaling_3d_r{r}.png")
        plt.close(fig)
    print(f"  -> scaling_3d_r{{0,1}}.{{pdf,png}}")


def plot_refinement_time_breakdown(data: pd.DataFrame, outdir: Path):
    """Stacked bar chart: time per refinement step for selected instance sizes."""
    solved = data[data["status"] == "SOLVED"].copy()

    # Group by (objects, frames, refinement), take mean
    grp = (
        solved.groupby(["objects", "frames", "refinement"])["solve_time_seconds"]
        .mean()
        .reset_index()
    )

    # Select a subset of representative instance sizes
    combos = grp.groupby(["objects", "frames"]).size().reset_index()
    combos["size"] = combos["objects"] * combos["frames"]
    combos = combos.sort_values("size")

    # Pick ~12 evenly spaced combos
    if len(combos) > 12:
        idx = np.linspace(0, len(combos) - 1, 12, dtype=int)
        combos = combos.iloc[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    x_labels = []
    x_pos = np.arange(len(combos))

    bottoms = np.zeros(len(combos))
    for r in sorted(grp["refinement"].unique()):
        heights = []
        for _, row in combos.iterrows():
            val = grp[
                (grp["objects"] == row["objects"])
                & (grp["frames"] == row["frames"])
                & (grp["refinement"] == r)
            ]["solve_time_seconds"]
            heights.append(val.values[0] if len(val) > 0 else 0)
        heights = np.array(heights)
        ax.bar(
            x_pos,
            heights,
            bottom=bottoms,
            color=REFINEMENT_COLORS.get(r, "gray"),
            label=REFINEMENT_LABELS.get(r, f"r={r}"),
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += heights

    for _, row in combos.iterrows():
        x_labels.append(f"o{int(row['objects'])}\nf{int(row['frames'])}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Time breakdown by refinement step (selected instances)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "time_breakdown_stacked.pdf")
    fig.savefig(outdir / "time_breakdown_stacked.png")
    plt.close(fig)
    print(f"  -> time_breakdown_stacked.{{pdf,png}}")


def plot_refinement_ratio_heatmap(summary: pd.DataFrame, outdir: Path):
    """Heatmap showing ratio r1/r0 (how much slower/faster refinement is vs initial)."""
    obj_vals = sorted(summary["objects"].unique())
    frame_vals = sorted(summary["frames"].unique())

    for rN in [1, 2, 3, 4]:
        r0_pvt = summary[summary["refinement"] == 0].pivot_table(
            index="objects", columns="frames", values="mean_solve_s"
        ).reindex(index=obj_vals, columns=frame_vals)

        rN_pvt = summary[summary["refinement"] == rN].pivot_table(
            index="objects", columns="frames", values="mean_solve_s"
        ).reindex(index=obj_vals, columns=frame_vals)

        ratio = rN_pvt / r0_pvt

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            ratio.values,
            aspect="auto",
            norm=LogNorm(vmin=0.01, vmax=max(100, np.nanmax(ratio.values))),
            cmap="RdYlGn_r",
            origin="lower",
        )
        ax.set_xticks(range(len(frame_vals)))
        ax.set_xticklabels(frame_vals, rotation=45, ha="right")
        ax.set_yticks(range(len(obj_vals)))
        ax.set_yticklabels(obj_vals)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Objects")
        ax.set_title(f"Time ratio r{rN} / r0 (green = refinement faster)")

        for i in range(len(obj_vals)):
            for j in range(len(frame_vals)):
                val = ratio.values[i, j]
                if not np.isnan(val):
                    txt = f"{val:.1f}×" if val < 100 else f"{val:.0f}×"
                    color = "white" if val > 10 or val < 0.1 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)

        fig.colorbar(im, ax=ax, label=f"r{rN} / r0")
        fig.tight_layout()
        fig.savefig(outdir / f"ratio_heatmap_r{rN}_vs_r0.pdf")
        fig.savefig(outdir / f"ratio_heatmap_r{rN}_vs_r0.png")
        plt.close(fig)

    print(f"  -> ratio_heatmap_r{{1,2,3,4}}_vs_r0.{{pdf,png}}")


def plot_density_vs_time(data: pd.DataFrame, outdir: Path):
    """Show how constraint density affects solve time."""
    solved = data[data["status"] == "SOLVED"].copy()
    solved["density"] = solved["density"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in sorted(solved["refinement"].unique()):
        sub = solved[solved["refinement"] == r]
        ax.scatter(
            sub["density"],
            sub["solve_time_seconds"],
            color=REFINEMENT_COLORS.get(r, "gray"),
            label=REFINEMENT_LABELS.get(r, f"r={r}"),
            alpha=0.5,
            s=15,
        )

    ax.set_xlabel("Constraint density")
    ax.set_ylabel("Solve time (s)")
    ax.set_yscale("log")
    ax.set_title("Solve time vs. constraint density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "density_vs_time.pdf")
    fig.savefig(outdir / "density_vs_time.png")
    plt.close(fig)
    print(f"  -> density_vs_time.{{pdf,png}}")


def plot_initial_vs_refinement_comparison(summary: pd.DataFrame, outdir: Path):
    """Key narrative plot: initial solve time vs refinement time, showing refinement is fast."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: absolute times, grouped by objects, x=frames
    ax = axes[0]
    obj_vals = sorted(summary["objects"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for oi, n_obj in enumerate(obj_vals):
        sub_r0 = summary[(summary["objects"] == n_obj) & (summary["refinement"] == 0)].sort_values("frames")
        # Average refinement time (r1-r4)
        sub_rN = summary[(summary["objects"] == n_obj) & (summary["refinement"] > 0)]
        sub_rN_avg = sub_rN.groupby("frames")["mean_solve_s"].mean().reset_index().sort_values("frames")

        marker = markers[oi % len(markers)]
        color = plt.cm.tab10(oi / max(len(obj_vals) - 1, 1))
        ax.plot(sub_r0["frames"], sub_r0["mean_solve_s"], f"{marker}-",
                color=color, label=f"o={n_obj} (r=0)", markersize=5, alpha=0.9)
        ax.plot(sub_rN_avg["frames"], sub_rN_avg["mean_solve_s"], f"{marker}--",
                color=color, label=f"o={n_obj} (r>0 avg)", markersize=4, alpha=0.6)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Solve time (s)")
    ax.set_yscale("log")
    ax.set_title("Initial (r=0, solid) vs. refinement (r>0 avg, dashed)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: ratio of mean(r>0) / r0
    ax = axes[1]
    for oi, n_obj in enumerate(obj_vals):
        sub_r0 = summary[(summary["objects"] == n_obj) & (summary["refinement"] == 0)].set_index("frames")
        sub_rN = summary[(summary["objects"] == n_obj) & (summary["refinement"] > 0)]
        sub_rN_avg = sub_rN.groupby("frames")["mean_solve_s"].mean()
        ratio = sub_rN_avg / sub_r0["mean_solve_s"]
        ratio = ratio.dropna().sort_index()

        marker = markers[oi % len(markers)]
        color = plt.cm.tab10(oi / max(len(obj_vals) - 1, 1))
        ax.plot(ratio.index, ratio.values, f"{marker}-",
                color=color, label=f"o={n_obj}", markersize=5)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Ratio: mean(r>0) / r0")
    ax.set_title("Refinement time ratio (< 1 means refinement is faster)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle("Initial solving vs. refinement — scaling behavior", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "initial_vs_refinement.pdf")
    fig.savefig(outdir / "initial_vs_refinement.png")
    plt.close(fig)
    print(f"  -> initial_vs_refinement.{{pdf,png}}")


def plot_key_narrative(summary: pd.DataFrame, outdir: Path):
    """Main figure: 3-panel showing (a) r=0 scaling, (b) density cost, (c) all refinements vs r=0."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    obj_vals = sorted(summary["objects"].unique())
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    def _obj_color(oi):
        return plt.cm.tab10(oi / max(len(obj_vals) - 1, 1))

    # --- Panel (a): r=0 is fast and scales well ---
    ax = axes[0]
    for oi, n_obj in enumerate(obj_vals):
        sub = summary[(summary["objects"] == n_obj) & (summary["refinement"] == 0)].sort_values("frames")
        ax.plot(sub["frames"], sub["mean_solve_s"], f"{markers[oi % len(markers)]}-",
                color=_obj_color(oi), label=f"{n_obj} obj", markersize=5)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Solve time (s)")
    ax.set_yscale("log")
    ax.set_title("(a) Initial solve (r=0, density 2--20%)\nis fast across all sizes")
    ax.legend(fontsize=7, title="Objects")
    ax.grid(True, alpha=0.3)
    ax.axhline(60, color="gray", ls=":", alpha=0.4)
    ax.text(max(summary["frames"]) * 0.95, 65, "1 min", ha="right", fontsize=7, color="gray")
    ax.axhline(300, color="gray", ls=":", alpha=0.4)
    ax.text(max(summary["frames"]) * 0.95, 330, "5 min", ha="right", fontsize=7, color="gray")

    # --- Panel (b): All refinements much slower due to density ---
    ax = axes[1]
    for oi, n_obj in enumerate(obj_vals):
        sub_r0 = summary[(summary["objects"] == n_obj) & (summary["refinement"] == 0)].set_index("frames")
        # Average over all refinement steps (r1-r4)
        sub_rN = summary[(summary["objects"] == n_obj) & (summary["refinement"] > 0)]
        sub_rN_avg = sub_rN.groupby("frames")["mean_solve_s"].mean()
        ratio = sub_rN_avg / sub_r0["mean_solve_s"]
        ratio = ratio.dropna().sort_index()
        if ratio.empty:
            continue

        ax.plot(ratio.index, ratio.values, f"{markers[oi % len(markers)]}-",
                color=_obj_color(oi), label=f"{n_obj} obj", markersize=5)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frames")
    ax.set_ylabel(r"Time ratio: mean($t_{r>0}$) / $t_{r=0}$")
    ax.set_yscale("log")
    ax.set_title("(b) Cost of higher constraint density\n"
                 "Refinement steps are 1--500x slower than r=0")
    ax.legend(fontsize=7, title="Objects")
    ax.grid(True, alpha=0.3)

    # --- Panel (c): Solve time across all refinement levels ---
    ax = axes[2]
    for r in sorted(summary["refinement"].unique()):
        sub = summary[summary["refinement"] == r].sort_values("instance_size")
        ax.plot(sub["instance_size"], sub["mean_solve_s"], "o-",
                color=REFINEMENT_COLORS.get(r, "gray"),
                label=REFINEMENT_LABELS.get(r, f"r={r}"),
                markersize=3, alpha=0.8)

    ax.set_xlabel("Instance size (objects × frames)")
    ax.set_ylabel("Solve time (s)")
    ax.set_yscale("log")
    ax.set_title("(c) Solve time by refinement level\n"
                 "r=0 consistently 1--2 orders of magnitude faster")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(60, color="gray", ls=":", alpha=0.4)
    ax.axhline(3600, color="gray", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outdir / "key_narrative.pdf")
    fig.savefig(outdir / "key_narrative.png")
    plt.close(fig)
    print(f"  -> key_narrative.{{pdf,png}}")


def plot_time_per_variable(summary: pd.DataFrame, outdir: Path):
    """Solve time normalized by number of decision variables (objects * frames)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in sorted(summary["refinement"].unique()):
        sub = summary[summary["refinement"] == r].sort_values("instance_size")
        normalized = sub["mean_solve_s"] / sub["instance_size"]
        ax.plot(
            sub["instance_size"],
            normalized,
            "o-",
            color=REFINEMENT_COLORS.get(r, "gray"),
            label=REFINEMENT_LABELS.get(r, f"r={r}"),
            markersize=4,
            alpha=0.8,
        )

    ax.set_xlabel("Instance size (objects × frames)")
    ax.set_ylabel("Solve time per variable (s)")
    ax.set_title("Normalized solve time: seconds per decision variable")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "time_per_variable.pdf")
    fig.savefig(outdir / "time_per_variable.png")
    plt.close(fig)
    print(f"  -> time_per_variable.{{pdf,png}}")


def plot_objects_vs_frames_asymmetry(summary: pd.DataFrame, outdir: Path):
    """Show that objects contribute more to difficulty than frames."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for r in [0, 1]:
        ax = axes[r]
        sub = summary[summary["refinement"] == r]

        # For fixed instance size ~1200, compare (60,20) vs (20,60) etc.
        # More generally: plot solve time for (o, f) pairs that give same size
        pvt = sub.pivot_table(index="objects", columns="frames", values="mean_solve_s")

        obj_vals = sorted(sub["objects"].unique())
        frame_vals = sorted(sub["frames"].unique())

        # Line for each fixed #frames, x=objects
        for fi, nf in enumerate(frame_vals):
            s = sub[sub["frames"] == nf].sort_values("objects")
            if len(s) < 2:
                continue
            ax.plot(s["objects"], s["mean_solve_s"], "o-",
                    label=f"f={nf}", markersize=4, alpha=0.7)

        ax.set_xlabel("Objects")
        ax.set_ylabel("Solve time (s)")
        ax.set_yscale("log")
        ax.set_title(f"{REFINEMENT_LABELS_SHORT[r]}: Scaling with objects (lines=fixed frames)")
        ax.legend(fontsize=7, title="Frames", ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Objects contribute more to difficulty than frames", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "objects_vs_frames_asymmetry.pdf")
    fig.savefig(outdir / "objects_vs_frames_asymmetry.png")
    plt.close(fig)
    print(f"  -> objects_vs_frames_asymmetry.{{pdf,png}}")


def plot_small_instance_refinement(data: pd.DataFrame, summary: pd.DataFrame, outdir: Path):
    """Focus on small instances (n={10,20}, T={20,40}): first-solution time per refinement."""
    SMALL_OBJECTS = [10, 20, 30, 40]
    SMALL_FRAMES = [10, 20, 30, 40]

    solved = data[
        (data["status"] == "SOLVED")
        & (data["objects"].isin(SMALL_OBJECTS))
        & (data["frames"].isin(SMALL_FRAMES))
    ].copy()

    if solved.empty:
        print("  -> skipping small_instance_refinement (no matching data)")
        return

    combos = [(o, f) for o in SMALL_OBJECTS for f in SMALL_FRAMES]
    n_combos = len(combos)

    # --- Plot: grouped bar chart, one group per (n, T), bars per refinement ---
    fig, ax = plt.subplots(figsize=(10, 5))
    refinements = sorted(solved["refinement"].unique())
    n_ref = len(refinements)
    bar_width = 0.8 / n_ref
    x = np.arange(n_combos)

    for ri, r in enumerate(refinements):
        means = []
        stds = []
        for o, f in combos:
            sub = solved[(solved["objects"] == o) & (solved["frames"] == f) & (solved["refinement"] == r)]
            vals = sub["first_solution_seconds"].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)

        offset = (ri - (n_ref - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=3,
            color=REFINEMENT_COLORS.get(r, "gray"),
            label=REFINEMENT_LABELS.get(r, f"r={r}"),
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={o}, T={f}" for o, f in combos])
    ax.set_ylabel("Time to first solution (s)")
    ax.set_title("Small instances: time to first solution by refinement level")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "small_instance_refinement.pdf")
    fig.savefig(outdir / "small_instance_refinement.png")
    plt.close(fig)
    print(f"  -> small_instance_refinement.{{pdf,png}}")

    # --- Line plot: x=refinement level, one line per (n, T) ---
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, (o, f) in enumerate(combos):
        means = []
        stds = []
        for r in refinements:
            sub = solved[(solved["objects"] == o) & (solved["frames"] == f) & (solved["refinement"] == r)]
            vals = sub["first_solution_seconds"].dropna()
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            stds.append(vals.std() if len(vals) > 1 else 0)
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(
            refinements, means, yerr=stds,
            fmt=f"{markers[ci % len(markers)]}-",
            capsize=4, markersize=6, alpha=0.85,
            label=f"n={o}, T={f}",
        )

    ax.set_xticks(refinements)
    ax.set_xticklabels([REFINEMENT_LABELS.get(r, f"r={r}") for r in refinements],
                       fontsize=8, rotation=15, ha="right")
    ax.set_xlabel("Refinement level")
    ax.set_ylabel("Time to first solution (s)")
    ax.set_title("Small instances: first-solution time across refinement levels")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "small_instance_refinement_lines.pdf")
    fig.savefig(outdir / "small_instance_refinement_lines.png")
    plt.close(fig)
    print(f"  -> small_instance_refinement_lines.{{pdf,png}}")

    # --- Heatmaps: one per refinement, rows=objects, cols=frames ---
    n_ref = len(refinements)
    fig, axes = plt.subplots(1, n_ref, figsize=(4 * n_ref, 4), squeeze=False)

    # Build pivot per refinement for consistent color scale
    grp = (
        solved.groupby(["objects", "frames", "refinement"])["first_solution_seconds"]
        .mean()
        .reset_index()
    )
    vmin = max(grp["first_solution_seconds"].min(), 0.01)
    vmax = grp["first_solution_seconds"].max()

    for idx, r in enumerate(refinements):
        ax = axes[0][idx]
        sub = grp[grp["refinement"] == r]
        pvt = sub.pivot_table(index="objects", columns="frames", values="first_solution_seconds")
        pvt = pvt.reindex(index=SMALL_OBJECTS, columns=SMALL_FRAMES)

        im = ax.imshow(
            pvt.values,
            aspect="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="YlOrRd",
            origin="lower",
        )
        ax.set_xticks(range(len(SMALL_FRAMES)))
        ax.set_xticklabels(SMALL_FRAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(SMALL_OBJECTS)))
        ax.set_yticklabels(SMALL_OBJECTS)
        ax.set_xlabel("Frames")
        if idx == 0:
            ax.set_ylabel("Objects")
        ax.set_title(REFINEMENT_LABELS.get(r, f"r={r}"))

        for i in range(len(SMALL_OBJECTS)):
            for j in range(len(SMALL_FRAMES)):
                val = pvt.values[i, j]
                if not np.isnan(val):
                    txt = format_time(val)
                    color = "white" if val > (vmax * 0.3) else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=12, color=color)

    # fig.colorbar(im, ax=axes[0].tolist(), label="Time to first solution (s)", shrink=0.8)
    # fig.suptitle("Small instances: first-solution time heatmaps by refinement level", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "small_instance_refinement_heatmaps.pdf")
    fig.savefig(outdir / "small_instance_refinement_heatmaps.png")
    fig.savefig(outdir / "small_instance_refinement_heatmaps.pgf")
    plt.close(fig)
    print(f"  -> small_instance_refinement_heatmaps.{{pdf,png}}")

    # --- Table: print + LaTeX ---
    rows = []
    for o, f in combos:
        for r in refinements:
            sub = solved[(solved["objects"] == o) & (solved["frames"] == f) & (solved["refinement"] == r)]
            vals = sub["first_solution_seconds"].dropna()
            rows.append({
                "n": o, "T": f,
                "refinement": r,
                "mean_first_sol_s": vals.mean() if len(vals) > 0 else np.nan,
                "std_first_sol_s": vals.std() if len(vals) > 1 else np.nan,
                "min_first_sol_s": vals.min() if len(vals) > 0 else np.nan,
                "max_first_sol_s": vals.max() if len(vals) > 0 else np.nan,
                "n_solved": len(vals),
            })
    tbl = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("SMALL INSTANCES: Time to first solution (seconds) by refinement")
    print("=" * 90)
    tbl_display = tbl.copy()
    for col in ["mean_first_sol_s", "std_first_sol_s", "min_first_sol_s", "max_first_sol_s"]:
        tbl_display[col] = tbl_display[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    print(tbl_display.to_string(index=False))

    tbl.to_csv(outdir / "small_instance_refinement.csv", index=False)

    # LaTeX table
    with open(outdir / "table_small_instance_refinement.tex", "w") as fout:
        fout.write("% Auto-generated by analyze_free_search_r5.py\n")
        fout.write("\\begin{table}[htbp]\n")
        fout.write("\\centering\n")
        fout.write("\\caption{Time to first solution (seconds) for small instances "
                    "($n \\in \\{10,20\\}$, $T \\in \\{20,40\\}$) by refinement level.}\n")
        fout.write("\\label{tab:small_instance_refinement}\n")
        fout.write("\\small\n")
        fout.write("\\begin{tabular}{rrrrrrr}\n")
        fout.write("\\toprule\n")
        fout.write("$n$ & $T$ & $r$ & Mean (s) & Std (s) & Min (s) & Max (s) \\\\\n")
        fout.write("\\midrule\n")
        prev_key = None
        for _, row in tbl.iterrows():
            cur_key = (row["n"], row["T"])
            if prev_key is not None and cur_key != prev_key:
                fout.write("\\midrule\n")
            prev_key = cur_key

            def fmt(v):
                if pd.isna(v):
                    return "---"
                return f"{v:.2f}"

            vals = [
                str(int(row["n"])),
                str(int(row["T"])),
                str(int(row["refinement"])),
                fmt(row["mean_first_sol_s"]),
                fmt(row["std_first_sol_s"]),
                fmt(row["min_first_sol_s"]),
                fmt(row["max_first_sol_s"]),
            ]
            fout.write(" & ".join(vals) + " \\\\\n")
        fout.write("\\bottomrule\n")
        fout.write("\\end{tabular}\n")
        fout.write("\\end{table}\n")
    print(f"  -> table_small_instance_refinement.tex")


def plot_small_instance_objective_tradeoff(results_dir: Path, outdir: Path):
    """Plot the time-vs-improvement tradeoff for small instances using intermediate data.

    x-axis: relative improvement over first solution (%)
    y-axis: time spent since first solution (s)
    Shows whether waiting beyond the first solution is worthwhile.
    """
    SMALL_OBJECTS = [10, 20, 30, 40]
    SMALL_FRAMES = [10, 20, 30, 40]

    intermediates = load_all_intermediates(results_dir)
    if intermediates.empty:
        print("  -> skipping objective_tradeoff (no intermediate data)")
        return

    # Filter to small instances and skip r=0 (feasibility only, objective=0)
    inter = intermediates[
        (intermediates["objects"].isin(SMALL_OBJECTS))
        & (intermediates["frames"].isin(SMALL_FRAMES))
        & (intermediates["refinement"] > 0)
    ].copy()

    if inter.empty:
        print("  -> skipping objective_tradeoff (no matching data)")
        return

    refinements = sorted(inter["refinement"].unique())

    # --- One subplot per refinement level ---
    n_ref = len(refinements)
    fig, axes = plt.subplots(1, n_ref, figsize=(5 * n_ref, 4.5), squeeze=False)

    combos = sorted(set(zip(inter["objects"], inter["frames"])))
    combo_markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*",
                     "o", "s", "^", "D"]

    for ri, r in enumerate(refinements):
        ax = axes[0][ri]
        sub_r = inter[inter["refinement"] == r]

        for ci, (o, f) in enumerate(combos):
            sub = sub_r[(sub_r["objects"] == o) & (sub_r["frames"] == f)]
            scenarios = sub["scenario"].unique()

            # Average across scenarios: collect all traces, interpolate on common x grid
            all_x = []
            all_y = []
            for sc in scenarios:
                trace = sub[sub["scenario"] == sc].sort_values("time")
                if len(trace) < 1:
                    continue
                first_obj = trace["objective"].iloc[0]
                first_time = trace["time"].iloc[0]
                if first_obj == 0:
                    continue  # can't compute relative improvement

                # relative improvement (%) and time since first solution
                rel_imp = (first_obj - trace["objective"]) / abs(first_obj) * 100
                dt = (trace["time"] - first_time)/first_time * 100

                all_x.append(rel_imp.values)
                all_y.append(dt.values)

            if not all_x:
                continue

            # Plot individual scenario traces with low alpha, then nothing to average
            for xi, yi in zip(all_x, all_y):
                ax.plot(xi, yi, f"{combo_markers[ci]}-",
                        color=plt.cm.tab10(ci / max(len(combos) - 1, 1)),
                        markersize=3, alpha=0.4, linewidth=0.8)

            # Invisible plot for legend entry
            ax.plot([], [], f"{combo_markers[ci]}-",
                    color=plt.cm.tab10(ci / max(len(combos) - 1, 1)),
                    label=f"n={o}, T={f}", markersize=4)

        ax.set_xlabel("Relative improvement over first solution (%)")
        if ri == 0:
            ax.set_ylabel("Time since first solution (s)")
        ax.set_title(REFINEMENT_LABELS.get(r, f"r={r}"))
        ax.grid(True, alpha=0.3)
        if ri == n_ref - 1:
            ax.legend(fontsize=6, loc="upper left", ncol=2)

    fig.suptitle("Solution improvement tradeoff: time cost of improving beyond first solution",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "small_instance_objective_tradeoff.pdf")
    fig.savefig(outdir / "small_instance_objective_tradeoff.png")
    plt.close(fig)
    print(f"  -> small_instance_objective_tradeoff.{{pdf,png}}")

    # --- Combined view: all refinements on one plot, averaged ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for ri, r in enumerate(refinements):
        sub_r = inter[inter["refinement"] == r]
        all_x = []
        all_y = []
        for sc in sub_r["scenario"].unique():
            for o, f in combos:
                trace = sub_r[(sub_r["objects"] == o) & (sub_r["frames"] == f)
                              & (sub_r["scenario"] == sc)].sort_values("time")
                if len(trace) < 1:
                    continue
                first_obj = trace["objective"].iloc[0]
                first_time = trace["time"].iloc[0]
                if first_obj == 0:
                    continue
                rel_imp = (first_obj - trace["objective"]) / abs(first_obj) * 100
                dt = (trace["time"] - first_time)/first_time * 100
                all_x.extend(rel_imp.values)
                all_y.extend(dt.values)

        if not all_x:
            continue
        # Sort by improvement for a cleaner line
        order = np.argsort(all_x)
        all_x = np.array(all_x)[order]
        all_y = np.array(all_y)[order]

        ax.scatter(all_x, all_y, s=12, alpha=0.5,
                   color=REFINEMENT_COLORS.get(r, "gray"),
                   label=REFINEMENT_LABELS.get(r, f"r={r}"))

    ax.set_xlabel("Relative improvement over first solution (%)")
    ax.set_ylabel("Time since first solution (s)")
    ax.set_title("Cost of optimality: additional time for solution improvement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "small_instance_objective_tradeoff_combined.pdf")
    fig.savefig(outdir / "small_instance_objective_tradeoff_combined.png")
    plt.close(fig)
    print(f"  -> small_instance_objective_tradeoff_combined.{{pdf,png}}")


def plot_cumulative_time(summary: pd.DataFrame, outdir: Path):
    """Show cumulative time across all refinement steps (total pipeline time)."""
    pvt = make_pivot_table(summary, "mean_solve_s")
    rcols = [c for c in pvt.columns if c.startswith("r")]
    pvt["total"] = pvt[rcols].sum(axis=1)
    pvt["r0_fraction"] = pvt["r0"] / pvt["total"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: total pipeline time vs instance size
    ax = axes[0]
    ax.plot(pvt["instance_size"], pvt["total"], "ko-", markersize=4, label="Total (r0+r1+r2+r3+r4)")
    ax.plot(pvt["instance_size"], pvt["r0"], "o-",
            color=REFINEMENT_COLORS[0], markersize=4, label="r=0 only")
    ax.set_xlabel("Instance size (objects × frames)")
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.set_title("Total pipeline time vs initial solve only")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: r0 as fraction of total
    ax = axes[1]
    ax.plot(pvt["instance_size"], pvt["r0_fraction"] * 100, "o-",
            color=REFINEMENT_COLORS[0], markersize=4)
    ax.set_xlabel("Instance size (objects × frames)")
    ax.set_ylabel("r=0 as % of total pipeline time")
    ax.set_title("Initial solve becomes negligible for large instances")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outdir / "cumulative_time.pdf")
    fig.savefig(outdir / "cumulative_time.png")
    plt.close(fig)
    print(f"  -> cumulative_time.{{pdf,png}}")


def write_latex_narrative_table(summary: pd.DataFrame, outdir: Path):
    """Write a compact LaTeX table focused on the narrative: r0 fast, r1 hard, r2-4 warm-start."""
    pvt = make_pivot_table(summary, "mean_solve_s")
    rcols = [c for c in pvt.columns if c.startswith("r")]

    # Compute ratios
    pvt["r1/r0"] = pvt["r1"] / pvt["r0"]
    pvt["r4/r1"] = pvt["r4"] / pvt["r1"]

    with open(outdir / "table_narrative.tex", "w") as f:
        f.write("% Auto-generated by analyze_free_search_r5.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Solve time scaling: the initial solve (r=0) with low constraint density is fast; "
                "all refinement steps (r=1--4) incur significant cost due to higher constraint density. "
                "All steps warm-start from the previous solution.}\n")
        f.write("\\label{tab:narrative}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{rrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Obj & Fr & Size & $t_{r=0}$ & $t_{r=1}$ & $t_{r=4}$ & "
                "$t_{r=1}/t_{r=0}$ & $t_{r=4}/t_{r=1}$ \\\\\n")
        f.write("\\midrule\n")
        for _, row in pvt.iterrows():
            def fmt(v):
                if pd.isna(v): return "---"
                if v < 60: return f"{v:.1f}\\,s"
                if v < 3600: return f"{v/60:.1f}\\,m"
                return f"{v/3600:.1f}\\,h"

            def fmtx(v):
                if pd.isna(v): return "---"
                return f"{v:.1f}$\\times$"

            vals = [
                str(int(row["objects"])),
                str(int(row["frames"])),
                str(int(row["instance_size"])),
                fmt(row.get("r0")),
                fmt(row.get("r1")),
                fmt(row.get("r4")),
                fmtx(row.get("r1/r0")),
                fmtx(row.get("r4/r1")),
            ]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  -> table_narrative.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading data from {RESULTS_DIR} ...")
    data = load_all_stats(RESULTS_DIR)
    print(f"  Loaded {len(data)} rows from {data['scenario'].nunique()} scenarios")
    print(f"  Objects: {sorted(data['objects'].unique())}")
    print(f"  Frames:  {sorted(data['frames'].unique())}")
    print(f"  Status:  {data['status'].value_counts().to_dict()}")

    summary = make_summary_table(data)
    pvt = make_pivot_table(summary)

    # --- Tables ---
    print_tables(summary, pvt)
    write_latex_tables(summary, pvt, OUTPUT_DIR)
    print(f"\nLaTeX tables written to {OUTPUT_DIR}/")

    # Also write summary CSV
    summary.to_csv(OUTPUT_DIR / "summary_solve_times.csv", index=False)
    pvt.to_csv(OUTPUT_DIR / "pivot_solve_times.csv", index=False)

    # --- Plots ---
    print("\nGenerating plots...")

    plot_low_density(summary, OUTPUT_DIR)
    plot_high_density(summary, OUTPUT_DIR)

    plot_key_narrative(summary, OUTPUT_DIR)
    plot_solve_time_vs_size(summary, OUTPUT_DIR)
    plot_solve_time_by_objects(summary, OUTPUT_DIR)
    plot_solve_time_by_frames(summary, OUTPUT_DIR)
    plot_heatmaps(summary, OUTPUT_DIR)
    plot_refinement_speedup(summary, OUTPUT_DIR)
    plot_first_solution_vs_total(summary, OUTPUT_DIR)
    plot_scaling_3d(summary, OUTPUT_DIR)
    plot_refinement_time_breakdown(data, OUTPUT_DIR)
    plot_refinement_ratio_heatmap(summary, OUTPUT_DIR)
    plot_density_vs_time(data, OUTPUT_DIR)
    plot_initial_vs_refinement_comparison(summary, OUTPUT_DIR)
    plot_time_per_variable(summary, OUTPUT_DIR)
    plot_objects_vs_frames_asymmetry(summary, OUTPUT_DIR)
    plot_cumulative_time(summary, OUTPUT_DIR)
    plot_small_instance_refinement(data, summary, OUTPUT_DIR)
    plot_small_instance_objective_tradeoff(RESULTS_DIR, OUTPUT_DIR)

    # --- Additional LaTeX tables ---
    write_latex_narrative_table(summary, OUTPUT_DIR)

    print(f"\nAll outputs written to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
