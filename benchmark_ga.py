#!/usr/bin/env python3
"""GA Scalability Benchmark

Systematically measures how the GA solver scales with increasing objects and
frames.  Every successful GA solution is validated via MiniZinc
(GlobalScenarioSolver.check_solution).

Outputs:
  benchmark_results.csv   — raw per-run data
  benchmark_analysis.png  — summary plots
  Console summary table
"""

import sys
import time
import random
import itertools
import csv
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Import solver components from the main module
# ---------------------------------------------------------------------------
from GUI_ScenaGen_GA import (
    GlobalScenarioSolver,
    Config,
    get_ra_string,
    get_qdc_string,
)
from ga_solver_jax import JAXGASolver

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
OBJECTS_GRID = [8, 10]
FRAMES_GRID = [8, 10]
REPETITIONS = 2  # different random seeds per (O, T)
TIMEOUT_S = 900  # per-run wall-clock timeout

CATEGORIES = ["car", "pedestrian", "truck", "bus"]
SPEED_CATS = ["not moving", "slow", "normal", "fast"]
MAP_LIMIT = Config.MAP_LIMIT  # 500

CSV_FILE = "benchmark_results.csv"
PLOT_FILE = "benchmark_analysis.png"


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------
def generate_scenario(num_objects: int, num_frames: int, seed: int):
    """Generate a random but guaranteed-solvable scenario.

    Returns (objects, ra_matrix, qdc_matrix, velocities, heading_map).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # --- 1. Create object list (1 ego + rest random) ---
    objects = [{"id": 1, "category": "ego"}]
    for i in range(2, num_objects + 1):
        objects.append({"id": i, "category": rng.choice(CATEGORIES)})

    # --- 2. Ground-truth positions per frame ---
    # Frame 0: random positions in [-MAP_LIMIT/2, MAP_LIMIT/2]
    half = MAP_LIMIT // 2
    gt_positions = {}  # (obj_id, frame) -> (x, y)
    gt_headings = {}  # (obj_id, frame) -> heading (0 or 90)
    gt_speeds = {}  # (obj_id, frame) -> speed_cat

    for obj in objects:
        oid = obj["id"]
        if obj["category"] == "ego":
            # ego always at origin in frame 0
            gt_positions[(oid, 0)] = (0, 0)
        else:
            gt_positions[(oid, 0)] = (
                rng.randint(-half, half),
                rng.randint(-half, half),
            )
        gt_headings[(oid, 0)] = rng.choice([0, 90])
        gt_speeds[(oid, 0)] = rng.choice(SPEED_CATS)

    # Subsequent frames: small random displacements respecting speed limits
    for t in range(1, num_frames):
        for obj in objects:
            oid = obj["id"]
            prev_x, prev_y = gt_positions[(oid, t - 1)]
            speed_cat = rng.choice(SPEED_CATS)
            gt_speeds[(oid, t)] = speed_cat
            max_move = Config.SPEED_LIMITS[speed_cat]
            if max_move == 0:
                dx, dy = 0, 0
            else:
                # Random displacement with Manhattan distance <= max_move
                dx = rng.randint(-max_move, max_move)
                remaining = max_move - abs(dx)
                dy = rng.randint(-remaining, remaining)
            new_x = max(-MAP_LIMIT, min(MAP_LIMIT, prev_x + dx))
            new_y = max(-MAP_LIMIT, min(MAP_LIMIT, prev_y + dy))
            gt_positions[(oid, t)] = (new_x, new_y)
            gt_headings[(oid, t)] = rng.choice([0, 90])

    # --- 3. Compute constraints from ground-truth ---
    ra_matrix = [set() for _ in range(num_frames)]
    qdc_matrix = [set() for _ in range(num_frames)]
    velocities = {}
    heading_map = {}

    for t in range(num_frames):
        # Build temporary object dicts for constraint computation
        obj_states = {}
        for obj in objects:
            oid = obj["id"]
            x, y = gt_positions[(oid, t)]
            h = gt_headings[(oid, t)]
            dims = Config.get_dimensions(obj["category"])
            L, W = dims
            is_vert = h == 90 or h == 1
            if is_vert:
                bw, bh = W, L
            else:
                bw, bh = L, W
            obj_states[oid] = {
                "id": oid,
                "cat": obj["category"],
                "x": x,
                "y": y,
                "heading": h,
                "x_min": x - bw / 2,
                "x_max": x + bw / 2,
                "y_min": y - bh / 2,
                "y_max": y + bh / 2,
            }

        # Pairwise RA and QDC constraints
        ids = [obj["id"] for obj in objects]
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                oid_i, oid_j = ids[i_idx], ids[j_idx]
                si = obj_states[oid_i]
                sj = obj_states[oid_j]

                # Allen relations (X and Y axes)
                ra_x = get_ra_string(
                    int(si["x_min"]),
                    int(si["x_max"]),
                    int(sj["x_min"]),
                    int(sj["x_max"]),
                )
                ra_y = get_ra_string(
                    int(si["y_min"]),
                    int(si["y_max"]),
                    int(sj["y_min"]),
                    int(sj["y_max"]),
                )
                ra_matrix[t].add((oid_i, oid_j, ra_x, ra_y))

                # QDC
                qdc = get_qdc_string(si, sj)
                qdc_matrix[t].add((oid_i, oid_j, qdc))

        # Velocities and heading map
        for obj in objects:
            oid = obj["id"]
            velocities[(oid, t)] = gt_speeds[(oid, t)]
            heading_map[(oid, t)] = 1 if gt_headings[(oid, t)] == 90 else 0

    return objects, ra_matrix, qdc_matrix, velocities, heading_map


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_single(num_objects, num_frames, seed):
    """Run one benchmark point. Returns a dict of metrics."""
    result = {
        "num_objects": num_objects,
        "num_frames": num_frames,
        "seed": seed,
        "ga_time_s": None,
        "ga_success": False,
        "ga_best_fitness": None,
        "ga_generations": None,
        "mzn_valid": None,
        "num_genes": None,
        "num_ra_constraints": 0,
        "num_qdc_constraints": 0,
        "timed_out": False,
        "solver": "jax",
    }

    # Generate scenario
    objects, ra_matrix, qdc_matrix, velocities, heading_map = generate_scenario(
        num_objects, num_frames, seed
    )

    # Count constraints
    result["num_ra_constraints"] = sum(len(s) for s in ra_matrix)
    result["num_qdc_constraints"] = sum(len(s) for s in qdc_matrix)

    sol_per_pop = max(1000, 100 * num_objects)

    # Create solver
    ga_solver = JAXGASolver(objects, num_frames)
    result["num_genes"] = ga_solver._gene_count()

    # Track generations and best fitness via callback
    gen_counter = {"count": 0, "best_fitness": None}

    def on_gen(ga_inst):
        gen_counter["count"] = ga_inst.generations_completed
        gen_counter["best_fitness"] = ga_inst.best_solution_fitness

    # Run with timeout
    print(
        f"\n{'=' * 60}\n"
        f"  O={num_objects}, T={num_frames}, seed={seed}, "
        f"timeout={TIMEOUT_S}s, pop={sol_per_pop}\n"
        f"{'=' * 60}"
    )

    # Set numpy seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    t0 = time.time()
    try:
        solution = ga_solver.solve(
            ra_matrix,
            qdc_matrix,
            velocities,
            heading_map,
            timeout=TIMEOUT_S,
            sol_per_pop=sol_per_pop,
            on_generation=on_gen,
        )
    except Exception:
        solution = None
        result["timed_out"] = True
        print(f"  TIMEOUT after {TIMEOUT_S}s")
    except Exception as e:
        solution = None
        print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    result["ga_time_s"] = round(elapsed, 2)
    result["ga_generations"] = gen_counter["count"]

    if solution is not None:
        result["ga_success"] = True
        result["ga_best_fitness"] = 0

        # Validate with MiniZinc
        print("  Validating with MiniZinc...")
        try:
            mzn_solver = GlobalScenarioSolver(objects, num_frames)
            is_valid, details = mzn_solver.check_solution(
                solution, ra_matrix, qdc_matrix, velocities, heading_map
            )
            result["mzn_valid"] = is_valid
            print(f"  MiniZinc: {details}")
            if not is_valid:
                print(f"  *** WARNING: GA solution FAILED MiniZinc validation! ***")
        except Exception as e:
            result["mzn_valid"] = False
            print(f"  MiniZinc check error: {e}")
    else:
        result["ga_success"] = False
        if gen_counter["best_fitness"] is not None:
            result["ga_best_fitness"] = -gen_counter["best_fitness"]  # penalty (positive = violation)
        else:
            result["ga_best_fitness"] = None

    print(
        f"  Result: success={result['ga_success']}, "
        f"time={result['ga_time_s']}s, "
        f"gens={result['ga_generations']}, "
        f"mzn={result['mzn_valid']}"
    )
    return result


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "num_objects",
    "num_frames",
    "seed",
    "solver",
    "ga_time_s",
    "ga_success",
    "ga_best_fitness",
    "ga_generations",
    "mzn_valid",
    "num_genes",
    "num_ra_constraints",
    "num_qdc_constraints",
    "timed_out",
]


def write_csv(results, path=CSV_FILE):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Analysis & plotting
# ---------------------------------------------------------------------------
def print_summary(results):
    """Print a console summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by (O, T)
    groups = defaultdict(list)
    for r in results:
        groups[(r["num_objects"], r["num_frames"])].append(r)

    # Header
    header = f"{'O':>4} {'T':>4} | {'Success':>8} | {'Avg Time':>10} | {'Avg Gens':>10} | {'MZN Valid':>10} | {'Timeouts':>9}"
    print(header)
    print("-" * len(header))

    mzn_failures = []
    for key in sorted(groups.keys()):
        runs = groups[key]
        n = len(runs)
        successes = sum(1 for r in runs if r["ga_success"])
        timeouts = sum(1 for r in runs if r["timed_out"])
        avg_time = sum(r["ga_time_s"] for r in runs if r["ga_time_s"] is not None) / n
        avg_gens = sum(r["ga_generations"] or 0 for r in runs) / n
        mzn_ok = sum(1 for r in runs if r["mzn_valid"] is True)
        mzn_total = sum(1 for r in runs if r["ga_success"])

        mzn_str = f"{mzn_ok}/{mzn_total}" if mzn_total > 0 else "N/A"

        # Flag any MiniZinc failures
        for r in runs:
            if r["ga_success"] and r["mzn_valid"] is False:
                mzn_failures.append(r)

        print(
            f"{key[0]:>4} {key[1]:>4} | "
            f"{successes}/{n:>5} | "
            f"{avg_time:>8.1f}s | "
            f"{avg_gens:>10.0f} | "
            f"{mzn_str:>10} | "
            f"{timeouts:>9}"
        )

    if mzn_failures:
        print(f"\n*** {len(mzn_failures)} MiniZinc VALIDATION FAILURES: ***")
        for r in mzn_failures:
            print(f"  O={r['num_objects']}, T={r['num_frames']}, seed={r['seed']}")

    total = len(results)
    total_success = sum(1 for r in results if r["ga_success"])
    total_mzn = sum(1 for r in results if r["mzn_valid"] is True)
    total_mzn_checked = sum(1 for r in results if r["ga_success"])
    total_timeout = sum(1 for r in results if r["timed_out"])

    print(f"\nTotal runs: {total}")
    print(f"GA successes: {total_success}/{total} ({100 * total_success / total:.1f}%)")
    print(f"MiniZinc validated: {total_mzn}/{total_mzn_checked}")
    print(f"Timeouts: {total_timeout}/{total}")

    return len(mzn_failures)


def make_plots(results, path=PLOT_FILE):
    """Produce summary plots and save to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GA Scalability Benchmark", fontsize=14, fontweight="bold")

    # Group data
    groups = defaultdict(list)
    for r in results:
        groups[(r["num_objects"], r["num_frames"])].append(r)

    # --- Plot 1: Solve time vs objects (lines per frame count) ---
    ax = axes[0, 0]
    for nf in FRAMES_GRID:
        xs, ys = [], []
        for no in OBJECTS_GRID:
            runs = groups.get((no, nf), [])
            successful = [r for r in runs if r["ga_success"]]
            if successful:
                xs.append(no)
                ys.append(np.mean([r["ga_time_s"] for r in successful]))
        if xs:
            ax.plot(xs, ys, "o-", label=f"T={nf}")
    ax.set_xlabel("Number of objects")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Solve Time vs Objects")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Solve time vs frames (lines per object count) ---
    ax = axes[0, 1]
    for no in OBJECTS_GRID:
        xs, ys = [], []
        for nf in FRAMES_GRID:
            runs = groups.get((no, nf), [])
            successful = [r for r in runs if r["ga_success"]]
            if successful:
                xs.append(nf)
                ys.append(np.mean([r["ga_time_s"] for r in successful]))
        if xs:
            ax.plot(xs, ys, "o-", label=f"O={no}")
    ax.set_xlabel("Number of frames")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Solve Time vs Frames")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Success rate heatmap ---
    ax = axes[1, 0]
    success_grid = np.full((len(OBJECTS_GRID), len(FRAMES_GRID)), np.nan)
    for i, no in enumerate(OBJECTS_GRID):
        for j, nf in enumerate(FRAMES_GRID):
            runs = groups.get((no, nf), [])
            if runs:
                success_grid[i, j] = (
                    sum(1 for r in runs if r["ga_success"]) / len(runs) * 100
                )

    im = ax.imshow(
        success_grid,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        origin="lower",
    )
    ax.set_xticks(range(len(FRAMES_GRID)))
    ax.set_xticklabels(FRAMES_GRID)
    ax.set_yticks(range(len(OBJECTS_GRID)))
    ax.set_yticklabels(OBJECTS_GRID)
    ax.set_xlabel("Number of frames")
    ax.set_ylabel("Number of objects")
    ax.set_title("Success Rate (%)")
    # Annotate cells
    for i in range(len(OBJECTS_GRID)):
        for j in range(len(FRAMES_GRID)):
            val = success_grid[i, j]
            if not np.isnan(val):
                color = "white" if val < 50 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax)

    # --- Plot 4: Solve time heatmap (mean, successful only) ---
    ax = axes[1, 1]
    time_grid = np.full((len(OBJECTS_GRID), len(FRAMES_GRID)), np.nan)
    for i, no in enumerate(OBJECTS_GRID):
        for j, nf in enumerate(FRAMES_GRID):
            runs = groups.get((no, nf), [])
            successful = [r for r in runs if r["ga_success"]]
            if successful:
                time_grid[i, j] = np.mean([r["ga_time_s"] for r in successful])

    im2 = ax.imshow(
        time_grid,
        aspect="auto",
        cmap="YlOrRd",
        origin="lower",
    )
    ax.set_xticks(range(len(FRAMES_GRID)))
    ax.set_xticklabels(FRAMES_GRID)
    ax.set_yticks(range(len(OBJECTS_GRID)))
    ax.set_yticklabels(OBJECTS_GRID)
    ax.set_xlabel("Number of frames")
    ax.set_ylabel("Number of objects")
    ax.set_title("Mean Solve Time (s, successful only)")
    for i in range(len(OBJECTS_GRID)):
        for j in range(len(FRAMES_GRID)):
            val = time_grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im2, ax=ax)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Plots saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_results = []
    total_runs = len(OBJECTS_GRID) * len(FRAMES_GRID) * REPETITIONS
    run_idx = 0

    print("GA Scalability Benchmark (JAX GA)")
    print(f"Objects: {OBJECTS_GRID}")
    print(f"Frames:  {FRAMES_GRID}")
    print(f"Reps:    {REPETITIONS}")
    print(f"Timeout: {TIMEOUT_S}s")
    print(f"Total runs: {total_runs}")
    print()

    for num_objects, num_frames in itertools.product(OBJECTS_GRID, FRAMES_GRID):
        for rep in range(REPETITIONS):
            run_idx += 1
            seed = num_objects * 10000 + num_frames * 100 + rep
            print(f"\n[{run_idx}/{total_runs}]", end="")

            result = run_single(num_objects, num_frames, seed)
            all_results.append(result)

            # Write CSV incrementally so we don't lose data on crash
            write_csv(all_results)

    # Final summary
    num_mzn_failures = print_summary(all_results)

    # Plots
    try:
        make_plots(all_results)
    except Exception as e:
        print(f"Warning: could not generate plots: {e}")

    # Exit code
    if num_mzn_failures > 0:
        print(f"\nEXIT 1: {num_mzn_failures} MiniZinc validation failure(s)")
        sys.exit(1)
    else:
        print("\nAll validated solutions passed MiniZinc check.")
        sys.exit(0)


if __name__ == "__main__":
    main()
