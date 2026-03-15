#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line version of the Qualitative Scenario Designer
Allows importing CSV files, generating random scenarios, and solving with various heuristics
"""

import argparse
import asyncio
from gc import collect
import json
import sys
import random
import copy
import itertools
import ast
import pandas as pd
import numpy as np
from minizinc import Instance, Model, Solver as MznSolver
import datetime
import nest_asyncio
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import math

nest_asyncio.apply()


# ==========================================
# 1. CONFIGURATION (same as GUI version)
# ==========================================
class Config:
    SCALE = 1
    BASE_CANVAS_SIZE = 400
    CENTER = BASE_CANVAS_SIZE // 2

    # Base dimensions for objects
    DIMENSIONS = {
        "car": (10, 5),
        "ego": (10, 5),
        "pedestrian": (3, 3),
        "bus": (20, 10),
        "truck": (25, 25),
    }

    SPEED_LIMITS = {"not moving": 2, "slow": 10, "normal": 22, "fast": 45}

    # These will be calculated dynamically
    MAP_LIMIT = None
    MANHATTAN_THRESHOLDS = {}
    COLORS = {
        "ego": "#2c3e50",
        "car": "#3498db",
        "pedestrian": "#e67e22",
        "bus": "#27ae60",
        "truck": "#8e44ad",
    }

    @classmethod
    def calculate_map_size(cls, num_objects, object_list=None):
        """
        Calculate appropriate map size based on number of objects and their dimensions
        """
        if object_list:
            # Get maximum object dimension
            max_dim = 0
            for obj in object_list:
                cat = obj.get("category", obj.get("cat", "car"))
                dims = cls.DIMENSIONS.get(cat)
                max_dim = max(max_dim, max(dims))
        else:
            # Use maximum possible dimension
            max_dim = max(max(dims) for dims in cls.DIMENSIONS.values())

        # Base map size: need enough space for objects plus distances between them
        # Each object needs its own space, plus gaps between them
        # Formula: (num_objects * max_dim * 2) + (num_objects * spacing_factor)
        spacing_factor = 1  # Multiplier for gaps between objects

        base_size = num_objects * max_dim * spacing_factor

        # Ensure minimum size
        min_size = 500
        map_size = max(min_size, base_size)

        # Round to nearest 100 for nicer numbers
        map_size = ((map_size + 99) // 100) * 100

        return map_size

    @classmethod
    def calculate_thresholds(cls, map_limit):
        """
        Calculate Manhattan distance thresholds based on map size
        """
        return {
            "very close": max(10, map_limit // 20),  # 5% of map
            "close": max(20, map_limit // 10),  # 10% of map
            "normal": max(50, map_limit // 5),  # 20% of map
            "far": max(100, map_limit // 2),  # 50% of map
            "very far": map_limit,  # 100% of map
        }


ALLEN_WEIGHTS = {
    "Equals": 10,
    "Meets": 9,
    "MetBy": 9,
    "Starts": 8,
    "StartedBy": 8,
    "Finishes": 8,
    "FinishedBy": 8,
    "Before": 5,
    "After": 5,
    "Overlaps": 3,
    "OverlappedBy": 3,
    "During": 2,
    "Contains": 2,
}


# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def get_ra_string(s1, e1, s2, e2):
    if e1 < s2:
        return "Before"
    if s1 > e2:
        return "After"
    if e1 == s2:
        return "Meets"
    if s1 == e2:
        return "MetBy"
    if s1 < s2 and e1 > s2 and e1 < e2:
        return "Overlaps"
    if s1 > s2 and s1 < e2 and e1 > e2:
        return "OverlappedBy"
    if s1 == s2 and e1 < e2:
        return "Starts"
    if s1 == s2 and e1 > e2:
        return "StartedBy"
    if s1 > s2 and e1 < e2:
        return "During"
    if s1 < s2 and e1 > e2:
        return "Contains"
    if e1 == e2 and s1 > s2:
        return "Finishes"
    if e1 == e2 and s1 < s2:
        return "FinishedBy"
    if s1 == s2 and e1 == e2:
        return "Equals"
    return "Unknown"


def get_qdc_string(o1, o2):
    """
    Calculate qualitative distance category based on closest points between bounding boxes
    """
    # Manual mode - calculate bbox from center and dimensions
    dims1 = Config.DIMENSIONS.get(o1.get("cat", "car"), (20, 10))
    dims2 = Config.DIMENSIONS.get(o2.get("cat", "car"), (20, 10))

    L1, W1 = dims1[0], dims1[1]
    L2, W2 = dims2[0], dims2[1]

    h_val1 = o1.get("heading", -1)
    h_val2 = o2.get("heading", -1)

    curr_w1, curr_h1 = (W1, L1) if (h_val1 == 90 or h_val1 == 1) else (L1, W1)
    curr_w2, curr_h2 = (W2, L2) if (h_val2 == 90 or h_val2 == 1) else (L2, W2)

    cx1, cy1 = o1["x"], o1.get("y", 0)
    cx2, cy2 = o2["x"], o2.get("y", 0)

    bbox1 = (cx1 - curr_w1 / 2, cx1 + curr_w1 / 2, cy1 - curr_h1 / 2, cy1 + curr_h1 / 2)
    bbox2 = (cx2 - curr_w2 / 2, cx2 + curr_w2 / 2, cy2 - curr_h2 / 2, cy2 + curr_h2 / 2)

    # Calculate minimum distance between bounding boxes
    x1_min, x1_max, y1_min, y1_max = bbox1
    x2_min, x2_max, y2_min, y2_max = bbox2

    # Calculate distance in x direction (0 if overlapping in x)
    if x1_max < x2_min:
        dx = x2_min - x1_max
    elif x2_max < x1_min:
        dx = x1_min - x2_max
    else:
        dx = 0  # Overlapping in x

    # Calculate distance in y direction (0 if overlapping in y)
    if y1_max < y2_min:
        dy = y2_min - y1_max
    elif y2_max < y1_min:
        dy = y1_min - y2_max
    else:
        dy = 0  # Overlapping in y

    # Euclidean distance between closest points
    distance = math.sqrt(dx * dx + dy * dy)

    # Categorize based on thresholds
    if distance <= Config.MANHATTAN_THRESHOLDS["very close"]:
        return "very close"
    if (
        distance <= Config.MANHATTAN_THRESHOLDS["close"]
        and distance > Config.MANHATTAN_THRESHOLDS["very close"]
    ):
        return "close"
    if (
        distance <= Config.MANHATTAN_THRESHOLDS["normal"]
        and distance > Config.MANHATTAN_THRESHOLDS["close"]
    ):
        return "normal"
    if (
        distance <= Config.MANHATTAN_THRESHOLDS["far"]
        and distance > Config.MANHATTAN_THRESHOLDS["normal"]
    ):
        return "far"
    return "very far"


# ==========================================
# 3. GLOBAL MATRIX SOLVER (MINIZINC VERSION)
# ==========================================
class GlobalScenarioSolver:
    def __init__(self, objects, num_frames):
        self.objects = objects
        self.num_frames = num_frames
        self.id_map = {obj["id"]: i + 1 for i, obj in enumerate(objects)}
        self.rev_map = {i + 1: obj["id"] for i, obj in enumerate(objects)}
        self.num_objs = len(objects)
        self._last_result = None

        # Category rank for size ordering
        self.rank_map = {"pedestrian": 2, "car": 3, "bus": 4, "truck": 4, "ego": 3}

    # -------------------------------------------------
    # MAIN SOLVE WRAPPER
    # -------------------------------------------------

    def solve_with_stats(
        self,
        ra_matrix,
        qdc_matrix,
        velocities,
        heading_map,
        solver_name="gecode",
        heuristic="default",
        timeout=None,
        prev_result=None,
    ):

        start = time.time()
        result, intermediate_solutions = self.solve(
            ra_matrix,
            qdc_matrix,
            velocities,
            heading_map,
            solver_name,
            heuristic,
            timeout,
            prev_result=prev_result,
        )
        end = time.time()

        stats = {
            "time": end - start,
            "status": "SOLVED" if result else "UNSAT",
            "solver": solver_name,
            "heuristic": heuristic,
            "num_objects": self.num_objs,
            "num_frames": self.num_frames,
            "map_limit": Config.MAP_LIMIT,
            "first_solution_time": intermediate_solutions[0]["time"]
            if intermediate_solutions
            else None,
            "intermediate_solutions": intermediate_solutions,
        }

        return result, stats

    # -------------------------------------------------
    # CORE SOLVER
    # -------------------------------------------------

    def solve(
        self,
        ra_matrix,
        qdc_matrix,
        velocities,
        heading_map,
        solver_name="gecode",
        heuristic="default",
        timeout=None,
        prev_result=None,
    ):

        original_map_limit = Config.MAP_LIMIT
        original_thresholds = Config.MANHATTAN_THRESHOLDS.copy()

        max_scale = 8
        scale = 1
        result = None

        while scale <= max_scale and result is None:
            map_limit = original_map_limit * scale
            Config.MAP_LIMIT = map_limit
            Config.MANHATTAN_THRESHOLDS = {
                k: int(v * scale) for k, v in original_thresholds.items()
            }

            result, intermediate_solutions = self._solve_with_map_size(
                ra_matrix,
                qdc_matrix,
                velocities,
                solver_name,
                heuristic,
                timeout,
                map_limit,
                prev_result=prev_result,
            )

            if result is None:
                scale *= 2

        Config.MAP_LIMIT = original_map_limit
        Config.MANHATTAN_THRESHOLDS = original_thresholds

        return result, intermediate_solutions

    # -------------------------------------------------
    # MINIZINC MODEL
    # -------------------------------------------------

    def _solve_with_map_size(
        self,
        ra_matrix,
        qdc_matrix,
        velocities,
        solver_name,
        heuristic,
        timeout,
        map_limit,
        prev_result=None,
    ):
        ego_solver_index = None
        for obj in self.objects:
            if obj["category"] == "ego":
                ego_solver_index = self.id_map[obj["id"]]
                break
        speed_to_value = Config.SPEED_LIMITS
        sorted_objs = sorted(self.objects, key=lambda o: self.id_map[o["id"]])
        size_rank = [self.rank_map.get(o["category"], 2) for o in sorted_objs]

        sorted_objs = sorted(self.objects, key=lambda o: self.id_map[o["id"]])

        speed_matrix = []

        for obj in sorted_objs:
            oid = obj["id"]
            row = []
            for t in range(self.num_frames):
                speed_cat = velocities.get((oid, t), "normal")
                row.append(speed_to_value.get(speed_cat, 22))
            speed_matrix.append(row)

        speed_limit_str = (
            "[|"
            + "|".join(", ".join(str(x) for x in row) for row in speed_matrix)
            + "|]"
        )

        mzn = f"""
        include "globals.mzn";
        int: T = {self.num_frames};
        int: O = {self.num_objs};
        int: MAP_LIMIT = {map_limit};
        
        set of int: FRAMES = 1..T;
        set of int: OBJS = 1..O;
        
        array[OBJS, FRAMES] of int: speed_limit = {speed_limit_str};
        array[OBJS] of int: size_rank = {size_rank};
        
        array[OBJS, FRAMES] of var -MAP_LIMIT..MAP_LIMIT: x_min;
        array[OBJS, FRAMES] of var -MAP_LIMIT..MAP_LIMIT: x_max;
        array[OBJS, FRAMES] of var -MAP_LIMIT..MAP_LIMIT: y_min;
        array[OBJS, FRAMES] of var -MAP_LIMIT..MAP_LIMIT: y_max;
        
        array[OBJS] of var 5..MAP_LIMIT div 4: size;
        
        array[OBJS, FRAMES] of var -MAP_LIMIT..MAP_LIMIT: cx;
        array[OBJS, FRAMES] of var -MAP_LIMIT..MAP_LIMIT: cy;
        % -----------------------
        % SQUARE CONSTRAINT
        % -----------------------

        constraint forall(o in OBJS, t in FRAMES)(
            x_max[o,t] - x_min[o,t] = size[o] /\\
            y_max[o,t] - y_min[o,t] = size[o] /\\
            x_min[o,t] < x_max[o,t] /\\
            y_min[o,t] < y_max[o,t]
        );
            % -----------------------
            % CENTER DEFINITIONS
            % -----------------------
            
            constraint forall(o in OBJS, t in FRAMES)(
                2 * cx[o,t] = x_min[o,t] + x_max[o,t] /\\
                2 * cy[o,t] = y_min[o,t] + y_max[o,t]
            );
        constraint forall( t in FRAMES)(
            cx[{ego_solver_index},t] = 0 /\\
            cy[{ego_solver_index},t] = 0
        );
        % -----------------------
        % SIZE ORDERING
        % -----------------------

        constraint forall(o1,o2 in OBJS where size_rank[o1] < size_rank[o2])(
            size[o1] + 2 <= size[o2]
        );
        % -----------------------
        % TEMPORAL SPEED CONSTRAINT
        % -----------------------
        
        constraint forall(o in OBJS, t in 1..T-1)(
            abs(cx[o,t+1] - cx[o,t]) +
            abs(cy[o,t+1] - cy[o,t])
            <= speed_limit[o,t]
        );
        """

        # -----------------------
        # ALLEN CONSTRAINTS
        # -----------------------

        def allen(rel, s1, e1, s2, e2):
            if rel == "Before":
                return f"{e1} < {s2}"
            if rel == "After":
                return f"{s1} > {e2}"
            if rel == "Meets":
                return f"{e1} = {s2}"
            if rel == "MetBy":
                return f"{s1} = {e2}"
            if rel == "Overlaps":
                return f"{s1} < {s2} /\\ {s2} < {e1} /\\ {e1} < {e2}"
            if rel == "OverlappedBy":
                return f"{s2} < {s1} /\\ {s1} < {e2} /\\ {e2} < {e1}"
            if rel == "During":
                return f"{s1} > {s2} /\\ {e1} < {e2}"
            if rel == "Contains":
                return f"{s1} < {s2} /\\ {e1} > {e2}"
            if rel == "Starts":
                return f"{s1} = {s2} /\\ {e1} < {e2}"
            if rel == "StartedBy":
                return f"{s1} = {s2} /\\ {e1} > {e2}"
            if rel == "Finishes":
                return f"{e1} = {e2} /\\ {s1} > {s2}"
            if rel == "FinishedBy":
                return f"{e1} = {e2} /\\ {s1} < {s2}"
            if rel == "Equals":
                return f"{s1} = {s2} /\\ {e1} = {e2}"
            return "true"
        def framewise_allen_variable_order(objects, num_frames, ra_matrix, id_map):

            num_objects = len(objects)
        
            frame_weight = {}
        
            for t in range(num_frames):
        
                weight = 0
        
                for (i, j, rx, ry) in ra_matrix[t]:
        
                    weight += ALLEN_WEIGHTS.get(rx, 1)
                    weight += ALLEN_WEIGHTS.get(ry, 1)
        
                frame_weight[t] = weight
        
            frames_sorted = sorted(frame_weight.keys(), key=lambda x: -frame_weight[x])
        
            var_order = []
        
            for t in frames_sorted:
        
                object_weight = {o["id"]:0 for o in objects}
        
                for (i, j, rx, ry) in ra_matrix[t]:
        
                    w = ALLEN_WEIGHTS.get(rx,1) + ALLEN_WEIGHTS.get(ry,1)
        
                    object_weight[i] += w
                    object_weight[j] += w
        
                objs_sorted = sorted(object_weight.keys(), key=lambda x: -object_weight[x])
        
                for oid in objs_sorted:
        
                    idx = id_map[oid]
        
                    var_order.append(f"cx[{idx},{t+1}]")
                    var_order.append(f"cy[{idx},{t+1}]")
        
            return var_order
        for t in range(self.num_frames):
            overlapping_objects = set()
            for i, j, rx, ry in ra_matrix[t]:
                idx1 = self.id_map[i]
                idx2 = self.id_map[j]

                cx = allen(
                    rx,
                    f"x_min[{idx1},{t + 1}]",
                    f"x_max[{idx1},{t + 1}]",
                    f"x_min[{idx2},{t + 1}]",
                    f"x_max[{idx2},{t + 1}]",
                )

                cy = allen(
                    ry,
                    f"y_min[{idx1},{t + 1}]",
                    f"y_max[{idx1},{t + 1}]",
                    f"y_min[{idx2},{t + 1}]",
                    f"y_max[{idx2},{t + 1}]",
                )

                mzn += f"\nconstraint ({cx}) /\\ ({cy});"

            # TODO To avoid overlaps, but it didn't show helpful for solving
            #     if rx not in ("Before", "After", "Meets", "MetBy") or ry not in ("Before", "After", "Meets", "MetBy"):
            #         overlapping_objects.add(idx1)
            #         overlapping_objects.add(idx2)

            # # We collect all non-overlapping objects to post `diffn`
            # non_overlapping = {"x": [], "y": [], "dx": [], "dy": []}
            # for oi in range(1, self.num_objs+1):
            #     if oi in overlapping_objects:
            #         continue

            #     non_overlapping["x"].append(f"x_min[{oi},{t + 1}]")
            #     non_overlapping["y"].append(f"y_min[{oi},{t + 1}]")
            #     non_overlapping["dx"].append(f"size[{oi}]")
            #     non_overlapping["dy"].append(f"size[{oi}]")
            # mzn += f"""\nconstraint diffn(
            #     [{','.join(non_overlapping['x'])}],
            #     [{','.join(non_overlapping['y'])}],
            #     [{','.join(non_overlapping['dx'])}],
            #     [{','.join(non_overlapping['dy'])}]
            # );"""

        # -----------------------
        # QDC (Manhattan gap)
        # -----------------------

        vc = Config.MANHATTAN_THRESHOLDS["very close"]
        cl = Config.MANHATTAN_THRESHOLDS["close"]
        nr = Config.MANHATTAN_THRESHOLDS["normal"]
        fa = Config.MANHATTAN_THRESHOLDS["far"]

        mzn += """
        function var int: dx(var int: a1, var int: a2, var int: b1, var int: b2) =
            max(0, max(a1 - b2, b1 - a2));

        function var int: dy(var int: a1, var int: a2, var int: b1, var int: b2) =
            max(0, max(a1 - b2, b1 - a2));
        """

        for t in range(self.num_frames):
            for i, j, q in qdc_matrix[t]:
                idx1 = self.id_map[i]
                idx2 = self.id_map[j]

                gap = f"""
                dx(x_min[{idx1},{t + 1}], x_max[{idx1},{t + 1}],
                   x_min[{idx2},{t + 1}], x_max[{idx2},{t + 1}])
                +
                dy(y_min[{idx1},{t + 1}], y_max[{idx1},{t + 1}],
                   y_min[{idx2},{t + 1}], y_max[{idx2},{t + 1}])
                """

                if q == "very close":
                    mzn += f"\nconstraint ( {gap} <= {vc});"
                elif q == "close":
                    mzn += f"\n constraint ({gap} > {vc} /\\ {gap} <= {cl});"
                    
                elif q == "normal":
                    mzn += f"\n constraint ( {gap} > {cl} /\\  {gap} <= {nr});"
                elif q == "far":
                    mzn += f"\n constraint ( {gap} > {nr} /\\ {gap} <= {fa});"
                else:
                    mzn += f"\n constraint {gap} > {fa}; "
        if heuristic == "frame-allen":

            var_order = framewise_allen_variable_order(self.objects, self.num_frames, ra_matrix)
            
            order_string = ", ".join(var_order)
            
            mzn += f"""
            
            array[int] of var int: VAR_ORDER = [{order_string}];
            
            solve :: int_search(
                    VAR_ORDER,
                    input_order,
                    indomain_min,
                    complete
            ) satisfy;
            
            """
            
         
        if prev_result is not None:
            cx_ref = [list() for _ in range(self.num_objs)]
            cy_ref = [list() for _ in range(self.num_objs)]
            for prev_frame in prev_result:
                for o in prev_frame:
                    cx = (o["x_min"] + o["x_max"]) // 2
                    cy = (o["y_min"] + o["y_max"]) // 2
                    cx_ref[o["id"] - 1].append(str(cx))
                    cy_ref[o["id"] - 1].append(str(cy))

            cx_ref_str = "[|" + "|".join(",".join(row) for row in cx_ref) + "|]"
            cy_ref_str = "[|" + "|".join(",".join(row) for row in cy_ref) + "|]"
            mzn += f"""\n
        % -----------------------
        % CONSISTENCY OBJECTIVE: MINIMIZE CHANGES TO PREVIOUS SOLUTION
        % -----------------------

        array[OBJS, FRAMES] of int: cx_ref = {cx_ref_str};
        array[OBJS, FRAMES] of int: cy_ref = {cy_ref_str};

        % TODO Do we need a way to exclude certain objects or frames? How would we mask?
        var int: objective = sum(o in OBJS, t in FRAMES)(abs(cx[o,t]-cx_ref[o,t]) + abs(cy[o,t]-cy_ref[o,t]));
        solve minimize objective;
        """
        else:
            mzn += "\nsolve satisfy;"

        # -----------------------
        # RUN SOLVER
        # -----------------------

        open("model.mzn", "w").write(mzn)

        model = Model()
        model.add_string(mzn)

        solver = MznSolver.lookup(solver_name)
        inst = Instance(solver, model)

        try:

            async def collect():
                results = []
                async for item in inst.solutions(
                    timeout=datetime.timedelta(seconds=timeout)
                    if timeout is not None
                    else None,
                    free_search=True,
                    processes=8,
                    intermediate_solutions=prev_result is not None,
                ):
                    if item.solution is None:
                        continue
                    results.append(item)

                return results

            all_results = asyncio.run(collect())
            for r in all_results:
                print(f"Obj.: {r.statistics['objective']} @ {r.statistics['time']}")

            if len(all_results) == 0:
                return None, []

            # Final result is the main result
            result = all_results[-1]
        except:
            # raise
            return None, []

        if not result.status.has_solution():
            return None, []

        intermediate_solution_stats = [
            {
                "objective": r.statistics["objective"],
                "time": r.statistics["time"].total_seconds(),
            }
            for r in all_results
        ]

        res_xmin = result["x_min"]
        res_xmax = result["x_max"]
        res_ymin = result["y_min"]
        res_ymax = result["y_max"]

        output = []

        for t in range(self.num_frames):
            frame = []
            for i_idx, obj_idx in enumerate(sorted(self.rev_map.keys())):
                oid = self.rev_map[obj_idx]
                orig = next(o for o in self.objects if o["id"] == oid)

                xmin = res_xmin[i_idx][t]
                xmax = res_xmax[i_idx][t]
                ymin = res_ymin[i_idx][t]
                ymax = res_ymax[i_idx][t]

                frame.append(
                    {
                        "id": oid,
                        "category": orig["category"],
                        "x_min": xmin,
                        "x_max": xmax,
                        "y_min": ymin,
                        "y_max": ymax,
                        "x": (xmin + xmax) // 2,
                        "y": (ymin + ymax) // 2,
                        "w": xmax - xmin,
                        "h": ymax - ymin,
                    }
                )
            output.append(frame)

        return output, intermediate_solution_stats


# ==========================================
# 4. SCENARIO GENERATOR AND SOLVER WRAPPER
# ==========================================
class CLIScenarioDesigner:
    def __init__(self):
        self.num_frames = 3
        self.objects = []
        self.speeds = {}
        self.headings = {}
        self.ra_matrix = None
        self.qdc_matrix = None
        self.object_positions = {}  # Store positions for each object in each frame
        self.log_messages = []  # For storing log messages

    def log(self, message):
        """Add message to log"""
        self.log_messages.append(message)
        print(message)

    def get_bbox_logic(self, obj):
        """
        Helper method to compute bounding box from object data
        Ensures pedestrians appear as squares (5x5)
        """
        dims = Config.DIMENSIONS.get(obj["cat"], "")
        L, W = dims[0], dims[1]  # L = length, W = width

        # Get heading value
        h_val = obj.get("heading", -1)

        # Determine orientation-based dimensions
        # For pedestrians, dimensions are already square (5x5), so orientation doesn't matter
        if obj["cat"] == "pedestrian":
            curr_w, curr_h = L, W  # Both are 5
        else:
            # For non-pedestrian objects, swap dimensions based on orientation
            if h_val == 90 or h_val == 1:  # Vertical orientation
                curr_w, curr_h = (
                    W,
                    L,
                )  # Width becomes the smaller dimension, height becomes length
            else:  # Horizontal orientation (0 or -1)
                curr_w, curr_h = L, W  # Width is length, height is width

        cx, cy = obj["x"], obj.get("y", 0)

        # Calculate bounding box
        x_min = cx - curr_w / 2
        x_max = cx + curr_w / 2
        y_min = cy - curr_h / 2
        y_max = cy + curr_h / 2

        return (x_min, x_max, y_min, y_max)

    def _spawn_ego_object(self):
        """
        Helper method to spawn ego object at origin
        """
        ego_id = 1

        # Add ego to objects list
        self.objects.append({"id": ego_id, "category": "ego", "name": "ego"})

        # Ego is always at origin (0,0) with horizontal heading
        for t in range(self.num_frames):
            self.object_positions[(ego_id, t)] = (0.0, 0.0)
            self.headings[(ego_id, t)] = 0  # Horizontal
            self.speeds[(ego_id, t)] = "not moving"  # Default speed

    def generate_random_scenario(
        self, num_objects, num_frames, include_ego=True, seed=None
    ):
        """
        Generate a random scenario by spawning objects and computing relations
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_frames = num_frames

        # Clear existing data
        self.objects = []
        self.speeds = {}
        self.headings = {}
        self.object_positions = {}

        # Calculate map size based on number of objects FIRST
        # Create a temporary object list to estimate dimensions
        temp_objects = []
        if include_ego:
            temp_objects.append({"category": "ego"})

        # Estimate categories for the new objects
        categories = ["car", "pedestrian", "bus", "truck"]
        weights = [0.25, 0.25, 0.25, 0.25]
        for _ in range(num_objects):
            cat = random.choices(categories, weights=weights)[0]
            temp_objects.append({"category": cat})

        # Calculate map size
        self.map_limit = Config.calculate_map_size(
            num_objects + (1 if include_ego else 0), temp_objects
        )
        Config.MAP_LIMIT = self.map_limit

        # Calculate Manhattan thresholds
        Config.MANHATTAN_THRESHOLDS = Config.calculate_thresholds(self.map_limit)

        print(f"\n{'=' * 60}")
        print(f"GENERATING RANDOM SCENARIO")
        print(f"{'=' * 60}")
        print(f"Number of objects: {num_objects}")
        print(f"Number of frames: {num_frames}")
        print(f"Include ego: {include_ego}")
        print(f"Map size: {self.map_limit} x {self.map_limit}")
        print(f"Distance thresholds:")
        print(f"  very close: ≤ {Config.MANHATTAN_THRESHOLDS['very close']}")
        print(
            f"  close: {Config.MANHATTAN_THRESHOLDS['very close']} < x ≤ {Config.MANHATTAN_THRESHOLDS['close']}"
        )
        print(
            f"  normal: {Config.MANHATTAN_THRESHOLDS['close']} < x ≤ {Config.MANHATTAN_THRESHOLDS['normal']}"
        )
        print(
            f"  far: {Config.MANHATTAN_THRESHOLDS['normal']} < x ≤ {Config.MANHATTAN_THRESHOLDS['far']}"
        )
        print(f"  very far: > {Config.MANHATTAN_THRESHOLDS['far']}")
        print(f"{'='=}")

        # Add ego if requested
        if include_ego:
            self._spawn_ego_object()

        # Spawn random objects with computed relations
        if num_objects > 0:
            self.spawn_random_objects_with_computed_relations(
                num_objects=num_objects, frames_list=list(range(num_frames))
            )

        return self.objects

    def spawn_random_objects_with_computed_relations(
        self, num_objects, frames_list=None
    ):
        """
        Spawn random objects and compute their relations based on positions
        """
        import random

        # If no frames specified, use all frames
        if frames_list is None:
            frames_list = list(range(self.num_frames))

        # Validate frames
        for f in frames_list:
            if f >= self.num_frames or f < 0:
                raise ValueError(f"Frame {f} out of range (0-{self.num_frames - 1})")

        # Find max ID
        max_id = 0
        for obj in self.objects:
            if obj["id"] > max_id:
                max_id = obj["id"]

        # Categories with weights (more balanced like GUI)
        categories = ["car", "pedestrian", "bus", "truck"]
        weights = [0.25, 0.25, 0.25, 0.25]

        new_ids = []
        new_objects = []

        # FIRST: Create all objects and generate positions
        for i in range(num_objects):
            new_id = max_id + i + 1
            new_ids.append(new_id)

            # Random category
            cat = random.choices(categories, weights=weights)[0]

            # Random heading (same for all frames)
            heading = random.choice([0, 1])

            # Get dimensions for boundary calculation
            dims = Config.DIMENSIONS.get(cat)
            print(f"Creating {cat} with dimensions: {dims}")

            # Calculate dimensions based on heading
            if heading == 1:  # Vertical
                width = dims[1]  # Width becomes the smaller dimension
                height = dims[0]  # Height becomes the larger dimension
            else:  # Horizontal
                width = dims[0]  # Width is the length
                height = dims[1]  # Height is the width

            half_width = width / 2
            half_height = height / 2

            # Calculate safe bounds (keep object fully inside map)
            min_x = -Config.MAP_LIMIT + half_width
            max_x = Config.MAP_LIMIT - half_width
            min_y = -Config.MAP_LIMIT + half_height
            max_y = Config.MAP_LIMIT - half_height

            # Store object info
            obj_info = {
                "id": new_id,
                "category": cat,
                "name": f"{cat}_{new_id}",
                "heading": heading,
            }
            new_objects.append(obj_info)

            # Generate random position
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            # Round for cleaner positions
            x = int(round(x, 1))
            y = int(round(y, 1))

            print(
                f"  Placed {cat} at ({x}, {y}) with heading {heading} (width={width}, height={height})"
            )

            # Store the same position for all frames
            for f_idx in frames_list:
                # Store position
                self.object_positions[(new_id, f_idx)] = (x, y)

                # Store heading (same for all frames)
                self.headings[(new_id, f_idx)] = heading

                # Store speed (random)
                speed = random.choice(list(Config.SPEED_LIMITS.keys()))
                self.speeds[(new_id, f_idx)] = speed

        # Add all new objects to self.objects
        self.objects.extend(new_objects)

        # Initialize matrices
        self.ra_matrix = [set() for _ in range(self.num_frames)]
        self.qdc_matrix = [set() for _ in range(self.num_frames)]

        # NOW: Compute relations for each frame based on actual positions
        for f_idx in frames_list:
            # Get all objects in this frame (including ego)
            frame_objects = []

            # Add all objects that have positions in this frame
            for obj in self.objects:
                obj_id = obj["id"]
                if (obj_id, f_idx) in self.object_positions:
                    pos = self.object_positions[(obj_id, f_idx)]
                    heading_val = self.headings.get((obj_id, f_idx), -1)
                    heading_deg = (
                        90 if heading_val == 1 else (0 if heading_val == 0 else -1)
                    )

                    frame_objects.append(
                        {
                            "id": obj_id,
                            "cat": obj["category"],
                            "x": pos[0],
                            "y": pos[1],
                            "heading": heading_deg,
                        }
                    )

            # Compute relations between all pairs in this frame
            for i, obj1 in enumerate(frame_objects):
                for obj2 in frame_objects[i + 1 :]:
                    # Compute RA relations from actual positions
                    bbox1 = self.get_bbox_logic(obj1)
                    bbox2 = self.get_bbox_logic(obj2)

                    ra_x = get_ra_string(bbox1[0], bbox1[1], bbox2[0], bbox2[1])
                    ra_y = get_ra_string(bbox1[2], bbox1[3], bbox2[2], bbox2[3])

                    # Add to ra_matrix
                    self.ra_matrix[f_idx].add((obj1["id"], obj2["id"], ra_x, ra_y))

                    # Compute QDC from actual positions
                    qdc = get_qdc_string(obj1, obj2)
                    self.qdc_matrix[f_idx].add((obj1["id"], obj2["id"], qdc))

        print(
            f"✓ Spawned {num_objects} objects in frames {frames_list} and computed relations from positions"
        )
        return new_ids

    def _map_category(self, name):
        """Maps dataset object names to categories."""
        n = str(name).lower().split("_")[0]
        if "ego" in n:
            return "ego"
        if "pedestrian" in n or "human" in n or "animal" in n:
            return "pedestrian"
        if "bus" in n:
            return "bus"
        if "truck" in n:
            return "truck"
        return "car"

    def load_from_csv(self, filepath, scene_name=None):

        df = pd.read_csv(filepath)
        df.columns=['0','frameidx','object_pair','1','2','3','distance_x','distance_y','speed_o1','speed_o2','4','5','6','7','8','RA','9','heading_o1','heading_o2','scene','actions','dynamics']
        df = df[df['frameidx'] % 20 == 0]
        df= df[df['object_pair'].apply(lambda x: 'ego' in x)]
        print(df.columns)
        if df.empty:
            raise ValueError("No rows found")
    
        # --------------------------------
        # Frames
        # --------------------------------
        frames = sorted(df["frameidx"].unique())
        self.num_frames = len(frames)
    
        frame_map = {f:i for i,f in enumerate(frames)}
    
        # --------------------------------
        # Extract objects
        # --------------------------------
        objects = set()
    
        for p in df["object_pair"]:
    
            o1,o2 = ast.literal_eval(p)
    
            objects.add(str(o1))
            objects.add(str(o2))
    
        obj_map = {name:i+1 for i,name in enumerate(sorted(objects))}
    
        self.objects = []
    
        for name,oid in obj_map.items():
    
            category = self._map_category(name)
    
            self.objects.append({
                "id": oid,
                "category": category,
                "name": name
            })
    
        # --------------------------------
        # Init matrices
        # --------------------------------
        self.ra_matrix = [set() for _ in range(self.num_frames)]
        self.qdc_matrix = [set() for _ in range(self.num_frames)]
    
        self.speeds = {}
        self.headings = {}
    
        # --------------------------------
        # RA mapping
        # --------------------------------
        ra_map = {
            "B":"Before",
            "BI":"After",
            "M":"Meets",
            "MI":"MetBy",
            "O":"Overlaps",
            "OI":"OverlappedBy",
            "D":"During",
            "DI":"Contains",
            "S":"Starts",
            "SI":"StartedBy",
            "F":"Finishes",
            "FI":"FinishedBy",
            "E":"Equals"
        }
    
        # --------------------------------
        # Heading parser
        # --------------------------------
        def parse_heading(h):
    
            h=str(h).lower()
    
            if "north" in h or "south" in h:
                return 1
    
            if "east" in h or "west" in h:
                return 0
    
            return -1
    
        # --------------------------------
        # Speed parser
        # --------------------------------
        def parse_speed(s):
    
            s=str(s).lower()
    
            if "not" in s:
                return "not moving"
    
            if "slow" in s:
                return "slow"
    
            if "fast" in s:
                return "fast"
            if "very fast" in s:
                return "fast"
    
            return "normal"
    
        # --------------------------------
        # Fill matrices
        # --------------------------------
        for _,row in df.iterrows():
    
            t = frame_map[row["frameidx"]]
    
            o1,o2 = ast.literal_eval(row["object_pair"])
    
            id1 = obj_map[str(o1)]
            id2 = obj_map[str(o2)]
    
            # ---------------- RA ----------------
            rx,ry = ast.literal_eval(row["RA"])

            rx = ra_map.get(rx[:-1],"")
            ry = ra_map.get(ry[:-1],"")
    
            self.ra_matrix[t].add((id1,id2,rx,ry))
    
            # ---------------- QDC ----------------
            dist = str(row["distance_x"]).lower()
    
            if "very" in dist:
                q="very close"
            elif "close" in dist:
                q="close"
            elif "far" in dist:
                q="far"
            elif 'very far' in dist:
                q='very far'
            else:
                q="normal"
    
            self.qdc_matrix[t].add((id1,id2,q))
    
            # ---------------- speeds ----------------
            self.speeds[(id1,t)] = parse_speed(row["speed_o1"])
            self.speeds[(id2,t)] = parse_speed(row["speed_o2"])
    
            # ---------------- headings ----------------
            self.headings[(id1,t)] = parse_heading(row["heading_o1"])
            self.headings[(id2,t)] = parse_heading(row["heading_o2"])
    
        print(f"Loaded {len(self.objects)} objects")
        print(f"{self.num_frames} frames")
        # --------------------------------
        # Initialize map size (missing!)
        # --------------------------------
        
        self.map_limit = Config.calculate_map_size(len(self.objects), self.objects)
        
        Config.MAP_LIMIT = self.map_limit
        
        Config.MANHATTAN_THRESHOLDS = Config.calculate_thresholds(self.map_limit)
        
        print(f"Map size initialized: {self.map_limit}")
        print("Distance thresholds:")
        for k,v in Config.MANHATTAN_THRESHOLDS.items():
            print(f"  {k}: {v}")
        return self.objects

    def _find_inconsistent_objects(
        self, obj_defs, ra_matrix, qdc_matrix, velocities, heading_map, n_frames
    ):
        """
        Find which objects are causing inconsistencies by trying to solve without them
        """
        inconsistent_objects = []

        # Find ego object
        ego_objects = [o for o in obj_defs if o["category"] == "ego"]
        if not ego_objects:
            print("Warning: No ego object found")
            return []

        ego = ego_objects[0]

        # First, try to solve without ego
        filtered_objs = [obj for obj in obj_defs if obj["id"] != ego["id"]]

        if not filtered_objs:
            print("No non-ego objects to check")
            return []

        # Create filtered matrices without ego
        filtered_ra = []
        filtered_qdc = []

        for t in range(n_frames):
            # Filter RA relations
            frame_ra = {
                rel
                for rel in ra_matrix[t]
                if rel[0] != ego["id"] and rel[1] != ego["id"]
            }
            filtered_ra.append(frame_ra)

            # Filter QDC relations
            frame_qdc = {
                rel
                for rel in qdc_matrix[t]
                if rel[0] != ego["id"] and rel[1] != ego["id"]
            }
            filtered_qdc.append(frame_qdc)

        # Filter velocities and headings
        filtered_velocities = {k: v for k, v in velocities.items() if k[0] != ego["id"]}
        filtered_headings = {k: v for k, v in heading_map.items() if k[0] != ego["id"]}

        # Try to solve without ego
        print("  Testing if non-ego objects are consistent among themselves...")
        solver = GlobalScenarioSolver(filtered_objs, n_frames)
        result, _ = solver.solve(
            filtered_ra,
            filtered_qdc,
            filtered_velocities,
            filtered_headings,
            solver_name="gecode",
            heuristic="default",
        )

        if not result:
            # Non-ego objects themselves are inconsistent
            print("  Non-ego objects are inconsistent among themselves")

            # Try removing each non-ego object one by one
            for obj_to_remove in filtered_objs:
                # Create filtered object list
                test_objs = [
                    obj for obj in filtered_objs if obj["id"] != obj_to_remove["id"]
                ]

                if not test_objs:
                    continue

                # Create filtered matrices
                test_ra = []
                test_qdc = []

                for t in range(n_frames):
                    # Filter RA relations
                    frame_ra = {
                        rel
                        for rel in ra_matrix[t]
                        if rel[0] != obj_to_remove["id"]
                        and rel[1] != obj_to_remove["id"]
                        and rel[0] != ego["id"]
                        and rel[1] != ego["id"]
                    }
                    test_ra.append(frame_ra)

                    # Filter QDC relations
                    frame_qdc = {
                        rel
                        for rel in qdc_matrix[t]
                        if rel[0] != obj_to_remove["id"]
                        and rel[1] != obj_to_remove["id"]
                        and rel[0] != ego["id"]
                        and rel[1] != ego["id"]
                    }
                    test_qdc.append(frame_qdc)

                # Filter velocities and headings
                test_velocities = {
                    k: v
                    for k, v in velocities.items()
                    if k[0] != obj_to_remove["id"] and k[0] != ego["id"]
                }
                test_headings = {
                    k: v
                    for k, v in heading_map.items()
                    if k[0] != obj_to_remove["id"] and k[0] != ego["id"]
                }

                # Try to solve without this object
                test_solver = GlobalScenarioSolver(test_objs, n_frames)
                test_result, _ = test_solver.solve(
                    test_ra,
                    test_qdc,
                    test_velocities,
                    test_headings,
                    solver_name="gecode",
                    heuristic="default",
                )

                if test_result:
                    # Removing this object makes the problem solvable
                    inconsistent_objects.append(obj_to_remove)
                    print(
                        f"  → Object {obj_to_remove['name']} (ID: {obj_to_remove['id']}) appears to be causing inconsistencies"
                    )
                    # Return immediately with the first found inconsistent object
                    # (we can add more sophisticated logic to find all, but this is efficient)
                    return inconsistent_objects

        return inconsistent_objects
    
    def solve(
        self, solver_name="gecode", heuristic="default", timeout=None, refinements=0
    ):
        """
        Solve the current scenario and return (result, stats)
        """
        all_results = []
        all_stats = []

        solver = GlobalScenarioSolver(self.objects, self.num_frames)

        full_ra_matrix = copy.deepcopy(self.ra_matrix)
        full_qdc_matrix = copy.deepcopy(self.qdc_matrix)

        def count_constraints(ra_matrix, qdc_matrix):
            ra_count = sum((len(r) for r in ra_matrix))
            qdc_count = sum((len(q) for q in qdc_matrix))
            return ra_count + qdc_count

        full_size = count_constraints(full_ra_matrix, full_qdc_matrix)

        def density(ra_matrix, qdc_matrix):
            current_size = count_constraints(ra_matrix, qdc_matrix)
            return (current_size / full_size) if full_size > 0 else 0

        constraint_queue = []

        # We fix the first and the final frame only.
        for t in range(1, self.num_frames - 1):
            # Enqueue all constraints for this frame before removal
            constraint_queue.extend(map(lambda x: (t, "ra", x), self.ra_matrix[t]))
            constraint_queue.extend(map(lambda x: (t, "qdc", x), self.qdc_matrix[t]))

            self.ra_matrix[t] = []
            self.qdc_matrix[t] = []

        # Shuffling to simulate random addition of constraints during refinement
        random.shuffle(constraint_queue)

        # We ramp density from 0% to 100% over the refinements.
        # That means first run is always almost empty
        densities = [0.0] + [((i + 1) / refinements) for i in range(refinements)]

        reduced_size = count_constraints(self.ra_matrix, self.qdc_matrix)

        result, stats = solver.solve_with_stats(
            self.ra_matrix,
            self.qdc_matrix,
            self.speeds,
            self.headings,
            solver_name=solver_name,
            heuristic=heuristic,
            timeout=timeout,
        )
        stats["density"] = density(self.ra_matrix, self.qdc_matrix)
        stats["refinement"] = 0
        all_results.append(result)
        all_stats.append(stats)
        # print(result)
        print(stats)

        print(
            f"Density: {reduced_size} constraints (reduced from {full_size}) ({stats['density']:.1%})"
        )

        if stats["status"] != "SOLVED":
            return all_results, all_stats

        for refine_iteration in range(1, refinements + 1):
            next_density = densities[refine_iteration]

            current_density = density(self.ra_matrix, self.qdc_matrix)

            if current_density < next_density:
                # Randomly add back constraints until we reach the next density level
                # Calculate number of missing constraints
                missing_constraints = full_size - count_constraints(
                    self.ra_matrix, self.qdc_matrix
                )
                target_constraints = int(full_size * next_density)
                num_constraints_to_add = target_constraints - (
                    full_size - missing_constraints
                )
            else:
                num_constraints_to_add = 1  # Just add one constraint if we're already above the target density

            constraints_to_add = constraint_queue[:num_constraints_to_add]
            constraint_queue = constraint_queue[num_constraints_to_add:]

            for t, constraint_type, constraint in constraints_to_add:
                if constraint_type == "ra":
                    self.ra_matrix[t].append(constraint)
                elif constraint_type == "qdc":
                    self.qdc_matrix[t].append(constraint)

            result2, stats2 = solver.solve_with_stats(
                self.ra_matrix,
                self.qdc_matrix,
                self.speeds,
                self.headings,
                solver_name=solver_name,
                heuristic=heuristic,
                timeout=timeout,
                prev_result=all_results[-1],
            )
            stats2["density"] = density(self.ra_matrix, self.qdc_matrix)
            stats2["refinement"] = refine_iteration
            all_results.append(result2)
            all_stats.append(stats2)
            # print(result2)
            print(stats2)

        return all_results, all_stats

    def export_to_csv(self, result, output_file):
        """
        Export solution to CSV file
        """
        data = []
        for t, frame in enumerate(result):
            for obj in frame:
                data.append(
                    {
                        "frame": t,
                        "object_id": obj["id"],
                        "category": obj["category"],
                        "x": obj["x"],
                        "y": obj["y"],
                        "width": obj["w"],
                        "height": obj["h"],
                        "x_min": obj["x_min"],
                        "x_max": obj["x_max"],
                        "y_min": obj["y_min"],
                        "y_max": obj["y_max"],
                    }
                )

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Exported to {output_file}")

    def save_qualitative_relations(self, output_file=None):
        """
        Save the qualitative relations (RA and QDC) for the current scenario to a CSV file
        """
        if self.ra_matrix is None or self.qdc_matrix is None:
            print("No relations to save")
            return False

        if output_file is None:
            output_file = (
                f"relations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        data = []
        for t in range(self.num_frames):
            # Get all RA relations for this frame
            ra_dict = {}
            for i, j, rx, ry in self.ra_matrix[t]:
                ra_dict[(min(i, j), max(i, j))] = (rx, ry)

            # Get all QDC relations for this frame
            qdc_dict = {}
            for i, j, q in self.qdc_matrix[t]:
                qdc_dict[(min(i, j), max(i, j))] = q

            # Combine all pairs
            all_pairs = set(ra_dict.keys()) | set(qdc_dict.keys())

            for obj1, obj2 in sorted(all_pairs):
                # Get object categories
                cat1 = next(
                    (obj["category"] for obj in self.objects if obj["id"] == obj1),
                    "unknown",
                )
                cat2 = next(
                    (obj["category"] for obj in self.objects if obj["id"] == obj2),
                    "unknown",
                )

                # Get RA relations
                rx, ry = ra_dict.get((obj1, obj2), (None, None))

                # Get QDC relation
                qdc = qdc_dict.get((obj1, obj2), None)

                # Get positions if available
                pos1 = self.object_positions.get((obj1, t), (None, None))
                pos2 = self.object_positions.get((obj2, t), (None, None))

                data.append(
                    {
                        "frame": t,
                        "object1_id": obj1,
                        "object1_category": cat1,
                        "object1_x": pos1[0] if pos1[0] is not None else "",
                        "object1_y": pos1[1] if pos1[1] is not None else "",
                        "object2_id": obj2,
                        "object2_category": cat2,
                        "object2_x": pos2[0] if pos2[0] is not None else "",
                        "object2_y": pos2[1] if pos2[1] is not None else "",
                        "ra_x": rx,
                        "ra_y": ry,
                        "qdc": qdc,
                    }
                )

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"✓ Qualitative relations saved to {output_file}")
        return True

    def save_solver_stats(
        self,
        stats,
        output_file=None,
        append=False,
        solver_name="gecode",
        heuristic="default",
        num_objects=None,
        num_frames=None,
        success=True,
    ):
        """
        Save solver statistics to a CSV file

        Args:
            stats: Statistics dictionary from solver
            output_file: Output CSV file path
            append: Whether to append to existing file
            solver_name: Name of solver used
            heuristic: Heuristic used
            num_objects: Number of objects (if None, uses len(self.objects))
            num_frames: Number of frames (if None, uses self.num_frames)
            success: Whether a solution was found
        """
        if output_file is None:
            output_file = (
                f"solver_stats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        if num_objects is None:
            num_objects = len(self.objects)
        if num_frames is None:
            num_frames = self.num_frames

        # Prepare statistics data
        stats_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "solver": solver_name,
            "heuristic": heuristic,
            "num_objects": num_objects,
            "num_frames": num_frames,
            "success": success,
            "refinement": stats.get("refinement", 0),
            "density": stats.get("density", None),
            "first_solution_seconds": stats.get("first_solution_time", None),
            "solve_time_seconds": stats.get("time", None),
            "flat_time_seconds": stats.get("flatTime", None),
            "init_time_seconds": stats.get("initTime", None),
            "solve_time_ms": stats.get("solveTime", None),
            "nodes": stats.get("nodes", None),
            "failures": stats.get("failures", None),
            "restarts": stats.get("restarts", None),
            "variables": stats.get("variables", None),
            "propagators": stats.get("propagators", None),
            "propagations": stats.get("propagations", None),
            "nogoods": stats.get("nogoods", None),
            "peak_depth": stats.get("peakDepth", None),
            "memory_used_mb": stats.get("memory", None),
            "status": stats.get("status", "UNKNOWN"),
        }

        # Create DataFrame
        df = pd.DataFrame([stats_data])

        # Save or append
        if append and os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"✓ Appended solver statistics to {output_file}")
        else:
            df.to_csv(output_file, index=False)
            print(f"✓ Solver statistics saved to {output_file}")

    def save_solver_stats_intermediate(
        self,
        stats,
        output_file=None,
        append=False,
        solver_name="gecode",
        heuristic="default",
        num_objects=None,
        num_frames=None,
        success=True,
    ):
        """
        Save solver statistics to a CSV file

        Args:
            stats: Statistics dictionary from solver
            output_file: Output CSV file path
            append: Whether to append to existing file
            solver_name: Name of solver used
            heuristic: Heuristic used
            num_objects: Number of objects (if None, uses len(self.objects))
            num_frames: Number of frames (if None, uses self.num_frames)
            success: Whether a solution was found
        """

        intermediate_solutions = stats.get("intermediate_solutions", [])

        if len(intermediate_solutions) == 0:
            return

        if output_file is None:
            output_file = (
                f"solver_stats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        if num_objects is None:
            num_objects = len(self.objects)
        if num_frames is None:
            num_frames = self.num_frames

        stats_data = []
        for solution in intermediate_solutions:
            stats_data.append(
                {
                    "solver": solver_name,
                    "heuristic": heuristic,
                    "refinement": stats.get("refinement", 0),
                    **solution,
                }
            )

        # Create DataFrame
        df = pd.DataFrame(stats_data)

        # Save or append
        if append and os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"✓ Appended solver statistics to {output_file}")
        else:
            df.to_csv(output_file, index=False)
            print(f"✓ Solver statistics saved to {output_file}")

    def print_summary(self, result=None):
        """
        Print summary of the scenario
        """
        print("\n" + "=" * 60)
        print("SCENARIO SUMMARY")
        print("=" * 60)
        print(f"Number of frames: {self.num_frames}")
        print(f"Number of objects: {len(self.objects)}")
        print("\nObjects:")
        for obj in self.objects:
            heading_str = {0: "Horizontal", 1: "Vertical", -1: "Auto"}.get(
                obj.get("heading", -1), "Unknown"
            )
            print(
                f"  - ID {obj['id']}: {obj['category']} ({obj.get('name', 'unknown')}) [Heading: {heading_str}]"
            )

        print("\nConstraints:")
        for t in range(self.num_frames):
            print(f"  Frame {t}:")
            if self.ra_matrix and t < len(self.ra_matrix):
                for i, j, rx, ry in list(self.ra_matrix[t])[:5]:  # Show first 5 only
                    print(f"    Object {i} <-> {j}: RA=({rx}, {ry})")
                if len(self.ra_matrix[t]) > 5:
                    print(f"    ... and {len(self.ra_matrix[t]) - 5} more RA relations")
            if self.qdc_matrix and t < len(self.qdc_matrix):
                for i, j, q in list(self.qdc_matrix[t])[:5]:  # Show first 5 only
                    print(f"    Object {i} <-> {j}: QDC={q}")
                if len(self.qdc_matrix[t]) > 5:
                    print(
                        f"    ... and {len(self.qdc_matrix[t]) - 5} more QDC relations"
                    )

        if result:
            print("\n" + "=" * 60)
            print("SOLUTION SUMMARY")
            print("=" * 60)
            for t, frame in enumerate(result):
                print(f"Frame {t}:")
                for obj in frame[:5]:  # Show first 5 objects only
                    print(
                        f"  Object {obj['id']} ({obj['category']}): "
                        f"pos=({obj['x']:.1f}, {obj['y']:.1f}), "
                        f"heading={obj['heading']}°"
                    )
                if len(frame) > 5:
                    print(f"  ... and {len(frame) - 5} more objects")

    # PLOTTING METHODS
    def plot_scenario(self, result=None, output_dir=None, show_plots=True, dpi=100):
        """
        Plot the scenario frames as matplotlib figures
        If result is None, plot the generated random scenario using object_positions
        """
        # Determine what to plot
        if result is not None:
            plot_data = result
            title_prefix = "Solved"
            print(f"Plotting solved scenario with {len(plot_data)} frames")
        else:
            print("No solution found - plotting generated random scenario")
            plot_data = self._construct_plot_data_with_dimensions()
            title_prefix = "Generated Random Scenario"

        if not plot_data:
            print("No data to plot")
            return []

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        figures = []

        # Create a figure for each frame
        for t, frame_data in enumerate(plot_data):
            # Create figure
            fig = Figure(figsize=(10, 10), dpi=dpi)
            ax = fig.add_subplot(111)

            # Set limits with some padding
            padding = Config.MAP_LIMIT * 0.1
            ax.set_xlim(-Config.MAP_LIMIT - padding, Config.MAP_LIMIT + padding)
            ax.set_ylim(-Config.MAP_LIMIT - padding, Config.MAP_LIMIT + padding)

            # Add grid
            ax.grid(True, alpha=0.3, linestyle="--")

            # Add axis lines
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

            # Set title
            ax.set_title(f"{title_prefix} - Frame {t}", fontsize=14, fontweight="bold")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            # Plot each object
            for obj in frame_data:
                # Get color based on category
                color = Config.COLORS.get(obj.get("category", obj.get("cat")), "gray")

                # Get bounding box coordinates using w and h if available
                if "w" in obj and "h" in obj and "x" in obj and "y" in obj:
                    # Use w, h and center (x,y) to calculate bounding box
                    x_center = obj["x"]
                    y_center = obj["y"]
                    width = obj["w"]
                    height = obj["h"]
                    x_min = x_center - width / 2
                    x_max = x_center + width / 2
                    y_min = y_center - height / 2
                    y_max = y_center + height / 2

                    obj_id = obj["id"]
                    category = obj.get("category", obj.get("cat", "unknown"))
                    heading = obj.get("heading", 0)

                    # Create rectangle patch
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        width,
                        height,
                        linewidth=2,
                        edgecolor="black",
                        facecolor=color,
                        alpha=0.8,
                    )
                    ax.add_patch(rect)

                    # Add object ID and category text
                    center_x = x_center
                    center_y = y_center

                    # Add text background for better readability
                    bbox_props = dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="black",
                    )
                    ax.text(
                        center_x,
                        center_y,
                        f"{category[:3].upper()}{obj_id}",
                        ha="center",
                        va="top",
                        fontsize=9,
                        fontweight="bold",
                    )

                    # Add heading indicator if available and not -1
                    if heading != -1 and heading != "unknown":
                        # Convert heading to degrees if needed
                        if isinstance(heading, str):
                            heading_val = 90 if heading.lower() == "vertical" else 0
                        else:
                            heading_val = heading

                        # Draw a small line to indicate heading direction
                        line_length = min(width, height) * 0.6
                        if heading_val == 0:  # Horizontal (East)
                            dx, dy = line_length, 0
                        elif heading_val == 90:  # Vertical (North)
                            dx, dy = 0, line_length
                        else:  # Unknown heading - no indicator
                            dx, dy = 0, 0

                        if dx != 0 or dy != 0:
                            ax.arrow(
                                center_x,
                                center_y,
                                dx,
                                dy,
                                head_width=line_length * 0.3,
                                head_length=line_length * 0.3,
                                fc="black",
                                ec="black",
                                alpha=0.7,
                            )

                elif "x_min" in obj:  # Solved data format with explicit bounds
                    x_min, x_max = obj["x_min"], obj["x_max"]
                    y_min, y_max = obj["y_min"], obj["y_max"]
                    width = x_max - x_min
                    height = y_max - y_min
                    obj_id = obj["id"]
                    category = obj["category"]
                    heading = obj.get("heading", 0)

                    # Create rectangle patch
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        width,
                        height,
                        linewidth=2,
                        edgecolor="black",
                        facecolor=color,
                        alpha=0.8,
                    )
                    ax.add_patch(rect)

                    # Add object ID and category text
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2

                    bbox_props = dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="black",
                    )
                    ax.text(
                        center_x,
                        center_y,
                        f"{category[:3].upper()}{obj_id}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        bbox=bbox_props,
                    )

            # Add legend for categories
            legend_elements = [
                Patch(facecolor=Config.COLORS.get("ego", "#2c3e50"), label="Ego"),
                Patch(facecolor=Config.COLORS.get("car", "#3498db"), label="Car"),
                Patch(
                    facecolor=Config.COLORS.get("pedestrian", "#e67e22"),
                    label="Pedestrian",
                ),
                Patch(facecolor=Config.COLORS.get("bus", "#27ae60"), label="Bus"),
                Patch(facecolor=Config.COLORS.get("truck", "#8e44ad"), label="Truck"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

            # Add frame information
            ax.text(
                0.02,
                0.98,
                f"Frame: {t}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Save if output directory specified
            if output_dir:
                filename = os.path.join(output_dir, f"frame_{t:03d}.png")
                fig.savefig(filename, dpi=dpi, bbox_inches="tight")
                print(f"Saved: {filename}")

            figures.append(fig)

        # Show plots if requested
        if show_plots:
            plt.show()
        else:
            # Close figures to free memory
            for fig in figures:
                plt.close(fig)

        return figures
    def reconstruct_and_compare(self, result, output_file=None):

        """
        Reconstruct qualitative relations from solver output
        and compare them with the original RA/QDC constraints.
        """
    
        reconstructed_ra = [set() for _ in range(self.num_frames)]
        reconstructed_qdc = [set() for _ in range(self.num_frames)]
    
        mismatches = []
    
        for t, frame in enumerate(result):
    
            # build objects list for relation computation
            objs = []
    
            for o in frame:
                objs.append({
                    "id": o["id"],
                    "cat": o["category"],
                    "x": o["x"],
                    "y": o["y"],
                    "heading": self.headings.get((o["id"], t), -1)
                })
    
            # compute relations
            for i, o1 in enumerate(objs):
                for o2 in objs[i+1:]:
                    if o2['cat']=='ego':
                        bbox1 = self.get_bbox_logic(o1)
                        bbox2 = self.get_bbox_logic(o2)
        
                        ra_x = get_ra_string(bbox1[0], bbox1[1], bbox2[0], bbox2[1])
                        ra_y = get_ra_string(bbox1[2], bbox1[3], bbox2[2], bbox2[3])
        
                        qdc = get_qdc_string(o1, o2)
        
                        reconstructed_ra[t].add((o1["id"], o2["id"], ra_x, ra_y))
                        reconstructed_qdc[t].add((o1["id"], o2["id"], qdc))
                        # check mismatch
                        if (o1["id"], o2["id"], ra_x, ra_y) not in self.ra_matrix[t]:
                            mismatches.append({
                                "frame": t,
                                "type": "RA",
                                "pair": (o1["id"], o2["id"]),
                                "expected": next(
                                    (r for r in self.ra_matrix[t]
                                     if r[0]==o1["id"] and r[1]==o2["id"]),
                                    None
                                ),
                                "reconstructed": (ra_x, ra_y)
                            })
        
                        if (o1["id"], o2["id"], qdc) not in self.qdc_matrix[t]:
                            mismatches.append({
                                "frame": t,
                                "type": "QDC",
                                "pair": (o1["id"], o2["id"]),
                                "expected": next(
                                    (r for r in self.qdc_matrix[t]
                                     if r[0]==o1["id"] and r[1]==o2["id"]),
                                    None
                                ),
                                "reconstructed": qdc
                            })
    
        print("\n==============================")
        print("RECONSTRUCTION CHECK")
        print("==============================")
    
        if len(mismatches) == 0:
            print("✓ Perfect reconstruction: solution matches qualitative graph")
        else:
            print(f"⚠ {len(mismatches)} mismatches found")
    
            for m in mismatches[:20]:
                print(m)
    
        # optionally save comparison
        if output_file:
    
            rows = []
    
            for m in mismatches:
                rows.append({
                    "frame": m["frame"],
                    "type": m["type"],
                    "object1": m["pair"][0],
                    "object2": m["pair"][1],
                    "expected": m["expected"],
                    "reconstructed": m["reconstructed"]
                })
    
            pd.DataFrame(rows).to_csv(output_file, index=False)
            print(f"Mismatch report saved to {output_file}")
    
        return mismatches
    def _construct_plot_data_with_dimensions(self):
        """
        Construct plot data for the generated random scenario including dimensions
        """
        if not self.objects:
            return []

        plot_data = []
        for t in range(self.num_frames):
            frame_objects = []
            for obj in self.objects:
                obj_id = obj["id"]

                if (obj_id, t) in self.object_positions:
                    pos = self.object_positions[(obj_id, t)]
                    heading_val = self.headings.get((obj_id, t), 0)

                    # Get dimensions from Config
                    dims = Config.DIMENSIONS.get(obj["category"], (20, 10))

                    # Calculate width and height based on heading
                    if heading_val == 1:  # Vertical
                        width = dims[1]  # Width becomes the smaller dimension
                        height = dims[0]  # Height becomes the larger dimension
                    else:  # Horizontal
                        width = dims[0]  # Width is the length
                        height = dims[1]  # Height is the width

                    # For pedestrians, ensure square shape
                    if obj["category"] == "pedestrian":
                        width, height = dims[0], dims[1]  # Both are 5

                    heading_deg = 90 if heading_val == 1 else 0

                    frame_objects.append(
                        {
                            "id": obj_id,
                            "category": obj["category"],
                            "x": pos[0],
                            "y": pos[1],
                            "w": width,
                            "h": height,
                            "heading": heading_deg,
                        }
                    )

            if frame_objects:
                plot_data.append(frame_objects)

        return plot_data
    def plot_cactus(self, df):

        plt.figure(figsize=(8,6))
    
        groups = df.groupby("objects")
    
        for obj, g in groups:
    
            g = g.sort_values("frames")
    
            plt.plot(
                g["frames"],
                g["time"],
                marker="o",
                label=f"{obj} objects"
            )
    
        plt.xlabel("Number of Frames")
        plt.ylabel("Solve Time (seconds)")
        plt.title("Scenario Reconstruction Performance")
    
        plt.legend()
    
        plt.grid(True)
    
        plt.tight_layout()
    
        plt.savefig("cactus_plot.png", dpi=300)
    
        print("Plot saved to cactus_plot.png")
    
        #plt.show()
    def run_folder_experiment(self, folder, solver="gecode", heuristic="default"):

        import glob
        import time
    
        results = []
    
        files = glob.glob(os.path.join(folder, "*.csv"))
    
        print(f"\nRunning experiment on {len(files)} scenes\n")
    
        for f in sorted(files):
    
            print("=================================")
            print("Scene:", f)
    
            start = time.time()
    
            self.load_from_csv(f)
    
            result, stats = self.solve(
                solver_name=solver,
                heuristic=heuristic,
                refinements=0
            )
    
            elapsed = stats[-1]["time"] if stats else None
    
            num_objects = len(self.objects)
            num_frames = self.num_frames
    
            results.append({
                "scene": os.path.basename(f),
                "objects": num_objects,
                "frames": num_frames,
                "time": elapsed
            })
    
            print(f"Objects: {num_objects}  Frames: {num_frames}  Time: {elapsed}")
    
            df = pd.DataFrame(results)
    
            df.to_csv("experiment_results_pandaset.csv", index=False)
    
            print("\nResults saved to experiment_results.csv")
    
            self.plot_cactus(df)
    def plot_all_frames(self, result=None, output_dir=None):
        """Plot all frames, using result if available, otherwise generated scenario"""
        return self.plot_scenario(result=result, output_dir=output_dir, show_plots=True)

    def create_animation(self, result=None, output_file=None, fps=2):
        """Create animation, using result if available, otherwise generated scenario"""
        import matplotlib.animation as animation

        if result is not None:
            plot_data = result
            title_prefix = "Solved"
        else:
            print("No solution found - creating animation of generated random scenario")
            plot_data = self._construct_plot_data_with_dimensions()
            title_prefix = "Generated Random Scenario"

        if not plot_data:
            print("No data to animate")
            return None

        fig = Figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        def update_frame(t):
            ax.clear()
            padding = Config.MAP_LIMIT * 0.1
            ax.set_xlim(-Config.MAP_LIMIT - padding, Config.MAP_LIMIT + padding)
            ax.set_ylim(-Config.MAP_LIMIT - padding, Config.MAP_LIMIT + padding)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
            ax.set_title(f"{title_prefix} - Frame {t}", fontsize=14, fontweight="bold")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            frame_data = plot_data[t]
            for obj in frame_data:
                color = Config.COLORS.get(
                    obj.get("category", obj.get("cat", "car")), "gray"
                )

                if "w" in obj and "h" in obj and "x" in obj and "y" in obj:
                    x_center = obj["x"]
                    y_center = obj["y"]
                    width = obj["w"]
                    height = obj["h"]

                    x_min = x_center - width / 2
                    y_min = y_center - height / 2

                    rect = patches.Rectangle(
                        (x_min, y_min),
                        width,
                        height,
                        linewidth=2,
                        edgecolor="black",
                        facecolor=color,
                        alpha=0.8,
                    )
                    ax.add_patch(rect)

                    center_x = x_center
                    center_y = y_center

                    bbox_props = dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.2,
                    )
                    ax.text(
                        center_x,
                        center_y,
                        f"{obj['category'][:3].upper()}{obj['id']}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                       # bbox=bbox_props,
                    )

            # Add legend
            legend_elements = [
                Patch(facecolor=Config.COLORS.get("ego", "#2c3e50"), label="Ego"),
                Patch(facecolor=Config.COLORS.get("car", "#3498db"), label="Car"),
                Patch(
                    facecolor=Config.COLORS.get("pedestrian", "#e67e22"),
                    label="Pedestrian",
                ),
                Patch(facecolor=Config.COLORS.get("bus", "#27ae60"), label="Bus"),
                Patch(facecolor=Config.COLORS.get("truck", "#8e44ad"), label="Truck"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        ani = animation.FuncAnimation(
            fig, update_frame, frames=len(plot_data), interval=1000 // fps, repeat=True
        )

        if output_file:
            if output_file.endswith(".gif"):
                ani.save(output_file, writer="pillow", fps=fps)
            else:
                ani.save(output_file, writer="ffmpeg", fps=fps)
            print(f"Animation saved to {output_file}")

        plt.show()
        return ani


# ==========================================
# 5. MAIN CLI ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Qualitative Scenario Designer - Command Line Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate random scenario with 5 objects over 10 frames
  python ScenaGen_CLI.py --generate --num-objects 5 --num-frames 10 --output result.csv
  
  # Generate and plot
  python ScenaGen_CLI.py --generate --num-objects 3 --num-frames 5 --plot
  
  # Generate and create animation
  python ScenaGen_CLI.py --generate --num-objects 3 --num-frames 5 --animate
  
  # Import from CSV and solve with inconsistency removal
  python ScenaGen_CLI.py --import-file data.csv --scene scene_001 --solver gecode --heuristic frame-wise --output result.csv
  
  # Compare different heuristics (append stats to same file)
  python ScenaGen_CLI.py --generate --num-objects 5 --num-frames 5 --seed 42 --solver gecode --heuristic default --stats-output comparison.csv --append-stats
  python ScenaGen_CLI.py --generate --num-objects 5 --num-frames 5 --seed 42 --solver gecode --heuristic domoverwdeg --stats-output comparison.csv --append-stats
        """,
    )

    # Input options
    #input_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--generate", action="store_false", help="Generate random scenario"
    )
    parser.add_argument(
        "--import-file",default="/home/nassim/local_nuscenes/SmartForce_scene-0002.csv", type=str, metavar="FILE", help="Import scenario from CSV file"
    )

    # Generation parameters
    parser.add_argument(
        "--num-objects",
        type=int,
        default=10,
        help="Number of objects to generate (default: 10)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=10, help="Number of time frames (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=11, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-ego",
        action="store_true",
        help="Do not include ego vehicle (default: include ego)",
    )

    # Import parameters
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name to import from CSV (if not specified, uses all data)",
    )

    # Solver options
    parser.add_argument(
        "--solver",
        type=str,
        default="cp-sat",
        help="MiniZinc solver to use (default: gecode)",
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        default="default",
        choices=[
            "default",
            "first-fail",
            "smallest",
            "largest",
            "frame-wise",
            "pair-wise",
            "domoverwdeg",
        ],
        help="Search heuristic (default: default)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Solver timeout in seconds"
    )
    parser.add_argument(
        "--import-folder",
        type=str,
        default=None,
        help="Run solver on every CSV scene in a folder"
    )
    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="scenario_result.csv",
        help="Output CSV file for solution (default: scenario_result.csv)",
    )
    parser.add_argument(
        "--relations-output",
        type=str,
        default=None,
        help="Output CSV file for qualitative relations (if not specified, uses [output]_relations.csv)",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Output CSV file for solver statistics (if not specified, uses [output]_stats.csv)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "--append-stats",
        action="store_true",
        help="Append statistics to existing stats file instead of overwriting",
    )

    # Plotting options
    parser.add_argument("--plot", action="store_true", help="Plot the scenario frames")
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save plots (if not specified, plots are only shown)",
    )
    parser.add_argument(
        "--animate", action="store_true", help="Create animation of the solution"
    )
    parser.add_argument(
        "--animate-output",
        type=str,
        default="scenario_animation.gif",
        help="Output file for animation (default: scenario_animation.gif)",
    )
    parser.add_argument(
        "--refinements",
        type=int,
        default=0,
        help="Number of refinement iterations (default: 0)",
    )

    args = parser.parse_args()
    args.FILE= 'SmartForce_scene-0001.csv'
    # Create designer
    designer = CLIScenarioDesigner()
    args.generate=False
    # Load or generate scenario
    if args.import_folder:

        designer.run_folder_experiment(
            args.import_folder,
            solver=args.solver,
            heuristic=args.heuristic
        )
    
        return
    if args.generate:
        print(
            f"Generating random scenario with {args.num_objects} objects over {args.num_frames} frames..."
        )
        designer.generate_random_scenario(
            num_objects=args.num_objects,
            num_frames=args.num_frames,
            include_ego=not args.no_ego,
            seed=args.seed,
        )
    elif args.import_file:
        print(
            f"Importing from {args.import_file}"
            + (f" (scene: {args.scene})" if args.scene else "")
        )
        designer.load_from_csv(args.import_file, args.scene)

    # Save qualitative relations if requested
    relations_output = args.relations_output
    if relations_output is None and args.output:
        # Default: use same base name as output with _relations suffix
        base = args.output.rsplit(".", 1)[0]
        relations_output = (
            f"{base}_o{args.num_objects}_f{args.num_frames}_s{args.seed}_relations.csv"
        )

    if relations_output:
        print("\nSaving qualitative relations...")
        designer.save_qualitative_relations(relations_output)

    # Print summary
    if args.verbose:
        designer.print_summary()

    heuristics_to_test = ["frame-wise"]
    solvers = ["cp-sat"]
    # Solve with statistics
    for s in solvers:
        for h in heuristics_to_test:
            print(f"\nSolving with {s} solver, heuristic: {h}...")
            all_result, all_stats = designer.solve(
                solver_name=s,
                heuristic=h,
                timeout=args.timeout,
                refinements=args.refinements,
            )



            for result, stats in zip(all_result, all_stats):
                # Save solver statistics
                stats_output = args.stats_output
                if stats_output is None and args.output:
                    base = args.output.rsplit(".", 1)[0]
                    stats_output = f"{base}_o{args.num_objects}_f{args.num_frames}_s{args.seed}_{s}_{h}_stats.csv"
                    interm_stats_output = f"{base}_o{args.num_objects}_f{args.num_frames}_s{args.seed}_{s}_{h}_intermediate.csv"

                if stats_output:
                    designer.save_solver_stats(
                        stats=stats,
                        output_file=stats_output,
                        append=args.append_stats,
                        solver_name=s,
                        heuristic=h,
                        num_objects=len(designer.objects),
                        num_frames=designer.num_frames,
                        success=(result is not None),
                    )
                    designer.save_solver_stats_intermediate(
                        stats=stats,
                        output_file=interm_stats_output,
                        append=args.append_stats,
                        solver_name=s,
                        heuristic=h,
                        num_objects=len(designer.objects),
                        num_frames=designer.num_frames,
                        success=(result is not None),
                    )

                # In main function, replace the plotting section:

                if result:
                    print("\n✓ Solution found!")
                    print(f"  Solve time: {stats.get('time', 0)} seconds")
                    if "nodes" in stats:
                        print(f"  Search nodes: {stats['nodes']}")
                    if "failures" in stats:
                        print(f"  Failures: {stats['failures']}")

                    
                    if args.verbose:
                        designer.print_summary(result)

                    # Export to CSV
                    refinement_iteration = stats.get("refinement", 0)
                    base = args.output.rsplit(".", 1)[0]
                    result_output = f"{base}_o{args.num_objects}_f{args.num_frames}_s{args.seed}_{s}_{h}_r{refinement_iteration}.csv"

                    designer.export_to_csv(result, result_output)
                    designer.reconstruct_and_compare(
                        result,
                        output_file="reconstruction_check.csv"
                    )
                    print(f"\nResult saved to {result_output}")

                    # Plot if requested
                    if args.plot:
                        print("\nPlotting solved scenario frames...")
                        designer.plot_scenario(
                            result=result, output_dir=args.plot_dir, show_plots=True
                        )

                    # Animate if requested
                    if args.animate:
                        print("\nCreating animation of solved scenario...")
                        designer.create_animation(
                            result=result, output_file=args.animate_output
                        )
                else:
                    print("\n✗ No solution found!")

                    # Still plot the generated random scenario if requested
                    if args.plot:
                        print(
                            "\nPlotting generated random scenario (no solution found)..."
                        )
                        designer.plot_scenario(
                            result=None,  # This will trigger plotting of generated scenario
                            output_dir=args.plot_dir,
                            show_plots=True,
                        )

                    # Animate generated scenario if requested
                    if args.animate:
                        print("\nCreating animation of generated random scenario...")
                        designer.create_animation(
                            result=None,  # This will trigger animation of generated scenario
                            output_file=args.animate_output,
                        )


if __name__ == "__main__":
    main()
