#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import math
import re
from pathlib import Path
import seaborn as sns


directory = Path("./results_generated_free_search_r5")

#%%

# Intermediate solutions

dfs = []
for f in directory.glob("*_intermediate.csv"):
    num_objs, num_frames, seed = map(int, re.search(r"o(\d+)_f(\d+)_s(\d+)_.*", f.name).groups())
    curdf = pd.read_csv(f)
    curdf["num_objs"] = num_objs
    curdf["num_frames"] = num_frames
    curdf["seed"] = seed
    dfs.append(curdf)

df = pd.concat(dfs)
df["instance"] = df.apply(lambda row: f"o{row['num_objs']}_f{row['num_frames']}_s{row['seed']}", axis=1)
df["solution_number"] = (
    df.groupby(["num_objs", "num_frames", "seed", "refinement"])["time"]
    .rank(method="first", ascending=True)
    .astype(int)
)
df

# %%
sns.lineplot(df, x="solution_number", y="time", hue="refinement", units="instance", estimator=None)
# %%

group_cols =["num_objs", "num_frames", "seed", "refinement"]

# 1. Sort by the grouping columns AND time (ascending)
# This ensures the 'first' row is rank 1, and the 'last' row is the maximum rank
df_sorted = df.sort_values(by=group_cols + ["time"])

# 2. Isolate the exact first and last rows per group, and set the index to your groups
first_rows = df_sorted.drop_duplicates(subset=group_cols, keep="first").set_index(group_cols)
last_rows  = df_sorted.drop_duplicates(subset=group_cols, keep="last").set_index(group_cols)

# 3. Calculate the ratios (Last row divided by First row)
# Note: If you want First / Last, simply reverse the division
ratios = last_rows[["time", "objective"]] / first_rows[["time", "objective"]]
ratios["time_diff"] = last_rows[["time"]] - first_rows[["time"]]


# 4. Keep only the groups that have more than 1 entry
group_sizes = df.groupby(group_cols).size()
ratios = ratios[group_sizes > 1]
# diffs = diffs[group_sizes > 1]

#  = diffs["time"]


# 5. (Optional) Rename columns for clarity
ratios = ratios.rename(columns={
    "time": "time_ratio", 
    "objective": "objective_ratio"
})

print(ratios)
