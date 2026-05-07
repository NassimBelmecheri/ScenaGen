"""JAX-accelerated Genetic Algorithm solver.

Vectorises fitness evaluation across the entire population using
jax.jit + jax.vmap.

Extracted from GUI_ScenaGen_GA.py for reuse by CLI and benchmark tools.
"""

import time as _time
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom


# ============================================================
# Module-level configuration (mutable via configure())
# ============================================================

MAP_LIMIT = 500

# Default thresholds — squared Euclidean, matching GUI Config.THRESHOLDS
THRESHOLDS = {
    "very close": (500 // 10) ** 2,   # 2500
    "close": (500 // 5) ** 2,         # 10000
    "normal": (500 // 2) ** 2,        # 62500
    "far": 500 ** 2,                  # 250000
    "very far": (2 * 500) ** 2,       # 1000000
}

# Size-rank per category (matching MiniZinc solver)
SIZE_RANK = {
    "pedestrian": 1,
    "car": 2,
    "ego": 2,
    "bus": 3,
    "truck": 3,
    "trafficcone": 1,
    "barrier": 1,
}


# ============================================================
# 1. Constants & code mappings
# ============================================================

ALLEN_NAMES = [
    "Before",
    "After",
    "Meets",
    "MetBy",
    "Overlaps",
    "OverlappedBy",
    "Starts",
    "StartedBy",
    "Finishes",
    "FinishedBy",
    "During",
    "Contains",
    "Equals",
]
ALLEN_TO_CODE = {name: i for i, name in enumerate(ALLEN_NAMES)}

QDC_NAMES = ["very close", "close", "normal", "far", "very far"]
QDC_TO_CODE = {name: i for i, name in enumerate(QDC_NAMES)}

# QDC thresholds — no relaxation (matching MiniZinc solver exactly)
_RELAXATION = 1.0


def _compute_qdc_thresholds():
    """Recompute the JAX QDC threshold arrays from the current THRESHOLDS.

    Returns (upper, lower) where:
      upper: (4,) — max gap for codes 0-3 (very_far has no upper bound)
      lower: (5,) — min gap for codes 0-4 (very_close has no lower bound)
    """
    upper = jnp.array(
        [
            int(THRESHOLDS["very close"] * _RELAXATION),
            int(THRESHOLDS["close"] * _RELAXATION),
            int(THRESHOLDS["normal"] * _RELAXATION),
            int(THRESHOLDS["far"] * _RELAXATION),
        ],
        dtype=jnp.float32,
    )
    lower = jnp.array(
        [
            -1.0,  # very_close: no lower bound (always satisfied)
            int(THRESHOLDS["very close"] * _RELAXATION),
            int(THRESHOLDS["close"] * _RELAXATION),
            int(THRESHOLDS["normal"] * _RELAXATION),
            int(THRESHOLDS["far"] * _RELAXATION),
        ],
        dtype=jnp.float32,
    )
    return upper, lower


QDC_UPPER_THRESHOLDS, QDC_LOWER_THRESHOLDS = _compute_qdc_thresholds()


def configure(map_limit=None, thresholds=None, dimensions=None):
    """Reconfigure module globals.

    Parameters
    ----------
    map_limit : int, optional
        New MAP_LIMIT value.
    thresholds : dict, optional
        Keys: "very close", "close", "normal", "far", "very far".
        Values are the raw threshold numbers (same unit as the existing
        THRESHOLDS dict — squared-Euclidean for the GUI, Manhattan for
        the CLI).
    dimensions : dict, optional
        Ignored (kept for backward compatibility). Bbox is now square,
        determined by size gene.
    """
    global MAP_LIMIT, THRESHOLDS, QDC_UPPER_THRESHOLDS, QDC_LOWER_THRESHOLDS

    if map_limit is not None:
        MAP_LIMIT = map_limit
    if thresholds is not None:
        THRESHOLDS = dict(thresholds)

    # Always recompute the JAX arrays
    QDC_UPPER_THRESHOLDS, QDC_LOWER_THRESHOLDS = _compute_qdc_thresholds()


# ============================================================
# 2. ConstraintData — pre-encoded JAX arrays only
# ============================================================
# num_objs, num_frames, ego_idx are NOT stored here because JAX
# pytree flattening converts Python ints into traced scalars,
# breaking reshape/slice operations that need static shapes.


class ConstraintData(NamedTuple):
    fixed_mask: jnp.ndarray  # (G,) bool
    fixed_vals: jnp.ndarray  # (G,) float32
    xy_mask: jnp.ndarray  # (G,) bool — all x,y genes
    delta_mask: jnp.ndarray  # (G,) bool — x,y genes at frames > 0 (relative deltas)
    size_mask: jnp.ndarray  # (G,) bool — size genes
    size_rank: jnp.ndarray  # (O,) int32 — category rank per object
    speed_values: jnp.ndarray  # (O, T) float32 — fixed speed limit per obj/frame
    ra_constraints: jnp.ndarray  # (T, R, 4) int32 — [i, j, rx_code, ry_code]
    ra_mask: jnp.ndarray  # (T, R) bool
    qdc_constraints: jnp.ndarray  # (T, Q, 3) int32 — [i, j, qdc_code]
    qdc_mask: jnp.ndarray  # (T, Q) bool


# ============================================================
# 3. Constraint encoding (Python-side, runs once)
# ============================================================


def encode_constraints(
    objects, num_frames, ra_matrix, qdc_matrix, velocities, speed_limits, fixed_genes
):
    """Pack constraint data into a ConstraintData for JAX.

    Returns (cdata, num_objs, num_frames, ego_idx) — the last three
    are plain Python ints to be used as static constants.

    speed_limits: dict mapping speed category name -> numeric limit value
                  e.g. {'not moving': 2, 'slow': 10, 'normal': 22, 'fast': 45}
    """
    num_objs = len(objects)
    id_map = {obj["id"]: i for i, obj in enumerate(objects)}

    # Ego index (must be computed before gene layout)
    ego_idx = -1
    for i, obj in enumerate(objects):
        if obj["category"] == "ego":
            ego_idx = i
            break

    n_pos_objs = num_objs - (1 if ego_idx >= 0 else 0)
    n_pos = 2 * n_pos_objs * num_frames
    n_genes = n_pos + num_objs  # position genes + size genes

    # Gene masks (ego position genes are not in the genotype)
    xy_mask = np.zeros(n_genes, dtype=bool)
    delta_mask = np.zeros(n_genes, dtype=bool)
    size_mask = np.zeros(n_genes, dtype=bool)
    idx = 0
    for _o in range(num_objs):
        if _o == ego_idx:
            continue  # ego position is not in the genotype
        for _t in range(num_frames):
            xy_mask[idx] = True
            xy_mask[idx + 1] = True
            if _t > 0:
                delta_mask[idx] = True
                delta_mask[idx + 1] = True
            idx += 2
    for k in range(num_objs):
        size_mask[idx + k] = True

    # Fixed genes
    fixed_mask_np = np.zeros(n_genes, dtype=bool)
    fixed_vals_np = np.zeros(n_genes, dtype=np.float32)
    for g, val in fixed_genes.items():
        fixed_mask_np[g] = True
        fixed_vals_np[g] = val

    # Size rank per object
    size_rank_arr = np.array(
        [SIZE_RANK.get(obj["category"], 2) for obj in objects], dtype=np.int32
    )

    # Speed values: (O, T) float32 — fixed per obj/frame from Config.SPEED_LIMITS
    speed_to_val = {"not moving": 2, "slow": 10, "normal": 22, "fast": 45}
    if speed_limits is not None:
        speed_to_val = dict(speed_limits)
    speed_values = np.full((num_objs, num_frames), 22.0, dtype=np.float32)
    for i, obj in enumerate(objects):
        for t in range(num_frames):
            cat = velocities.get((obj["id"], t), "normal")
            speed_values[i, t] = speed_to_val.get(cat, 22)

    # RA constraints: pad to max per frame
    max_ra = max((len(ra_matrix[t]) for t in range(num_frames)), default=0)
    max_ra = max(max_ra, 1)
    ra_arr = np.zeros((num_frames, max_ra, 4), dtype=np.int32)
    ra_mask_np = np.zeros((num_frames, max_ra), dtype=bool)
    for t in range(num_frames):
        for k, entry in enumerate(ra_matrix[t]):
            oid_i, oid_j, rx, ry = entry
            ra_arr[t, k] = [
                id_map[oid_i],
                id_map[oid_j],
                ALLEN_TO_CODE.get(rx, 0),
                ALLEN_TO_CODE.get(ry, 0),
            ]
            ra_mask_np[t, k] = True

    # QDC constraints: pad to max per frame
    max_qdc = max((len(qdc_matrix[t]) for t in range(num_frames)), default=0)
    max_qdc = max(max_qdc, 1)
    qdc_arr = np.zeros((num_frames, max_qdc, 3), dtype=np.int32)
    qdc_mask_np = np.zeros((num_frames, max_qdc), dtype=bool)
    for t in range(num_frames):
        for k, entry in enumerate(qdc_matrix[t]):
            oid_i, oid_j, q = entry
            qdc_arr[t, k] = [id_map[oid_i], id_map[oid_j], QDC_TO_CODE.get(q, 0)]
            qdc_mask_np[t, k] = True

    cdata = ConstraintData(
        fixed_mask=jnp.array(fixed_mask_np),
        fixed_vals=jnp.array(fixed_vals_np),
        xy_mask=jnp.array(xy_mask),
        delta_mask=jnp.array(delta_mask),
        size_mask=jnp.array(size_mask),
        size_rank=jnp.array(size_rank_arr),
        speed_values=jnp.array(speed_values),
        ra_constraints=jnp.array(ra_arr),
        ra_mask=jnp.array(ra_mask_np),
        qdc_constraints=jnp.array(qdc_arr),
        qdc_mask=jnp.array(qdc_mask_np),
    )
    return cdata, num_objs, num_frames, ego_idx


# ============================================================
# 4. JAX fitness sub-functions (all branchless)
# ============================================================


def _decode_solution(solution, num_objs, num_frames, ego_idx):
    """Decode flat gene vector into positions and sizes.

    Ego position genes are not in the genotype; ego is re-inserted at
    (0, 0) for all frames during decoding.

    Other genes encode frame-0 as absolute position and frames 1+ as
    relative deltas.  Cumulative sum along the time axis converts to
    absolute.

    Sizes are snapped to even values to match MiniZinc's integer
    constraint: `2*cx = x_min + x_max` requires even bbox width.

    num_objs, num_frames, ego_idx must be static Python ints.
    """
    n_pos_objs = num_objs - 1 if ego_idx >= 0 else num_objs
    n_pos = 2 * n_pos_objs * num_frames
    pos_genes = solution[:n_pos].reshape(n_pos_objs, num_frames, 2)
    positions = jnp.cumsum(pos_genes, axis=1)

    if ego_idx >= 0:
        ego_pos = jnp.zeros((1, num_frames, 2))
        positions = jnp.concatenate(
            [positions[:ego_idx], ego_pos, positions[ego_idx:]], axis=0
        )

    raw_sizes = solution[n_pos : n_pos + num_objs]
    sizes = 2.0 * jnp.floor(raw_sizes / 2.0)  # snap to even
    return positions, sizes


def _penalty_size_ordering(sizes, size_rank):
    """Enforce size[o1]+2 <= size[o2] when size_rank[o1] < size_rank[o2].

    Vectorised: for all (i,j) pairs, compute penalty when rank[i] < rank[j].
    """
    n = sizes.shape[0]
    # (n,1) vs (1,n) broadcasts to (n,n)
    rank_i = size_rank[:, None]
    rank_j = size_rank[None, :]
    size_i = sizes[:, None]
    size_j = sizes[None, :]
    should_constrain = rank_i < rank_j  # (n, n) bool
    violation = jnp.maximum(0.0, size_i + 2.0 - size_j)  # need size_i+2 <= size_j
    return jnp.sum(jnp.where(should_constrain, violation, 0.0))


def _compute_bboxes(positions, sizes):
    """Compute square bounding boxes for all objects in all frames.

    sizes: (O,) — one size per object (square bbox side, always even
           after _decode_solution snapping).
    Returns: (O, T, 4) — [x_min, x_max, y_min, y_max]

    Matches MiniZinc: x_max - x_min = size, 2*cx = x_min + x_max.
    """
    half = sizes / 2.0  # exact since sizes are even

    x = positions[:, :, 0]  # (O, T)
    y = positions[:, :, 1]  # (O, T)

    half_exp = half[:, None]  # (O, 1) → broadcasts to (O, T)

    x_min = x - half_exp
    x_max = x + half_exp
    y_min = y - half_exp
    y_max = y + half_exp

    return jnp.stack([x_min, x_max, y_min, y_max], axis=-1)


def _penalty_speed_movement(positions, speed_values, num_frames):
    """Movement between consecutive frames must not exceed speed limit.

    speed_values: (O, T) — fixed speed limit per object per frame.
    num_frames is a static Python int.
    """
    if num_frames <= 1:
        return 0.0

    def body(t, acc):
        dx = jnp.abs(positions[:, t + 1, 0] - positions[:, t, 0])
        dy = jnp.abs(positions[:, t + 1, 1] - positions[:, t, 1])
        lim = speed_values[:, t]
        excess = (dx + dy) - lim
        return acc + jnp.sum(jnp.maximum(0.0, excess))

    return jax.lax.fori_loop(0, num_frames - 1, body, 0.0)


def _allen_violation_all(s1, e1, s2, e2):
    """Compute violations for all 13 Allen relations simultaneously.

    Returns (13,) array where index matches ALLEN_TO_CODE.
    Strict (tol=0) matching MiniZinc solver.
    """
    # Before: e1 < s2  →  violation = max(0, e1 - s2 + 1)
    v_before = jnp.maximum(0.0, e1 - s2 + 1.0)
    # After: s1 > e2  →  violation = max(0, e2 - s1 + 1)
    v_after = jnp.maximum(0.0, e2 - s1 + 1.0)
    # Meets: e1 = s2
    v_meets = jnp.abs(e1 - s2)
    # MetBy: s1 = e2
    v_metby = jnp.abs(s1 - e2)
    # Overlaps: s1 < s2 /\ s2 < e1 /\ e1 < e2
    v_overlaps = (
        jnp.maximum(0.0, s1 - s2 + 1.0)
        + jnp.maximum(0.0, s2 - e1 + 1.0)
        + jnp.maximum(0.0, e1 - e2 + 1.0)
    )
    # OverlappedBy: s2 < s1 /\ s1 < e2 /\ e2 < e1
    v_overlappedby = (
        jnp.maximum(0.0, s2 - s1 + 1.0)
        + jnp.maximum(0.0, s1 - e2 + 1.0)
        + jnp.maximum(0.0, e2 - e1 + 1.0)
    )
    # Starts: s1 = s2 /\ e1 < e2
    v_starts = jnp.abs(s1 - s2) + jnp.maximum(0.0, e1 - e2 + 1.0)
    # StartedBy: s1 = s2 /\ e1 > e2
    v_startedby = jnp.abs(s1 - s2) + jnp.maximum(0.0, e2 - e1 + 1.0)
    # Finishes: e1 = e2 /\ s1 > s2
    v_finishes = jnp.abs(e1 - e2) + jnp.maximum(0.0, s2 - s1 + 1.0)
    # FinishedBy: e1 = e2 /\ s1 < s2
    v_finishedby = jnp.abs(e1 - e2) + jnp.maximum(0.0, s1 - s2 + 1.0)
    # During: s1 > s2 /\ e1 < e2
    v_during = jnp.maximum(0.0, s2 - s1 + 1.0) + jnp.maximum(
        0.0, e1 - e2 + 1.0
    )
    # Contains: s1 < s2 /\ e1 > e2
    v_contains = jnp.maximum(0.0, s1 - s2 + 1.0) + jnp.maximum(
        0.0, e2 - e1 + 1.0
    )
    # Equals: s1 = s2 /\ e1 = e2
    v_equals = jnp.abs(s1 - s2) + jnp.abs(e1 - e2)

    return jnp.array(
        [
            v_before,
            v_after,
            v_meets,
            v_metby,
            v_overlaps,
            v_overlappedby,
            v_starts,
            v_startedby,
            v_finishes,
            v_finishedby,
            v_during,
            v_contains,
            v_equals,
        ]
    )


def _penalty_allen_single(constraint, bboxes_t):
    """Penalty for a single RA constraint in one frame."""
    i, j = constraint[0], constraint[1]
    rx_code, ry_code = constraint[2], constraint[3]
    bb_i = bboxes_t[i]
    bb_j = bboxes_t[j]

    vx_all = _allen_violation_all(bb_i[0], bb_i[1], bb_j[0], bb_j[1])
    vy_all = _allen_violation_all(bb_i[2], bb_i[3], bb_j[2], bb_j[3])
    return vx_all[rx_code] + vy_all[ry_code]


def _penalty_allen(bboxes, ra_constraints, ra_mask, num_frames):
    """Total Allen RA penalty across all frames. num_frames is static."""

    def frame_penalty(t):
        bboxes_t = bboxes[:, t, :]
        constraints_t = ra_constraints[t]
        mask_t = ra_mask[t]
        violations = jax.vmap(lambda c: _penalty_allen_single(c, bboxes_t))(
            constraints_t
        )
        return jnp.sum(violations * mask_t)

    def body(t, acc):
        return acc + frame_penalty(t)

    return jax.lax.fori_loop(0, num_frames, body, 0.0)


def _qdc_violation(gap, qdc_code):
    """QDC distance violation for a single pair.

    Matches MiniZinc two-sided constraints:
      very_close (0): gap <= vc
      close      (1): gap > vc  AND gap <= cl
      normal     (2): gap > cl  AND gap <= nr
      far        (3): gap > nr  AND gap <= fa
      very_far   (4): gap > fa
    """
    is_very_far = qdc_code == 4
    safe_upper_code = jnp.minimum(qdc_code, 3)
    upper_thresh = QDC_UPPER_THRESHOLDS[safe_upper_code]

    # Upper bound: gap <= upper_thresh (not for very_far)
    penalty_upper = jnp.where(is_very_far, 0.0, jnp.maximum(0.0, gap - upper_thresh))

    # Lower bound: gap > lower_thresh (strict, matching MiniZinc integer >)
    # For very_close (code 0), lower_thresh is -1 so this is always 0
    lower_thresh = QDC_LOWER_THRESHOLDS[qdc_code]
    penalty_lower = jnp.maximum(0.0, lower_thresh - gap + 1.0)

    return penalty_upper + penalty_lower


def _penalty_qdc_single(constraint, bboxes_t):
    """QDC penalty for a single constraint in one frame."""
    i, j = constraint[0], constraint[1]
    qdc_code = constraint[2]
    bb_i = bboxes_t[i]
    bb_j = bboxes_t[j]

    dx = jnp.maximum(0.0, jnp.maximum(bb_i[0] - bb_j[1], bb_j[0] - bb_i[1]))
    dy = jnp.maximum(0.0, jnp.maximum(bb_i[2] - bb_j[3], bb_j[2] - bb_i[3]))
    gap = dx + dy
    return _qdc_violation(gap, qdc_code)


def _penalty_qdc(bboxes, qdc_constraints, qdc_mask, num_frames):
    """Total QDC penalty across all frames. num_frames is static."""

    def frame_penalty(t):
        bboxes_t = bboxes[:, t, :]
        constraints_t = qdc_constraints[t]
        mask_t = qdc_mask[t]
        violations = jax.vmap(lambda c: _penalty_qdc_single(c, bboxes_t))(constraints_t)
        return jnp.sum(violations * mask_t)

    def body(t, acc):
        return acc + frame_penalty(t)

    return jax.lax.fori_loop(0, num_frames, body, 0.0)


def _penalty_bounds(positions):
    """Positions must be within [-MAP_LIMIT, MAP_LIMIT]."""
    return jnp.sum(jnp.maximum(0.0, jnp.abs(positions) - MAP_LIMIT))


# ============================================================
# 5. Fitness function factory
# ============================================================


def make_fitness_batch(num_objs, num_frames, ego_idx):
    """Create a JIT-compiled, vmapped fitness function.

    num_objs, num_frames, ego_idx are baked in as Python constants
    so JAX can use them for static shapes/slices during tracing.
    """

    def fitness_single(solution, cdata):
        positions, sizes = _decode_solution(solution, num_objs, num_frames, ego_idx)

        bboxes = _compute_bboxes(positions, sizes)

        penalty = 0.0
        penalty += _penalty_size_ordering(sizes, cdata.size_rank)
        penalty += _penalty_speed_movement(
            positions, cdata.speed_values, num_frames
        )
        penalty += _penalty_allen(
            bboxes, cdata.ra_constraints, cdata.ra_mask, num_frames
        )
        penalty += _penalty_qdc(
            bboxes, cdata.qdc_constraints, cdata.qdc_mask, num_frames
        )
        penalty += _penalty_bounds(positions)
        return penalty

    return jax.jit(jax.vmap(fitness_single, in_axes=(0, None)))


# ============================================================
# 6. GA parameters & operators (JIT-compiled)
# ============================================================


@dataclass(frozen=True)
class GAParams:
    """All tunable GA hyperparameters in one place.

    Defaults reproduce the original hardcoded behaviour.
    """

    # --- population fractions ---
    parent_frac: float = 0.5
    elite_frac: float = 0.05

    # --- tournament selection ---
    tournament_k: int = 7

    # --- mutation ---
    mutation_rate: float = 0.1
    mutation_step_factor: float = 0.05
    mutation_step_min: float = 20
    mutation_decay: float = 0.95  # step *= (1 - progress * decay)

    # --- size gene mutation ---
    size_step_init: float = 10.0  # initial size step
    size_step_min: float = 2.0  # minimum size step
    size_decay: float = 0.7  # size_step *= (1 - progress * decay)

    # --- delta (relative-position) gene range ---
    delta_range: float = 60.0  # max magnitude for frame-to-frame deltas

    # --- stagnation restart (0 = disabled) ---
    stagnation_limit: int = 200  # gens without improvement before restart
    restart_diversity_frac: float = 0.5  # fraction of pop replaced on restart

    # --- sync/callback control ---
    check_interval: int = 1  # generations between host syncs

    # --- optional: gradient-based local search (default OFF) ---
    local_search: bool = False
    local_search_top_k: int = 10
    local_search_steps: int = 5
    local_search_lr: float = 1.0

    # --- optional: constraint-aware seeding (default OFF) ---
    smart_init: bool = False
    smart_init_frac: float = 0.2  # fraction of pop with heuristic placement


def make_tournament_select(K=5):
    """Factory: return a JIT-compiled tournament selection with tournament size K."""

    @partial(jax.jit, static_argnums=(2,))
    def _tournament_select(key, fitness, num_parents, pop):
        pop_size = fitness.shape[0]

        def select_one(key_i):
            candidates = jrandom.randint(key_i, (K,), 0, pop_size)
            candidate_fitness = fitness[candidates]
            best = jnp.argmin(candidate_fitness)
            return candidates[best]

        keys = jrandom.split(key, num_parents)
        parent_indices = jax.vmap(select_one)(keys)
        return pop[parent_indices]

    return _tournament_select


# Default instance (K=5) for backward compat
tournament_select = make_tournament_select(5)


@partial(jax.jit, static_argnums=(2,))
def uniform_crossover(key, parents, num_offspring, fixed_mask, fixed_vals):
    """Uniform crossover producing num_offspring from parents."""
    n_parents = parents.shape[0]
    n_genes = parents.shape[1]

    def make_child(key_i):
        k1, k2, k3 = jrandom.split(key_i, 3)
        p1_idx = jrandom.randint(k1, (), 0, n_parents)
        p2_idx = jrandom.randint(k2, (), 0, n_parents)
        mask = jrandom.randint(k3, (n_genes,), 0, 2)
        child = jnp.where(mask, parents[p1_idx], parents[p2_idx])
        child = jnp.where(fixed_mask, fixed_vals, child)
        return child

    keys = jrandom.split(key, num_offspring)
    return jax.vmap(make_child)(keys)


def make_block_crossover(num_objs, num_frames, ego_idx=-1):
    """Factory: per-object block crossover for relative encoding.

    For each child, each object's full trajectory (all frames) comes from
    one parent.  This preserves cumulative-sum coherence that uniform
    gene-level crossover would destroy.

    Ego position genes are excluded from the genotype, so only
    non-ego objects have trajectory genes to cross over.
    """
    n_pos_objs = num_objs - 1 if ego_idx >= 0 else num_objs
    genes_per_obj_pos = 2 * num_frames  # x,y per frame
    n_pos_genes = genes_per_obj_pos * n_pos_objs

    # Map position-object index -> original object index
    _pos_to_orig = jnp.array(
        [o for o in range(num_objs) if o != ego_idx], dtype=jnp.int32
    )

    @partial(jax.jit, static_argnums=(2,))
    def _crossover(key, parents, num_offspring, fixed_mask, fixed_vals):
        n_parents = parents.shape[0]
        n_genes = parents.shape[1]

        def make_child(key_i):
            k1, k2, k3 = jrandom.split(key_i, 3)
            p1_idx = jrandom.randint(k1, (), 0, n_parents)
            p2_idx = jrandom.randint(k2, (), 0, n_parents)

            # Per-object random parent choice
            obj_choice = jrandom.randint(k3, (num_objs,), 0, 2)

            # Map each gene to its object's choice
            gene_idx = jnp.arange(n_genes)
            # Position genes: map to original object index
            pos_obj = jnp.minimum(gene_idx // genes_per_obj_pos, n_pos_objs - 1)
            obj_for_pos = _pos_to_orig[pos_obj]
            # Size genes: obj index = gene - n_pos_genes
            obj_for_size = jnp.clip(gene_idx - n_pos_genes, 0, num_objs - 1)
            is_size = gene_idx >= n_pos_genes
            obj_for_gene = jnp.where(is_size, obj_for_size, obj_for_pos)
            use_p2 = obj_choice[obj_for_gene]

            child = jnp.where(use_p2, parents[p2_idx], parents[p1_idx])
            child = jnp.where(fixed_mask, fixed_vals, child)
            return child

        keys = jrandom.split(key, num_offspring)
        return jax.vmap(make_child)(keys)

    return _crossover


def make_mutate_population(
    mutation_rate=0.2,
    step_factor=0.2,
    step_min=3.0,
    delta_range=60.0,
    decay=0.9,
    size_step_init=10.0,
    size_step_min=2.0,
    size_decay=0.7,
):
    """Factory: return a JIT-compiled mutation operator with the given params."""
    ml = MAP_LIMIT
    size_max = ml // 4

    dr = delta_range

    @jax.jit
    def _mutate_population(key, population, cdata, progress):
        step = jnp.maximum(step_min, ml * step_factor * (1.0 - progress * decay))
        delta_step = jnp.maximum(dr * 0.08, dr * 0.25 * (1.0 - progress * decay))
        sz_step = jnp.maximum(
            size_step_min, size_step_init * (1.0 - progress * size_decay)
        )

        n_genes = population.shape[1]

        def mutate_one(key_i, individual):
            k1, k2, k3, k4 = jrandom.split(key_i, 4)

            mutable = ~cdata.fixed_mask
            mutation_probs = jnp.where(mutable, mutation_rate, 0.0)
            do_mutate = jrandom.bernoulli(k1, mutation_probs)

            abs_xy_delta = jnp.floor(
                jrandom.uniform(k2, (n_genes,), minval=-step, maxval=step + 1)
            )
            delta_xy_delta = jnp.floor(
                jrandom.uniform(k4, (n_genes,), minval=-delta_step, maxval=delta_step + 1)
            )
            size_delta = jnp.floor(
                jrandom.uniform(
                    k3, (n_genes,), minval=-sz_step, maxval=sz_step + 1
                )
            )

            abs_xy_new = jnp.clip(individual + abs_xy_delta, -ml, ml)
            delta_new = jnp.clip(individual + delta_xy_delta, -dr, dr)
            size_new = jnp.clip(individual + size_delta, 5, size_max)

            mutated = jnp.where(
                cdata.delta_mask,
                delta_new,
                jnp.where(
                    cdata.xy_mask,
                    abs_xy_new,
                    jnp.where(cdata.size_mask, size_new, individual),
                ),
            )

            result = jnp.where(do_mutate, mutated, individual)
            result = jnp.where(cdata.fixed_mask, cdata.fixed_vals, result)
            return result

        keys = jrandom.split(key, population.shape[0])
        return jax.vmap(mutate_one)(keys, population)

    return _mutate_population


# Default instance for backward compat
mutate_population = make_mutate_population()


@partial(jax.jit, static_argnums=(3,))
def elitism_merge(old_pop, old_fitness, new_pop, num_elite):
    """Keep top num_elite from old population, fill rest with new_pop."""
    sorted_indices = jnp.argsort(old_fitness)
    elite = old_pop[sorted_indices[:num_elite]]
    pop_size = old_pop.shape[0]
    rest = new_pop[: pop_size - num_elite]
    return jnp.concatenate([elite, rest], axis=0)


# ============================================================
# 7. JAX-side random initialisation helper
# ============================================================


def _random_init_jax(
    key,
    count,
    n_genes,
    xy_mask,
    delta_mask,
    size_mask,
    fixed_mask,
    fixed_vals,
    ml=None,
    delta_range=60.0,
):
    """Generate *count* random individuals entirely on the JAX side."""
    if ml is None:
        ml = MAP_LIMIT
    size_max = ml // 4
    k1, k2, k3 = jrandom.split(key, 3)

    # Absolute xy genes (frame 0): full map range
    abs_xy_vals = jrandom.uniform(k1, (count, n_genes), minval=-ml, maxval=ml + 1)
    abs_xy_vals = jnp.floor(abs_xy_vals)

    # Delta xy genes (frames > 0): small range
    dr = delta_range
    delta_vals = jrandom.normal(k2, (count, n_genes)) * (dr / 3.0)
    delta_vals = jnp.floor(jnp.clip(delta_vals, -dr, dr))

    # Combine: delta_mask selects delta range, remaining xy_mask selects absolute range
    xy_vals = jnp.where(delta_mask, delta_vals, abs_xy_vals)

    # Size genes: [5, size_max]
    size_vals = jrandom.uniform(k3, (count, n_genes), minval=5, maxval=size_max + 1)
    size_vals = jnp.floor(size_vals)

    pop = jnp.where(
        xy_mask,
        xy_vals,
        jnp.where(size_mask, size_vals, 0.0),
    )

    pop = jnp.where(fixed_mask, fixed_vals[None, :], pop)
    return pop


# ============================================================
# 8. Heuristic seeding (optional, numpy, runs once)
# ============================================================


def _heuristic_seed_numpy(
    count, objects, num_frames, ego_idx, id_map,
    qdc_matrix, n_genes, ml, delta_range,
):
    """Create *count* heuristic individuals using QDC distance hints.

    Strategy per individual:
    1. Place objects based on QDC constraints for frame 0: for each pair
       (i,j) with QDC distance d, place j at distance ~sqrt(threshold)
       from i at a random angle.
    2. For frames 1+, use small random deltas.
    3. Randomize sizes within valid range.
    4. Encode back to gene vector (frame 0 = absolute, frames 1+ = deltas).
    """
    num_objs = len(objects)
    n_pos_objs = num_objs - (1 if ego_idx >= 0 else 0)
    size_max = ml // 4
    dr = delta_range

    # Build threshold lookup: qdc_code -> approximate distance
    thresh_map = {
        0: int(THRESHOLDS["very close"] ** 0.5),   # sqrt of squared threshold
        1: int(THRESHOLDS["close"] ** 0.5),
        2: int(THRESHOLDS["normal"] ** 0.5),
        3: int(THRESHOLDS["far"] ** 0.5),
        4: int(THRESHOLDS["far"] ** 0.5),  # very_far: unconstrained, use far
    }

    pop = np.zeros((count, n_genes), dtype=np.float32)

    for ind in range(count):
        # Place objects for frame 0
        positions = np.zeros((num_objs, 2), dtype=np.float32)
        placed = np.zeros(num_objs, dtype=bool)

        # Ego is always at origin
        if ego_idx >= 0:
            positions[ego_idx] = [0, 0]
            placed[ego_idx] = True

        # Use QDC constraints from frame 0 to guide placement
        try:
            qdc_frame0 = qdc_matrix[0]
        except (KeyError, IndexError):
            qdc_frame0 = []
        for entry in qdc_frame0:
            oid_i, oid_j, q = entry
            i_idx = id_map.get(oid_i, -1)
            j_idx = id_map.get(oid_j, -1)
            if i_idx < 0 or j_idx < 0:
                continue
            qdc_code = QDC_TO_CODE.get(q, 2)
            target_dist = thresh_map.get(qdc_code, ml // 2)

            if placed[i_idx] and not placed[j_idx]:
                angle = np.random.uniform(0, 2 * np.pi)
                dist = target_dist * np.random.uniform(0.5, 1.0)
                positions[j_idx, 0] = np.clip(
                    positions[i_idx, 0] + dist * np.cos(angle), -ml, ml
                )
                positions[j_idx, 1] = np.clip(
                    positions[i_idx, 1] + dist * np.sin(angle), -ml, ml
                )
                placed[j_idx] = True
            elif placed[j_idx] and not placed[i_idx]:
                angle = np.random.uniform(0, 2 * np.pi)
                dist = target_dist * np.random.uniform(0.5, 1.0)
                positions[i_idx, 0] = np.clip(
                    positions[j_idx, 0] + dist * np.cos(angle), -ml, ml
                )
                positions[i_idx, 1] = np.clip(
                    positions[j_idx, 1] + dist * np.sin(angle), -ml, ml
                )
                placed[i_idx] = True

        # Place remaining objects randomly
        for o in range(num_objs):
            if not placed[o]:
                positions[o, 0] = np.random.randint(-ml, ml + 1)
                positions[o, 1] = np.random.randint(-ml, ml + 1)

        # Encode into gene vector
        gene_idx = 0
        for o in range(num_objs):
            if o == ego_idx:
                continue
            # Frame 0: absolute position
            pop[ind, gene_idx] = int(positions[o, 0])
            pop[ind, gene_idx + 1] = int(positions[o, 1])
            gene_idx += 2
            # Frames 1+: small random deltas
            for _t in range(1, num_frames):
                pop[ind, gene_idx] = np.random.randint(
                    max(-int(dr), -int(dr)), int(dr) + 1
                )
                pop[ind, gene_idx + 1] = np.random.randint(-int(dr), int(dr) + 1)
                gene_idx += 2

        # Size genes: random within valid range
        for o in range(num_objs):
            pop[ind, gene_idx + o] = np.random.randint(5, size_max + 1)

    return jnp.array(pop)


# ============================================================
# 9. Local search factory (optional, gradient-based)
# ============================================================


def make_local_search(num_objs, num_frames, ego_idx, top_k, n_steps, lr):
    """Gradient-based refinement of top-K individuals."""

    def fitness_single(solution, cdata):
        positions, sizes = _decode_solution(solution, num_objs, num_frames, ego_idx)
        bboxes = _compute_bboxes(positions, sizes)
        penalty = 0.0
        penalty += _penalty_size_ordering(sizes, cdata.size_rank)
        penalty += _penalty_speed_movement(positions, cdata.speed_values, num_frames)
        penalty += _penalty_allen(
            bboxes, cdata.ra_constraints, cdata.ra_mask, num_frames
        )
        penalty += _penalty_qdc(
            bboxes, cdata.qdc_constraints, cdata.qdc_mask, num_frames
        )
        penalty += _penalty_bounds(positions)
        return penalty

    grad_fn = jax.grad(fitness_single)

    @jax.jit
    def local_search(population, fitness, cdata):
        top_indices = jnp.argsort(fitness)[:top_k]
        top_pop = population[top_indices]

        def refine_one(sol, cdata):
            def body(i, s):
                g = grad_fn(s, cdata)
                return jnp.where(
                    cdata.fixed_mask, cdata.fixed_vals,
                    jnp.clip(s - lr * g, -MAP_LIMIT, MAP_LIMIT),
                )
            return jax.lax.fori_loop(0, n_steps, body, sol)

        refined = jax.vmap(refine_one, in_axes=(0, None))(top_pop, cdata)
        refined = jnp.round(refined)  # snap to integers
        return population.at[top_indices].set(refined)

    return local_search


# ============================================================
# 10. Fused generation step factory
# ============================================================


def make_generation_step(
    num_objs, num_frames, ego_idx,
    num_parents, num_offspring, num_elite,
    tournament_k, mutation_rate, step_factor, step_min,
    delta_range, decay, size_step_init, size_step_min, size_decay,
    local_search_fn=None,
):
    """Create a single JIT-compiled function for one full GA generation.

    Composes selection, block crossover, mutation, elitism merge, and
    fitness evaluation into one fused JIT call, reducing dispatch overhead
    from 5 calls per generation to 1.
    """
    _select = make_tournament_select(tournament_k)
    _crossover = make_block_crossover(num_objs, num_frames, ego_idx)
    _mutate = make_mutate_population(
        mutation_rate, step_factor, step_min,
        delta_range, decay, size_step_init, size_step_min, size_decay,
    )
    _fitness_batch = make_fitness_batch(num_objs, num_frames, ego_idx)

    @jax.jit
    def generation_step(key, population, fitness, cdata, progress):
        key, k_sel, k_cross, k_mut = jrandom.split(key, 4)
        parents = _select(k_sel, fitness, num_parents, population)
        offspring = _crossover(
            k_cross, parents, num_offspring,
            cdata.fixed_mask, cdata.fixed_vals,
        )
        offspring = _mutate(k_mut, offspring, cdata, progress)
        population = elitism_merge(population, fitness, offspring, num_elite)
        fitness = _fitness_batch(population, cdata)

        if local_search_fn is not None:  # resolved at trace time
            population = local_search_fn(population, fitness, cdata)
            fitness = _fitness_batch(population, cdata)

        best_fitness = jnp.min(fitness)
        return key, population, fitness, best_fitness

    return generation_step


# ============================================================
# 11. JAXGASolver class
# ============================================================


class JAXGASolver:
    """JAX-accelerated GA solver."""

    def __init__(self, objects, num_frames):
        self.objects = objects
        self.num_frames = num_frames
        self.num_objs = len(objects)
        self.id_map = {obj["id"]: i for i, obj in enumerate(objects)}
        self.rev_map = {i: obj["id"] for i, obj in enumerate(objects)}

    def _gene_count(self):
        ego_count = sum(1 for obj in self.objects if obj["category"] == "ego")
        n_pos_objs = self.num_objs - ego_count
        return 2 * n_pos_objs * self.num_frames + self.num_objs

    def _gene_indices(self):
        ego_idx = next(
            (i for i, obj in enumerate(self.objects) if obj["category"] == "ego"),
            -1,
        )
        xy_idx = []
        delta_idx = []
        idx = 0
        for _o in range(self.num_objs):
            if _o == ego_idx:
                continue
            for _t in range(self.num_frames):
                xy_idx.append(idx)
                xy_idx.append(idx + 1)
                if _t > 0:
                    delta_idx.append(idx)
                    delta_idx.append(idx + 1)
                idx += 2
        size_idx = list(range(idx, idx + self.num_objs))
        return xy_idx, delta_idx, size_idx

    def _decode(self, solution):
        """Decode flat gene vector — numpy side.

        Ego position is not in the genotype; re-inserted as (0, 0).
        Other genes encode frame-0 as absolute, frames 1+ as relative
        deltas.  Cumulative sum converts to absolute positions.
        Sizes are snapped to even to match MiniZinc integer constraints.

        Returns (positions, sizes).
        """
        ego_idx = next(
            (i for i, obj in enumerate(self.objects) if obj["category"] == "ego"),
            -1,
        )
        positions = []
        idx = 0
        for _o in range(self.num_objs):
            if _o == ego_idx:
                positions.append([(0, 0)] * self.num_frames)
                continue
            pos_frames = []
            cum_x, cum_y = 0, 0
            for _t in range(self.num_frames):
                cum_x += int(solution[idx])
                cum_y += int(solution[idx + 1])
                pos_frames.append((cum_x, cum_y))
                idx += 2
            positions.append(pos_frames)
        sizes = [2 * (int(solution[idx + k]) // 2) for k in range(self.num_objs)]
        return positions, sizes

    def _build_output(self, positions, sizes):
        """Convert decoded solution to GlobalScenarioSolver output format.

        Uses square bounding boxes from sizes (matching MiniZinc model).
        Sizes are already even from _decode.
        """
        output = []
        for t in range(self.num_frames):
            frame_data = []
            for i, obj in enumerate(self.objects):
                x, y = positions[i][t]
                sz = sizes[i]
                half = sz // 2
                x_min = x - half
                x_max = x + half  # = x_min + sz since sz is even
                y_min = y - half
                y_max = y + half
                frame_data.append(
                    {
                        "id": obj["id"],
                        "category": obj["category"],
                        "x": x,
                        "y": y,
                        "w": sz,
                        "h": sz,
                        "heading": 0,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                    }
                )
            output.append(frame_data)
        return output

    def _build_fixed_genes(self, heading_map):
        """Build dict of gene_index -> fixed_value.

        Ego position is no longer in the genotype (re-inserted during
        decoding), so no fixed genes are needed for it.
        """
        return {}

    def solve(
        self,
        ra_matrix,
        qdc_matrix,
        velocities,
        heading_map,
        solver_name="GA",
        heuristic="default",
        timeout=60.0,
        sol_per_pop=4000,
        on_generation=None,
        ga_params=None,
        speed_limits=None,
    ):
        """Run the JAX GA and return a solution (or None).

        timeout: maximum wall-clock seconds for the GA loop.
        speed_limits: dict mapping speed category -> numeric value,
                      e.g. {'not moving': 2, 'slow': 10, 'normal': 22, 'fast': 45}.
                      If None, defaults are used.
        """
        if ga_params is None:
            ga_params = GAParams()

        ml = MAP_LIMIT
        dr = ga_params.delta_range
        n_genes = self._gene_count()

        fixed_genes = self._build_fixed_genes(heading_map)

        # Encode constraints
        cdata, num_objs, num_frames, ego_idx = encode_constraints(
            self.objects,
            self.num_frames,
            ra_matrix,
            qdc_matrix,
            velocities,
            speed_limits,
            fixed_genes,
        )

        # Derive counts from fractions
        num_parents = max(2, int(sol_per_pop * ga_params.parent_frac))
        num_elite = max(1, int(sol_per_pop * ga_params.elite_frac))
        num_offspring = sol_per_pop - num_elite

        # --- JAX random init (replaces slow numpy double loop) ---
        key = jrandom.PRNGKey(np.random.randint(0, 2**31))
        key, k_init = jrandom.split(key)
        population = _random_init_jax(
            k_init, sol_per_pop, n_genes,
            cdata.xy_mask, cdata.delta_mask, cdata.size_mask,
            cdata.fixed_mask, cdata.fixed_vals, ml, dr,
        )

        # --- Optional: constraint-aware seeding ---
        if ga_params.smart_init:
            n_smart = max(1, int(sol_per_pop * ga_params.smart_init_frac))
            smart_pop = _heuristic_seed_numpy(
                n_smart, self.objects, self.num_frames, ego_idx,
                self.id_map, qdc_matrix, n_genes, ml, dr,
            )
            population = population.at[:n_smart].set(smart_pop)

        print(
            f"JAX GA: starting ({self.num_objs} objects, "
            f"{self.num_frames} frames, {n_genes} genes, "
            f"{len(fixed_genes)} fixed, pop={sol_per_pop}, "
            f"timeout={timeout}s)"
        )

        # --- Optional: local search ---
        local_search_fn = None
        if ga_params.local_search:
            local_search_fn = make_local_search(
                num_objs, num_frames, ego_idx,
                ga_params.local_search_top_k,
                ga_params.local_search_steps,
                ga_params.local_search_lr,
            )

        # --- Build fused generation step (1 JIT call instead of 5) ---
        generation_step = make_generation_step(
            num_objs, num_frames, ego_idx,
            num_parents, num_offspring, num_elite,
            ga_params.tournament_k,
            ga_params.mutation_rate, ga_params.mutation_step_factor,
            ga_params.mutation_step_min, dr, ga_params.mutation_decay,
            ga_params.size_step_init, ga_params.size_step_min,
            ga_params.size_decay,
            local_search_fn=local_search_fn,
        )

        # JIT-warm fitness evaluation
        fitness_batch = make_fitness_batch(num_objs, num_frames, ego_idx)
        print("JAX GA: JIT compiling...")
        fitness = fitness_batch(population, cdata)
        fitness.block_until_ready()
        print("JAX GA: JIT compilation done.")

        best_fitness_val = float(jnp.min(fitness))
        initial_best_fitness = max(best_fitness_val, 1e-6)  # avoid div-by-zero
        stagnation_counter = 0
        check_interval = ga_params.check_interval

        class _GenInfo:
            pass

        t_start = _time.monotonic()
        gen = 0
        while True:
            elapsed = _time.monotonic() - t_start
            if elapsed >= timeout:
                break

            # Progress measured by how close best fitness is to 0:
            # 0 = no improvement from initial, 1 = solved (fitness=0)
            progress = jnp.float32(
                max(0.0, min(1.0, 1.0 - best_fitness_val / initial_best_fitness))
            )
            key, population, fitness, best_fit_jax = generation_step(
                key, population, fitness, cdata, progress,
            )
            gen += 1

            # Only sync to host every check_interval generations
            should_check = (gen % check_interval == 0)

            if should_check:
                new_best = float(best_fit_jax)
                if new_best <= 0.0:
                    best_fitness_val = new_best
                    print(f"JAX GA: converged at generation {gen}")
                    break

                # Stagnation detection (check_interval granularity)
                if ga_params.stagnation_limit > 0:
                    if new_best < best_fitness_val - 1e-6:
                        stagnation_counter = 0
                    else:
                        stagnation_counter += check_interval

                    if stagnation_counter >= ga_params.stagnation_limit:
                        n_replace = max(
                            1, int(sol_per_pop * ga_params.restart_diversity_frac)
                        )
                        key, k_restart = jrandom.split(key)
                        fresh = _random_init_jax(
                            k_restart, n_replace, n_genes,
                            cdata.xy_mask, cdata.delta_mask, cdata.size_mask,
                            cdata.fixed_mask, cdata.fixed_vals, ml, dr,
                        )
                        sorted_idx = jnp.argsort(fitness)
                        keep = population[sorted_idx[: sol_per_pop - n_replace]]
                        population = jnp.concatenate([keep, fresh], axis=0)
                        fitness = fitness_batch(population, cdata)
                        stagnation_counter = 0
                        print(
                            f"JAX GA: restart at gen {gen} "
                            f"(replaced {n_replace}/{sol_per_pop}) "
                            f"/ best fitness = {new_best:.2f}"
                        )

                best_fitness_val = new_best

            # Callback (uses potentially stale best_fitness_val between checks)
            if on_generation is not None:
                info = _GenInfo()
                info.generations_completed = gen
                info.best_solution_fitness = -best_fitness_val
                on_generation(info)

        # Extract best
        best_idx = int(jnp.argmin(fitness))
        best_solution = np.array(population[best_idx])
        best_penalty = float(fitness[best_idx])

        total_time = _time.monotonic() - t_start
        print(
            f"JAX GA: finished — best penalty = {best_penalty}, "
            f"generation {gen}, {total_time:.1f}s"
        )

        if best_penalty <= 0.0:
            positions, sizes = self._decode(best_solution)
            print(f"JAX GA: sizes = {sizes}")
            return self._build_output(positions, sizes)
        else:
            print(f"JAX GA: no valid solution found (best violation = {best_penalty})")
            return None

    def solve_all(
        self,
        ra_matrix,
        qdc_matrix,
        velocities,
        heading_map,
        solver_name="GA",
        heuristic="default",
        num_solutions=None,
        timeout=60.0,
        on_generation=None,
    ):
        """Find multiple solutions by running the GA repeatedly."""
        import random

        target = num_solutions if num_solutions else 5
        all_solutions = []
        per_run_timeout = timeout / max(1, target)

        for run_idx in range(target * 2):
            if len(all_solutions) >= target:
                break
            random.seed(run_idx * 42 + 7)
            np.random.seed(run_idx * 42 + 7)
            print(
                f"JAX GA multi-solution: run {run_idx + 1} "
                f"(found {len(all_solutions)}/{target} so far)"
            )
            sol = self.solve(
                ra_matrix,
                qdc_matrix,
                velocities,
                heading_map,
                timeout=per_run_timeout,
                on_generation=on_generation,
            )
            if sol is not None:
                all_solutions.append(sol)

        return all_solutions if all_solutions else None


# ============================================================
# 12. Pure-Python Constraint Checker (mirrors MiniZinc model)
# ============================================================


class ConstraintChecker:
    """Evaluates a candidate solution against all constraints in pure Python.
    Returns a non-negative violation score (0 = fully valid).

    Mirrors MiniZinc solver model exactly: square bbox, size ordering,
    ego at origin all frames, fixed speed limits, strict Allen, two-sided
    QDC constraints, no relaxation.
    """

    def __init__(
        self, objects, num_frames, ra_matrix, qdc_matrix, velocities,
        speed_limits=None,
    ):
        self.objects = objects
        self.num_frames = num_frames
        self.ra_matrix = ra_matrix
        self.qdc_matrix = qdc_matrix
        self.velocities = velocities
        self.num_objs = len(objects)
        self.id_map = {obj["id"]: i for i, obj in enumerate(objects)}

        # Size rank per object
        self.size_rank = [SIZE_RANK.get(obj["category"], 2) for obj in objects]

        # Fixed speed limits (from Config.SPEED_LIMITS)
        speed_to_val = {"not moving": 2, "slow": 10, "normal": 22, "fast": 45}
        if speed_limits is not None:
            speed_to_val = dict(speed_limits)
        self.speed_to_val = speed_to_val

        # Find ego object index
        self.ego_idx = None
        for i, obj in enumerate(objects):
            if obj["category"] == "ego":
                self.ego_idx = i
                break

        # QDC upper thresholds — no relaxation (matching MiniZinc solver exactly)
        self.qdc_upper_thresholds = {
            "very close": int(THRESHOLDS["very close"]),
            "close": int(THRESHOLDS["close"]),
            "normal": int(THRESHOLDS["normal"]),
            "far": int(THRESHOLDS["far"]),
        }
        # QDC lower thresholds (strict >): close requires gap > vc, etc.
        self.qdc_lower_thresholds = {
            "close": int(THRESHOLDS["very close"]),
            "normal": int(THRESHOLDS["close"]),
            "far": int(THRESHOLDS["normal"]),
            "very far": int(THRESHOLDS["far"]),
        }

    @staticmethod
    def compute_bbox(x, y, size):
        """Compute square bounding box from center + size (matches MiniZinc).
        Size should already be even from _decode()."""
        half = size // 2
        x_min = x - half
        x_max = x + half
        y_min = y - half
        y_max = y + half
        return x_min, x_max, y_min, y_max

    @staticmethod
    def check_allen_relation(rel, s1, e1, s2, e2):
        """Check if an Allen relation holds between two intervals.
        Returns 0 if satisfied, >0 violation magnitude otherwise.
        Strict (tol=0) matching MiniZinc solver."""
        if rel == "Before":
            return max(0, e1 - s2 + 1)  # need e1 < s2
        if rel == "After":
            return max(0, e2 - s1 + 1)  # need s1 > e2
        if rel == "Meets":
            return abs(e1 - s2)  # need e1 = s2
        if rel == "MetBy":
            return abs(s1 - e2)  # need s1 = e2
        if rel == "Overlaps":
            v = 0
            v += max(0, s1 - s2 + 1)  # s1 < s2
            v += max(0, s2 - e1 + 1)  # s2 < e1
            v += max(0, e1 - e2 + 1)  # e1 < e2
            return v
        if rel == "OverlappedBy":
            v = 0
            v += max(0, s2 - s1 + 1)  # s2 < s1
            v += max(0, s1 - e2 + 1)  # s1 < e2
            v += max(0, e2 - e1 + 1)  # e2 < e1
            return v
        if rel == "Starts":
            return abs(s1 - s2) + max(0, e1 - e2 + 1)
        if rel == "StartedBy":
            return abs(s1 - s2) + max(0, e2 - e1 + 1)
        if rel == "Finishes":
            return abs(e1 - e2) + max(0, s2 - s1 + 1)
        if rel == "FinishedBy":
            return abs(e1 - e2) + max(0, s1 - s2 + 1)
        if rel == "During":
            return max(0, s2 - s1 + 1) + max(0, e1 - e2 + 1)
        if rel == "Contains":
            return max(0, s1 - s2 + 1) + max(0, e2 - e1 + 1)
        if rel == "Equals":
            return abs(s1 - s2) + abs(e1 - e2)
        return 0  # unknown relation

    def evaluate(self, positions, sizes):
        """Evaluate total constraint violation for a candidate solution.

        positions: list of shape [num_objs][num_frames] -> (x, y)
        sizes: list of shape [num_objs] -> int (square bbox side)

        Returns: non-negative float (0 = valid solution).
        """
        penalty = 0.0

        # --- 1. Size ordering: sizes[o1]+2 <= sizes[o2] when rank[o1] < rank[o2] ---
        for i in range(self.num_objs):
            for j in range(self.num_objs):
                if self.size_rank[i] < self.size_rank[j]:
                    excess = sizes[i] + 2 - sizes[j]
                    if excess > 0:
                        penalty += excess

        # --- 2. Ego at origin (ALL frames) ---
        if self.ego_idx is not None:
            for t in range(self.num_frames):
                ex = positions[self.ego_idx][t][0]
                ey = positions[self.ego_idx][t][1]
                penalty += abs(ex) + abs(ey)

        # Pre-compute all bounding boxes
        bboxes = []  # [obj_idx][frame] -> (x_min, x_max, y_min, y_max)
        for i in range(self.num_objs):
            obj_bboxes = []
            for t in range(self.num_frames):
                x, y = positions[i][t]
                obj_bboxes.append(self.compute_bbox(x, y, sizes[i]))
            bboxes.append(obj_bboxes)

        # --- 3. Speed / movement constraints (fixed speed limits) ---
        for i, obj in enumerate(self.objects):
            for t in range(self.num_frames - 1):
                speed_cat = self.velocities.get((obj["id"], t), "normal")
                lim = self.speed_to_val.get(speed_cat, 22)
                dx = abs(positions[i][t + 1][0] - positions[i][t][0])
                dy = abs(positions[i][t + 1][1] - positions[i][t][1])
                excess = (dx + dy) - lim
                if excess > 0:
                    penalty += excess

        # --- 4. Allen RA constraints (strict, no tolerance) ---
        for t in range(self.num_frames):
            for entry in self.ra_matrix[t]:
                oid_i, oid_j, rx, ry = entry
                idx_i = self.id_map[oid_i]
                idx_j = self.id_map[oid_j]
                bb_i = bboxes[idx_i][t]
                bb_j = bboxes[idx_j][t]
                penalty += self.check_allen_relation(
                    rx, bb_i[0], bb_i[1], bb_j[0], bb_j[1]
                )
                penalty += self.check_allen_relation(
                    ry, bb_i[2], bb_i[3], bb_j[2], bb_j[3]
                )

        # --- 5. QDC distance constraints (two-sided, matching MiniZinc) ---
        for t in range(self.num_frames):
            for entry in self.qdc_matrix[t]:
                oid_i, oid_j, q = entry
                idx_i = self.id_map[oid_i]
                idx_j = self.id_map[oid_j]
                bb_i = bboxes[idx_i][t]
                bb_j = bboxes[idx_j][t]
                dx = max(0, max(bb_i[0] - bb_j[1], bb_j[0] - bb_i[1]))
                dy = max(0, max(bb_i[2] - bb_j[3], bb_j[2] - bb_i[3]))
                gap = dx + dy
                # Upper bound (not for very_far)
                upper = self.qdc_upper_thresholds.get(q)
                if upper is not None:
                    excess = gap - upper
                    if excess > 0:
                        penalty += excess
                # Lower bound (strict >): close needs gap > vc, etc.
                lower = self.qdc_lower_thresholds.get(q)
                if lower is not None:
                    deficit = lower - gap + 1
                    if deficit > 0:
                        penalty += deficit

        # --- 6. Bounds ---
        ml = MAP_LIMIT
        for i in range(self.num_objs):
            for t in range(self.num_frames):
                x, y = positions[i][t]
                penalty += max(0, abs(x) - ml)
                penalty += max(0, abs(y) - ml)

        return penalty
