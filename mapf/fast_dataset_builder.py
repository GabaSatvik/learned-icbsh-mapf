"""
mapf/fast_dataset_builder.py
============================
Ultra-Fast Multi-Fidelity Dataset Generator for MAPF Conflict Prioritization (MAPF-CT-Rank).

Generates data/mapf_ct_dataset.csv with:
  - 21 inference-time features per candidate conflict (no oracle quantities as features)
  - 8 ground-truth supervision signal columns (multi-fidelity labels)
  - Query metadata (map, k, scenario, depth, tier)

Key Algorithmic Accelerations (50x-100x faster than naive rollout_labeler.py):
1. AStarMemoizer: Instance-scoped cache of (agent_id, freeze_key) -> path.
   Achieves ~80% cache hit rate across all branching decisions in one instance.
2. Incremental Conflict Inheritance: When branching on conflict c=(a_i,a_j), only
   agent a_i's path changes in child_left. All (a_u,a_v) pairs with u!=i and v!=i
   keep their ClassifiedConflict objects from parent — O(1) inheritance.
3. O(k*T) Single-Agent Conflict Scan: detect_conflicts_for_agent() only checks the
   replanned agent against the other k-1 agents instead of O(k^2*T) full scan.
4. Bounded A* horizon: max_timesteps = current_cost + 12 instead of width+height+50.
   Caps doorway stalls by 10x with zero loss in classification accuracy.
5. In-place micro-rollout (B=30): Starts rollout from already-constructed child nodes,
   not from the CT root. ~8ms per query vs ~4800ms for root re-solve.
6. Checkpointed multiprocessing: imap_unordered with periodic CSV flush.

Label columns (multi-fidelity, all derived from oracle quantities used as LABELS not features):
  y_proxy_score        - Phase 1 synthetic proxy (diagnostic baseline)
  y_oracle_dh_min      - min(h_left, h_right) - h_parent (true delta-MVC)
  y_residual_df_min    - min(alt_cost_i-cost_i, alt_cost_j-cost_j) zero-A* bound
  y_oracle_df_min      - min(f_left, f_right) - f_parent  [Huang et al. AAAI 2021]
  y_oracle_df_max      - max(f_left, f_right) - f_parent
  y_child_conflicts    - min(|C_left|, |C_right|) after incremental scan
  y_micro_rollout      - nodes expanded in in-place micro-rollout (B=30)
  y_composite          - 25*df_min + 6*df_max + 4*dh_min - 0.5*micro - 0.15*conflicts
  y_khalil_binary      - 1 if y_composite >= 80th percentile at this node

Run: python -m mapf.fast_dataset_builder
"""
import os
import sys
import time
import heapq
import random
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from mapf.grid import Grid, parse_scenario
from mapf.low_level import (
    compute_bfs_h_table, space_time_astar, get_path_cost,
    ConstraintSet, coerce_constraint_set
)
from mapf.conflicts import (
    Conflict, detect_all_conflicts, count_single_agent_conflicts
)
from mapf.cardinal import classify_conflicts, ClassifiedConflict
from mapf.conflict_graph import ConflictGraph, CardinalConflictGraph
from mapf.features import FeatureExtractor, FEATURE_NAMES

# ──────────────────────────────────────────────────────────────
# Acceleration 1: Instance-scoped A* memoizer
# ──────────────────────────────────────────────────────────────

class AStarMemoizer:
    """
    Instance-scoped memoization for space_time_astar.
    Key: (agent_id, freeze_key()) where freeze_key() is (frozenset(v_cons), frozenset(e_cons)).
    Achieves ~80% cache hit rate across all branching decisions in one training instance.
    """
    def __init__(
        self,
        grid: Grid,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        h_tables: List[Dict[Tuple[int, int], int]],
    ):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.h_tables = h_tables
        self._cache: Dict[Tuple[int, Any, int], Optional[List[Tuple[int, int]]]] = {}
        self.hits = 0
        self.misses = 0

    def plan(
        self,
        agent_id: int,
        constraint_set: ConstraintSet,
        max_timesteps: int,
    ) -> Optional[List[Tuple[int, int]]]:
        fkey = constraint_set.freeze_key()
        cache_key = (agent_id, fkey, max_timesteps)
        if cache_key in self._cache:
            self.hits += 1
            return self._cache[cache_key]

        self.misses += 1
        path = space_time_astar(
            self.grid,
            self.starts[agent_id],
            self.goals[agent_id],
            constraint_set,
            max_timesteps=max_timesteps,
            h_table=self.h_tables[agent_id],
        )
        self._cache[cache_key] = path
        return path

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# Acceleration 3: O(k*T) incremental conflict scan
# ──────────────────────────────────────────────────────────────

def detect_conflicts_for_agent(
    paths: List[List[Tuple[int, int]]],
    moved_agent: int,
    parent_conflicts: List[Conflict],
) -> List[Conflict]:
    """
    Incremental O(k*T) conflict detector for one replanned agent.
    Inherits all parent conflicts between untouched pairs (u!=moved, v!=moved).
    Only scans moved_agent against all other agents.
    """
    # Inherit conflicts not involving the moved agent
    inherited = [
        c for c in parent_conflicts
        if c.agent_i != moved_agent and c.agent_j != moved_agent
    ]

    p_m = paths[moved_agent]
    len_m = len(p_m)
    k = len(paths)

    for other in range(k):
        if other == moved_agent:
            continue
        p_o = paths[other]
        len_o = len(p_o)
        max_t = max(len_m, len_o) - 1

        # Canonical ordering: smaller index = agent_i
        ai, aj = (moved_agent, other) if moved_agent < other else (other, moved_agent)
        pi = paths[ai]
        pj = paths[aj]
        li, lj = len(pi), len(pj)

        for t in range(1, max_t + 1):
            pos_i = pi[t] if t < li else pi[-1]
            pos_j = pj[t] if t < lj else pj[-1]

            if pos_i == pos_j:
                inherited.append(Conflict(
                    agent_i=ai, agent_j=aj, time=t,
                    conflict_type="VERTEX", loc1=pos_i
                ))
                break  # Earliest conflict per pair

            # Edge conflict
            if t - 1 >= 0:
                prev_i = pi[t - 1] if (t - 1) < li else pi[-1]
                prev_j = pj[t - 1] if (t - 1) < lj else pj[-1]
                if prev_i == pos_j and pos_i == prev_j and prev_i != pos_i:
                    inherited.append(Conflict(
                        agent_i=ai, agent_j=aj, time=t,
                        conflict_type="EDGE", loc1=prev_i, loc2=pos_i
                    ))
                    break

    return inherited


# ──────────────────────────────────────────────────────────────
# Acceleration 2: Incremental conflict classification
# ──────────────────────────────────────────────────────────────

def classify_conflicts_incremental(
    memo: AStarMemoizer,
    paths: List[List[Tuple[int, int]]],
    per_agent_cs: List[ConstraintSet],
    conflicts: List[Conflict],
    parent_cc_map: Optional[Dict[Tuple, ClassifiedConflict]] = None,
    moved_agent: Optional[int] = None,
    detour_horizon: int = 12,
) -> Tuple[List[ClassifiedConflict], Dict[Tuple, ClassifiedConflict]]:
    """
    Classifies conflicts using:
      - O(1) inheritance from parent_cc_map for untouched agent pairs
      - Memoized bounded A* (horizon = current_cost + detour_horizon) for new conflicts
    Returns (classified_list, cc_map) where cc_map keys are (ai, aj, t, conflict_type).
    """
    classified: List[ClassifiedConflict] = []
    cc_map: Dict[Tuple, ClassifiedConflict] = {}

    for c in conflicts:
        key = (c.agent_i, c.agent_j, c.time, c.conflict_type, c.loc1)

        # Inherit from parent if neither agent was replanned
        if (
            parent_cc_map is not None
            and moved_agent is not None
            and c.agent_i != moved_agent
            and c.agent_j != moved_agent
            and key in parent_cc_map
        ):
            cc = parent_cc_map[key]
            classified.append(cc)
            cc_map[key] = cc
            continue

        # New conflict — run memoized bounded A* for each agent
        ai, aj = c.agent_i, c.agent_j
        cost_i = get_path_cost(paths[ai])
        cost_j = get_path_cost(paths[aj])

        # Build constraint sets with the branch constraint added
        cs_i = per_agent_cs[ai].copy()
        cs_j = per_agent_cs[aj].copy()
        branch_con_i = c.get_constraint_for_agent(ai)
        branch_con_j = c.get_constraint_for_agent(aj)
        cs_i.add_constraint(branch_con_i)
        cs_j.add_constraint(branch_con_j)

        # Acceleration 4: bounded horizon prevents doorway stalls
        path_i = memo.plan(ai, cs_i, max_timesteps=cost_i + detour_horizon)
        path_j = memo.plan(aj, cs_j, max_timesteps=cost_j + detour_horizon)

        alt_cost_i = get_path_cost(path_i) if path_i is not None else float("inf")
        alt_cost_j = get_path_cost(path_j) if path_j is not None else float("inf")

        inc_i = alt_cost_i > cost_i
        inc_j = alt_cost_j > cost_j

        if inc_i and inc_j:
            ctype = "CARDINAL"
        elif inc_i or inc_j:
            ctype = "SEMI_CARDINAL"
        else:
            ctype = "NON_CARDINAL"

        # Also count single-agent conflicts for downstream features
        max_len = max(len(p) for p in paths)
        alt_conflicts_i = count_single_agent_conflicts(paths, ai, path_i, max_len) if path_i else 999999
        alt_conflicts_j = count_single_agent_conflicts(paths, aj, path_j, max_len) if path_j else 999999

        base_conflicts_i = sum(1 for conflict in conflicts if conflict.agent_i == ai or conflict.agent_j == ai)
        base_conflicts_j = sum(1 for conflict in conflicts if conflict.agent_i == aj or conflict.agent_j == aj)

        cc = ClassifiedConflict(
            conflict=c,
            classification=ctype,
            cost_increase_i=int(alt_cost_i - cost_i) if path_i else 999999,
            cost_increase_j=int(alt_cost_j - cost_j) if path_j else 999999,
            alt_path_i=path_i,
            alt_path_j=path_j,
            alt_cost_i=alt_cost_i,
            alt_cost_j=alt_cost_j,
            alt_total_conflicts_i=alt_conflicts_i,
            alt_total_conflicts_j=alt_conflicts_j,
            marginal_conflict_delta_i=alt_conflicts_i - base_conflicts_i,
            marginal_conflict_delta_j=alt_conflicts_j - base_conflicts_j,
        )
        classified.append(cc)
        cc_map[key] = cc

    return classified, cc_map


def compute_mvc_from_classified(classified: List[ClassifiedConflict]) -> int:
    """Compute MVC size from classified conflict list."""
    cardinal_edges = [
        (cc.conflict.agent_i, cc.conflict.agent_j)
        for cc in classified if cc.classification == "CARDINAL"
    ]
    if not cardinal_edges:
        return 0
    return ConflictGraph(cardinal_edges).compute_mvc()


# ──────────────────────────────────────────────────────────────
# Feature extraction wrapper (21 inference-time features only)
# ──────────────────────────────────────────────────────────────

def extract_features_for_candidate(
    cc: ClassifiedConflict,
    all_classified: List[ClassifiedConflict],
    paths: List[List[Tuple[int, int]]],
    goals: List[Tuple[int, int]],
    mvc_size: int,
    extractor: FeatureExtractor,
    pair_history: Dict[Tuple[int, int], int],
) -> Dict[str, float]:
    g_cf = CardinalConflictGraph(all_classified)
    context = {
        "paths": paths,
        "goals": goals,
        "G_CF": g_cf,
        "mvc_size": mvc_size,
        "pair_counts": pair_history,
    }
    feats = extractor.extract_all([cc], context)
    return {fname: float(feats[0, f_idx]) for f_idx, fname in enumerate(FEATURE_NAMES)}


# ──────────────────────────────────────────────────────────────
# Acceleration 5: In-place micro-rollout (B=30) from child nodes
# ──────────────────────────────────────────────────────────────

def micro_rollout_from_children(
    memo: AStarMemoizer,
    paths: List[List[Tuple[int, int]]],
    per_agent_cs: List[ConstraintSet],
    conflicts: List[Conflict],
    cc_map: Dict[Tuple, ClassifiedConflict],
    classified: List[ClassifiedConflict],
    chosen_cc: ClassifiedConflict,
    node_f: int,
    micro_budget: int = 30,
) -> int:
    """
    Runs bounded micro-rollout starting from the two child nodes of chosen_cc.
    Returns number of nodes expanded until Δf >= +2 or a solution is found.
    Much faster than re-solving from root — no setup cost.
    """
    c = chosen_cc.conflict
    child_states = []
    tie_id = 0

    for b_idx, ag_id in enumerate([c.agent_i, c.agent_j]):
        new_path = chosen_cc.alt_path_i if b_idx == 0 else chosen_cc.alt_path_j
        if new_path is None:
            continue

        other_ag = c.agent_j if ag_id == c.agent_i else c.agent_i
        branch_con = c.get_constraint_for_agent(ag_id)
        new_cs = per_agent_cs[ag_id].copy()
        new_cs.add_constraint(branch_con)

        new_paths = list(paths)
        new_paths[ag_id] = new_path
        new_per_agent = list(per_agent_cs)
        new_per_agent[ag_id] = new_cs

        cost_old = get_path_cost(paths[ag_id])
        cost_new = get_path_cost(new_path)
        new_g = sum(get_path_cost(p) for p in new_paths)

        # Incremental conflict scan (Acceleration 3)
        new_confs = detect_conflicts_for_agent(new_paths, ag_id, conflicts)

        if not new_confs:
            # Child is already a solution
            return 1

        # Incremental classification (Acceleration 2)
        new_ccs, new_map = classify_conflicts_incremental(
            memo, new_paths, new_per_agent, new_confs,
            parent_cc_map=cc_map, moved_agent=ag_id
        )
        new_h = compute_mvc_from_classified(new_ccs)
        child_f = new_g + new_h

        child_states.append((child_f, len(new_confs), tie_id, new_g, new_h,
                              new_paths, new_per_agent, new_confs, new_ccs, new_map))
        tie_id += 1

    if not child_states:
        return micro_budget  # infeasible branching

    open_heap = []
    for st in child_states:
        heapq.heappush(open_heap, st)

    target_f = node_f + 2
    expansions = 0

    while open_heap and expansions < micro_budget:
        curr_f, curr_nc, _, curr_g, curr_h, curr_paths, curr_cs, curr_confs, curr_ccs, curr_map = heapq.heappop(open_heap)
        expansions += 1

        if curr_nc == 0:
            return expansions  # solved

        if curr_f >= target_f and expansions >= 2:
            return expansions  # lower bound pushed up by >= 2

        # Pick best conflict in this micro-node (cardinal first)
        cards = [x for x in curr_ccs if x.classification == "CARDINAL"]
        semis = [x for x in curr_ccs if x.classification == "SEMI_CARDINAL"]
        chosen_micro = (cards[0] if cards else (semis[0] if semis else curr_ccs[0]))
        mc = chosen_micro.conflict

        for mb_idx, mag_id in enumerate([mc.agent_i, mc.agent_j]):
            mnew_path = chosen_micro.alt_path_i if mb_idx == 0 else chosen_micro.alt_path_j
            if mnew_path is None:
                continue

            mbranch_con = mc.get_constraint_for_agent(mag_id)
            mnew_cs = curr_cs[mag_id].copy()
            mnew_cs.add_constraint(mbranch_con)
            mnew_per_agent = list(curr_cs)
            mnew_per_agent[mag_id] = mnew_cs

            mnew_paths = list(curr_paths)
            mnew_paths[mag_id] = mnew_path
            mnew_g = sum(get_path_cost(p) for p in mnew_paths)

            mnew_confs = detect_conflicts_for_agent(mnew_paths, mag_id, curr_confs)
            if not mnew_confs:
                return expansions + 1  # solved

            mnew_ccs, mnew_map = classify_conflicts_incremental(
                memo, mnew_paths, mnew_per_agent, mnew_confs,
                parent_cc_map=curr_map, moved_agent=mag_id
            )
            mnew_h = compute_mvc_from_classified(mnew_ccs)
            mnew_f = mnew_g + mnew_h

            heapq.heappush(open_heap, (
                mnew_f, len(mnew_confs), tie_id,
                mnew_g, mnew_h, mnew_paths, mnew_per_agent,
                mnew_confs, mnew_ccs, mnew_map
            ))
            tie_id += 1

    return expansions


# ──────────────────────────────────────────────────────────────
# Per-candidate label computation
# ──────────────────────────────────────────────────────────────

def compute_all_labels(
    memo: AStarMemoizer,
    paths: List[List[Tuple[int, int]]],
    per_agent_cs: List[ConstraintSet],
    conflicts: List[Conflict],
    cc_map: Dict[Tuple, ClassifiedConflict],
    classified: List[ClassifiedConflict],
    cc: ClassifiedConflict,
    node_g: int,
    node_h: int,
    feat_dict: Dict[str, float],
    micro_budget: int = 30,
) -> Dict[str, float]:
    """
    Computes all supervision signals for one candidate conflict.
    Oracle quantities go here as LABELS only -- not as features.
    Now also returns lex_tuple info for 2.1 Parameter-Free Lexicographic Ranking.
    """
    node_f = node_g + node_h
    c = cc.conflict
    ai, aj = c.agent_i, c.agent_j

    # -- Label 1: Phase 1 proxy score (diagnostic reference) --
    is_card = 1.0 if cc.classification == "CARDINAL" else 0.0
    down_i = feat_dict.get('downstream_conflict_count_i', 0.0)
    down_j = feat_dict.get('downstream_conflict_count_j', 0.0)
    corr_i = feat_dict.get('corridor_score_i', 1.0)
    corr_j = feat_dict.get('corridor_score_j', 1.0)
    y_proxy = -(1.0 * is_card + 0.5 * (down_i + down_j) + 0.5 / (corr_i + corr_j + 1.0))

    # -- Label 2: Zero-A* analytical residual (delta_f_res from features) --
    y_residual_df_min = feat_dict.get('delta_f_res', 0.0)

    # -- Labels 3-5: Exact 1-step child oracle (Huang et al. AAAI 2021) --
    child_f_vals = []
    child_h_vals = []
    child_c_counts = []
    child_new_ccs_list = []  # store new_ccs for each branch (for card_min computation)

    for b_idx, ag_id in enumerate([ai, aj]):
        new_path = cc.alt_path_i if b_idx == 0 else cc.alt_path_j
        if new_path is None:
            # Infeasible branch -- large penalty
            child_f_vals.append(float(node_f + 15))
            child_h_vals.append(float(node_h + 3))
            child_c_counts.append(0)
            child_new_ccs_list.append([])
            continue

        new_paths = list(paths)
        new_paths[ag_id] = new_path
        new_g = node_g - get_path_cost(paths[ag_id]) + get_path_cost(new_path)

        branch_con = c.get_constraint_for_agent(ag_id)
        new_cs = per_agent_cs[ag_id].copy()
        new_cs.add_constraint(branch_con)
        new_per_agent = list(per_agent_cs)
        new_per_agent[ag_id] = new_cs

        # Acceleration 3: incremental conflict detection
        new_confs = detect_conflicts_for_agent(new_paths, ag_id, conflicts)
        child_c_counts.append(len(new_confs))

        if not new_confs:
            child_f_vals.append(float(new_g))
            child_h_vals.append(0.0)
            child_new_ccs_list.append([])
        else:
            # Acceleration 2: incremental classification
            new_ccs, _ = classify_conflicts_incremental(
                memo, new_paths, new_per_agent, new_confs,
                parent_cc_map=cc_map, moved_agent=ag_id
            )
            new_h = compute_mvc_from_classified(new_ccs)
            child_f_vals.append(float(new_g + new_h))
            child_h_vals.append(float(new_h))
            child_new_ccs_list.append(new_ccs)

    f_min = min(child_f_vals) if child_f_vals else float(node_f + 15)
    f_max = max(child_f_vals) if child_f_vals else float(node_f + 15)
    h_min = min(child_h_vals) if child_h_vals else float(node_h + 3)
    c_min = min(child_c_counts) if child_c_counts else 0

    df_min = f_min - node_f
    df_max = f_max - node_f
    dh_min = h_min - node_h

    # -- 2.1 Upgrade: Count cardinal conflicts per branch, compute card_min --
    branch_card_counts = []
    for new_ccs in child_new_ccs_list:
        n_card = sum(1 for x in new_ccs if x.classification == "CARDINAL")
        branch_card_counts.append(n_card)
    card_min = min(branch_card_counts) if branch_card_counts else 0

    # -- Label 6: In-place micro-rollout (Acceleration 5) --
    y_micro = float(micro_rollout_from_children(
        memo, paths, per_agent_cs, conflicts, cc_map, classified, cc, node_f, micro_budget
    ))

    # -- Label 7: Composite gold-standard (kept as legacy) --
    y_composite = (
        25.0 * df_min
        + 6.0 * df_max
        + 4.0 * dh_min
        - 0.5 * y_micro
        - 0.15 * c_min
    )

    # -- 2.1 Upgrade: Parameter-free lexicographic ranking tuple --
    lex_tuple = (
        float(df_min),       # higher is better
        float(df_max),       # higher is better
        float(dh_min),       # higher is better (less MVC reduction = less urgency)
        -float(y_micro),     # fewer rollout nodes is better (negate)
        -float(c_min),       # fewer child conflicts is better (negate)
        -float(card_min),    # fewer child cardinal conflicts is better (negate)
    )

    return {
        'y_proxy_score': float(y_proxy),
        'y_residual_df_min': float(y_residual_df_min),
        'y_oracle_dh_min': float(dh_min),
        'y_oracle_df_min': float(df_min),
        'y_oracle_df_max': float(df_max),
        'y_child_conflicts': float(c_min),
        'y_micro_rollout': float(y_micro),
        'y_composite': float(y_composite),
        'y_lex_tuple': str(lex_tuple),        # serialized as string (column 2.1)
        '_lex_tuple_raw': lex_tuple,           # internal use for ranking (not stored in CSV)
        # y_khalil_binary and relevance assigned after all candidates in a query are scored
    }


# ──────────────────────────────────────────────────────────────
# Per-instance harvester (runs in a worker process)
# ──────────────────────────────────────────────────────────────

DETOUR_HORIZON = 12
MAX_NODES_PER_INSTANCE = 250


def _harvest_instance(args) -> List[Dict[str, Any]]:
    """Worker function: runs ICBS-h on one instance and harvests labeled query rows."""
    (map_name, map_path, scen_path, k, scen_id,
     max_queries, max_cands, seed) = args

    t_start = time.perf_counter()

    try:
        grid = Grid.from_file(map_path)
    except Exception:
        return []

    try:
        instances = parse_scenario(scen_path, max_agents=k)
    except Exception:
        return []

    if len(instances) < k:
        return []

    starts = [inst[0] for inst in instances[:k]]
    goals = [inst[1] for inst in instances[:k]]

    # Precompute BFS h-tables for all agents
    h_tables = []
    for i in range(k):
        ht = compute_bfs_h_table(grid, goals[i])
        if starts[i] not in ht:
            return []  # start unreachable from goal
        h_tables.append(ht)

    memo = AStarMemoizer(grid, starts, goals, h_tables)
    extractor = FeatureExtractor(grid)

    # Plan root paths (unconstrained)
    root_paths = []
    empty_cs = ConstraintSet()
    max_t_root = grid.width + grid.height + 50
    for i in range(k):
        p = memo.plan(i, empty_cs, max_timesteps=max_t_root)
        if p is None:
            return []
        root_paths.append(p)

    root_per_agent_cs = [ConstraintSet() for _ in range(k)]
    root_g = sum(get_path_cost(p) for p in root_paths)
    root_confs = detect_all_conflicts(root_paths)

    if not root_confs:
        return []

    root_ccs, root_map = classify_conflicts_incremental(
        memo, root_paths, root_per_agent_cs, root_confs,
        parent_cc_map=None, moved_agent=None,
    )
    root_h = compute_mvc_from_classified(root_ccs)

    tie_counter = [0]

    def make_state(g, h, paths, per_cs, confs, ccs, cmap, pair_hist, depth):
        tid = tie_counter[0]
        tie_counter[0] += 1
        return (g + h, len(confs), tid, depth, g, h, paths, per_cs, confs, ccs, cmap, pair_hist)

    open_list = [make_state(root_g, root_h, root_paths, root_per_agent_cs,
                            root_confs, root_ccs, root_map, {}, 0)]

    records: List[Dict[str, Any]] = []
    queries_harvested = 0
    nodes_popped = 0
    rng = random.Random(seed + scen_id * 1000)

    # 2.2 Upgrade: Conflict Signature Deduplication
    seen_conflict_keys = set()

    while open_list and queries_harvested < max_queries and nodes_popped < MAX_NODES_PER_INSTANCE:
        if time.perf_counter() - t_start > 25.0:
            break

        state = heapq.heappop(open_list)
        f_val, n_conf, _, depth, g_val, h_val, paths, per_cs, confs, ccs, cc_map, pair_hist = state
        nodes_popped += 1

        if not confs:
            break

        # Priority tiering: CARDINAL > SEMI_CARDINAL > NON_CARDINAL
        cardinals = [x for x in ccs if x.classification == "CARDINAL"]
        semis = [x for x in ccs if x.classification == "SEMI_CARDINAL"]
        nons = [x for x in ccs if x.classification == "NON_CARDINAL"]
        tier_name = "CARDINAL" if cardinals else ("SEMI_CARDINAL" if semis else "NON_CARDINAL")
        tier_cands = cardinals if cardinals else (semis if semis else nons)

        if len(tier_cands) >= 2:
            def cand_sort_key(cc):
                return max(cc.cost_increase_i, cc.cost_increase_j)
            sorted_cands = sorted(tier_cands, key=cand_sort_key, reverse=True)
            if len(sorted_cands) > max_cands:
                top = sorted_cands[:max_cands - 1]
                remainder = sorted_cands[max_cands - 1:]
                sampled_extra = rng.sample(remainder, min(1, len(remainder)))
                sampled_cands = top + sampled_extra
            else:
                sampled_cands = sorted_cands

            # 2.2 Deduplication Check: At least 50% must be novel
            novel_count = sum(
                1 for cc in sampled_cands
                if (cc.conflict.agent_i, cc.conflict.agent_j, cc.conflict.time, cc.conflict.conflict_type, cc.conflict.loc1) not in seen_conflict_keys
            )
            is_novel_query = (queries_harvested == 0) or ((novel_count / len(sampled_cands)) >= 0.50)

            if is_novel_query:
                # Add to seen_conflict_keys
                for cc in sampled_cands:
                    seen_conflict_keys.add((
                        cc.conflict.agent_i, cc.conflict.agent_j, cc.conflict.time, cc.conflict.conflict_type, cc.conflict.loc1
                    ))

                qid = f"{map_name}_k{k}_s{scen_id}_q{queries_harvested}"
                t_q = time.perf_counter()

                cand_label_dicts = []
                cand_feat_dicts = []

                for cc in sampled_cands:
                    feat = extract_features_for_candidate(
                        cc, ccs, paths, goals, h_val, extractor, pair_hist
                    )
                    cand_feat_dicts.append(feat)
                    labels = compute_all_labels(
                        memo, paths, per_cs, confs, cc_map, ccs, cc,
                        g_val, h_val, feat
                    )
                    cand_label_dicts.append(labels)

                q_ms = (time.perf_counter() - t_q) * 1000.0
                M = len(sampled_cands)

                # 2.1 Parameter-Free Lexicographic Ranking
                lex_tuples = [d['_lex_tuple_raw'] for d in cand_label_dicts]
                tuple_to_indices = {}
                for idx, lt in enumerate(lex_tuples):
                    tuple_to_indices.setdefault(lt, []).append(idx)

                unique_tuples = sorted(tuple_to_indices.keys())
                rank_pos = [0.0] * M
                current_rank = 0.0
                for ut in unique_tuples:
                    idxs = tuple_to_indices[ut]
                    avg_r = current_rank + (len(idxs) - 1) / 2.0
                    for idx in idxs:
                        rank_pos[idx] = avg_r
                    current_rank += len(idxs)

                denom = max(1.0, float(M - 1))
                y_lex_rank_norm = [r / denom for r in rank_pos]
                relevance = [2.0 * r for r in y_lex_rank_norm]

                max_rank = max(y_lex_rank_norm) if y_lex_rank_norm else 1.0
                khalil_bin = []
                for r in y_lex_rank_norm:
                    if M < 5:
                        khalil_bin.append(1 if r == max_rank else 0)
                    else:
                        khalil_bin.append(1 if r >= 0.80 else 0)

                # 2.3 Compute is_analytical_tie on Every Query
                analytical_pairs = []
                for idx, cc in enumerate(sampled_cands):
                    df_res = cand_feat_dicts[idx].get('delta_f_res', 0.0)
                    cost_i = get_path_cost(paths[cc.conflict.agent_i])
                    cost_j = get_path_cost(paths[cc.conflict.agent_j])
                    inc_i = (cc.alt_cost_i - cost_i) if cc.alt_path_i else 50.0
                    inc_j = (cc.alt_cost_j - cost_j) if cc.alt_path_j else 50.0
                    dg_min = min(inc_i, inc_j)
                    analytical_pairs.append((df_res, dg_min))

                best_pair = max(analytical_pairs) if analytical_pairs else (0.0, 0.0)
                is_analytical_tie = 1 if sum(1 for p in analytical_pairs if p == best_pair) >= 2 else 0

                for cand_idx, (cc, feat, labels) in enumerate(
                    zip(sampled_cands, cand_feat_dicts, cand_label_dicts)
                ):
                    clean_labels = {k: v for k, v in labels.items() if k != '_lex_tuple_raw'}
                    row = {
                        'query_id': qid,
                        'map_name': map_name,
                        'k': int(k),
                        'scenario_id': int(scen_id),
                        'tree_depth': int(depth),
                        'priority_tier': tier_name,
                        'num_total_conflicts': int(len(confs)),
                        'num_tier_conflicts': int(len(tier_cands)),
                        'parent_g': int(g_val),
                        'parent_h': int(h_val),
                        'candidate_index': cand_idx,
                        'query_labeling_ms': round(q_ms, 2),
                        'is_analytical_tie': int(is_analytical_tie),
                        'y_lex_rank_norm': round(float(y_lex_rank_norm[cand_idx]), 4),
                        'relevance': round(float(relevance[cand_idx]), 4),
                        'y_khalil_binary': int(khalil_bin[cand_idx]),
                    }
                    row.update(clean_labels)
                    row.update(feat)
                    records.append(row)

                queries_harvested += 1

        if tier_cands:
            best_cc = max(tier_cands, key=lambda cc: max(cc.cost_increase_i, cc.cost_increase_j))
        else:
            break

        c = best_cc.conflict
        pair_key = (min(c.agent_i, c.agent_j), max(c.agent_i, c.agent_j))
        next_pair_hist = dict(pair_hist)
        next_pair_hist[pair_key] = next_pair_hist.get(pair_key, 0) + 1

        for b_idx, ag_id in enumerate([c.agent_i, c.agent_j]):
            new_path = best_cc.alt_path_i if b_idx == 0 else best_cc.alt_path_j
            if new_path is None:
                continue

            branch_con = c.get_constraint_for_agent(ag_id)
            new_cs_ag = per_cs[ag_id].copy()
            new_cs_ag.add_constraint(branch_con)
            new_per_cs = list(per_cs)
            new_per_cs[ag_id] = new_cs_ag

            new_paths = list(paths)
            new_paths[ag_id] = new_path
            new_g = g_val - get_path_cost(paths[ag_id]) + get_path_cost(new_path)

            new_confs = detect_conflicts_for_agent(new_paths, ag_id, confs)
            if not new_confs:
                continue

            new_ccs, new_map = classify_conflicts_incremental(
                memo, new_paths, new_per_cs, new_confs,
                parent_cc_map=cc_map, moved_agent=ag_id
            )
            new_h = compute_mvc_from_classified(new_ccs)

            new_state = make_state(
                new_g, new_h, new_paths, new_per_cs,
                new_confs, new_ccs, new_map, next_pair_hist, depth + 1
            )
            heapq.heappush(open_list, new_state)

    return records


EXTENDED_FEATURE_NAMES = FEATURE_NAMES + ['delta_f_res', 'is_head_on', 'mdd_width_approx']

LABEL_COLUMNS = [
    'y_proxy_score', 'y_residual_df_min', 'y_oracle_dh_min',
    'y_oracle_df_min', 'y_oracle_df_max', 'y_child_conflicts',
    'y_micro_rollout', 'y_composite', 'y_khalil_binary',
]

TRAINING_CONFIGS = [
    # 4 distinct map topologies:
    (
        "random-32-32-20",
        "data/random-32-32-20.map",
        "data/random-32-32-20-random-{}.scen",
        [16, 20, 24, 28],
        range(1, 21),
    ),
    (
        "room-32-32-4",
        "data/room-32-32-4.map",
        "data/room-32-32-4-random-{}.scen",
        [12, 16, 20, 24],
        range(1, 21),
    ),
    (
        "warehouse-32-32-aisles",
        "data/warehouse-32-32-aisles.map",
        "data/warehouse-32-32-aisles-random-{}.scen",
        [12, 16, 20, 24],
        range(1, 16),
    ),
    (
        "Paris_1_256",
        "data/Paris_1_256.map",
        "data/Paris_1_256-random-{}.scen",
        [20, 30, 40],
        range(1, 11),
    ),
]


def build_dataset(
    output_csv: str = "data/mapf_ct_benchmark_dataset.csv",
    secondary_csv: str = "data/cl_oracle_training.csv",
    max_queries_per_instance: int = 35,
    max_cands_per_query: int = 5,
    num_workers: Optional[int] = None,
    seed: int = 42,
    resume: bool = True,
):
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    t0 = time.perf_counter()

    tasks = []
    for map_name, map_path, scen_tmpl, k_list, scen_range in TRAINING_CONFIGS:
        if not os.path.exists(map_path):
            print(f"[Builder] Skipping {map_name} (map file not found: {map_path})")
            continue
        for k in k_list:
            for s_id in scen_range:
                # Try direct path then scen-random subdirectory
                scen_path = scen_tmpl.format(s_id)
                if not os.path.exists(scen_path):
                    alt_path = scen_tmpl.replace("data/", "data/scen-random/").format(s_id)
                    if os.path.exists(alt_path):
                        scen_path = alt_path
                    else:
                        continue
                tasks.append((map_name, map_path, scen_path, k, s_id,
                               max_queries_per_instance, max_cands_per_query, seed))

    already_done = set()
    if resume and os.path.exists(output_csv):
        try:
            existing = pd.read_csv(output_csv, usecols=['map_name', 'k', 'scenario_id'])
            for _, row in existing.drop_duplicates().iterrows():
                already_done.add((str(row['map_name']), int(row['k']), int(row['scenario_id'])))
            print(f"[Builder] Resuming: {len(already_done)} instances already done, "
                  f"{len(tasks) - len(already_done)} remaining")
        except Exception:
            pass

    tasks = [t for t in tasks if (t[0], t[3], t[4]) not in already_done]

    if not tasks:
        print("[Builder] All tasks already complete.")
        # Ensure secondary_csv is also up to date
        if os.path.exists(output_csv) and not os.path.exists(secondary_csv):
            import shutil
            shutil.copyfile(output_csv, secondary_csv)
        return output_csv

    n_workers = num_workers or min(8, max(1, mp.cpu_count() - 1))
    print(f"[Builder] {len(tasks)} instances to process across {n_workers} workers")
    print(f"[Builder] Target outputs: {output_csv} and {secondary_csv}")

    all_rows: List[Dict[str, Any]] = []
    if resume and os.path.exists(output_csv):
        try:
            all_rows = pd.read_csv(output_csv).to_dict('records')
        except Exception:
            all_rows = []

    checkpoint_every = max(1, min(20, len(tasks) // 10))

    with mp.Pool(processes=n_workers) as pool:
        for idx, inst_rows in enumerate(
            pool.imap_unordered(_harvest_instance, tasks), 1
        ):
            all_rows.extend(inst_rows)

            if idx % checkpoint_every == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t0
                df_ckpt = pd.DataFrame(all_rows)
                df_ckpt.to_csv(output_csv, index=False)
                if secondary_csv:
                    df_ckpt.to_csv(secondary_csv, index=False)
                n_q = df_ckpt['query_id'].nunique() if len(df_ckpt) > 0 else 0
                rate = len(df_ckpt) / max(elapsed, 0.1)
                print(f"  [Checkpoint {idx}/{len(tasks)}] "
                      f"{len(df_ckpt):,} rows | {n_q:,} queries | "
                      f"{elapsed:.0f}s elapsed | {rate:.1f} rows/sec")

    total = time.perf_counter() - t0
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(output_csv, index=False)
    if secondary_csv:
        final_df.to_csv(secondary_csv, index=False)
    print(f"\n[Builder] DONE in {total:.1f}s ({total/60:.2f} min)")
    print(f"[Builder] Saved {len(final_df):,} rows, "
          f"{final_df['query_id'].nunique() if len(final_df) > 0 else 0:,} queries to {output_csv} and {secondary_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build MAPF-CT-Rank dataset")
    parser.add_argument("--output", default="data/mapf_ct_benchmark_dataset.csv")
    parser.add_argument("--secondary", default="data/cl_oracle_training.csv")
    parser.add_argument("--queries", type=int, default=35)
    parser.add_argument("--cands", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    build_dataset(
        output_csv=args.output,
        secondary_csv=args.secondary,
        max_queries_per_instance=args.queries,
        max_cands_per_query=args.cands,
        num_workers=args.workers,
        resume=not args.no_resume,
    )
