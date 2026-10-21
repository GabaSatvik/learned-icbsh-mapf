"""
Conflict detection and constraint representation for Multi-Agent Path Finding.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class Constraint:
    agent: int
    time: int
    loc1: Tuple[int, int]
    loc2: Optional[Tuple[int, int]] = None
    conflict_type: str = "VERTEX"  # "VERTEX" or "EDGE"


@dataclass(frozen=True)
class Conflict:
    agent_i: int
    agent_j: int
    time: int
    conflict_type: str  # "VERTEX" or "EDGE"
    loc1: Tuple[int, int]
    loc2: Optional[Tuple[int, int]] = None

    @property
    def agent_1(self) -> int:
        return self.agent_i

    @property
    def agent_2(self) -> int:
        return self.agent_j

    @property
    def loc_1(self) -> Tuple[int, int]:
        return self.loc1

    @property
    def loc_2(self) -> Optional[Tuple[int, int]]:
        return self.loc2

    def get_constraint_for_agent(
        self, agent_id: int, other_agent: Optional[int] = None
    ) -> Constraint:
        ctype = self.conflict_type.upper()
        if ctype == "VERTEX":
            return Constraint(
                agent=agent_id,
                time=self.time,
                loc1=self.loc1,
                loc2=None,
                conflict_type="VERTEX",
            )
        else:
            if agent_id == self.agent_i:
                return Constraint(
                    agent=agent_id,
                    time=self.time,
                    loc1=self.loc1,
                    loc2=self.loc2,
                    conflict_type="EDGE",
                )
            else:
                return Constraint(
                    agent=agent_id,
                    time=self.time,
                    loc1=self.loc2 if self.loc2 is not None else self.loc1,
                    loc2=self.loc1,
                    conflict_type="EDGE",
                )


def get_agent_pos_at_t(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
    if t < len(path):
        return path[t]
    return path[-1]


def detect_first_conflict(paths: List[List[Tuple[int, int]]]) -> Optional[Conflict]:
    """Find the earliest conflict between any pair of agents."""
    num_agents = len(paths)
    if num_agents < 2:
        return None
    max_t = max(len(p) for p in paths) - 1
    if max_t <= 0:
        return None

    for t in range(1, max_t + 1):
        # 1. Vertex conflicts at timestep t
        pos_map = {}
        for a in range(num_agents):
            pos = path_pos(paths[a], t)
            if pos in pos_map:
                return Conflict(
                    agent_i=pos_map[pos],
                    agent_j=a,
                    time=t,
                    conflict_type="VERTEX",
                    loc1=pos,
                )
            pos_map[pos] = a

        # 2. Edge conflicts between t-1 and t
        for a1 in range(num_agents):
            p1_prev = path_pos(paths[a1], t - 1)
            p1_curr = path_pos(paths[a1], t)
            for a2 in range(a1 + 1, num_agents):
                p2_prev = path_pos(paths[a2], t - 1)
                p2_curr = path_pos(paths[a2], t)
                if p1_prev == p2_curr and p1_curr == p2_prev and p1_prev != p1_curr:
                    return Conflict(
                        agent_i=a1,
                        agent_j=a2,
                        time=t,
                        conflict_type="EDGE",
                        loc1=p1_prev,
                        loc2=p1_curr,
                    )
    return None


def path_pos(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
    return path[t] if t < len(path) else path[-1]


def detect_all_conflicts(paths: List[List[Tuple[int, int]]]) -> List[Conflict]:
    """Find all vertex and edge conflicts across all pairs of agents and timesteps."""
    conflicts: List[Conflict] = []
    num_agents = len(paths)
    if num_agents < 2:
        return conflicts
    max_t = max(len(p) for p in paths) - 1
    if max_t <= 0:
        return conflicts

    for t in range(1, max_t + 1):
        for a1 in range(num_agents):
            p1 = paths[a1]
            pos1 = p1[t] if t < len(p1) else p1[-1]
            prev1 = p1[t - 1] if (t - 1) < len(p1) else p1[-1]
            for a2 in range(a1 + 1, num_agents):
                p2 = paths[a2]
                pos2 = p2[t] if t < len(p2) else p2[-1]
                prev2 = p2[t - 1] if (t - 1) < len(p2) else p2[-1]

                if pos1 == pos2:
                    conflicts.append(
                        Conflict(
                            agent_i=a1,
                            agent_j=a2,
                            time=t,
                            conflict_type="VERTEX",
                            loc1=pos1,
                        )
                    )
                elif prev1 == pos2 and pos1 == prev2 and prev1 != pos1:
                    conflicts.append(
                        Conflict(
                            agent_i=a1,
                            agent_j=a2,
                            time=t,
                            conflict_type="EDGE",
                            loc1=prev1,
                            loc2=pos1,
                        )
                    )
    return conflicts


def count_single_agent_conflicts(
    paths: List[List[Tuple[int, int]]],
    agent_id: int,
    alt_path: List[Tuple[int, int]],
    max_path_len: int,
) -> int:
    """
    Fast O(T * k) count of conflicts between `alt_path` (for `agent_id`) and all other agents.
    Avoids O(T * k^2) full all-pairs scan during 1-step conflict classification.
    """
    num_agents = len(paths)
    max_t = max(max_path_len, len(alt_path)) - 1
    if max_t <= 0:
        return 0

    cnt = 0
    len_a = len(alt_path)
    last_a = alt_path[-1]

    for other in range(num_agents):
        if other == agent_id:
            continue
        p_o = paths[other]
        len_o = len(p_o)
        last_o = p_o[-1]
        tmax_pair = max(len_a, len_o) - 1
        for t in range(1, tmax_pair + 1):
            pos_a = alt_path[t] if t < len_a else last_a
            pos_o = p_o[t] if t < len_o else last_o
            if pos_a == pos_o:
                cnt += 1
            else:
                prev_a = alt_path[t - 1] if (t - 1) < len_a else last_a
                prev_o = p_o[t - 1] if (t - 1) < len_o else last_o
                if prev_a == pos_o and pos_a == prev_o and prev_a != pos_a:
                    cnt += 1
    return cnt


find_first_conflict = detect_first_conflict
find_all_conflicts = detect_all_conflicts
