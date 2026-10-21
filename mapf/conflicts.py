"""
Conflict detection and representation for Multi-Agent Path Finding.
Supports both (agent_1, agent_2) and (agent_i, agent_j) naming conventions.
"""
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class Constraint:
    agent: int
    time: int
    constraint_type: str  # 'VERTEX' or 'EDGE'
    loc_1: Tuple[int, int]
    loc_2: Optional[Tuple[int, int]] = None

    @property
    def loc(self) -> Tuple[int, int]:
        return self.loc_1


@dataclass(frozen=True)
class Conflict:
    agent_1: int
    agent_2: int
    time: int
    conflict_type: str  # 'VERTEX' or 'EDGE'
    loc_1: Tuple[int, int]
    loc_2: Optional[Tuple[int, int]] = None

    @property
    def agent_i(self) -> int:
        return self.agent_1

    @property
    def agent_j(self) -> int:
        return self.agent_2

    @property
    def loc_i(self) -> Tuple[int, int]:
        return self.loc_1

    @property
    def loc_j(self) -> Optional[Tuple[int, int]]:
        return self.loc_2

    def get_constraint_for_agent(self, agent: int, other_agent: int = -1) -> Constraint:
        ctype = self.conflict_type.upper()
        if ctype == "VERTEX":
            return Constraint(
                agent=agent,
                time=self.time,
                constraint_type="VERTEX",
                loc_1=self.loc_1,
            )
        else:
            # Edge conflict
            if agent == self.agent_1:
                loc_from, loc_to = self.loc_1, self.loc_2
            else:
                loc_from, loc_to = self.loc_2, self.loc_1
            return Constraint(
                agent=agent,
                time=self.time,
                constraint_type="EDGE",
                loc_1=loc_from,
                loc_2=loc_to,
            )


def get_agent_pos_at_t(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
    if t < len(path):
        return path[t]
    return path[-1] if path else (0, 0)


def detect_all_conflicts(paths: List[List[Tuple[int, int]]]) -> List[Conflict]:
    """Find all conflicts across all pairs of agents and timesteps."""
    conflicts = []
    num_agents = len(paths)
    if num_agents <= 1:
        return conflicts

    max_t = max(len(p) for p in paths) if paths else 0

    for t in range(1, max_t + 1):
        for a1 in range(num_agents):
            pos1 = get_agent_pos_at_t(paths[a1], t)
            prev1 = get_agent_pos_at_t(paths[a1], t - 1)

            for a2 in range(a1 + 1, num_agents):
                pos2 = get_agent_pos_at_t(paths[a2], t)
                prev2 = get_agent_pos_at_t(paths[a2], t - 1)

                if pos1 == pos2:
                    conflicts.append(
                        Conflict(
                            agent_1=a1,
                            agent_2=a2,
                            time=t,
                            conflict_type="VERTEX",
                            loc_1=pos1,
                            loc_2=pos1,
                        )
                    )

                if prev1 == pos2 and pos1 == prev2 and prev1 != pos1:
                    conflicts.append(
                        Conflict(
                            agent_1=a1,
                            agent_2=a2,
                            time=t,
                            conflict_type="EDGE",
                            loc_1=prev1,
                            loc_2=pos1,
                        )
                    )
    return conflicts


find_all_conflicts = detect_all_conflicts


def find_first_conflict(paths: List[List[Tuple[int, int]]]) -> Optional[Conflict]:
    """Find the earliest conflict between any pair of agents."""
    num_agents = len(paths)
    if num_agents <= 1:
        return None
    max_t = max(len(p) for p in paths) if paths else 0

    for t in range(1, max_t + 1):
        # 1. Vertex conflicts
        pos_map = {}
        for a in range(num_agents):
            pos = get_agent_pos_at_t(paths[a], t)
            if pos in pos_map:
                return Conflict(
                    agent_1=pos_map[pos],
                    agent_2=a,
                    time=t,
                    conflict_type="VERTEX",
                    loc_1=pos,
                    loc_2=pos,
                )
            pos_map[pos] = a

        # 2. Edge conflicts
        for a1 in range(num_agents):
            p1_prev = get_agent_pos_at_t(paths[a1], t - 1)
            p1_curr = get_agent_pos_at_t(paths[a1], t)
            for a2 in range(a1 + 1, num_agents):
                p2_prev = get_agent_pos_at_t(paths[a2], t - 1)
                p2_curr = get_agent_pos_at_t(paths[a2], t)
                if p1_prev == p2_curr and p1_curr == p2_prev and p1_prev != p1_curr:
                    return Conflict(
                        agent_1=a1,
                        agent_2=a2,
                        time=t,
                        conflict_type="EDGE",
                        loc_1=p1_prev,
                        loc_2=p1_curr,
                    )
    return None
