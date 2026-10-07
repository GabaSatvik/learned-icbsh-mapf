"""
Conflict detection and representation for Multi-Agent Path Finding.
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Conflict:
    agent_1: int
    agent_2: int
    time: int
    conflict_type: str  # 'vertex' or 'edge'
    loc_1: Tuple[int, int]
    loc_2: Optional[Tuple[int, int]] = None  # only used for edge conflicts


def get_agent_pos_at_t(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
    if t < len(path):
        return path[t]
    return path[-1]  # Agent remains at goal position


def find_first_conflict(paths: List[List[Tuple[int, int]]]) -> Optional[Conflict]:
    """Find the earliest conflict between any pair of agents."""
    num_agents = len(paths)
    max_t = max(len(p) for p in paths) if paths else 0

    for t in range(1, max_t + 1):
        # 1. Check vertex conflicts at timestep t
        pos_map = {}
        for a in range(num_agents):
            pos = get_agent_pos_at_t(paths[a], t)
            if pos in pos_map:
                return Conflict(
                    agent_1=pos_map[pos],
                    agent_2=a,
                    time=t,
                    conflict_type='vertex',
                    loc_1=pos
                )
            pos_map[pos] = a

        # 2. Check edge conflicts between timestep t-1 and t
        for a1 in range(num_agents):
            p1_prev = get_agent_pos_at_t(paths[a1], t - 1)
            p1_curr = get_agent_pos_at_t(paths[a1], t)
            for a2 in range(a1 + 1, num_agents):
                p2_prev = get_agent_pos_at_t(paths[a2], t - 1)
                p2_curr = get_agent_pos_at_t(paths[a2], t)

                # Head-on collision across the same edge in opposite directions
                if p1_prev == p2_curr and p1_curr == p2_prev and p1_prev != p1_curr:
                    return Conflict(
                        agent_1=a1,
                        agent_2=a2,
                        time=t,
                        conflict_type='edge',
                        loc_1=p1_prev,
                        loc_2=p1_curr
                    )
    return None


def find_all_conflicts(paths: List[List[Tuple[int, int]]]) -> List[Conflict]:
    """Find all conflicts across all pairs of agents and timesteps."""
    conflicts = []
    num_agents = len(paths)
    max_t = max(len(p) for p in paths) if paths else 0

    for t in range(1, max_t + 1):
        for a1 in range(num_agents):
            pos1 = get_agent_pos_at_t(paths[a1], t)
            prev1 = get_agent_pos_at_t(paths[a1], t - 1)

            for a2 in range(a1 + 1, num_agents):
                pos2 = get_agent_pos_at_t(paths[a2], t)
                prev2 = get_agent_pos_at_t(paths[a2], t - 1)

                if pos1 == pos2:
                    conflicts.append(Conflict(
                        agent_1=a1,
                        agent_2=a2,
                        time=t,
                        conflict_type='vertex',
                        loc_1=pos1
                    ))

                if prev1 == pos2 and pos1 == prev2 and prev1 != pos1:
                    conflicts.append(Conflict(
                        agent_1=a1,
                        agent_2=a2,
                        time=t,
                        conflict_type='edge',
                        loc_1=prev1,
                        loc_2=pos1
                    ))
    return conflicts
