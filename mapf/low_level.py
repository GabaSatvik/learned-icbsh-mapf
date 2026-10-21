"""
Low-level space-time A* pathfinding for individual agents with dynamic constraints.
"""
import heapq
from typing import List, Tuple, Set, Dict, Optional, Any, Union
from mapf.grid import Grid


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def compute_bfs_h_table(grid: Grid, goal: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """
    Reverse BFS from goal to compute exact shortest distance to goal ignoring dynamic obstacles.
    """
    dist: Dict[Tuple[int, int], int] = {goal: 0}
    queue = [goal]
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        d = dist[curr]
        for nxt in grid.get_neighbors(curr[0], curr[1]):
            if nxt not in dist:
                dist[nxt] = d + 1
                queue.append(nxt)
    return dist


compute_heuristic_map = compute_bfs_h_table


def get_path_cost(path: Optional[List[Tuple[int, int]]]) -> int:
    return len(path) - 1 if path else 0


def _parse_constraints(constraints: Any) -> Tuple[Set[Tuple[int, int, int]], Set[Tuple[int, int, int, int, int]]]:
    """
    Parses various constraint formats into:
      vertex_constraints: Set[(x, y, t)]
      edge_constraints: Set[(from_x, from_y, to_x, to_y, t)]
    """
    v_cons = set()
    e_cons = set()
    if not constraints:
        return v_cons, e_cons

    for c in constraints:
        if hasattr(c, "constraint_type"):
            # Constraint object
            ctype = str(c.constraint_type).lower()
            if ctype in ("vertex", "v"):
                loc = c.loc_1 if hasattr(c, "loc_1") else c.loc
                v_cons.add((loc[0], loc[1], c.time))
            else:
                l1 = c.loc_1 if hasattr(c, "loc_1") else c.loc[0]
                l2 = c.loc_2 if hasattr(c, "loc_2") else c.loc[1]
                e_cons.add((l1[0], l1[1], l2[0], l2[1], c.time))
        elif isinstance(c, tuple):
            if len(c) == 3 and isinstance(c[1], str):
                # (agent_id, type_str, data)
                _, ctype, data = c
                if str(ctype).lower() == "vertex":
                    v_cons.add(data)
                else:
                    e_cons.add(data)
            elif len(c) == 3:
                # (x, y, t)
                v_cons.add(c)
            elif len(c) == 5:
                # (fx, fy, tx, ty, t)
                e_cons.add(c)
    return v_cons, e_cons


def space_time_astar(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    constraints: Any = None,
    max_timesteps: int = 1024,
    h_table: Optional[Dict[Tuple[int, int], int]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    Space-time A* search for a single agent.
    Returns path as [pos_at_t0, pos_at_t1, ...] or None.
    """
    vertex_constraints, edge_constraints = _parse_constraints(constraints)

    if (start[0], start[1], 0) in vertex_constraints:
        return None

    if not grid.is_valid(start[0], start[1]) or not grid.is_valid(goal[0], goal[1]):
        return None

    if h_table is None:
        h_table = compute_bfs_h_table(grid, goal)

    if start not in h_table:
        return None

    max_c_time = 0
    for (_, _, t) in vertex_constraints:
        if t > max_c_time:
            max_c_time = t
    for (_, _, _, _, t) in edge_constraints:
        if t > max_c_time:
            max_c_time = t

    h0 = h_table.get(start, manhattan_distance(start, goal))
    # open_set item: (f_score, g_score, (x, y, timestep), path)
    open_set = [(h0, 0, (start[0], start[1], 0), [start])]
    closed_set: Set[Tuple[int, int, int]] = set()

    while open_set:
        f, g, (x, y, t), path = heapq.heappop(open_set)

        if (x, y) == goal and t >= max_c_time:
            return path

        state_key = (x, y, t)
        if state_key in closed_set:
            continue
        closed_set.add(state_key)

        if t >= max_timesteps:
            continue

        possible_moves = grid.get_neighbors(x, y) + [(x, y)]
        next_t = t + 1

        for nx, ny in possible_moves:
            if (nx, ny, next_t) in vertex_constraints:
                continue
            if (x, y, nx, ny, next_t) in edge_constraints:
                continue
            if (nx, ny, next_t) in closed_set:
                continue

            next_g = g + 1
            next_h = h_table.get((nx, ny), manhattan_distance((nx, ny), goal))
            next_f = next_g + next_h
            heapq.heappush(open_set, (next_f, next_g, (nx, ny, next_t), path + [(nx, ny)]))

    return None


class SpaceTimeAStar:
    def __init__(self, grid: Grid):
        self.grid = grid
        self._h_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}

    def get_h(self, loc: Tuple[int, int], goal: Tuple[int, int]) -> int:
        if goal not in self._h_cache:
            self._h_cache[goal] = compute_bfs_h_table(self.grid, goal)
        return self._h_cache[goal].get(loc, manhattan_distance(loc, goal))

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        vertex_constraints: Set[Tuple[int, int, int]],
        edge_constraints: Set[Tuple[int, int, int, int, int]],
        max_timestep: int = 1024,
    ) -> Optional[List[Tuple[int, int]]]:
        if goal not in self._h_cache:
            self._h_cache[goal] = compute_bfs_h_table(self.grid, goal)
        all_cons = list(vertex_constraints) + list(edge_constraints)
        return space_time_astar(
            self.grid, start, goal, all_cons, max_timestep, self._h_cache[goal]
        )
