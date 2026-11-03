"""
Grid and scenario parsing for Moving AI benchmark format.
Provides both GridMap (row, col) and legacy Grid interfaces, plus load_map and load_scenarios.
"""
import os
import random
from collections import deque
from typing import List, Tuple, Set, Dict, Optional, Union


class GridMap:
    """
    2D 4-connected grid map.
    Supports initialization via either:
      - GridMap(width, height, passable_2d_bool_list)
      - GridMap(width, height, obstacles_set)
    """
    def __init__(
        self,
        width: int,
        height: int,
        passable_or_obstacles: Optional[Union[List[List[bool]], Set[Tuple[int, int]]]] = None,
    ):
        self.width = width
        self.height = height
        if isinstance(passable_or_obstacles, list):
            self.height = len(passable_or_obstacles)
            self.width = len(passable_or_obstacles[0]) if self.height > 0 else width
            self.passable = [ [bool(cell) for cell in row] for row in passable_or_obstacles ]
            self.obstacles: Set[Tuple[int, int]] = set()
            for r in range(self.height):
                for c in range(self.width):
                    if not self.passable[r][c]:
                        self.obstacles.add((r, c))
        else:
            obs = set(passable_or_obstacles) if passable_or_obstacles is not None else set()
            self.obstacles = obs
            self.passable = [
                [(r, c) not in self.obstacles for c in range(self.width)]
                for r in range(self.height)
            ]

    def is_passable(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width and self.passable[r][c]

    def is_valid(self, r: int, c: int) -> bool:
        return self.is_passable(r, c)

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return 4-connected traversable neighbors."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and self.passable[nr][nc]:
                neighbors.append((nr, nc))
        return neighbors

    @classmethod
    def from_file(cls, filepath: str) -> "GridMap":
        with open(filepath, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        height = 0
        width = 0
        map_start = 0

        for i, line in enumerate(lines):
            if line.startswith("height"):
                height = int(line.split()[1])
            elif line.startswith("width"):
                width = int(line.split()[1])
            elif line.startswith("map"):
                map_start = i + 1
                break

        passable = [[True for _ in range(width)] for _ in range(height)]
        for r, row_line in enumerate(lines[map_start : map_start + height]):
            for c, char in enumerate(row_line[:width]):
                if char in "@OTW":
                    passable[r][c] = False

        return cls(width, height, passable)


Grid = GridMap


def load_map(filepath: str) -> GridMap:
    return GridMap.from_file(filepath)


def parse_scenario(
    filepath: str, max_agents: Optional[int] = None
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Parses a Moving AI .scen file.
    Note: Moving AI .scen format stores (start_col, start_row, goal_col, goal_row) in columns 4..7.
    Since GridMap uses (row, col) = (y, x), we store (sy, sx) and (gy, gx) so coordinates match
    GridMap.from_file row/col indexing.
    """
    instances = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version"):
                continue
            parts = line.split()
            if len(parts) >= 9:
                sx, sy = int(parts[4]), int(parts[5])
                gx, gy = int(parts[6]), int(parts[7])
                instances.append(((sy, sx), (gy, gx)))
                if max_agents is not None and len(instances) >= max_agents:
                    break
    return instances


def _resolve_scen_file_for_idx(base_scen_path: str, scen_idx: int) -> Optional[str]:
    """
    Finds the scenario file corresponding to scen_idx (1-indexed).
    Checks:
      1. Exact numbered file alongside base_scen_path (e.g. data/random-32-32-20-random-{scen_idx}.scen)
      2. Numbered file in data/scen-random/
      3. Fallback to base_scen_path if it exists
    """
    import re
    dirname = os.path.dirname(base_scen_path)
    basename = os.path.basename(base_scen_path)
    replaced = re.sub(r"-random-\d+\.scen$", f"-random-{scen_idx}.scen", basename)

    cand1 = os.path.join(dirname, replaced)
    if os.path.exists(cand1):
        return cand1

    cand2 = os.path.join(dirname, "scen-random", replaced)
    if os.path.exists(cand2):
        return cand2

    if os.path.exists(base_scen_path):
        return base_scen_path

    return None


def load_scenarios(
    scen_path: str, k: int, max_scenarios: int = 25
) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Loads up to `max_scenarios` scenario instances of `k` agents each.
    Returns a list of (starts, goals) tuples.
    """
    scenarios: List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = []
    for s_idx in range(1, max_scenarios + 1):
        resolved = _resolve_scen_file_for_idx(scen_path, s_idx)
        if resolved is None:
            break
        if resolved != scen_path or s_idx == 1:
            pairs = parse_scenario(resolved, max_agents=None)
            if len(pairs) < k:
                break
            if resolved != scen_path or s_idx == 1:
                selected = pairs[:k]
            else:
                rng = random.Random(1000 + s_idx)
                selected = rng.sample(pairs, k)
        else:
            all_pairs = parse_scenario(resolved, max_agents=None)
            if len(all_pairs) < k:
                break
            # Deterministic slice/sample for scenario s_idx if using a single consolidated .scen file
            offset = ((s_idx - 1) * 7) % max(1, len(all_pairs) - k + 1)
            rng = random.Random(10000 + s_idx * 131)
            pool = all_pairs[offset:] + all_pairs[:offset]
            # Ensure unique starts and unique goals among k agents
            starts_used = set()
            goals_used = set()
            selected = []
            indices = list(range(len(pool)))
            if s_idx > 1:
                rng.shuffle(indices)
            for idx in indices:
                st, gl = pool[idx]
                if st not in starts_used and gl not in goals_used:
                    starts_used.add(st)
                    goals_used.add(gl)
                    selected.append((st, gl))
                    if len(selected) == k:
                        break
            if len(selected) < k:
                break

        starts = [p[0] for p in selected]
        goals = [p[1] for p in selected]
        scenarios.append((starts, goals))

    return scenarios
