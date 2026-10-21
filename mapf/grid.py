"""
Grid and scenario parsing for Moving AI benchmark format.
Supports both legacy Grid and modern GridMap interfaces.
"""
import os
import glob
from typing import List, Tuple, Set, Dict, Optional, Union


class Grid:
    """
    2D 4-connected grid map.
    Traversable cells are '.' and 'G' (and 'S').
    Obstacles are '@', 'O', 'T', 'W'.
    Coordinates are (x, y) where x is horizontal/column [0..width-1],
    and y is vertical/row [0..height-1].
    """
    def __init__(self, width: int, height: int, obstacles: Set[Tuple[int, int]]):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

    def is_passable(self, x: int, y: int) -> bool:
        return self.is_valid(x, y)

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Return 4-connected traversable neighbors."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    @classmethod
    def from_file(cls, filepath: str) -> "Grid":
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

        obstacles = set()
        for y, row_line in enumerate(lines[map_start:map_start + height]):
            for x, char in enumerate(row_line[:width]):
                if char in "@OTW":
                    obstacles.add((x, y))

        return cls(width, height, obstacles)


class GridMap(Grid):
    """
    GridMap supporting 2D passable boolean matrix or obstacle set.
    """
    def __init__(self, width: int, height: int, passable_or_obstacles: Union[List[List[bool]], Set[Tuple[int, int]]]):
        if isinstance(passable_or_obstacles, (set, frozenset)):
            super().__init__(width, height, passable_or_obstacles)
            self.passable = [
                [self.is_valid(x, y) for x in range(width)]
                for y in range(height)
            ]
        else:
            self.width = width
            self.height = height
            self.passable = passable_or_obstacles
            obstacles = set()
            for y in range(height):
                for x in range(width):
                    # Support both (y, x) indexing and (x, y) indexing if symmetric
                    is_free = False
                    if y < len(passable_or_obstacles) and x < len(passable_or_obstacles[y]):
                        is_free = bool(passable_or_obstacles[y][x])
                    if not is_free:
                        obstacles.add((x, y))
            self.obstacles = obstacles


def load_map(filepath: str) -> GridMap:
    """Load a Moving AI .map file into a GridMap."""
    g = Grid.from_file(filepath)
    return GridMap(g.width, g.height, g.obstacles)


def parse_scenario(filepath: str, max_agents: Optional[int] = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Parses a Moving AI .scen file.
    Returns list of ((start_x, start_y), (goal_x, goal_y)).
    """
    instances = []
    if not os.path.exists(filepath):
        return instances
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version"):
                continue
            parts = line.split()
            if len(parts) >= 9:
                sx, sy = int(parts[4]), int(parts[5])
                gx, gy = int(parts[6]), int(parts[7])
                instances.append(((sx, sy), (gx, gy)))
                if max_agents is not None and len(instances) >= max_agents:
                    break
    return instances


def load_scenarios(filepath: str, k: int, max_scenarios: Optional[int] = None) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Loads scenarios for a benchmark instance.
    Returns a list of scenarios: [(starts_list, goals_list), ...] where each starts_list has k elements.
    If sibling scenario files exist (e.g. random-1.scen, random-2.scen, ...), loads each scenario file.
    Otherwise, chunks the single scenario file.
    """
    n_needed = max_scenarios if max_scenarios is not None else 25
    scenarios = []

    # Check for individual scenario files
    # E.g., if filepath is "data/random-32-32-20-random-1.scen" or "data/scen-random/random-32-32-20-random-1.scen"
    base_dir = os.path.dirname(filepath)
    fname = os.path.basename(filepath)
    
    # Try finding sibling files like random-32-32-20-random-{s}.scen
    import re
    match = re.search(r"^(.*-random-)(\d+)(\.scen)$", fname)
    if match:
        prefix, _, suffix = match.groups()
        found_siblings = True
        for s_idx in range(1, n_needed + 1):
            cand_paths = [
                os.path.join(base_dir, f"{prefix}{s_idx}{suffix}"),
                os.path.join(base_dir, "scen-random", f"{prefix}{s_idx}{suffix}"),
                os.path.join("data", f"{prefix}{s_idx}{suffix}"),
                os.path.join("data", "scen-random", f"{prefix}{s_idx}{suffix}"),
            ]
            cand = next((p for p in cand_paths if os.path.exists(p)), None)
            if cand is not None:
                insts = parse_scenario(cand, max_agents=k)
                if len(insts) >= k:
                    scenarios.append(([x[0] for x in insts[:k]], [x[1] for x in insts[:k]]))
            else:
                found_siblings = False
                break
        if len(scenarios) >= n_needed:
            return scenarios[:n_needed]

    # Fallback: slice from the specified file
    all_pairs = parse_scenario(filepath)
    if not all_pairs:
        return []

    # Create n_needed slices
    for s_idx in range(n_needed):
        start_offset = s_idx * k
        if start_offset + k <= len(all_pairs):
            slice_pairs = all_pairs[start_offset : start_offset + k]
        else:
            # Wrap around with deterministic offset
            slice_pairs = [all_pairs[(start_offset + i) % len(all_pairs)] for i in range(k)]
        scenarios.append(([p[0] for p in slice_pairs], [p[1] for p in slice_pairs]))

    return scenarios
