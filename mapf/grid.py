"""
Grid and scenario parsing for Moving AI benchmark format.
"""
from typing import List, Tuple, Set, Dict, Optional


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
        self.obstacles = obstacles

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

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


def parse_scenario(filepath: str, max_agents: Optional[int] = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Parses a Moving AI .scen file.
    Returns list of ((start_x, start_y), (goal_x, goal_y)).
    """
    instances = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("version"):
                continue
            parts = line.split()
            if len(parts) >= 9:
                # parts[4]: start_x, parts[5]: start_y, parts[6]: goal_x, parts[7]: goal_y
                sx, sy = int(parts[4]), int(parts[5])
                gx, gy = int(parts[6]), int(parts[7])
                instances.append(((sx, sy), (gx, gy)))
                if max_agents is not None and len(instances) >= max_agents:
                    break
    return instances
