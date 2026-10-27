"""
Bug 5 Diagnostic: Paris_1_256 NO_ROOT_PATH Investigation.

Tests a minimal 2-agent instance on the Paris_1_256 map to determine exactly
why space_time_astar fails to find a path. Prints:
  - Parsed grid dimensions and obstacle count
  - Whether start/goal cells are passable
  - Whether BFS h-table reaches the goal from start
  - Raw A* result with detailed diagnostics

Run: python -m mapf.diagnose_paris
"""
import os
import sys
import time
from collections import deque
from typing import Optional

# Ensure mapf package is importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mapf.grid import Grid, parse_scenario
from mapf.low_level import compute_bfs_h_table, space_time_astar, ConstraintSet


MAP_PATH = "data/Paris_1_256.map"
SCEN_PATH = "data/scen-random/Paris_1_256-random-1.scen"


def diagnose_paris():
    print("=" * 60)
    print("Paris_1_256 Diagnostic (Bug 5 Investigation)")
    print("=" * 60)

    # 1. Check files exist
    if not os.path.exists(MAP_PATH):
        print(f"[ERROR] Map file not found: {MAP_PATH}")
        print("  Download it from movingai.com or run download_data.py")
        return

    # 2. Load and inspect grid
    print(f"\n[1] Loading grid from {MAP_PATH}...")
    try:
        grid = Grid.from_file(MAP_PATH)
        print(f"  Grid dimensions: width={grid.width}, height={grid.height}")
        print(f"  Total cells: {grid.width * grid.height:,}")
        print(f"  Obstacle count: {len(grid.obstacles):,}")
        passable = grid.width * grid.height - len(grid.obstacles)
        print(f"  Passable cells: {passable:,} ({100*passable/(grid.width*grid.height):.1f}%)")
    except Exception as e:
        print(f"  [ERROR] Failed to load grid: {e}")
        return

    # 3. Load scenario file
    if not os.path.exists(SCEN_PATH):
        print(f"\n[2] Scenario file not found: {SCEN_PATH}")
        print("  Using hardcoded test coordinates instead.")
        # Pick a few passable cells manually by scanning the grid
        passable_cells = [
            (x, y) for y in range(grid.height) for x in range(grid.width)
            if grid.is_valid(x, y) and (x, y) not in grid.obstacles
        ]
        if len(passable_cells) < 4:
            print("  [ERROR] Not enough passable cells for a test instance")
            return
        test_cases = [
            (passable_cells[0], passable_cells[len(passable_cells) // 2]),
            (passable_cells[1], passable_cells[len(passable_cells) // 3]),
        ]
    else:
        print(f"\n[2] Loading scenarios from {SCEN_PATH}...")
        try:
            instances = parse_scenario(SCEN_PATH, max_agents=4)
            print(f"  Loaded {len(instances)} agent start/goal pairs")
            test_cases = [(inst[0], inst[1]) for inst in instances[:2]]
        except Exception as e:
            print(f"  [ERROR] Failed to parse scenario: {e}")
            return

    # 4. Test BFS h-table construction and A* for each test pair
    for idx, (start, goal) in enumerate(test_cases):
        print(f"\n[3.{idx+1}] Test pair {idx+1}: start={start}, goal={goal}")

        # Check passability
        start_pass = grid.is_valid(start[0], start[1]) and (start, ) not in [grid.obstacles]
        # Use the actual grid method
        start_ok = grid.is_valid(start[0], start[1])
        goal_ok = grid.is_valid(goal[0], goal[1])
        print(f"  start passable: {start_ok} | goal passable: {goal_ok}")

        if not start_ok or not goal_ok:
            print("  [SKIP] Start or goal is not passable — this is the root cause of NO_ROOT_PATH")
            continue

        # Build BFS h-table from goal
        print(f"  Building BFS h-table from goal {goal}...")
        t0 = time.perf_counter()
        h_table = compute_bfs_h_table(grid, goal)
        elapsed = time.perf_counter() - t0
        print(f"  h-table size: {len(h_table):,} cells reachable from goal (in {elapsed:.3f}s)")

        if start not in h_table:
            print(f"  [ROOT CAUSE] Start {start} is NOT reachable from goal {goal} via BFS!")
            print(f"  This means the map is disconnected or the scenario assigns agents to")
            print(f"  different connected components. This IS the NO_ROOT_PATH bug.")
            # Find which component each belongs to
            def bfs_component(seed):
                visited = {seed}
                q = deque([seed])
                while q:
                    r, c = q.popleft()
                    for nr, nc in grid.get_neighbors(r, c):
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                return visited

            comp_goal = bfs_component(goal)
            comp_start = bfs_component(start)
            print(f"  Goal component size: {len(comp_goal):,} cells")
            print(f"  Start component size: {len(comp_start):,} cells")
            print(f"  Same component: {start in comp_goal}")
            continue
        else:
            print(f"  start h-value: {h_table[start]} steps to goal")

        # Run A*
        print(f"  Running space_time_astar from {start} to {goal}...")
        t0 = time.perf_counter()
        path = space_time_astar(
            grid, start, goal,
            constraints=ConstraintSet(),
            max_timesteps=grid.width + grid.height + 50,
            h_table=h_table
        )
        elapsed = time.perf_counter() - t0
        if path is None:
            print(f"  [FAIL] A* returned None after {elapsed:.3f}s")
            print(f"  Possible causes: max_timesteps too small, expansion limit hit")
        else:
            print(f"  [OK] A* found path of length {len(path)-1} steps in {elapsed:.3f}s")
            print(f"  Path: {path[:5]}{'...' if len(path) > 5 else ''}")

    print("\n" + "=" * 60)
    print("Diagnosis complete. See above for root cause of NO_ROOT_PATH.")
    print("=" * 60)


if __name__ == "__main__":
    diagnose_paris()
