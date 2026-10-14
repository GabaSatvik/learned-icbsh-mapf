"""
Verification script: Test Grid, Low-Level A*, Standard CBS, and ICBS-h on small hand-solvable instances and random-32-32-20 map.
"""
from mapf.grid import Grid, parse_scenario
from mapf.high_level_cbs import StandardCBS
from mapf.icbs_h import ICBS_H
from mapf.logger import ConflictDataLogger


def test_handcrafted_instances():
    print("=== Testing Handcrafted 2-Agent Instance ===")
    # 5x5 empty grid, agents cross paths head-on
    grid = Grid(5, 5, set())
    starts = [(0, 2), (4, 2)]
    goals = [(4, 2), (0, 2)]

    cbs = StandardCBS(grid)
    cbs_sol, cbs_stats = cbs.solve(starts, goals)
    print(f"Standard CBS: cost={cbs_stats.get('cost')}, expanded={cbs_stats.get('expanded')}, time={cbs_stats.get('time'):.4f}s")
    assert cbs_sol is not None, "CBS failed on handcrafted instance"

    icbs = ICBS_H(grid)
    icbs_sol, icbs_stats = icbs.solve(starts, goals)
    print(f"ICBS-h:       cost={icbs_stats.get('cost')}, expanded={icbs_stats.get('expanded')}, time={icbs_stats.get('time'):.4f}s")
    assert icbs_sol is not None, "ICBS-h failed on handcrafted instance"
    assert cbs_stats['cost'] == icbs_stats['cost'], "Costs must match for optimal solvers!"
    print("Handcrafted test passed! Both found optimal cost:", cbs_stats['cost'])


def test_movingai_map():
    print("\n=== Testing on random-32-32-20 Benchmark Map ===")
    map_file = "data/random-32-32-20.map"
    scen_file = "data/scen-random/random-32-32-20-random-1.scen"

    grid = Grid.from_file(map_file)
    print(f"Grid loaded: {grid.width}x{grid.height}, obstacles count={len(grid.obstacles)}")

    instances = parse_scenario(scen_file, max_agents=4)
    starts = [inst[0] for inst in instances]
    goals = [inst[1] for inst in instances]
    print(f"Solving 4 agents: starts={starts}, goals={goals}")

    logger = ConflictDataLogger(log_dir="logs")
    icbs = ICBS_H(grid, logger=logger)
    sol, stats = icbs.solve(starts, goals, time_limit=15.0)

    print(f"ICBS-h Result: cost={stats.get('cost')}, expanded={stats.get('expanded')}, time={stats.get('time'):.4f}s")
    if sol is not None:
        print("Paths found successfully!")
        for i, p in enumerate(sol):
            print(f"  Agent {i} length {len(p)}: {p[:3]} ... {p[-3:]}")

    csv_path = logger.save_to_csv("test_log.csv")
    print(f"Logger test complete. Saved to {csv_path}")


if __name__ == "__main__":
    test_handcrafted_instances()
    test_movingai_map()
