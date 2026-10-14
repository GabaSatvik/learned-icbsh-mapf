"""
Script to generate training data for Learning-to-Rank (LTR).
Runs ICBS-h across multiple scenarios and agent counts on the chosen map (random-32-32-20),
extracts features, and saves them to logs/ltr_training_data.csv.
"""
import os
import glob
from mapf.grid import Grid, parse_scenario
from mapf.icbs_h import ICBS_H
from mapf.logger import ConflictDataLogger


def collect_training_data(
    map_path: str = "data/random-32-32-20.map",
    scen_dir: str = "data/scen-random",
    agent_counts: list = [4, 6, 8, 10],
    num_scenarios: int = 15,
    output_csv: str = "ltr_training_data.csv"
):
    print(f"Loading map: {map_path}")
    grid = Grid.from_file(map_path)
    logger = ConflictDataLogger(log_dir="logs")
    solver = ICBS_H(grid, logger=logger)

    scen_files = sorted(glob.glob(os.path.join(scen_dir, "*.scen")))[:num_scenarios]
    print(f"Found {len(scen_files)} scenario files for data collection.")

    total_solved = 0
    total_attempted = 0

    for s_idx, scen_file in enumerate(scen_files):
        scen_name = os.path.basename(scen_file)
        print(f"\n--- Scenario {s_idx + 1}/{len(scen_files)}: {scen_name} ---")

        for k in agent_counts:
            instances = parse_scenario(scen_file, max_agents=k)
            if len(instances) < k:
                continue

            starts = [inst[0] for inst in instances]
            goals = [inst[1] for inst in instances]

            total_attempted += 1
            meta = {'scenario': scen_name, 'agents': k}

            paths, stats = solver.solve(starts, goals, time_limit=10.0, metadata=meta)

            if paths is not None:
                total_solved += 1
                print(f"  [Agents={k:2d}] SOLVED | Nodes={stats['expanded']:3d} | Cost={stats['cost']} | Time={stats['time']:.2f}s")
            else:
                print(f"  [Agents={k:2d}] TIMEOUT/FAIL | Time={stats['time']:.2f}s")

    saved_path = logger.save_to_csv(output_csv)
    print(f"\n[DONE] Successfully collected training data across {total_solved}/{total_attempted} instances.")
    print(f"Dataset location: {saved_path}")


if __name__ == "__main__":
    collect_training_data()
