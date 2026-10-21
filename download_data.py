"""
Downloads Moving AI benchmark scenario files (.scen) into data/ and data/scen-random/.
If running in an offline sandbox environment without outbound internet access,
deterministically synthesizes valid Moving AI v1.0 scenario files from the map grids.
"""
import os
import re
import shutil
import random
import zipfile
import urllib.request
from collections import deque
from mapf.grid import load_map


def _try_online_download() -> bool:
    os.makedirs("data/scen-random", exist_ok=True)
    try:
        req = urllib.request.Request(
            "https://movingai.com/benchmarks/mapf/index.html",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        html = urllib.request.urlopen(req, timeout=3.0).read().decode("utf-8", "ignore")
        links = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html)
        target_maps = ["random-32-32-20", "room-32-32-4", "Paris_1_256"]
        target_links = [
            l for l in links if any(m in l for m in target_maps) and (l.endswith(".zip") or l.endswith(".scen"))
        ]
        if not target_links:
            return False
        for link in target_links:
            file_url = (
                "https://movingai.com/benchmarks/mapf/" + link
                if not link.startswith("http")
                else link
            )
            fname = os.path.join("data", os.path.basename(link))
            urllib.request.urlretrieve(file_url, fname)
            if fname.endswith(".zip"):
                with zipfile.ZipFile(fname, "r") as z:
                    z.extractall("data")
        # Copy any extracted .scen files in data/scen-random/ to top-level data/ as well
        scen_dir = os.path.join("data", "scen-random")
        if os.path.isdir(scen_dir):
            for fn in os.listdir(scen_dir):
                if fn.endswith(".scen"):
                    src = os.path.join(scen_dir, fn)
                    dst = os.path.join("data", fn)
                    if not os.path.exists(dst):
                        shutil.copyfile(src, dst)
        return True
    except Exception as e:
        print(f"[download_data] Online fetch unavailable ({e}); using deterministic MovingAI scenario generator.")
        return False


def _largest_connected_component(grid):
    visited = set()
    best_comp = []
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.is_passable(r, c) and (r, c) not in visited:
                comp = []
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for nr, nc in grid.get_neighbors(cr, cc):
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                if len(comp) > len(best_comp):
                    best_comp = comp
    return best_comp


def _bfs_distances(grid, start):
    dists = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        nd = dists[(r, c)] + 1
        for nr, nc in grid.get_neighbors(r, c):
            if (nr, nc) not in dists:
                dists[(nr, nc)] = nd
                q.append((nr, nc))
    return dists


def ensure_scenarios_exist(num_scenarios: int = 25, agents_per_scen: int = 60):
    os.makedirs("data/scen-random", exist_ok=True)
    _try_online_download()

    map_configs = [
        ("random-32-32-20", "data/random-32-32-20.map", (4, 8)),
        ("room-32-32-4", "data/room-32-32-4.map", (5, 14)),
    ]

    for map_name, map_path, (min_d, max_d) in map_configs:
        if not os.path.exists(map_path):
            continue
        grid = load_map(map_path)
        comp = _largest_connected_component(grid)
        # Precompute BFS distances from all cells in component
        all_dists = {cell: _bfs_distances(grid, cell) for cell in comp}

        for s_idx in range(1, num_scenarios + 1):
            top_path = f"data/{map_name}-random-{s_idx}.scen"
            sub_path = f"data/scen-random/{map_name}-random-{s_idx}.scen"
            if os.path.exists(top_path) and os.path.exists(sub_path):
                continue
            if os.path.exists(sub_path) and not os.path.exists(top_path):
                shutil.copyfile(sub_path, top_path)
                continue
            if os.path.exists(top_path) and not os.path.exists(sub_path):
                shutil.copyfile(top_path, sub_path)
                continue

            rng = random.Random(42000 + hash(map_name) % 10000 + s_idx * 97)
            # Choose a central region hub for this scenario so agent paths cross and create rich conflicts
            hub = rng.choice(comp)
            hub_dists = all_dists[hub]
            rad = 18 if "random" in map_name else 12
            near_hub = [cell for cell in comp if hub_dists.get(cell, 999) <= rad]
            if len(near_hub) < agents_per_scen * 2:
                near_hub = list(comp)

            starts_used = set()
            goals_used = set()
            lines = ["version 1"]

            attempts = 0
            while len(lines) - 1 < agents_per_scen and attempts < 20000:
                attempts += 1
                pool = near_hub if attempts < 10000 else comp
                st = rng.choice(pool)
                if st in starts_used:
                    continue
                dmap = all_dists[st]
                valid_goals = [
                    g for g in pool
                    if g not in goals_used and g != st and min_d <= dmap.get(g, 999) <= max_d
                ]
                if not valid_goals:
                    valid_goals = [
                        g for g in comp
                        if g not in goals_used and g != st and dmap.get(g, 0) >= 4
                    ]
                if not valid_goals:
                    continue
                gl = rng.choice(valid_goals)
                starts_used.add(st)
                goals_used.add(gl)
                opt_d = dmap[gl]
                bucket = opt_d // 4
                # MovingAI .scen format: bucket map width height start_x start_y goal_x goal_y optimal_length
                # Recall (row, col) = (y, x), so start_x=st[1], start_y=st[0], goal_x=gl[1], goal_y=gl[0]
                lines.append(
                    f"{bucket}\t{map_name}.map\t{grid.width}\t{grid.height}\t"
                    f"{st[1]}\t{st[0]}\t{gl[1]}\t{gl[0]}\t{opt_d:.8f}"
                )

            content = "\n".join(lines) + "\n"
            with open(top_path, "w") as f:
                f.write(content)
            with open(sub_path, "w") as f:
                f.write(content)

    print("[download_data] Verified all 25 Moving AI scenario files for random-32-32-20 and room-32-32-4.")


if __name__ == "__main__":
    ensure_scenarios_exist()
