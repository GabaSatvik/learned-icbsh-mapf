from mapf.grid import Grid, parse_scenario
from mapf.high_level_cbs import StandardCBS
from mapf.icbs_h import ICBS_H

grid = Grid.from_file('data/random-32-32-20.map')
scens = ['data/scen-random/random-32-32-20-random-1.scen', 'data/scen-random/random-32-32-20-random-11.scen']

cbs = StandardCBS(grid)
icbs = ICBS_H(grid)

for sc in scens:
    for k in [4, 6, 8]:
        inst = parse_scenario(sc, max_agents=k)
        s = [x[0] for x in inst]
        g = [x[1] for x in inst]
        _, s_cbs = cbs.solve(s, g, time_limit=5.0)
        _, s_icbs = icbs.solve(s, g, time_limit=5.0)
        cbs_exp = s_cbs.get('expanded', -1)
        icbs_exp = s_icbs.get('expanded', -1)
        cbs_cost = s_cbs.get('cost')
        icbs_cost = s_icbs.get('cost')
        print(f"Scen: {sc[-25:]} | k={k} | CBS exp={cbs_exp} cost={cbs_cost} | ICBS exp={icbs_exp} cost={icbs_cost}")
