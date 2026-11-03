"""Independent solution checks; deliberately does not call conflict detection."""


def validate_instance(grid, starts, goals):
    if len(starts) != len(goals):
        raise ValueError('Starts and goals must have equal lengths')
    if any(not grid.is_passable(*p) for p in list(starts) + list(goals)):
        raise ValueError('All endpoints must be traversable cells')
    return len(set(starts)) == len(starts) and len(set(goals)) == len(goals)


def validate_solution(grid, starts, goals, paths, cost):
    if not validate_instance(grid, starts, goals) or len(paths) != len(starts):
        raise ValueError('Invalid instance or path count')
    for i, path in enumerate(paths):
        if not path or path[0] != starts[i] or path[-1] != goals[i]:
            raise ValueError('Path endpoints do not match instance')
        for cell in path:
            if not grid.is_passable(*cell):
                raise ValueError('Path enters a blocked cell')
        for a, b in zip(path, path[1:]):
            if abs(a[0]-b[0]) + abs(a[1]-b[1]) > 1:
                raise ValueError('Nonadjacent path step')
    if sum(len(p)-1 for p in paths) != cost:
        raise ValueError('Reported SOC differs from path lengths')
    for t in range(max(map(len, paths), default=0)):
        pos = [p[min(t, len(p)-1)] for p in paths]
        if len(set(pos)) != len(pos):
            raise ValueError('Vertex collision')
        if t:
            prev = [p[min(t-1, len(p)-1)] for p in paths]
            if any(pos[i] == prev[j] and pos[j] == prev[i]
                   for i in range(len(paths)) for j in range(i+1, len(paths))):
                raise ValueError('Edge swap')
    return True


def validate_constrained_paths(grid, starts, goals, paths, constraints):
    """Check individual saved paths, including constraints after goal arrival.

    Inter-agent collisions are allowed in a CT state. Optimal path lengths
    are checked separately by the continuation solver's low-level searches.
    """
    if not (len(paths) == len(starts) == len(goals) == len(constraints)):
        raise ValueError('Saved state sizes differ')
    for start, goal, path, cs in zip(starts, goals, paths, constraints):
        if not path or path[0] != start or path[-1] != goal:
            raise ValueError('Saved path endpoints differ')
        if any(not grid.is_passable(*p) for p in path):
            raise ValueError('Saved path enters obstacle')
        if any(abs(a[0]-b[0])+abs(a[1]-b[1]) > 1 for a,b in zip(path,path[1:])):
            raise ValueError('Saved path has illegal move')
        at = lambda t: path[min(t, len(path)-1)]
        for r,c,t in cs.vertex_constraints:
            if t < 0 or at(t) == (r,c):
                raise ValueError('Saved path violates vertex constraint')
        for r,c,rr,cc,t in cs.edge_constraints:
            if t < 1 or (at(t-1),at(t)) == ((r,c),(rr,cc)):
                raise ValueError('Saved path violates edge constraint')
    return True


