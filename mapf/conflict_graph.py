"""
Conflict Graph construction and Minimum Vertex Cover (MVC) computation
as defined in ICBS-h (Felner et al., 2018).
"""
from typing import List, Tuple, Set, Dict, Any


class ConflictGraph:
    """
    Graph G = (V, E) where:
    - Vertices V are agents involved in cardinal conflicts
    - Edges E are cardinal conflicts between pairs of agents
    """
    def __init__(self, edges: List[Tuple[int, int]]):
        self.edges = list(set((min(u, v), max(u, v)) for u, v in edges if u != v))
        self.vertices = set()
        self.adj: Dict[int, Set[int]] = {}
        for u, v in self.edges:
            self.vertices.add(u)
            self.vertices.add(v)
            if u not in self.adj:
                self.adj[u] = set()
            if v not in self.adj:
                self.adj[v] = set()
            self.adj[u].add(v)
            self.adj[v].add(u)

    def degree(self, node: int) -> int:
        return len(self.adj.get(node, set()))

    def compute_mvc(self) -> int:
        """
        Computes the exact size of the Minimum Vertex Cover via branch-and-bound.
        """
        if not self.edges:
            return 0
        return self._mvc_recursive(self.edges)

    def _mvc_recursive(self, edges: List[Tuple[int, int]]) -> int:
        if not edges:
            return 0

        # Branch on the edge containing the node with highest remaining degree
        deg_map: Dict[int, int] = {}
        for u, v in edges:
            deg_map[u] = deg_map.get(u, 0) + 1
            deg_map[v] = deg_map.get(v, 0) + 1

        best_edge = edges[0]
        max_sum_deg = -1
        for u, v in edges:
            s = deg_map[u] + deg_map[v]
            if s > max_sum_deg:
                max_sum_deg = s
                best_edge = (u, v)

        u, v = best_edge
        # Option 1: Cover edge by selecting u
        edges_no_u = [e for e in edges if e[0] != u and e[1] != u]
        mvc_u = 1 + self._mvc_recursive(edges_no_u)

        # Option 2: Cover edge by selecting v
        edges_no_v = [e for e in edges if e[0] != v and e[1] != v]
        mvc_v = 1 + self._mvc_recursive(edges_no_v)

        return min(mvc_u, mvc_v)


def compute_residual_mvc_size(G_CF: Any, agent: int) -> int:
    """Computes MVC size of G_CF after removing the specified agent node."""
    if hasattr(G_CF, "edges"):
        remaining_edges = [(u, v) for u, v in G_CF.edges if u != agent and v != agent]
        if not remaining_edges:
            return 0
        return ConflictGraph(remaining_edges).compute_mvc()
    return 0


def compute_h4_and_graph(classified: List[Any]) -> Tuple[int, ConflictGraph]:
    """
    Constructs the cardinal conflict graph G_CF from a list of ClassifiedConflict
    objects and returns (mvc_size, G_CF).
    """
    cardinal_edges = []
    for item in classified:
        if hasattr(item, "conflict_type"):
            ctype = str(item.conflict_type).upper()
            c = item.conflict
        elif isinstance(item, tuple) and len(item) == 2:
            c, ctype = item[0], str(item[1]).upper()
        else:
            continue

        if ctype == "CARDINAL":
            agent_i = c.agent_i if hasattr(c, "agent_i") else c.agent_1
            agent_j = c.agent_j if hasattr(c, "agent_j") else c.agent_2
            cardinal_edges.append((agent_i, agent_j))

    G_CF = ConflictGraph(cardinal_edges)
    h_val = G_CF.compute_mvc()
    return h_val, G_CF
