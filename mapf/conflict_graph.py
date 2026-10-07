"""
Conflict Graph construction and Minimum Vertex Cover (MVC) computation
as defined in ICBS-h (Felner et al., 2018).
"""
from typing import List, Tuple, Set, Dict


class ConflictGraph:
    """
    Graph G = (V, E) where:
    - Vertices V are agents involved in cardinal conflicts
    - Edges E are cardinal conflicts between pairs of agents
    """
    def __init__(self, edges: List[Tuple[int, int]]):
        self.edges = list(set((min(u, v), max(u, v)) for u, v in edges if u != v))
        self.vertices = set()
        for u, v in self.edges:
            self.vertices.add(u)
            self.vertices.add(v)

    def compute_mvc(self) -> int:
        """
        Computes the exact size of the Minimum Vertex Cover.
        Since MAPF conflict graphs for small to medium agent counts are small,
        branch-and-bound / recursion is exact and fast.
        """
        if not self.edges:
            return 0

        return self._mvc_recursive(self.edges)

    def _mvc_recursive(self, edges: List[Tuple[int, int]]) -> int:
        if not edges:
            return 0

        # Pick the edge with highest degree endpoint to branch efficiently
        u, v = edges[0]

        # Option 1: Include u in vertex cover (remove all edges incident to u)
        edges_without_u = [e for e in edges if e[0] != u and e[1] != u]
        mvc_u = 1 + self._mvc_recursive(edges_without_u)

        # Option 2: Include v in vertex cover (remove all edges incident to v)
        edges_without_v = [e for e in edges if e[0] != v and e[1] != v]
        mvc_v = 1 + self._mvc_recursive(edges_without_v)

        return min(mvc_u, mvc_v)
