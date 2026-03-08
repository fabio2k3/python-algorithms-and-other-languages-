"""
Iterative Deepening Search (IDS) - Búsqueda de Profundización Iterativa
========================================================================
Ejecuta múltiples DFS con límite de profundidad creciente (0, 1, 2, ...).
Combina la eficiencia en memoria de DFS con la completitud de BFS.

Propiedades:
  - Completo:   Sí
  - Óptimo:     Sí (si costos uniformes)
  - Tiempo:     O(b^d)
  - Espacio:    O(b*d)
"""


def _depth_limited_search(graph: dict, node, goal, limit, path, cost, visited):
    """DFS con límite de profundidad."""
    if node == goal:
        return path, cost

    if limit == 0:
        return None, float('inf')   # límite alcanzado, no es fallo definitivo

    best_path, best_cost = None, float('inf')

    for neighbor, edge_cost in graph.get(node, []):
        if neighbor not in visited:
            visited.add(neighbor)
            result_path, result_cost = _depth_limited_search(
                graph, neighbor, goal,
                limit - 1, path + [neighbor], cost + edge_cost, visited
            )
            visited.discard(neighbor)

            if result_path is not None and result_cost < best_cost:
                best_path, best_cost = result_path, result_cost

    return best_path, best_cost


def ids(graph: dict, start, goal, max_depth: int = 100):
    """
    Búsqueda de profundización iterativa (IDS).

    Args:
        graph:     dict  -> {nodo: [(vecino, costo), ...]}
        start:          -> nodo inicial
        goal:           -> nodo objetivo
        max_depth: int  -> límite máximo de profundidad (seguridad)

    Returns:
        (path, cost) si hay solución, (None, inf) si no.
    """
    for limit in range(max_depth + 1):
        visited = {start}
        path, cost = _depth_limited_search(
            graph, start, goal, limit, [start], 0, visited
        )
        if path is not None:
            print(f"  Solución encontrada en profundidad límite = {limit}")
            return path, cost

    return None, float('inf')


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('D', 2), ('E', 5)],
        'C': [('A', 4), ('F', 3)],
        'D': [('B', 2)],
        'E': [('B', 5), ('G', 2)],
        'F': [('C', 3), ('G', 1)],
        'G': [('E', 2), ('F', 1)],
    }

    path, cost = ids(graph, 'A', 'G')
    print("=== IDS ===")
    print(f"Camino encontrado : {' -> '.join(path)}")
    print(f"Costo total       : {cost}")
