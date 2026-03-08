"""
Breadth-First Search (BFS) - Búsqueda en Anchura
=================================================
Expande el nodo menos profundo de la frontera, explorando por niveles.
Utiliza una cola (FIFO).

Propiedades:
  - Completo:   Sí
  - Óptimo:     Sí (si los costos son uniformes)
  - Tiempo:     O(b^s)
  - Espacio:    O(b^s)
"""

from collections import deque


def bfs(graph: dict, start, goal):
    """
    Búsqueda en anchura (BFS).

    Args:
        graph: dict  -> {nodo: [(vecino, costo), ...]}
        start:       -> nodo inicial
        goal:        -> nodo objetivo

    Returns:
        (path, cost) si hay solución, (None, inf) si no.
    """
    # Cola: (nodo_actual, camino_hasta_aquí, costo_acumulado)
    queue = deque([(start, [start], 0)])
    visited = {start}

    while queue:
        node, path, cost = queue.popleft()

        if node == goal:
            return path, cost

        for neighbor, edge_cost in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], cost + edge_cost))

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

    path, cost = bfs(graph, 'A', 'G')
    print("=== BFS ===")
    print(f"Camino encontrado : {' -> '.join(path)}")
    print(f"Costo total       : {cost}")
