"""
Depth-First Search (DFS) - Búsqueda en Profundidad
===================================================
Expande siempre el nodo más profundo de la frontera actual.
Utiliza una pila (LIFO).

Propiedades:
  - Completo:   Sí (si el espacio es finito y se evitan ciclos)
  - Óptimo:     No
  - Tiempo:     O(b^m)
  - Espacio:    O(b*m)
"""

from collections import defaultdict


def dfs(graph: dict, start, goal):
    """
    Búsqueda en profundidad (DFS) con detección de ciclos.

    Args:
        graph: dict  -> {nodo: [(vecino, costo), ...]}
        start:       -> nodo inicial
        goal:        -> nodo objetivo

    Returns:
        (path, cost) si hay solución, (None, inf) si no.
    """
    # Pila: (nodo_actual, camino_hasta_aquí, costo_acumulado)
    stack = [(start, [start], 0)]
    visited = set()

    while stack:
        node, path, cost = stack.pop()

        if node == goal:
            return path, cost

        if node in visited:
            continue
        visited.add(node)

        for neighbor, edge_cost in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor], cost + edge_cost))

    return None, float('inf')


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Grafo de ejemplo (no dirigido)
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('D', 2), ('E', 5)],
        'C': [('A', 4), ('F', 3)],
        'D': [('B', 2)],
        'E': [('B', 5), ('G', 2)],
        'F': [('C', 3), ('G', 1)],
        'G': [('E', 2), ('F', 1)],
    }

    path, cost = dfs(graph, 'A', 'G')
    print("=== DFS ===")
    print(f"Camino encontrado : {' -> '.join(path)}")
    print(f"Costo total       : {cost}")
