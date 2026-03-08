"""
Uniform Cost Search (UCS) - Búsqueda de Costo Uniforme
=======================================================
Expande el nodo con menor costo acumulado g(n).
Utiliza una cola de prioridad ordenada por costo.

Propiedades:
  - Completo:   Sí (con costos positivos)
  - Óptimo:     Sí
  - Tiempo:     O(b^(C*/ε))
  - Espacio:    O(b^(C*/ε))
"""

import heapq


def ucs(graph: dict, start, goal):
    """
    Búsqueda de costo uniforme (UCS).

    Args:
        graph: dict  -> {nodo: [(vecino, costo), ...]}
        start:       -> nodo inicial
        goal:        -> nodo objetivo

    Returns:
        (path, cost) si hay solución, (None, inf) si no.
    """
    # Heap: (costo_acumulado, contador_desempate, nodo, camino)
    counter = 0
    heap = [(0, counter, start, [start])]
    visited = {}  # nodo -> mejor costo conocido

    while heap:
        cost, _, node, path = heapq.heappop(heap)

        # Si ya procesamos este nodo con menor costo, ignorar
        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost

        if node == goal:
            return path, cost

        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in visited or visited[neighbor] > new_cost:
                counter += 1
                heapq.heappush(heap, (new_cost, counter, neighbor, path + [neighbor]))

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

    path, cost = ucs(graph, 'A', 'G')
    print("=== UCS ===")
    print(f"Camino encontrado : {' -> '.join(path)}")
    print(f"Costo total       : {cost}")
