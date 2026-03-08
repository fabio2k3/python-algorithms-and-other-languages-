"""
A* Search - Búsqueda A Estrella
================================
Combina el costo real acumulado g(n) con la heurística h(n):
    f(n) = g(n) + h(n)

La heurística debe ser:
  - Admisible:   0 <= h(n) <= h*(n)   → óptimo en árboles
  - Consistente: h(n) <= c(n,n') + h(n')  → óptimo en grafos

Propiedades:
  - Completo:   Sí (costos positivos)
  - Óptimo:     Sí (heurística admisible/consistente)
  - Tiempo:     O(b^d) en el mejor caso
  - Espacio:    O(b^d)  ← principal limitación
"""

import heapq


def astar(graph: dict, start, goal, heuristic: dict):
    """
    Búsqueda A* con heurística proporcionada.

    Args:
        graph:     dict -> {nodo: [(vecino, costo), ...]}
        start:          -> nodo inicial
        goal:           -> nodo objetivo
        heuristic: dict -> {nodo: valor_h}  (h(goal) debe ser 0)

    Returns:
        (path, cost) si hay solución, (None, inf) si no.
    """
    h = heuristic.get

    # Heap: (f, contador_desempate, g, nodo, camino)
    counter = 0
    heap = [(h(start, float('inf')), counter, 0, start, [start])]
    best_g = {start: 0}   # mejor g(n) conocido

    while heap:
        f, _, g, node, path = heapq.heappop(heap)

        # Ignorar si ya encontramos un camino mejor a este nodo
        if g > best_g.get(node, float('inf')):
            continue

        if node == goal:
            return path, g

        for neighbor, edge_cost in graph.get(node, []):
            new_g = g + edge_cost
            if new_g < best_g.get(neighbor, float('inf')):
                best_g[neighbor] = new_g
                new_f = new_g + h(neighbor, float('inf'))
                counter += 1
                heapq.heappush(heap, (new_f, counter, new_g, neighbor, path + [neighbor]))

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

    # Heurística admisible (estimación al objetivo 'G')
    heuristic = {'A': 6, 'B': 5, 'C': 4, 'D': 7, 'E': 2, 'F': 1, 'G': 0}

    path, cost = astar(graph, 'A', 'G', heuristic)
    print("=== A* ===")
    print(f"Camino encontrado : {' -> '.join(path)}")
    print(f"Costo total       : {cost}")
