"""
IDA* (Iterative-Deepening A*) - A* con Profundización Iterativa
================================================================
Variante de A* con memoria lineal. En lugar de un límite de profundidad,
usa un umbral (threshold) sobre el valor f(n) = g(n) + h(n).

Algoritmo:
  1. threshold = h(start)
  2. DFS podando ramas donde f(n) > threshold
  3. Si no hay solución, el nuevo threshold = min f podado
  4. Repetir hasta encontrar objetivo

Propiedades:
  - Completo:   Sí (costos positivos, heurística admisible)
  - Óptimo:     Sí (heurística admisible)
  - Tiempo:     Similar a A* (puede re-expandir nodos)
  - Espacio:    O(b*d)  ← ventaja principal sobre A*
"""


def _search(graph, node, g, threshold, goal, heuristic, path):
    """Función recursiva de búsqueda con umbral."""
    f = g + heuristic.get(node, float('inf'))

    if f > threshold:
        return f, None, float('inf')   # poda: devolver f superado

    if node == goal:
        return -1, path, g             # -1 indica solución encontrada

    minimum = float('inf')

    for neighbor, edge_cost in graph.get(node, []):
        if neighbor not in path:       # evitar ciclos
            result, sol_path, sol_cost = _search(
                graph, neighbor, g + edge_cost,
                threshold, goal, heuristic, path + [neighbor]
            )
            if result == -1:
                return -1, sol_path, sol_cost
            if result < minimum:
                minimum = result

    return minimum, None, float('inf')


def idastar(graph: dict, start, goal, heuristic: dict):
    """
    Búsqueda IDA*.

    Args:
        graph:     dict -> {nodo: [(vecino, costo), ...]}
        start:          -> nodo inicial
        goal:           -> nodo objetivo
        heuristic: dict -> {nodo: valor_h}

    Returns:
        (path, cost) si hay solución, (None, inf) si no.
    """
    threshold = heuristic.get(start, 0)
    path = [start]

    while True:
        result, sol_path, sol_cost = _search(
            graph, start, 0, threshold, goal, heuristic, path
        )

        if result == -1:
            print(f"  Solución encontrada con threshold final = {threshold}")
            return sol_path, sol_cost

        if result == float('inf'):
            return None, float('inf')   # no hay solución

        print(f"  Threshold {threshold} insuficiente → nuevo threshold = {result}")
        threshold = result


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

    heuristic = {'A': 6, 'B': 5, 'C': 4, 'D': 7, 'E': 2, 'F': 1, 'G': 0}

    path, cost = idastar(graph, 'A', 'G', heuristic)
    print("=== IDA* ===")
    print(f"Camino encontrado : {' -> '.join(path)}")
    print(f"Costo total       : {cost}")
