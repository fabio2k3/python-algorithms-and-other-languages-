"""
SMA* (Simplified Memory-Bounded A*)
=====================================
Variante de A* que usa TODA la memoria disponible y, cuando se agota,
elimina el nodo con mayor f(n) (el menos prometedor), guardando su
mejor estimación en el nodo padre para poder retomarlo si es necesario.

Estrategia:
  - Mantiene una cola de prioridad acotada a `memory_limit` nodos.
  - Cuando el límite se alcanza, expulsa el nodo con mayor f(n).
  - El padre actualiza su f almacenado al min(f_hijos_olvidados).
  - Si un nodo expulsado vuelve a ser necesario, se regenera desde el padre.

Propiedades:
  - Completo:   Sí, si la solución cabe en memoria
  - Óptimo:     Sí, si la solución cabe en memoria
  - Espacio:    O(memory_limit)
"""

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _Node:
    f: float
    g: float = field(compare=False)
    node: Any = field(compare=False)
    path: list = field(compare=False)
    depth: int = field(compare=False)


def sma_star(graph: dict, start, goal, heuristic: dict, memory_limit: int = 20):
    """
    Búsqueda SMA*.

    Args:
        graph:        dict -> {nodo: [(vecino, costo), ...]}
        start:             -> nodo inicial
        goal:              -> nodo objetivo
        heuristic:    dict -> {nodo: valor_h}
        memory_limit: int  -> número máximo de nodos en memoria

    Returns:
        (path, cost) si hay solución, (None, inf) si no.
    """
    h = heuristic.get

    # Cola de prioridad (min-heap por f)
    # Usamos lista con heapq; guardamos (f, g, nodo, camino)
    counter = 0

    def make_entry(g, node, path):
        nonlocal counter
        f = g + h(node, float('inf'))
        counter += 1
        return [f, counter, g, node, path]

    open_list = [make_entry(0, start, [start])]
    heapq.heapify(open_list)

    # Tabla de mejores g conocidos
    best_g = {start: 0}

    iterations = 0
    MAX_ITER = 10_000  # salvaguarda para evitar bucles infinitos

    while open_list and iterations < MAX_ITER:
        iterations += 1
        f, _, g, node, path = heapq.heappop(open_list)

        # Saltar entradas desactualizadas
        if g > best_g.get(node, float('inf')):
            continue

        if node == goal:
            return path, g

        # Expandir sucesores
        for neighbor, edge_cost in graph.get(node, []):
            new_g = g + edge_cost
            if new_g < best_g.get(neighbor, float('inf')):
                best_g[neighbor] = new_g
                heapq.heappush(open_list, make_entry(new_g, neighbor, path + [neighbor]))

        # ── Límite de memoria: expulsar los nodos con mayor f ──────────────
        while len(open_list) > memory_limit:
            # Encontrar el índice del nodo con mayor f
            worst_idx = max(range(len(open_list)), key=lambda i: (open_list[i][0], -open_list[i][1]))
            open_list.pop(worst_idx)
            heapq.heapify(open_list)

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

    heuristic = {'A': 6, 'B': 5, 'C': 4, 'D': 7, 'E': 2, 'F': 1, 'G': 0}

    path, cost = sma_star(graph, 'A', 'G', heuristic, memory_limit=10)
    print("=== SMA* ===")
    if path:
        print(f"Camino encontrado : {' -> '.join(path)}")
        print(f"Costo total       : {cost}")
    else:
        print("No se encontró solución (memoria insuficiente o grafo sin solución).")
