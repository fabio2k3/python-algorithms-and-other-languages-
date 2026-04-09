"""
Algoritmos de Búsqueda - IA
BFS, DFS, UCS, Greedy, A*
"""

from collections import deque
import heapq


# ─────────────────────────────────────────────
#  GRAFO DE EJEMPLO (compartido por todos)
# ─────────────────────────────────────────────
#
#  A -1- B -2- D
#  |         /
#  3       1
#  |     /
#  C -4- E -2- F
#
# Grafo no dirigido con pesos

GRAFO = {
    'A': [('B', 1), ('C', 3)],
    'B': [('A', 1), ('D', 2)],
    'C': [('A', 3), ('E', 4)],
    'D': [('B', 2), ('E', 1)],
    'E': [('D', 1), ('C', 4), ('F', 2)],
    'F': [('E', 2)],
}

# Heurística h(n) = estimación de distancia al objetivo 'F'
HEURISTICA = {
    'A': 7,
    'B': 6,
    'C': 4,
    'D': 4,
    'E': 2,
    'F': 0,
}


# ─────────────────────────────────────────────
#  1. BFS — Breadth-First Search
# ─────────────────────────────────────────────
def bfs(grafo, inicio, objetivo):
    """
    Explora nivel por nivel (FIFO).
    - Completo: sí
    - Óptimo: sí (si los costos son uniformes)
    - Tiempo/Espacio: O(b^d)
    """
    cola = deque()
    cola.append((inicio, [inicio]))      # (nodo_actual, camino)
    visitados = set()
    visitados.add(inicio)

    while cola:
        nodo, camino = cola.popleft()

        print(f"  Visitando: {nodo}  |  Cola: {[c for c,_ in cola]}")

        if nodo == objetivo:
            return camino

        for vecino, _ in grafo[nodo]:
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append((vecino, camino + [vecino]))

    return None   # no encontrado


# ─────────────────────────────────────────────
#  2. DFS — Depth-First Search
# ─────────────────────────────────────────────
def dfs(grafo, inicio, objetivo):
    """
    Explora en profundidad primero (LIFO / recursivo).
    - Completo: sí (en grafos finitos)
    - Óptimo: NO
    - Tiempo: O(b^m)  |  Espacio: O(b*m)
    """
    visitados = set()

    def _dfs(nodo, camino):
        visitados.add(nodo)
        print(f"  Visitando: {nodo}  |  Pila (camino): {camino}")

        if nodo == objetivo:
            return camino

        for vecino, _ in grafo[nodo]:
            if vecino not in visitados:
                resultado = _dfs(vecino, camino + [vecino])
                if resultado:
                    return resultado
        return None

    return _dfs(inicio, [inicio])


# ─────────────────────────────────────────────
#  3. UCS — Uniform Cost Search
# ─────────────────────────────────────────────
def ucs(grafo, inicio, objetivo):
    """
    Expande el nodo de MENOR COSTO acumulado (cola de prioridad).
    - Completo: sí
    - Óptimo: sí
    - Tiempo/Espacio: O(b^(1 + C*/ε))
    """
    # heap: (costo_acumulado, nodo, camino)
    heap = [(0, inicio, [inicio])]
    visitados = {}   # nodo -> menor costo visto

    while heap:
        costo, nodo, camino = heapq.heappop(heap)

        print(f"  Visitando: {nodo}  |  Costo acumulado: {costo}")

        if nodo in visitados:
            continue
        visitados[nodo] = costo

        if nodo == objetivo:
            return camino, costo

        for vecino, peso in grafo[nodo]:
            if vecino not in visitados:
                heapq.heappush(heap, (costo + peso, vecino, camino + [vecino]))

    return None, float('inf')


# ─────────────────────────────────────────────
#  4. Greedy Best-First Search
# ─────────────────────────────────────────────
def greedy(grafo, inicio, objetivo, h):
    """
    Expande el nodo con MENOR HEURÍSTICA h(n).
    Solo mira qué tan cerca parece estar el objetivo (no el costo real).
    - Completo: NO (puede quedar en bucles)
    - Óptimo: NO
    - Rápido pero puede engañarse
    """
    heap = [(h[inicio], inicio, [inicio])]
    visitados = set()

    while heap:
        h_val, nodo, camino = heapq.heappop(heap)

        print(f"  Visitando: {nodo}  |  h(n)={h_val}")

        if nodo in visitados:
            continue
        visitados.add(nodo)

        if nodo == objetivo:
            return camino

        for vecino, _ in grafo[nodo]:
            if vecino not in visitados:
                heapq.heappush(heap, (h[vecino], vecino, camino + [vecino]))

    return None


# ─────────────────────────────────────────────
#  5. A* — A-Star Search
# ─────────────────────────────────────────────
def a_star(grafo, inicio, objetivo, h):
    """
    f(n) = g(n) + h(n)
      g(n) = costo real acumulado desde el inicio
      h(n) = estimación heurística hasta el objetivo
    - Completo: sí
    - Óptimo: sí (si h es admisible, es decir h(n) <= h_real(n))
    - El más usado en la práctica
    """
    # heap: (f, g, nodo, camino)
    heap = [(h[inicio], 0, inicio, [inicio])]
    mejor_g = {}   # nodo -> menor g visto

    while heap:
        f, g, nodo, camino = heapq.heappop(heap)

        print(f"  Visitando: {nodo}  |  g={g}, h={h[nodo]}, f={f}")

        if nodo in mejor_g and mejor_g[nodo] <= g:
            continue
        mejor_g[nodo] = g

        if nodo == objetivo:
            return camino, g

        for vecino, peso in grafo[nodo]:
            nuevo_g = g + peso
            nuevo_f = nuevo_g + h[vecino]
            heapq.heappush(heap, (nuevo_f, nuevo_g, vecino, camino + [vecino]))

    return None, float('inf')


# ─────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    inicio, objetivo = 'A', 'F'
    separador = "─" * 45

    print(f"\n{'='*45}")
    print(f"  Buscando camino de '{inicio}' a '{objetivo}'")
    print(f"{'='*45}")

    print(f"\n{separador}")
    print("  BFS")
    print(separador)
    camino = bfs(GRAFO, inicio, objetivo)
    print(f"  Camino: {' → '.join(camino)}\n")

    print(f"{separador}")
    print("  DFS")
    print(separador)
    camino = dfs(GRAFO, inicio, objetivo)
    print(f"  Camino: {' → '.join(camino)}\n")

    print(f"{separador}")
    print("  UCS")
    print(separador)
    camino, costo = ucs(GRAFO, inicio, objetivo)
    print(f"  Camino: {' → '.join(camino)}  |  Costo: {costo}\n")

    print(f"{separador}")
    print("  Greedy")
    print(separador)
    camino = greedy(GRAFO, inicio, objetivo, HEURISTICA)
    print(f"  Camino: {' → '.join(camino)}\n")

    print(f"{separador}")
    print("  A*")
    print(separador)
    camino, costo = a_star(GRAFO, inicio, objetivo, HEURISTICA)
    print(f"  Camino: {' → '.join(camino)}  |  Costo: {costo}\n")