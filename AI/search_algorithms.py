"""
Búsquedas no informadas: DFS, BFS, UCS, IDS
------------------------------------------
Implementaciones en modo 'grafo' (evitan repetir nodos) por defecto.
Representación de grafo:
- graph: dict[node] -> list of (neighbor, cost)   # usado por UCS; para DFS/BFS se puede ignorar el coste
- Las funciones aceptan `goal_test` como callable para comprobar si un estado es objetivo.

Autor: (generado) — Comentarios profesionales en español.
Notas generales:
- Las funciones están diseñadas para ser reutilizables en proyectos de IA/robótica/sistemas de búsqueda.
- Para problemas grandes o con requisitos específicos (p. ej. memoria, trazado de nodos expandidos),
  conviene adaptar la gestión de `parent`, `visited` y los filtros de expansión.
"""

from collections import deque
import heapq
from typing import Dict, List, Tuple, Any, Optional, Set

# -------------------------
# Utilidades comunes
# -------------------------
def reconstruct_path(parent: Dict[Any, Any], start: Any, goal: Any) -> List[Any]:
    """
    Reconstruye el camino desde `start` hasta `goal` usando el diccionario `parent`.

    Parámetros
    ----------
    parent : Dict[Any, Any]
        Mapa que para cada nodo almacena su padre (nodo desde el que fue alcanzado).
    start : Any
        Nodo inicial de la búsqueda.
    goal : Any
        Nodo objetivo (desde donde comenzamos a reconstruir hacia atrás).

    Retorno
    -------
    List[Any]
        Lista de nodos que representa el camino desde `start` hasta `goal`.
        Devuelve una lista vacía si no se puede reconstruir un camino válido.
    """
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()
    # Verificación defensiva: el primer nodo debe ser el inicio
    if path and path[0] == start:
        return path
    return []  # no hay camino válido

# -------------------------
# DFS - Depth First Search
# -------------------------
def dfs(graph: Dict[Any, List[Tuple[Any, float]]],
        start: Any,
        goal_test,
        use_graph_search: bool = True) -> Optional[List[Any]]:
    """
    Búsqueda en profundidad (iterativa, con pila).

    Descripción
    -----------
    Realiza una búsqueda en profundidad desde `start` hasta que `goal_test(node)` sea True.
    Por defecto evita reexpandir nodos ya visitados (modo grafo). Si `use_graph_search` es False,
    el algoritmo se comporta como búsqueda en árbol (no previene ciclos).

    Parámetros
    ----------
    graph : Dict[Any, List[Tuple[Any, float]]]
        Lista de adyacencia: para cada nodo lista de pares (vecino, coste). El coste se ignora para DFS.
    start : Any
        Estado inicial.
    goal_test : callable
        Función que recibe un nodo y devuelve True si es objetivo.
    use_graph_search : bool, opcional
        Si True usa un conjunto `visited` para evitar reexpansiones (comportamiento de grafo).

    Retorno
    -------
    Optional[List[Any]]
        Camino desde start hasta el nodo objetivo si se encuentra; None en caso contrario.

    Complejidad
    ----------
    - Tiempo: O(b^m) en el peor caso (b = factor de ramificación, m = profundidad máxima explorada).
    - Espacio: O(m) para la pila (iterativo).
    """
    # la pila contiene tuplas (nodo, padre_por_compatibilidad) aunque el padre real se guarda en `parent`
    stack = [(start, None)]
    parent = {start: None}
    visited: Set[Any] = set()

    while stack:
        node, _ = stack.pop()
        if use_graph_search:
            if node in visited:
                continue
            visited.add(node)

        # comprobación de meta al extraer el nodo (estrategia LIFO)
        if goal_test(node):
            return reconstruct_path(parent, start, node)

        # expandir vecinos: usamos reversed para mantener un orden lógico de exploración
        for neighbor, _cost in reversed(graph.get(node, [])):
            if use_graph_search and neighbor in visited:
                continue
            # registramos el padre la primera vez que descubrimos el vecino
            if neighbor not in parent:
                parent[neighbor] = node
            stack.append((neighbor, node))
    return None

# -------------------------
# BFS - Breadth First Search
# -------------------------
def bfs(graph: Dict[Any, List[Tuple[Any, float]]],
        start: Any,
        goal_test,
        use_graph_search: bool = True) -> Optional[List[Any]]:
    """
    Búsqueda en anchura.

    Descripción
    -----------
    Explora el espacio por niveles (cola FIFO). Es completa y óptima si todas las aristas
    tienen coste uniforme (p. ej. coste = 1).

    Parámetros
    ----------
    graph : Dict[Any, List[Tuple[Any, float]]]
        Lista de adyacencia. El valor de coste se ignora para la lógica de BFS.
    start : Any
        Estado inicial.
    goal_test : callable
        Función que determina si un nodo es objetivo.
    use_graph_search : bool, opcional
        Si True marca nodos visitados para evitar reexpansiones (recomendado para grafos).

    Retorno
    -------
    Optional[List[Any]]
        Camino desde start hasta el primer nodo objetivo encontrado; None si no existe.

    Complejidad
    ----------
    - Tiempo: O(b^d) donde d es la profundidad de la solución.
    - Espacio: O(b^d) (BFS almacena la frontera completa para cada nivel).
    """
    queue = deque([start])
    parent = {start: None}
    visited: Set[Any] = {start} if use_graph_search else set()

    while queue:
        node = queue.popleft()
        if goal_test(node):
            return reconstruct_path(parent, start, node)

        for neighbor, _cost in graph.get(node, []):
            if use_graph_search:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
            # registramos el padre la primera vez que se descubre el vecino
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    return None

# -------------------------
# UCS - Uniform Cost Search
# -------------------------
def ucs(graph: Dict[Any, List[Tuple[Any, float]]],
        start: Any,
        goal_test) -> Optional[Tuple[List[Any], float]]:
    """
    Búsqueda de costo uniforme (Uniform Cost Search).

    Descripción
    -----------
    Encuentra la ruta de coste mínimo desde `start` hasta un nodo que cumpla `goal_test`,
    asumiendo costes no negativos en las aristas. Emplea una cola de prioridad (heap) ordenada
    por el coste acumulado g(n).

    Parámetros
    ----------
    graph : Dict[Any, List[Tuple[Any, float]]]
        Mapa de adyacencia: nodo -> lista de (vecino, coste).
    start : Any
        Nodo inicial.
    goal_test : callable
        Función que determina si un nodo es objetivo.

    Retorno
    -------
    Optional[Tuple[List[Any], float]]
        Tupla (camino, coste_total) si se encuentra una solución; None si no existe.

    Observaciones importantes
    ------------------------
    - Para garantizar optimalidad, TODOS los costes deben ser >= 0.
    - Un nodo se considera expandido (cerrado) la primera vez que es extraído del heap con su menor coste.
      Por ello se utiliza `visited` solo al extraer de la frontera.
    - Si las aristas tienen costes negativos, el algoritmo no es correcto.

    Complejidad
    ----------
    - Depende fuertemente de la estructura de costes; en el peor caso se aproxima a O(b^(C*/ε)),
      con C* coste óptimo y ε coste mínimo positivo.
    """
    frontier = []  # heap de tuplas (g_cost, nodo)
    heapq.heappush(frontier, (0.0, start))
    parent: Dict[Any, Any] = {start: None}
    g_costs: Dict[Any, float] = {start: 0.0}
    visited: Set[Any] = set()  # nodos extraídos definitivamente (con su coste mínimo)

    while frontier:
        g, node = heapq.heappop(frontier)
        # Si ya cerramos este nodo (se extrajo con menor coste antes), lo ignoramos
        if node in visited:
            continue
        visited.add(node)

        if goal_test(node):
            # Al extraer un objetivo, su coste g es el coste mínimo alcanzable
            return reconstruct_path(parent, start, node), g

        for neighbor, cost in graph.get(node, []):
            new_g = g + cost
            # Si encontramos un camino más barato hacia `neighbor`, actualizamos su coste y padre
            if neighbor not in g_costs or new_g < g_costs[neighbor]:
                g_costs[neighbor] = new_g
                parent[neighbor] = node
                heapq.heappush(frontier, (new_g, neighbor))
    return None

# -------------------------
# IDS - Iterative Deepening Search
# -------------------------
def dls_limited(graph: Dict[Any, List[Tuple[Any, float]]],
                node,
                goal_test,
                limit: int,
                parent: Dict[Any, Any],
                visited: Set[Any]) -> Tuple[bool, bool]:
    """
    Depth-Limited Search (DLS) auxiliar para IDS.

    Descripción
    -----------
    Realiza una DFS limitada por profundidad `limit`. Devuelve una tupla (found, cutoff_occurred):
    - found: True si se encontró el objetivo en esta llamada.
    - cutoff_occurred: True si alguna rama alcanzó el límite (indica posibilidad de solución a mayor profundidad).

    Parámetros
    ----------
    graph : Dict[Any, List[Tuple[Any, float]]]
        Grafo de adyacencia.
    node : Any
        Nodo actual (raíz de la subbúsqueda recursiva).
    goal_test : callable
        Prueba de objetivo.
    limit : int
        Límite de profundidad actual (0 = no expandir más).
    parent : Dict[Any, Any]
        Diccionario compartido para registrar padres (se modifica durante la recursión).
    visited : Set[Any]
        Conjunto de nodos visitados (no usado activamente aquí, pero queda para extensiones).

    Retorno
    -------
    Tuple[bool, bool]
        (found, cutoff_occurred) descritos arriba.

    Advertencias
    -----------
    - Esta implementación gestiona ciclos de forma sencilla evitando reusar entradas ya en `parent`.
      En grafos con ciclos complejos puede ser necesario control adicional (p. ej. detectar ancestros).
    """
    if goal_test(node):
        return True, False
    if limit == 0:
        # alcanzamos el límite sin encontrar solución en esta rama -> corte
        return False, True

    cutoff_occurred = False
    for neighbor, _cost in graph.get(node, []):
        # evitamos volver por el mismo enlace padre -> prevención básica de ciclos inmediatos
        if neighbor in parent:
            continue
        parent[neighbor] = node
        found, cut = dls_limited(graph, neighbor, goal_test, limit - 1, parent, visited)
        if found:
            return True, False
        if cut:
            cutoff_occurred = True
        # si no se encontró en la rama, revertimos la asignación para explorar otras ramas
        del parent[neighbor]
    return False, cutoff_occurred

def ids(graph: Dict[Any, List[Tuple[Any, float]]],
        start: Any,
        goal_test,
        max_depth: int = 1000) -> Optional[List[Any]]:
    """
    Iterative Deepening Search (IDS).

    Descripción
    -----------
    Ejecuta repetidamente búsquedas DLS con límites crecientes hasta encontrar la solución
    o alcanzar `max_depth`. IDS combina la baja memoria de DFS con la completitud de BFS
    para espacios de búsqueda con profundidad finita.

    Parámetros
    ----------
    graph : Dict[Any, List[Tuple[Any, float]]]
        Grafo de adyacencia.
    start : Any
        Nodo inicial.
    goal_test : callable
        Función de test de objetivo.
    max_depth : int, opcional
        Profundidad máxima a probar (por seguridad).

    Retorno
    -------
    Optional[List[Any]]
        Camino desde start hasta el nodo objetivo, o None si no se encuentra solución
        hasta max_depth.

    Observaciones
    ------------
    - IDS es particularmente adecuada cuando la solución está relativamente cerca y la memoria es limitada.
    - En grafos con ciclos conviene asegurar mecanismos para evitar reexploración infinita; la implementación
      aquí evita ciclos inmediatos mediante `parent`, pero podría requerir mejoras para grafos densos.
    """
    for depth in range(max_depth + 1):
        parent = {start: None}
        visited: Set[Any] = set()
        found, cutoff = dls_limited(graph, start, goal_test, depth, parent, visited)
        if found:
            # buscamos el nodo objetivo entre las claves de parent (goal_test puede aplicarse a varios estados)
            goal_node = None
            for n in parent:
                if goal_test(n):
                    goal_node = n
                    break
            # comprobación defensiva: en circunstancias raras intentamos otra búsqueda ligera
            if goal_node is None:
                for n in parent.keys():
                    if goal_test(n):
                        goal_node = n
                        break
            if goal_node is None:
                return None
            return reconstruct_path(parent, start, goal_node)
        if not cutoff:
            # No hubo corte en esta iteración y no se encontró solución: no existe solución a mayor profundidad.
            return None
    # alcanzado max_depth sin encontrar solución
    return None