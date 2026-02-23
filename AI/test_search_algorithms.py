import pytest
from search_algorithms import dfs, bfs, ucs, ids

# ==========================================================
# GRAFOS DE PRUEBA
# ==========================================================

# 1️⃣ Grafo trivial (un solo nodo)
graph_single = {
    'A': []
}

# 2️⃣ Grafo lineal simple
graph_linear = {
    'A': [('B', 1)],
    'B': [('C', 1)],
    'C': [('D', 1)],
    'D': []
}

# 3️⃣ Grafo con múltiples caminos (para probar optimalidad)
graph_multiple_paths = {
    'A': [('B', 1), ('C', 5)],
    'B': [('D', 1)],
    'C': [('D', 1)],
    'D': []
}

# 4️⃣ Grafo con ciclo
graph_cycle = {
    'A': [('B', 1)],
    'B': [('C', 1)],
    'C': [('A', 1), ('D', 1)],
    'D': []
}

# 5️⃣ Grafo sin solución
graph_no_solution = {
    'A': [('B', 1)],
    'B': [],
    'C': []
}


# ==========================================================
# TESTS DFS
# ==========================================================

def test_dfs_single_node():
    result = dfs(graph_single, 'A', lambda x: x == 'A')
    assert result == ['A']

def test_dfs_linear():
    result = dfs(graph_linear, 'A', lambda x: x == 'D')
    assert result == ['A', 'B', 'C', 'D']

def test_dfs_cycle():
    result = dfs(graph_cycle, 'A', lambda x: x == 'D')
    assert result == ['A', 'B', 'C', 'D']

def test_dfs_no_solution():
    result = dfs(graph_no_solution, 'A', lambda x: x == 'C')
    assert result is None


# ==========================================================
# TESTS BFS
# ==========================================================

def test_bfs_linear():
    result = bfs(graph_linear, 'A', lambda x: x == 'D')
    assert result == ['A', 'B', 'C', 'D']

def test_bfs_shortest_path():
    result = bfs(graph_multiple_paths, 'A', lambda x: x == 'D')
    # BFS debe encontrar el camino más corto en número de pasos
    assert result == ['A', 'B', 'D']

def test_bfs_no_solution():
    result = bfs(graph_no_solution, 'A', lambda x: x == 'C')
    assert result is None


# ==========================================================
# TESTS UCS
# ==========================================================

def test_ucs_optimal_path():
    path, cost = ucs(graph_multiple_paths, 'A', lambda x: x == 'D')
    assert path == ['A', 'B', 'D']
    assert cost == 2

def test_ucs_linear():
    path, cost = ucs(graph_linear, 'A', lambda x: x == 'D')
    assert path == ['A', 'B', 'C', 'D']
    assert cost == 3

def test_ucs_no_solution():
    result = ucs(graph_no_solution, 'A', lambda x: x == 'C')
    assert result is None


# ==========================================================
# TESTS IDS
# ==========================================================

def test_ids_linear():
    result = ids(graph_linear, 'A', lambda x: x == 'D', max_depth=10)
    assert result == ['A', 'B', 'C', 'D']

def test_ids_cycle():
    result = ids(graph_cycle, 'A', lambda x: x == 'D', max_depth=10)
    assert result == ['A', 'B', 'C', 'D']

def test_ids_no_solution():
    result = ids(graph_no_solution, 'A', lambda x: x == 'C', max_depth=5)
    assert result is None