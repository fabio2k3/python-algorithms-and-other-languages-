"""
test_algorithms.py
==================
Suite de tests completa para todos los algoritmos de búsqueda:
  DFS, BFS, UCS, IDS, A*, IDA*, SMA*

Casos de prueba:
  1. Grafo simple (solución clara)
  2. Grafo con múltiples caminos (verifica optimalidad)
  3. Grafo sin solución
  4. Grafo con un solo nodo (start == goal)
  5. Grafo lineal (cadena)
  6. Grafo denso (muchas conexiones)

Ejecutar:
    python test_algorithms.py
"""

import sys
import traceback
from dfs     import dfs
from bfs     import bfs
from ucs     import ucs
from ids     import ids
from astar   import astar
from idastar import idastar
from smastar import sma_star


# ══════════════════════════════════════════════════════════════════════════════
#  Grafos y heurísticas de prueba
# ══════════════════════════════════════════════════════════════════════════════

# Grafo 1: simple, 7 nodos
GRAPH_SIMPLE = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('D', 2), ('E', 5)],
    'C': [('A', 4), ('F', 3)],
    'D': [('B', 2)],
    'E': [('B', 5), ('G', 2)],
    'F': [('C', 3), ('G', 1)],
    'G': [('E', 2), ('F', 1)],
}
H_SIMPLE = {'A': 6, 'B': 5, 'C': 4, 'D': 7, 'E': 2, 'F': 1, 'G': 0}
# Camino óptimo: A->B->E->G (costo 8) o A->C->F->G (costo 8)
OPTIMAL_SIMPLE = 8

# Grafo 2: múltiples caminos con distinto costo
GRAPH_MULTI = {
    'S': [('A', 1), ('B', 5)],
    'A': [('S', 1), ('C', 2), ('D', 8)],
    'B': [('S', 5), ('D', 2)],
    'C': [('A', 2), ('G', 10)],
    'D': [('A', 8), ('B', 2), ('G', 3)],
    'G': [('C', 10), ('D', 3)],
}
H_MULTI = {'S': 7, 'A': 6, 'B': 4, 'C': 5, 'D': 3, 'G': 0}
# Camino óptimo: S->A->C->G? No. S->B->D->G = 5+2+3=10; S->A->D->G=1+8+3=12; S->B->D->G=10
OPTIMAL_MULTI = 10

# Grafo 3: sin solución (nodo objetivo desconectado)
GRAPH_NO_SOLUTION = {
    'A': [('B', 1)],
    'B': [('A', 1)],
    'C': [],   # componente aislada
}
H_NO_SOLUTION = {'A': 1, 'B': 0, 'C': 0}

# Grafo 4: start == goal
GRAPH_SELF = {'X': [('Y', 3)], 'Y': [('X', 3)]}
H_SELF = {'X': 0, 'Y': 3}

# Grafo 5: cadena lineal
GRAPH_CHAIN = {
    '1': [('2', 1)],
    '2': [('1', 1), ('3', 1)],
    '3': [('2', 1), ('4', 1)],
    '4': [('3', 1), ('5', 1)],
    '5': [('4', 1)],
}
H_CHAIN = {'1': 4, '2': 3, '3': 2, '4': 1, '5': 0}
OPTIMAL_CHAIN = 4

# Grafo 6: denso
GRAPH_DENSE = {
    'A': [('B', 2), ('C', 5), ('D', 1)],
    'B': [('A', 2), ('C', 1), ('E', 4)],
    'C': [('A', 5), ('B', 1), ('D', 2), ('E', 1), ('F', 3)],
    'D': [('A', 1), ('C', 2), ('F', 6)],
    'E': [('B', 4), ('C', 1), ('F', 1)],
    'F': [('C', 3), ('D', 6), ('E', 1)],
}
H_DENSE = {'A': 5, 'B': 3, 'C': 2, 'D': 4, 'E': 1, 'F': 0}
OPTIMAL_DENSE = 5   # A->B->C->E->F = 2+1+1+1=5


# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades de test
# ══════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
SKIP = "\033[93m~ SKIP\033[0m"

results_summary = []


def run_test(algo_name, func, *args, expected_cost=None, expect_none=False, **kwargs):
    """
    Ejecuta una función de búsqueda y valida el resultado.

    - expected_cost: costo óptimo esperado (None = no verificar)
    - expect_none:   True si se espera que no haya solución
    """
    label = f"{algo_name:<10}"
    try:
        result = func(*args, **kwargs)
        path, cost = result

        if expect_none:
            if path is None:
                status = PASS
                detail = "Sin solución (correcto)"
            else:
                status = FAIL
                detail = f"Se esperaba None, se obtuvo camino {path} con costo {cost}"
        else:
            if path is None:
                status = FAIL
                detail = "No encontró solución (se esperaba una)"
            elif expected_cost is not None and abs(cost - expected_cost) > 1e-9:
                status = FAIL
                detail = f"Costo {cost} ≠ esperado {expected_cost} | Camino: {path}"
            else:
                status = PASS
                detail = f"Costo = {cost} | Camino: {' -> '.join(str(p) for p in path)}"

    except Exception:
        status = FAIL
        detail = traceback.format_exc().strip().split('\n')[-1]

    results_summary.append(status == PASS or PASS in status)
    print(f"  [{status}] {label} | {detail}")


def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print('═'*65)


# ══════════════════════════════════════════════════════════════════════════════
#  TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_grafo_simple():
    section("TEST 1 · Grafo simple (A → G)")
    run_test("DFS",   dfs,     GRAPH_SIMPLE, 'A', 'G')
    run_test("BFS",   bfs,     GRAPH_SIMPLE, 'A', 'G')
    run_test("UCS",   ucs,     GRAPH_SIMPLE, 'A', 'G',      expected_cost=OPTIMAL_SIMPLE)
    run_test("IDS",   ids,     GRAPH_SIMPLE, 'A', 'G')
    run_test("A*",    astar,   GRAPH_SIMPLE, 'A', 'G', H_SIMPLE, expected_cost=OPTIMAL_SIMPLE)
    run_test("IDA*",  idastar, GRAPH_SIMPLE, 'A', 'G', H_SIMPLE, expected_cost=OPTIMAL_SIMPLE)
    run_test("SMA*",  sma_star,GRAPH_SIMPLE, 'A', 'G', H_SIMPLE, expected_cost=OPTIMAL_SIMPLE)


def test_optimalidad():
    section("TEST 2 · Optimalidad (múltiples caminos, S → G)")
    # DFS e IDS no garantizan optimalidad → no verificamos su costo
    run_test("DFS",   dfs,     GRAPH_MULTI, 'S', 'G')
    run_test("BFS",   bfs,     GRAPH_MULTI, 'S', 'G')
    run_test("UCS",   ucs,     GRAPH_MULTI, 'S', 'G',      expected_cost=OPTIMAL_MULTI)
    run_test("IDS",   ids,     GRAPH_MULTI, 'S', 'G')
    run_test("A*",    astar,   GRAPH_MULTI, 'S', 'G', H_MULTI, expected_cost=OPTIMAL_MULTI)
    run_test("IDA*",  idastar, GRAPH_MULTI, 'S', 'G', H_MULTI, expected_cost=OPTIMAL_MULTI)
    run_test("SMA*",  sma_star,GRAPH_MULTI, 'S', 'G', H_MULTI, expected_cost=OPTIMAL_MULTI)


def test_sin_solucion():
    section("TEST 3 · Sin solución (A → C, componentes distintas)")
    run_test("DFS",   dfs,     GRAPH_NO_SOLUTION, 'A', 'C', expect_none=True)
    run_test("BFS",   bfs,     GRAPH_NO_SOLUTION, 'A', 'C', expect_none=True)
    run_test("UCS",   ucs,     GRAPH_NO_SOLUTION, 'A', 'C', expect_none=True)
    run_test("IDS",   ids,     GRAPH_NO_SOLUTION, 'A', 'C', expect_none=True)
    run_test("A*",    astar,   GRAPH_NO_SOLUTION, 'A', 'C', H_NO_SOLUTION, expect_none=True)
    run_test("IDA*",  idastar, GRAPH_NO_SOLUTION, 'A', 'C', H_NO_SOLUTION, expect_none=True)
    run_test("SMA*",  sma_star,GRAPH_NO_SOLUTION, 'A', 'C', H_NO_SOLUTION, expect_none=True)


def test_start_igual_goal():
    section("TEST 4 · start == goal  (X → X)")
    run_test("DFS",   dfs,     GRAPH_SELF, 'X', 'X', expected_cost=0)
    run_test("BFS",   bfs,     GRAPH_SELF, 'X', 'X', expected_cost=0)
    run_test("UCS",   ucs,     GRAPH_SELF, 'X', 'X', expected_cost=0)
    run_test("IDS",   ids,     GRAPH_SELF, 'X', 'X', expected_cost=0)
    run_test("A*",    astar,   GRAPH_SELF, 'X', 'X', H_SELF, expected_cost=0)
    run_test("IDA*",  idastar, GRAPH_SELF, 'X', 'X', H_SELF, expected_cost=0)
    run_test("SMA*",  sma_star,GRAPH_SELF, 'X', 'X', H_SELF, expected_cost=0)


def test_cadena_lineal():
    section("TEST 5 · Cadena lineal (1 → 5)")
    run_test("DFS",   dfs,     GRAPH_CHAIN, '1', '5')
    run_test("BFS",   bfs,     GRAPH_CHAIN, '1', '5', expected_cost=OPTIMAL_CHAIN)
    run_test("UCS",   ucs,     GRAPH_CHAIN, '1', '5', expected_cost=OPTIMAL_CHAIN)
    run_test("IDS",   ids,     GRAPH_CHAIN, '1', '5')
    run_test("A*",    astar,   GRAPH_CHAIN, '1', '5', H_CHAIN, expected_cost=OPTIMAL_CHAIN)
    run_test("IDA*",  idastar, GRAPH_CHAIN, '1', '5', H_CHAIN, expected_cost=OPTIMAL_CHAIN)
    run_test("SMA*",  sma_star,GRAPH_CHAIN, '1', '5', H_CHAIN, expected_cost=OPTIMAL_CHAIN)


def test_grafo_denso():
    section("TEST 6 · Grafo denso (A → F)")
    run_test("DFS",   dfs,     GRAPH_DENSE, 'A', 'F')
    run_test("BFS",   bfs,     GRAPH_DENSE, 'A', 'F')
    run_test("UCS",   ucs,     GRAPH_DENSE, 'A', 'F', expected_cost=OPTIMAL_DENSE)
    run_test("IDS",   ids,     GRAPH_DENSE, 'A', 'F')
    run_test("A*",    astar,   GRAPH_DENSE, 'A', 'F', H_DENSE, expected_cost=OPTIMAL_DENSE)
    run_test("IDA*",  idastar, GRAPH_DENSE, 'A', 'F', H_DENSE, expected_cost=OPTIMAL_DENSE)
    run_test("SMA*",  sma_star,GRAPH_DENSE, 'A', 'F', H_DENSE, expected_cost=OPTIMAL_DENSE)


# ══════════════════════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    total  = len(results_summary)
    passed = sum(results_summary)
    failed = total - passed
    pct    = 100 * passed / total if total else 0

    print(f"\n{'═'*65}")
    print(f"  RESUMEN FINAL")
    print(f"{'═'*65}")
    print(f"  Total tests : {total}")
    print(f"  Pasados     : \033[92m{passed}\033[0m")
    print(f"  Fallados    : \033[91m{failed}\033[0m")
    print(f"  Resultado   : {pct:.1f}%")
    print('═'*65)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    print("\n" + "█"*65)
    print("  SUITE DE TESTS — ALGORITMOS DE BÚSQUEDA EN IA")
    print("  DFS · BFS · UCS · IDS · A* · IDA* · SMA*")
    print("█"*65)

    test_grafo_simple()
    test_optimalidad()
    test_sin_solucion()
    test_start_igual_goal()
    test_cadena_lineal()
    test_grafo_denso()

    print_summary()
