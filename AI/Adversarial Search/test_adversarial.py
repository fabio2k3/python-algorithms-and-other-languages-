"""
test_adversarial.py
===================
Suite de tests completa para todos los algoritmos de Búsqueda Adversarial:
  1. Minimax
  2. Alpha-Beta Pruning
  3. Expectiminimax
  4. Depth-Limited Minimax + Función de Evaluación
  5. Quiescence Search

Los tests verifican:
  - Correctitud del valor calculado en la raíz.
  - Consistencia entre Minimax y Alpha-Beta (deben coincidir siempre).
  - Correctitud del cálculo esperado en Expectiminimax.
  - Comportamiento en árboles degenerados (un solo hijo, un solo nivel).
  - Casos borde: nodo terminal directo, profundidad 0.

Ejecutar:
    python test_adversarial.py
"""

import sys
import traceback

from minimax              import (minimax_value, minimax_decision,
                                   terminal as mm_terminal,
                                   max_node  as mm_max,
                                   min_node  as mm_min)

from alpha_beta           import (alpha_beta_value, alpha_beta_decision,
                                   terminal as ab_terminal,
                                   max_node  as ab_max,
                                   min_node  as ab_min)

from expectiminimax       import (expectiminimax, expectiminimax_decision,
                                   terminal    as em_terminal,
                                   max_node    as em_max,
                                   min_node    as em_min,
                                   chance_node as em_chance)

from depth_limited_minimax import (depth_limited_minimax, depth_limited_decision,
                                    quiescence_search,
                                    terminal  as dl_terminal,
                                    make_node as dl_node)


# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades
# ══════════════════════════════════════════════════════════════════════════════

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"
results_summary = []


def check(label: str, got, expected, tol: float = 1e-9):
    """Compara got con expected (con tolerancia para floats)."""
    try:
        ok = abs(float(got) - float(expected)) <= tol
    except Exception:
        ok = (got == expected)

    status = PASS if ok else FAIL
    results_summary.append(ok)
    detail = f"got={got}  expected={expected}"
    print(f"  [{status}] {label:<55} | {detail}")


def section(title: str):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print('═'*70)


def safe_run(fn, *args, **kwargs):
    """Ejecuta fn y devuelve (resultado, None) o (None, error_msg)."""
    try:
        return fn(*args, **kwargs), None
    except Exception:
        return None, traceback.format_exc().strip().split('\n')[-1]


# ══════════════════════════════════════════════════════════════════════════════
#  TESTS MINIMAX
# ══════════════════════════════════════════════════════════════════════════════

def test_minimax():
    section("1 · MINIMAX")

    # T1: árbol clásico 2 niveles
    #        MAX
    #       / | \
    #     MIN MIN MIN
    #     /\   |  /\
    #    3  5  2  9  1
    # MIN→3, MIN→2, MIN→1  →  MAX→3
    t1 = mm_max(
        mm_min(mm_terminal(3), mm_terminal(5)),
        mm_min(mm_terminal(2)),
        mm_min(mm_terminal(9), mm_terminal(1)),
    )
    check("Minimax T1: valor raíz", minimax_value(t1), 3)

    # T2: profundidad 3
    #         MAX
    #        /   \
    #       MIN   MIN
    #      / \   / \
    #    MAX MAX MAX MAX
    #    /\  /\  /\  /\
    #   5 4 3 6 2 1 8  7
    # MAX hoja→ 5,4→5; 3,6→6; 2,1→2; 8,7→8
    # MIN→ min(5,6)=5;  min(2,8)=2
    # MAX→ max(5,2)=5
    t2 = mm_max(
        mm_min(mm_max(mm_terminal(5), mm_terminal(4)),
               mm_max(mm_terminal(3), mm_terminal(6))),
        mm_min(mm_max(mm_terminal(2), mm_terminal(1)),
               mm_max(mm_terminal(8), mm_terminal(7))),
    )
    check("Minimax T2: árbol profundidad 3", minimax_value(t2), 5)

    # T3: un solo hijo en cada nivel  MAX→MIN→terminal(7)
    t3 = mm_max(mm_min(mm_terminal(7)))
    check("Minimax T3: un solo hijo", minimax_value(t3), 7)

    # T4: raíz es terminal directamente
    t4 = mm_terminal(42)
    check("Minimax T4: nodo terminal en raíz", minimax_value(t4), 42)

    # T5: todos los terminales iguales
    t5 = mm_max(mm_min(mm_terminal(5), mm_terminal(5)), mm_min(mm_terminal(5)))
    check("Minimax T5: todos iguales → 5", minimax_value(t5), 5)

    # T6: MAX con valores negativos
    # MIN1 → min(-3, -1) = -3
    # MIN2 → min(-2, -5) = -5
    # MAX  → max(-3, -5) = -3
    t6 = mm_max(mm_min(mm_terminal(-3), mm_terminal(-1)),
                mm_min(mm_terminal(-2), mm_terminal(-5)))
    check("Minimax T6: valores negativos → -3", minimax_value(t6), -3)

    # T7: minimax_decision devuelve el hijo correcto
    t7 = mm_max(
        mm_min(mm_terminal(3)),
        mm_min(mm_terminal(10)),
        mm_min(mm_terminal(1)),
    )
    _, val7 = minimax_decision(t7)
    check("Minimax T7: decision → valor 10 (MAX elige rama con valor 10)", val7, 10)


# ══════════════════════════════════════════════════════════════════════════════
#  TESTS ALPHA-BETA
# ══════════════════════════════════════════════════════════════════════════════

def test_alpha_beta():
    section("2 · ALPHA-BETA PRUNING")

    INF = float('inf')

    # Los mismos árboles que Minimax → resultado debe ser idéntico

    t1 = ab_max(
        ab_min(ab_terminal(3), ab_terminal(5)),
        ab_min(ab_terminal(2)),
        ab_min(ab_terminal(9), ab_terminal(1)),
    )
    check("Alpha-Beta T1: igual que Minimax → 3",
          alpha_beta_value(t1, -INF, INF), 3)

    t2 = ab_max(
        ab_min(ab_max(ab_terminal(5), ab_terminal(4)),
               ab_max(ab_terminal(3), ab_terminal(6))),
        ab_min(ab_max(ab_terminal(2), ab_terminal(1)),
               ab_max(ab_terminal(8), ab_terminal(7))),
    )
    check("Alpha-Beta T2: árbol profundidad 3 → 5",
          alpha_beta_value(t2, -INF, INF), 5)

    t3 = ab_max(ab_min(ab_terminal(7)))
    check("Alpha-Beta T3: un solo hijo → 7",
          alpha_beta_value(t3, -INF, INF), 7)

    t4 = ab_terminal(42)
    check("Alpha-Beta T4: terminal en raíz → 42",
          alpha_beta_value(t4, -INF, INF), 42)

    t5 = ab_max(ab_min(ab_terminal(-3), ab_terminal(-1)),
                ab_min(ab_terminal(-2), ab_terminal(-5)))
    check("Alpha-Beta T5: valores negativos → -3",
          alpha_beta_value(t5, -INF, INF), -3)

    # T6: Consistencia Minimax vs Alpha-Beta en árbol amplio
    import minimax as MM
    import alpha_beta as AB

    deep = MM.max_node(
        MM.min_node(MM.max_node(MM.terminal(5), MM.terminal(4)),
                    MM.max_node(MM.terminal(3), MM.terminal(6))),
        MM.min_node(MM.max_node(MM.terminal(2), MM.terminal(1)),
                    MM.max_node(MM.terminal(8), MM.terminal(7))),
    )
    deep_ab = AB.max_node(
        AB.min_node(AB.max_node(AB.terminal(5), AB.terminal(4)),
                    AB.max_node(AB.terminal(3), AB.terminal(6))),
        AB.min_node(AB.max_node(AB.terminal(2), AB.terminal(1)),
                    AB.max_node(AB.terminal(8), AB.terminal(7))),
    )
    mm_val = MM.minimax_value(deep)
    ab_val = AB.alpha_beta_value(deep_ab, -INF, INF)
    check("Alpha-Beta T6: consistencia con Minimax puro", mm_val, ab_val)

    # T7: alpha_beta_decision devuelve valor correcto
    t7 = ab_max(
        ab_min(ab_terminal(3)),
        ab_min(ab_terminal(10)),
        ab_min(ab_terminal(1)),
    )
    _, val7 = alpha_beta_decision(t7)
    check("Alpha-Beta T7: decision → valor 10", val7, 10)


# ══════════════════════════════════════════════════════════════════════════════
#  TESTS EXPECTIMINIMAX
# ══════════════════════════════════════════════════════════════════════════════

def test_expectiminimax():
    section("3 · EXPECTIMINIMAX")

    # T1: MAX → 2 ramas CHANCE (50/50)
    # E(izq) = 0.5*3 + 0.5*9 = 6.0
    # E(der) = 0.5*2 + 0.5*7 = 4.5
    # MAX → 6.0
    t1 = em_max(
        em_chance([0.5, 0.5], em_terminal(3), em_terminal(9)),
        em_chance([0.5, 0.5], em_terminal(2), em_terminal(7)),
    )
    check("Expectiminimax T1: MAX sobre CHANCE (50/50) → 6.0",
          expectiminimax(t1), 6.0)

    # T2: probabilidades asimétricas
    # E = 0.3*3 + 0.7*9 = 0.9 + 6.3 = 7.2
    t2 = em_max(
        em_chance([0.3, 0.7], em_terminal(3), em_terminal(9)),
    )
    check("Expectiminimax T2: CHANCE (0.3/0.7) → 7.2",
          expectiminimax(t2), 7.2)

    # T3: árbol con MIN y CHANCE
    # MIN → min(5,2) = 2
    # MAX(en CHANCE) → max(6,1) = 6
    # CHANCE: 0.3*6 + 0.7*4 = 1.8 + 2.8 = 4.6
    # Raíz MAX → max(2, 4.6) = 4.6
    t3 = em_max(
        em_min(em_terminal(5), em_terminal(2)),
        em_chance([0.3, 0.7],
                  em_max(em_terminal(6), em_terminal(1)),
                  em_terminal(4)),
    )
    check("Expectiminimax T3: árbol complejo MAX/MIN/CHANCE → 4.6",
          expectiminimax(t3), 4.6)

    # T4: CHANCE de un solo resultado (probabilidad 1.0)
    t4 = em_max(em_chance([1.0], em_terminal(7)))
    check("Expectiminimax T4: CHANCE con prob=1.0 → 7.0",
          expectiminimax(t4), 7.0)

    # T5: terminal directo
    t5 = em_terminal(99)
    check("Expectiminimax T5: terminal → 99",
          expectiminimax(t5), 99)

    # T6: CHANCE con 3 resultados equiprobables
    # E = (1/3)*2 + (1/3)*5 + (1/3)*8 = 5.0
    t6 = em_max(
        em_chance([1/3, 1/3, 1/3], em_terminal(2), em_terminal(5), em_terminal(8))
    )
    check("Expectiminimax T6: CHANCE 3 equiprobables → 5.0",
          expectiminimax(t6), 5.0)

    # T7: decision correcta
    t7 = em_max(
        em_chance([0.5, 0.5], em_terminal(1), em_terminal(3)),   # E=2
        em_chance([0.5, 0.5], em_terminal(5), em_terminal(9)),   # E=7
    )
    _, val7 = expectiminimax_decision(t7)
    check("Expectiminimax T7: decision → elige rama con E=7.0", val7, 7.0)


# ══════════════════════════════════════════════════════════════════════════════
#  TESTS DEPTH-LIMITED MINIMAX + QUIESCENCE
# ══════════════════════════════════════════════════════════════════════════════

def test_depth_limited():
    section("4 · DEPTH-LIMITED MINIMAX + FUNCIÓN DE EVALUACIÓN")

    WEIGHTS = {'material': 1.0, 'mobility': 0.1}

    def feat(m, mob):
        return {'material': m, 'mobility': mob}

    # T1: profundidad 0 → usa Eval directamente en hijos
    root = dl_node('MAX', feat(0, 0), True,
        dl_node('MIN', feat(3, 10), True),
        dl_node('MIN', feat(1, 5), True),
    )
    # depth=1: evalúa hijos (son MIN a depth=0) → depth_limited_minimax(child, 0, W)
    # = Eval(MIN con feat(3,10)) = 3*1 + 10*0.1 = 4.0
    # = Eval(MIN con feat(1,5))  = 1*1 + 5*0.1  = 1.5
    # MAX → 4.0
    _, val = depth_limited_decision(root, depth=1, weights=WEIGHTS)
    check("Depth-Limited T1: depth=1, MAX elige hijo con Eval=4.0", val, 4.0)

    # T2: nodo terminal se respeta aunque depth > 0
    root2 = dl_node('MAX', feat(0, 0), True,
        dl_terminal(10),
        dl_terminal(3),
    )
    _, val2 = depth_limited_decision(root2, depth=5, weights=WEIGHTS)
    check("Depth-Limited T2: terminales se respetan independiente del depth", val2, 10)

    # T3: árbol 3 niveles, comparar con Minimax puro en terminales
    root3 = dl_node('MAX', feat(0, 0), True,
        dl_node('MIN', feat(0, 0), True,
            dl_terminal(5), dl_terminal(2)),
        dl_node('MIN', feat(0, 0), True,
            dl_terminal(9), dl_terminal(4)),
    )
    # MIN → min(5,2)=2; min(9,4)=4  →  MAX → max(2,4) = 4
    _, val3 = depth_limited_decision(root3, depth=5, weights=WEIGHTS)
    check("Depth-Limited T3: árbol 3 niveles → 4", val3, 4)


def test_quiescence():
    section("5 · QUIESCENCE SEARCH")

    WEIGHTS = {'material': 1.0, 'mobility': 0.1}

    def feat(m, mob):
        return {'material': m, 'mobility': mob}

    # T1: posición estable → igual que depth-limited normal
    root_stable = dl_node('MAX', feat(0, 0), True,
        dl_node('MIN', feat(2, 5), True),  # stable=True
        dl_node('MIN', feat(1, 3), True),  # stable=True
    )
    _, val_std  = depth_limited_decision(root_stable, depth=1, weights=WEIGHTS,
                                          use_quiescence=False)
    _, val_q    = depth_limited_decision(root_stable, depth=1, weights=WEIGHTS,
                                          use_quiescence=True)
    check("Quiescence T1: posición estable, resultado igual con y sin QS",
          val_std, val_q)

    # T2: posición inestable tiene hijos → QS los evalúa
    unstable_child = dl_node('MIN', feat(0, 0), False,  # stable=False
        dl_terminal(8),
        dl_terminal(2),
    )
    stable_child = dl_node('MIN', feat(3, 0), True)

    root2 = dl_node('MAX', feat(0, 0), True,
        unstable_child,
        stable_child,
    )
    # Sin QS: el nodo inestable se evalúa directamente con Eval → feat(0,0)=0
    #         el nodo estable → Eval = 3.0   MAX → 3.0
    _, val_no_q = depth_limited_decision(root2, depth=1, weights=WEIGHTS,
                                          use_quiescence=False)

    # Con QS: nodo inestable se extiende → MIN(8,2)=2; estable→Eval=3  MAX→3
    _, val_qs   = depth_limited_decision(root2, depth=1, weights=WEIGHTS,
                                          use_quiescence=True)

    # Ambos deben dar 3.0 pero por razones distintas
    check("Quiescence T2: con QS MAX elige correctamente entre estable e inestable",
          val_qs, 3.0)

    # T3: quiescence sobre nodo terminal no cambia nada
    t3 = dl_terminal(77)
    val_t3 = quiescence_search(t3, WEIGHTS)
    check("Quiescence T3: terminal → 77.0", val_t3, 77.0)


# ══════════════════════════════════════════════════════════════════════════════
#  TESTS DE CONSISTENCIA CRUZADA
# ══════════════════════════════════════════════════════════════════════════════

def test_cross_consistency():
    section("6 · CONSISTENCIA CRUZADA (Minimax == Alpha-Beta)")

    import minimax  as MM
    import alpha_beta as AB

    INF = float('inf')

    def build_mm(depth, values):
        """Construye árbol MM alternando MAX/MIN hasta agotar values."""
        if depth == 0 or len(values) == 1:
            return MM.terminal(values[0])
        mid = len(values) // 2
        left  = build_mm(depth - 1, values[:mid])
        right = build_mm(depth - 1, values[mid:])
        if depth % 2 == 0:
            return MM.max_node(left, right)
        else:
            return MM.min_node(left, right)

    def build_ab(depth, values):
        if depth == 0 or len(values) == 1:
            return AB.terminal(values[0])
        mid = len(values) // 2
        left  = build_ab(depth - 1, values[:mid])
        right = build_ab(depth - 1, values[mid:])
        if depth % 2 == 0:
            return AB.max_node(left, right)
        else:
            return AB.min_node(left, right)

    test_cases = [
        ([3, 5, 2, 9, 1, 8, 4, 7], 3, "8 hojas, depth 3"),
        ([10, 1, 5, 2],             2, "4 hojas, depth 2"),
        ([-5, -3, -8, -1],          2, "valores negativos"),
        ([0, 0, 0, 0],              2, "todos cero"),
    ]

    for values, depth, label in test_cases:
        tree_mm = build_mm(depth, values)
        tree_ab = build_ab(depth, values)
        mm_val = MM.minimax_value(tree_mm)
        ab_val = AB.alpha_beta_value(tree_ab, -INF, INF)
        check(f"Consistencia: {label}", mm_val, ab_val)


# ══════════════════════════════════════════════════════════════════════════════
#  RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    total  = len(results_summary)
    passed = sum(results_summary)
    failed = total - passed
    pct    = 100 * passed / total if total else 0

    print(f"\n{'═'*70}")
    print(f"  RESUMEN FINAL")
    print(f"{'═'*70}")
    print(f"  Total tests : {total}")
    print(f"  Pasados     : \033[92m{passed}\033[0m")
    print(f"  Fallados    : \033[91m{failed}\033[0m")
    print(f"  Resultado   : {pct:.1f}%")
    print('═'*70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  SUITE DE TESTS — ALGORITMOS DE BÚSQUEDA ADVERSARIAL")
    print("  Minimax · Alpha-Beta · Expectiminimax · Depth-Limited · Quiescence")
    print("█"*70)

    test_minimax()
    test_alpha_beta()
    test_expectiminimax()
    test_depth_limited()
    test_quiescence()
    test_cross_consistency()

    print_summary()
