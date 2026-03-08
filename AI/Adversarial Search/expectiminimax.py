"""
Expectiminimax - Juegos Estocásticos
======================================
Extensión de Minimax para juegos con elementos de azar (Backgammon, dados, cartas).
Introduce nodos de azar (CHANCE) además de MAX y MIN.

Fórmula del nodo CHANCE:
    V(s) = Σ P(r) · V(s')
  donde P(r) es la probabilidad del resultado r y s' el estado resultante.

Tipos de nodo:
  - MAX:    elige la acción que maximiza V(s).
  - MIN:    elige la acción que minimiza V(s).
  - CHANCE: devuelve el valor esperado de sus hijos (ponderado por probabilidad).
  - TERMINAL: devuelve su utilidad directamente.

Representación del árbol:
  {
    'type':        'MAX' | 'MIN' | 'CHANCE' | 'TERMINAL',
    'utility':     int | None,   # solo TERMINAL
    'children':    [nodo, ...],  # MAX / MIN / CHANCE
    'probs':       [float, ...]  # solo CHANCE: probabilidades de cada hijo
  }

Propiedades:
  - Completo:   Sí (árbol finito)
  - Óptimo:     Sí (bajo la distribución de probabilidad dada)
  - Tiempo:     O(b^m · n^m)  donde n = número de resultados aleatorios
  - Espacio:    O(b·m)
"""


def expectiminimax(node: dict) -> float:
    """
    Calcula recursivamente el valor Expectiminimax del nodo.

    Args:
        node: dict con 'type', 'utility', 'children', 'probs'

    Returns:
        Valor óptimo esperado (float).
    """
    t = node['type']

    if t == 'TERMINAL':
        return float(node['utility'])

    if t == 'MAX':
        return max(expectiminimax(child) for child in node['children'])

    if t == 'MIN':
        return min(expectiminimax(child) for child in node['children'])

    if t == 'CHANCE':
        probs    = node['probs']
        children = node['children']
        assert abs(sum(probs) - 1.0) < 1e-9, "Las probabilidades deben sumar 1."
        return sum(p * expectiminimax(c) for p, c in zip(probs, children))

    raise ValueError(f"Tipo de nodo desconocido: {t!r}")


def expectiminimax_decision(root: dict):
    """
    Devuelve la acción (hijo) óptima para el jugador MAX en la raíz.

    Args:
        root: nodo raíz de tipo MAX

    Returns:
        (mejor_hijo, valor_esperado)
    """
    assert root['type'] == 'MAX', "La raíz debe ser un nodo MAX."
    best_child = None
    best_value = float('-inf')

    for child in root['children']:
        val = expectiminimax(child)
        if val > best_value:
            best_value = val
            best_child = child

    return best_child, best_value


# ── Helpers para construir nodos ─────────────────────────────────────────────

def terminal(utility) -> dict:
    return {'type': 'TERMINAL', 'utility': utility, 'children': [], 'probs': []}

def max_node(*children) -> dict:
    return {'type': 'MAX',    'utility': None, 'children': list(children), 'probs': []}

def min_node(*children) -> dict:
    return {'type': 'MIN',    'utility': None, 'children': list(children), 'probs': []}

def chance_node(probs, *children) -> dict:
    return {'type': 'CHANCE', 'utility': None, 'children': list(children), 'probs': list(probs)}


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    #
    # MAX elige entre dos acciones; cada acción va a un nodo CHANCE
    # que simula el lanzamiento de un dado de 2 caras (50%-50%).
    #
    #                 MAX
    #              /       \
    #          CHANCE      CHANCE
    #          (0.5/0.5)   (0.5/0.5)
    #           /   \       /    \
    #          3     9     2      7
    #
    # Rama izquierda:  E = 0.5*3 + 0.5*9 = 6.0
    # Rama derecha:    E = 0.5*2 + 0.5*7 = 4.5
    # MAX elige rama izquierda → valor esperado = 6.0
    #
    tree = max_node(
        chance_node([0.5, 0.5], terminal(3), terminal(9)),
        chance_node([0.5, 0.5], terminal(2), terminal(7)),
    )

    best, value = expectiminimax_decision(tree)
    print("=== EXPECTIMINIMAX ===")
    print(f"Valor esperado óptimo : {value}")
    print(f"MAX debe elegir la rama con E = {value}")

    # Árbol más complejo con nodos MIN
    #
    #              MAX
    #           /       \
    #         MIN        CHANCE (0.3/0.7)
    #        /   \         /      \
    #       5     2       MAX      4
    #                    /   \
    #                   6     1
    #
    # MIN elige min(5,2) = 2
    # MAX (hijo de CHANCE) elige max(6,1) = 6
    # CHANCE: 0.3*6 + 0.7*4 = 1.8 + 2.8 = 4.6
    # Raíz MAX elige max(2, 4.6) = 4.6
    #
    complex_tree = max_node(
        min_node(terminal(5), terminal(2)),
        chance_node([0.3, 0.7],
                    max_node(terminal(6), terminal(1)),
                    terminal(4)),
    )
    _, complex_value = expectiminimax_decision(complex_tree)
    print(f"\nÁrbol complejo — valor esperado óptimo : {complex_value:.2f}  (esperado: 4.60)")
