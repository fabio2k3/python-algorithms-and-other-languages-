"""
Minimax - Búsqueda Adversarial
==============================
Algoritmo de búsqueda en profundidad para juegos de suma cero con dos jugadores:
  - MAX: busca maximizar su utilidad.
  - MIN: busca minimizar la utilidad de MAX.

El valor Minimax de un nodo se calcula recursivamente:
  - Nodo MAX: V(s) = max{ V(s') | s' hijo de s }
  - Nodo MIN: V(s) = min{ V(s') | s' hijo de s }
  - Nodo terminal: V(s) = utilidad(s)

Representación del árbol de juego:
  Cada nodo es un dict con:
    {
      'player':   'MAX' | 'MIN',
      'terminal': bool,
      'utility':  int | None,     # solo si terminal=True
      'children': [nodo, ...]     # lista de nodos hijos
    }

Propiedades:
  - Completo:   Sí (árbol finito)
  - Óptimo:     Sí (ambos jugadores juegan óptimamente)
  - Tiempo:     O(b^m)
  - Espacio:    O(b*m)
"""


def minimax_value(node: dict) -> int:
    """
    Calcula recursivamente el valor Minimax del nodo.

    Args:
        node: dict con 'player', 'terminal', 'utility', 'children'

    Returns:
        Valor entero (utilidad óptima para MAX).
    """
    if node['terminal']:
        return node['utility']

    if node['player'] == 'MAX':
        return max(minimax_value(child) for child in node['children'])
    else:  # MIN
        return min(minimax_value(child) for child in node['children'])


def minimax_decision(node: dict):
    """
    Devuelve la acción (hijo) óptima para el jugador MAX en la raíz.

    Args:
        node: nodo raíz (debe ser jugador MAX)

    Returns:
        (mejor_hijo, valor_minimax)
    """
    best_child = None
    best_value = float('-inf')

    for child in node['children']:
        val = minimax_value(child)
        if val > best_value:
            best_value = val
            best_child = child

    return best_child, best_value


# ── Helpers para construir árboles ────────────────────────────────────────────

def terminal(utility: int) -> dict:
    """Crea un nodo terminal con utilidad dada."""
    return {'player': None, 'terminal': True, 'utility': utility, 'children': []}


def max_node(*children) -> dict:
    """Crea un nodo MAX con los hijos dados."""
    return {'player': 'MAX', 'terminal': False, 'utility': None, 'children': list(children)}


def min_node(*children) -> dict:
    """Crea un nodo MIN con los hijos dados."""
    return {'player': 'MIN', 'terminal': False, 'utility': None, 'children': list(children)}


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    #
    #         MAX
    #        / | \
    #      MIN MIN MIN
    #     /\   |   /\
    #    3  5  2  9  1
    #
    tree = max_node(
        min_node(terminal(3), terminal(5)),
        min_node(terminal(2)),
        min_node(terminal(9), terminal(1)),
    )

    best, value = minimax_decision(tree)
    print("=== MINIMAX ===")
    print(f"Valor Minimax de la raíz : {value}")
    print(f"MIN elegirá (rama 1)→3, (rama 2)→2, (rama 3)→1  →  MAX elige rama 1 (valor 3)")
