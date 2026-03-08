"""
Alpha-Beta Pruning - Poda Alpha-Beta
=====================================
Optimización del algoritmo Minimax que elimina ramas que no pueden
influir en la decisión final, manteniendo el mismo resultado.

Valores clave:
  α (alpha): mejor valor que MAX puede garantizarse en el camino actual.
             Se inicializa en -∞ y solo sube.
  β (beta):  mejor valor que MIN puede garantizarse en el camino actual.
             Se inicializa en +∞ y solo baja.

Regla de poda:
  - En nodo MIN: si el valor hallado ≤ α → podar (MAX nunca escogerá esta rama).
  - En nodo MAX: si el valor hallado ≥ β → podar (MIN nunca escogerá esta rama).

Propiedades:
  - Produce el MISMO resultado que Minimax puro.
  - Valores de nodos intermedios podados pueden ser incorrectos,
    pero la decisión en la raíz es siempre correcta.
  - Mejor caso (ordenamiento perfecto): O(b^(m/2))
  - Peor caso (sin ordenamiento):       O(b^m)
  - Espacio: O(b*m)
"""


def alpha_beta_value(node: dict, alpha: float, beta: float) -> int:
    """
    Calcula el valor Minimax con poda Alpha-Beta.

    Args:
        node:  dict con 'player', 'terminal', 'utility', 'children'
        alpha: mejor garantía actual de MAX (-inf al inicio)
        beta:  mejor garantía actual de MIN (+inf al inicio)

    Returns:
        Valor entero (puede ser inexacto en nodos podados, correcto en raíz).
    """
    if node['terminal']:
        return node['utility']

    if node['player'] == 'MAX':
        value = float('-inf')
        for child in node['children']:
            value = max(value, alpha_beta_value(child, alpha, beta))
            alpha = max(alpha, value)
            if value >= beta:          # poda β: MIN nunca elegirá este camino
                break
        return value

    else:  # MIN
        value = float('+inf')
        for child in node['children']:
            value = min(value, alpha_beta_value(child, alpha, beta))
            beta = min(beta, value)
            if value <= alpha:         # poda α: MAX nunca elegirá este camino
                break
        return value


def alpha_beta_decision(node: dict):
    """
    Devuelve la acción (hijo) óptima para el jugador MAX usando poda Alpha-Beta.

    Args:
        node: nodo raíz (debe ser jugador MAX)

    Returns:
        (mejor_hijo, valor)
    """
    best_child = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta  = float('+inf')

    for child in node['children']:
        val = alpha_beta_value(child, alpha, beta)
        if val > best_value:
            best_value = val
            best_child = child
        alpha = max(alpha, best_value)

    return best_child, best_value


# ── Helpers (mismos que minimax.py, reutilizables) ────────────────────────────

def terminal(utility: int) -> dict:
    return {'player': None, 'terminal': True, 'utility': utility, 'children': []}

def max_node(*children) -> dict:
    return {'player': 'MAX', 'terminal': False, 'utility': None, 'children': list(children)}

def min_node(*children) -> dict:
    return {'player': 'MIN', 'terminal': False, 'utility': None, 'children': list(children)}


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    #
    # Árbol clásico de demostración de poda:
    #
    #              MAX
    #           /   |   \
    #         MIN  MIN  MIN
    #        / \    |   / \
    #       3   5   2  9   1
    #
    # Minimax puro evalúa todos los nodos (7).
    # Alpha-Beta poda algunos y llega al mismo resultado.
    #
    tree = max_node(
        min_node(terminal(3), terminal(5)),
        min_node(terminal(2)),
        min_node(terminal(9), terminal(1)),
    )

    best, value = alpha_beta_decision(tree)
    print("=== ALPHA-BETA PRUNING ===")
    print(f"Valor óptimo (raíz MAX) : {value}")
    print("Resultado idéntico a Minimax pero con menos evaluaciones de nodos.")

    # Árbol más profundo para ver mejor el efecto de la poda
    #
    #              MAX
    #           /       \
    #         MIN        MIN
    #        /   \      /   \
    #      MAX  MAX   MAX   MAX
    #      /\   /\    /\    /\
    #     5 4  3 6   2 1   8  7
    #
    deep_tree = max_node(
        min_node(
            max_node(terminal(5), terminal(4)),
            max_node(terminal(3), terminal(6)),
        ),
        min_node(
            max_node(terminal(2), terminal(1)),
            max_node(terminal(8), terminal(7)),
        ),
    )
    _, deep_value = alpha_beta_decision(deep_tree)
    print(f"\nÁrbol profundo — valor Minimax con poda α-β : {deep_value}")
