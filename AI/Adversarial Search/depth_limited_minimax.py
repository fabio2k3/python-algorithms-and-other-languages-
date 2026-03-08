"""
Minimax con Profundidad Limitada + Función de Evaluación + Quiescence Search
=============================================================================
En juegos reales (Ajedrez, Damas) explorar hasta estados terminales es inviable.
Se aplican dos técnicas:

1. BÚSQUEDA CON PROFUNDIDAD LIMITADA (Depth-Limited Minimax)
   - Se corta la búsqueda a profundidad d.
   - Los nodos no terminales en el corte se evalúan con Eval(s):
       Eval(s) = w1*f1(s) + w2*f2(s) + ... + wn*fn(s)
   - Sustituye a la función de utilidad terminal.

2. QUIESCENCE SEARCH (Búsqueda de Reposo)
   - Evita evaluar posiciones "inestables" (capturas pendientes, jaques).
   - En el corte de profundidad, si la posición es inestable se extiende
     la búsqueda hasta que se estabiliza.
   - Previene el "error del horizonte": no evaluar justo antes de una
     captura de pieza importante.

Representación del nodo de juego (ejemplo tipo ajedrez simplificado):
  {
    'player':   'MAX' | 'MIN',
    'terminal': bool,
    'utility':  int | None,
    'stable':   bool,           # True = posición estable (sin capturas pendientes)
    'features': dict,           # características para Eval(s)
    'children': [nodo, ...]
  }
"""


# ── Función de evaluación heurística ─────────────────────────────────────────

def eval_function(node: dict, weights: dict) -> float:
    """
    Función de evaluación lineal:
        Eval(s) = Σ w_i * f_i(s)

    Args:
        node:    nodo del juego con 'features' dict
        weights: dict de pesos {nombre_feature: peso}

    Returns:
        Valor heurístico (float). Positivo favorece MAX, negativo a MIN.
    """
    return sum(weights.get(k, 0) * v for k, v in node['features'].items())


# ── Minimax con profundidad limitada ─────────────────────────────────────────

def depth_limited_minimax(node: dict, depth: int, weights: dict) -> float:
    """
    Minimax con límite de profundidad.

    Args:
        node:    nodo del árbol de juego
        depth:   profundidad restante
        weights: pesos para la función de evaluación

    Returns:
        Valor del nodo.
    """
    # Caso base: terminal o profundidad agotada
    if node['terminal']:
        return float(node['utility'])

    if depth == 0:
        return eval_function(node, weights)

    if node['player'] == 'MAX':
        return max(depth_limited_minimax(c, depth - 1, weights) for c in node['children'])
    else:
        return min(depth_limited_minimax(c, depth - 1, weights) for c in node['children'])


# ── Quiescence Search ─────────────────────────────────────────────────────────

def quiescence_search(node: dict, weights: dict, max_extra_depth: int = 4) -> float:
    """
    Minimax con profundidad limitada + extensión en posiciones inestables.

    Si al alcanzar el corte de profundidad el nodo NO es 'stable',
    la búsqueda se extiende hasta que la posición sea estable
    (o hasta max_extra_depth niveles adicionales).

    Args:
        node:            nodo del árbol
        weights:         pesos para Eval(s)
        max_extra_depth: extensión máxima en posiciones inestables

    Returns:
        Valor evaluado.
    """
    def _search(node, depth):
        if node['terminal']:
            return float(node['utility'])

        # Corte normal: posición estable → usar Eval
        if depth == 0 and node.get('stable', True):
            return eval_function(node, weights)

        # Corte con posición inestable → extender un nivel más
        if depth == 0 and not node.get('stable', True):
            if max_extra_depth <= 0:
                return eval_function(node, weights)
            # Extender solo los hijos "inestables" relevantes
            if not node['children']:
                return eval_function(node, weights)
            if node['player'] == 'MAX':
                return max(_search(c, 0) for c in node['children'])
            else:
                return min(_search(c, 0) for c in node['children'])

        if node['player'] == 'MAX':
            return max(_search(c, depth - 1) for c in node['children'])
        else:
            return min(_search(c, depth - 1) for c in node['children'])

    return _search(node, max_extra_depth)


def depth_limited_decision(root: dict, depth: int, weights: dict,
                           use_quiescence: bool = False):
    """
    Devuelve la acción óptima para MAX con profundidad limitada.

    Args:
        root:            nodo raíz (MAX)
        depth:           profundidad de búsqueda
        weights:         pesos heurísticos
        use_quiescence:  si True, usa Quiescence Search en los cortes

    Returns:
        (mejor_hijo, valor)
    """
    best_child = None
    best_value = float('-inf')

    for child in root['children']:
        if use_quiescence:
            val = quiescence_search(child, weights, max_extra_depth=depth - 1)
        else:
            val = depth_limited_minimax(child, depth - 1, weights)

        if val > best_value:
            best_value = val
            best_child = child

    return best_child, best_value


# ── Helpers ───────────────────────────────────────────────────────────────────

def terminal(utility: int) -> dict:
    return {
        'player': None, 'terminal': True, 'utility': utility,
        'stable': True, 'features': {}, 'children': []
    }

def make_node(player: str, features: dict, stable: bool, *children) -> dict:
    return {
        'player': player, 'terminal': False, 'utility': None,
        'stable': stable, 'features': features, 'children': list(children)
    }


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pesos para la función de evaluación
    weights = {
        'material_advantage': 1.0,   # diferencia de piezas
        'mobility':           0.1,   # movimientos disponibles
        'center_control':     0.3,   # control del centro
    }

    # Árbol de juego con características
    tree = make_node('MAX', {}, True,
        make_node('MIN', {'material_advantage': 2, 'mobility': 3, 'center_control': 1}, True,
            terminal(5),
            make_node('MAX', {'material_advantage': -1, 'mobility': 5, 'center_control': 2}, True,
                terminal(2), terminal(4)
            ),
        ),
        make_node('MIN', {'material_advantage': 1, 'mobility': 2, 'center_control': 3}, False,  # inestable
            terminal(7),
            terminal(3),
        ),
    )

    # Sin quiescence
    _, val_std = depth_limited_decision(tree, depth=3, weights=weights, use_quiescence=False)
    print("=== DEPTH-LIMITED MINIMAX ===")
    print(f"Valor (sin Quiescence) : {val_std}")

    # Con quiescence (extiende posiciones inestables)
    _, val_q = depth_limited_decision(tree, depth=3, weights=weights, use_quiescence=True)
    print("\n=== QUIESCENCE SEARCH ===")
    print(f"Valor (con Quiescence) : {val_q}")
    print("La Quiescence Search evita evaluar posiciones inestables directamente.")
