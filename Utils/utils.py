def matriz_a_grafo(matriz, nodos=None, dirigido=False, es_binaria=False):
    """
    Convierte una matriz de adyacencia en lista de adyacencia.

    matriz[i][j]:
        - peso (si es ponderado)
        - 1/0 (si es binario)

    Parámetros:
    - nodos: nombres de nodos (['A','B','C']...)
    - dirigido: si el grafo es dirigido
    - es_binaria: si la matriz es de 0/1
    """

    n = len(matriz)

    # Si no pasas nombres → A, B, C...
    if nodos is None:
        nodos = [chr(ord('A') + i) for i in range(n)]

    grafo = {nodo: [] for nodo in nodos}

    for i in range(n):
        for j in range(n):
            valor = matriz[i][j]

            # Caso 1: matriz binaria
            if es_binaria:
                if valor == 1:
                    grafo[nodos[i]].append((nodos[j], 1))

            # Caso 2: matriz con pesos
            else:
                if valor != -1 and i != j:
                    grafo[nodos[i]].append((nodos[j], valor))

    return grafo


def grid_a_grafo(grid, diagonales=True):
    filas = len(grid)
    cols = len(grid[0])

    grafo = {}

    # movimientos
    movimientos = [
        (-1, 0), (1, 0), (0, -1), (0, 1)  # 4 direcciones
    ]

    if diagonales:
        movimientos += [
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    for i in range(filas):
        for j in range(cols):
            if grid[i][j] == -1:
                continue  # obstáculo

            nodo = (i, j)
            grafo[nodo] = []

            for di, dj in movimientos:
                ni, nj = i + di, j + dj

                # dentro del grid
                if 0 <= ni < filas and 0 <= nj < cols:
                    if grid[ni][nj] != -1:

                        # costo
                        if diagonales and abs(di) + abs(dj) == 2:
                            costo = 1.4  # aproximación de sqrt(2)
                        else:
                            costo = 1

                        grafo[nodo].append(((ni, nj), costo))

    return grafo