from AI.algorithms_search import bfs, dfs, ucs, a_star, greedy
from Utils.utils import grid_a_grafo, matriz_a_grafo

matrizCost = [
    [0,2,5,-1],
    [3,0,-1,4],
    [1,3,0,-1],
    [3,5,2,0]
]

matrizDist = [
    [0,1,1,1],
    [1,0,1,1],
    [1,0,0,1],
    [1,0,0,0]
]

# EJERCICIO 1
# Dado una matriz , buscar el camino de menor longitud de Nodo a otro

grafo_dist = matriz_a_grafo(matrizDist, es_binaria=True)
camino = bfs(grafo_dist, 'A', 'D')
print(camino)



# EJERCICIO 2
# Dado una matriz , buscar el camino de costo mínimo de Nodo a otro

grafo_cost = matriz_a_grafo(matrizCost)
camino_cost = ucs(grafo_cost, 'A', 'D')
print(camino_cost)