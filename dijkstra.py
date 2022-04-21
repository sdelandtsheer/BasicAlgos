# Dijkstra's algorithm
# Find the shortest (least-cost) path between source and target nodes of a weighted graph

import numpy as np


def dijkstra(graph: np.array = None, source_idx: int = None, target_idx: int = None):
    graph = np.where(graph == 0, np.inf, graph)
    (rows, cols) = graph.shape
    dist = np.array([np.inf for _ in range(cols)])  # total shortest cost from source
    prev = np.array([np.inf for _ in range(cols)])  # previous node on shortest path
    queue = list(range(cols))  # all nodes in the queue
    dist[source_idx] = 0

    while queue:  # while not empty
        u = queue[0]
        for q in queue:
            if dist[q] < dist[u]:
                u = q
        queue.remove(u)

        u_neighbors = np.argwhere(graph[u, :] < np.inf)
        for v in u_neighbors:
            alt = dist[u] + graph[u, v]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    # return the path
    shortest_path = []
    temp = target_idx
    if prev[temp] < np.inf:
        shortest_path.append(temp)
        while temp != source_idx:
            print(shortest_path)
            print(prev)
            print(temp)
            n = int(prev[temp])
            shortest_path.append(n)
            temp = n
            print(n)

    return shortest_path[::-1], dist[target_idx]


# graph as a matrix:
G = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 2, 0],  # node 0
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # node 1
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # node 2
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # node 3
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # node 4
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # node 5
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # node 6
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # node 7
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # node 8
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # node 9
              ])
source = 0
target = 9

path, cost = dijkstra(G, source, target)

print(f"path found: {path} with cost {cost}")



