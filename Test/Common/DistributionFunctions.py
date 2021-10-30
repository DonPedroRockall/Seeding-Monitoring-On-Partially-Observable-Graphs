import random

import networkx


def DegreeDistribution(graph: networkx.Graph, nodes_to_hide: int):
    to_hide = list()
    while len(to_hide) < nodes_to_hide:
        random_edge = random.choice(list(graph.edges()))
        node = random_edge[int(random.random())]
        if node not in to_hide:
            to_hide.append(node)
    return to_hide


def UniformDistribution(graph: networkx.Graph, nodes_to_hide: int):
    to_hide = list()
    while len(to_hide) < nodes_to_hide:
        node = random.choice(list(graph.nodes()))
        if node not in to_hide:
            to_hide.append(node)
    return to_hide
