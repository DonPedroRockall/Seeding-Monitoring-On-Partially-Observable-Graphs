import random

import networkx


def DegreeDistribution(graph: networkx.Graph):
    random_edge = random.choice(list(graph.edges()))
    return random_edge[round(random.random(), 0)]


def UniformDistribution(graph: networkx.Graph):
    return random.choice(list(graph.nodes()))
