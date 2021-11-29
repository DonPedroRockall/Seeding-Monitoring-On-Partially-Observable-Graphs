import random
import networkx
from enum import Enum


def DegreeDistribution(graph: networkx.Graph, nodes_to_hide: int):
    """Samples nodes to select for hiding based on the degree of the node.
    Nodes of the graph that have an higher degree centrality are more likely to be selected and hidden"""
    to_hide = list()
    while len(to_hide) < nodes_to_hide:
        random_edge = random.choice(list(graph.edges()))
        node = random_edge[int(random.random())]
        if node not in to_hide:
            to_hide.append(node)
    return to_hide


def UniformDistribution(graph: networkx.Graph, nodes_to_hide: int):
    """Samples nodes to select for hiding from a uniform distribution.
    All nodes of the graph have the same probability of being selected and hidden"""
    to_hide = list()
    while len(to_hide) < nodes_to_hide:
        node = random.choice(list(graph.nodes()))
        if node not in to_hide:
            to_hide.append(node)
    return to_hide


class ENodeHidingSelectionFunction(Enum):
    EDegreeDistribution = {"name": "Degree Distribution", "function": DegreeDistribution, "short_name": "DEG"}
    EUniformDistribution = {"name": "Uniform Distribution", "function": UniformDistribution, "short_name": "UNIF"}
