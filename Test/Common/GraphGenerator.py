import random

import networkx as nx
import networkx.convert_matrix
import numpy
from enum import Enum

from KronFit.KroneckerFit import GenerateSKG, InstantiateGraph


def RandomConnectedDirectedGraph(num_nodes):
    graph = nx.DiGraph()

    graph.add_nodes_from(list(range(num_nodes)))

    while not nx.is_strongly_connected(graph):
        u = random.choice(list(graph.nodes()))
        v = u
        while u == v:
            v = random.choice(list(graph.nodes()))
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)

    return graph


def GNCConnectedDirectedGraph(num_nodes, batch=5):
    """
    Generates a GNC random directed graph using networkx, and checks its connectivity
    This function continuously adds edges in batches until the graph has reached strong connectivity
    :param num_nodes:           Number of nodes
    :param batch:               Number of edges to add if the graph is not strongly connected
    :return:
    """
    graph = nx.generators.gnc_graph(num_nodes, create_using=nx.DiGraph)

    while not nx.is_strongly_connected(graph):
        u = random.choice(list(graph.nodes()))
        i = 0
        while i <= batch:
            v = random.choice(list(graph.nodes()))
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
            i += 1
    return graph


def CorePeripheryDirectedGraph(power, a=0.99, b=0.7, c=0.5, d=0.01):
    """
    Generates a directed graph with a Core-Periphery structure.
    Core-Periphery graphs tend to be fitted by KronFit with an initiator matrix that has a large a, very small d and
    average c and d (Leskovec J. et Al, "Kronecker Graphs: An Approach to Modeling Networks", chapter 7. Discussion,
    pag 1033.
    So we exploit this to create a directed graph with core-periphery structure
    :param power:           Kronecker power. This function only creates graphs with a number of node that is a power of 2
    :param a:               Param of initiator matrix
    :param b:               Param of initiator matrix
    :param c:               Param of initiator matrix
    :param d:               Param of initiator matrix
    :return:                Core-Periphery graph
    """
    P = GenerateSKG(numpy.ndarray([[a, b], [c, d]]), power)
    A = InstantiateGraph(P)
    return networkx.convert_matrix.from_numpy_array(A, create_using=networkx.DiGraph)


class EGraphGenerationFunction(Enum):
    ERandomConnectedDirectedGraph = {"name": "Random Connected graph", "function": RandomConnectedDirectedGraph, "short_name": "rnd"}
    EGNCConnectedDirectedGraph = {"name": "GNC Connected graph", "function": GNCConnectedDirectedGraph, "short_name": "gnc"}
    ECorePeripheryDirectedGraph = {"name": "Core-Periphery graph", "function": CorePeripheryDirectedGraph, "short_name": "c-p"}
