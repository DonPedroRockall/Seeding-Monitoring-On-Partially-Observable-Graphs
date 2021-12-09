import random

import networkx as nx
import networkx.convert_matrix
import numpy
from enum import Enum

from KronFit.KroneckerFit import GenerateSKG, InstantiateGraph


def RandomConnectedDirectedGraph(num_nodes, **kwargs):
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


def GNCConnectedDirectedGraph(num_nodes, **kwargs):
    """
    Generates a GNC random directed graph using networkx, and checks its connectivity
    This function continuously adds edges in batches until the graph has reached strong connectivity
    :param num_nodes:           Number of nodes
    :param kwargs:              Keyword argument to pass to the generation function
     - batch:                   Number of edges to add if the graph is not strongly connected. Defaults to 5
    :return:
    """
    graph = nx.generators.gnc_graph(num_nodes, create_using=nx.DiGraph)

    if kwargs is None or "batch" not in kwargs:
        kwargs = {"batch": 5}

    while not nx.is_strongly_connected(graph):
        u = random.choice(list(graph.nodes()))
        i = 0
        while i <= kwargs["batch"]:
            v = random.choice(list(graph.nodes()))
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
            i += 1
    return graph


def CorePeripheryDirectedGraph(min_num_nodes, **kwargs):
    """
    Generates a directed graph with a Core-Periphery structure.
    Core-Periphery graphs tend to be fitted by KronFit with an initiator matrix that has a large a, very small d and
    average c and d (Leskovec J. et Al, "Kronecker Graphs: An Approach to Modeling Networks", chapter 7. Discussion,
    pag 1033.
    So we exploit this to create a directed graph with core-periphery structure
    :param min_num_nodes:       Kronecker power. This function only creates graphs with a number of node that is a
                                power of 2. The power is automatically calculated
    :param a:                   Param of initiator matrix
    :param b:                   Param of initiator matrix
    :param c:                   Param of initiator matrix
    :param d:                   Param of initiator matrix
    :return:                    Core-Periphery graph
    """
    power = 1
    while 2 ** power < min_num_nodes:
        power += 1

    if kwargs == None:
        kwargs = {}
    a = 0.99 if "a" not in kwargs else kwargs["a"]
    b = 0.75 if "b" not in kwargs else kwargs["b"]
    c = 0.65 if "c" not in kwargs else kwargs["c"]
    d = 0.01 if "d" not in kwargs else kwargs["d"]

    P = GenerateSKG(numpy.asarray([[a, b], [c, d]], dtype=float), power)
    A = InstantiateGraph(P)
    return networkx.convert_matrix.from_numpy_array(A, create_using=networkx.DiGraph)


def RandomSparseDirectedGraph(num_nodes, **kwargs):
    """
    Generates a graph that is sparse, that is, the nodes have a maximum degree
    :param num_nodes:           Number of nodes of the graph
    :param kwargs:              Keyword argument for this function
                                - minimum_in_degree: int -> maximum number of edges having a node N as target [default 2]
                                - maximum_in_degree: int -> maximum number of edges having a node N as target [default 10]
                                - minimum_out_degree: int -> maximum number of edges having a node N as source [default 2]
                                - maximum_out_degree: int -> maximum number of edges having a node N as source [default 10]
    :return:
    """

    max_in_deg = 10 if kwargs is None or "maximum_in_degree" not in kwargs else kwargs["maximum_in_degree"]
    max_out_deg = 10 if kwargs is None or "maximum_out_degree" not in kwargs else kwargs["maximum_out_degree"]
    min_in_deg = 2 if kwargs is None or "minimum_in_degree" not in kwargs else kwargs["minimum_in_degree"]
    min_out_deg = 2 if kwargs is None or "minimum_out_degree" not in kwargs else kwargs["minimum_out_degree"]
    graph = networkx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    # Iterate over each node
    for node in graph.nodes():

        # Add a random number of out edges between min_out_deg to max_out_deg
        k_in = random.randint(min_out_deg, max_out_deg)
        while graph.out_degree(node) < k_in:
            new_node = random.choice(list(graph.nodes()))
            if new_node != node and graph.in_degree(new_node) < max_in_deg:
                graph.add_edge(node, new_node)

        # Add a random number of in edges between min_in_deg to max_in_deg
        k_out = random.randint(min_in_deg, max_in_deg)
        while graph.in_degree(node) < k_out:
            new_node = random.choice(list(graph.nodes()))
            if new_node != node and graph.out_degree(new_node) < max_out_deg:
                graph.add_edge(new_node, node)

    return graph


class EGraphGenerationFunction(Enum):
    ERandomConnectedDirectedGraph = {"name": "Random Connected graph", "function": RandomConnectedDirectedGraph,
                                     "short_name": "RND"}
    EGNCConnectedDirectedGraph = {"name": "GNC Connected graph", "function": GNCConnectedDirectedGraph,
                                  "short_name": "GNC"}
    ECorePeripheryDirectedGraph = {"name": "Core-Periphery graph", "function": CorePeripheryDirectedGraph,
                                   "short_name": "C-P"}
    ERandomSparseDirectedGraph = {"name": "Sparse Directed Graph", "function": RandomConnectedDirectedGraph,
                                 "short_name": "SDG"}
