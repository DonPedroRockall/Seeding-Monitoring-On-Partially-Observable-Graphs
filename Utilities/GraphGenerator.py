import random

import networkx as nx


def RandomConnectedDirectedGraph(num_nodes, min_num_edges):
    graph = nx.DiGraph()

    graph.add_nodes_from(list(range(num_nodes)))

    while graph.number_of_edges() < min_num_edges or not nx.is_strongly_connected(graph):
        u = random.choice(list(graph.nodes()))
        v = u
        while u == v:
            v = random.choice(list(graph.nodes()))
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)

    return graph