import random

import networkx as nx


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


def GNCConnectedDirectedGraph(num_nodes):
    graph = nx.generators.gnc_graph(num_nodes, create_using=nx.DiGraph)

    while not nx.is_strongly_connected(graph):
        u = random.choice(list(graph.nodes()))
        i = 0
        while i <= 5:
            v = random.choice(list(graph.nodes()))
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
            i += 1

    print("Random graph generated")

    return graph
