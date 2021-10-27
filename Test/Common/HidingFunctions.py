import random

import networkx


def TotalNodeClosure(graph: networkx.Graph, nodes_to_hide: list):
    """Removes all the edges from the graph that are incident to each node in nodes_to_hide parameter"""
    for node in nodes_to_hide:
        if node in graph:
            #for edge in list(graph.edges(node)):
            #    graph.remove_edge(edge[0], edge[1])
            graph.remove_node(node)
    return graph


def PartialNodeClosure(graph: networkx.Graph, nodes_to_hide: list, hide_probability: float = 0.5):
    """Removes some edges at random for each node in nodes_to_hide list"""
    for node in nodes_to_hide:
        if node in graph:
            for edge in list(graph.edges(node)):
                if random.random() < hide_probability:
                    graph.remove_edge(edge[0], edge[1])
    return graph
