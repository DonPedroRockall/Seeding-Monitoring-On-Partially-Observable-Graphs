import random
import networkx
from enum import Enum


def TotalNodeClosure(graph: networkx.Graph, nodes_to_hide: list, **kwargs):
    """Removes all the edges from the graph that are incident to each node in nodes_to_hide parameter"""
    for node in nodes_to_hide:
        if node in graph:
            # for edge in list(graph.edges(node)):
            #    graph.remove_edge(edge[0], edge[1])
            graph.remove_node(node)
    return graph


def PartialNodeClosure(graph: networkx.Graph, nodes_to_hide: list, **kwargs):
    """Removes some edges at random for each node in nodes_to_hide list. Pass kwargs with "hide_probability" to set the
    hiding probability for each out-edge for each node of nodes_to_hide"""
    for node in nodes_to_hide:
        if node in graph:
            for edge in list(graph.edges(node)):
                if random.random() < kwargs["hide_probability"]:
                    graph.remove_edge(edge[0], edge[1])
    return graph


class EClosureFunction(Enum):
    ETotalClosure = {"name": "Total Closure", "function": TotalNodeClosure, "short_name": "total"}
    EPartialClosure = {"name": "Partial Closure", "function": PartialNodeClosure, "short_name": "part"}
