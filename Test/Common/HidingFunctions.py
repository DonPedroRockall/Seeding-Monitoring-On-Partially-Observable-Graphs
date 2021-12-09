import random
import networkx
from queue import Queue
from enum import Enum

from Common.ColorPrints import cprint, bcolors


def TotalNodeClosure(graph: networkx.DiGraph, nodes_to_hide: list, **kwargs):
    """Removes all the edges from the graph that are incident to each node in nodes_to_hide parameter"""
    for node in nodes_to_hide:
        if node in graph:
            # for edge in list(graph.edges(node)):
            #    graph.remove_edge(edge[0], edge[1])
            graph.remove_node(node)
    return graph


def PartialNodeClosure(graph: networkx.DiGraph, nodes_to_hide: list, **kwargs):
    """Removes some edges at random for each node in nodes_to_hide list. Pass kwargs with "hide_probability" to set the
    hiding probability for each out-edge for each node of nodes_to_hide"""
    for node in nodes_to_hide:
        if node in graph:
            for edge in list(graph.edges(node)):
                if random.random() < kwargs["hide_probability"]:
                    graph.remove_edge(edge[0], edge[1])
    return graph


def CrawlerClosure(graph: networkx.DiGraph, nodes_to_hide: list, **kwargs):
    """Marks nodes as "to_remove" without removing them, and then performs a crawling over the graph from a starting
    node to simulate a real scenario.
    kwargs:
        "start_node" -> <node>:             the starting node for the crawling. Must be a node of the graph, raises
        AttributeError otherwise if kwargs is None or empty, then a random node from the largest connected component is chosen
    """

    if kwargs is None or "starting_node" not in kwargs:
        list_cc = list(networkx.strongly_connected_components(graph))
        list_cc.sort(key=len, reverse=True)
        start_node = None

        # Find the largest connected component that has at least one valid node to be chosen as crawler starting node
        for connected_component in list_cc:
            possible_choices = list(set(connected_component).difference(set(nodes_to_hide)))
            if len(possible_choices) > 0:
                start_node = random.choice(possible_choices)
                break

        if start_node is None:
            cprint(bcolors.FAIL, "Failed to perform crawling. There is no partition that has at least one non-hidden "
                                 "node, thus crawling could not be started")

    else:
        start_node = kwargs["starting_node"]

    print(start_node)

    pq = Queue()
    pq.put(start_node)
    visited = [start_node]

    while pq.qsize() > 0:

        cur_node = pq.get()

        # Iterate over the neighbors of the node
        for node in graph[cur_node]:
            # If node is already visited, skip, otherwise, add to the visited nodes
            if node not in visited:
                visited.append(node)
                # But if the node is a node to hide, then do not iterate over its neighbors
                if node not in nodes_to_hide:
                    pq.put(node)

    # Iterate over the nodes and remove the non-visited ones
    for node in list(graph.nodes()):
        if node not in visited:
            graph.remove_node(node)

    return graph



class EClosureFunction(Enum):
    ETotalClosure = {"name": "Total Closure", "function": TotalNodeClosure, "short_name": "TOTAL"}
    EPartialClosure = {"name": "Partial Closure", "function": PartialNodeClosure, "short_name": "PART"}
    ECrawlerClosure = {"name": "Crawler Closure", "function": CrawlerClosure, "short_name": "CRAWL"}
