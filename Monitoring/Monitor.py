import copy
import random

import networkx
import networkx as nx


def PerformIndependentCascade(graph: networkx.DiGraph, sources: list, max_iter=-1):
    """
    Indipendent Cascade Model: Each edge has a weight in [0, 1] that represents the probability that the edges spreads
    the misinformation from the source to the target
    :param graph:           The Graph on which to perform the dynamics
    :param sources:         Misinformation sources
    :return:                The set of infected nodes and a dict representing at which iteration each node has been infected
    """
    infected_nodes = set(sources)
    infected_at = dict()
    iteration = 0

    while True:
        # Build list of neighbors
        infected_at[iteration] = set()
        for infected in infected_nodes:
            for node in graph.neighbors(infected):
                if node not in infected_nodes:
                    weight = graph.get_edge_data(infected, node)["weight"]
                    if random.random() < weight:
                        infected_at[iteration].add(node)
        for node in infected_at[iteration]:
            infected_nodes.add(node)

        # Check for termination condition
        if len(infected_at[iteration]) == 0 or (max_iter != -1 and iteration >= max_iter):
            return infected_nodes, infected_at

        iteration += 1


def GetInfectedSubgraph(graph: networkx.DiGraph, sources: list):
    """
    Runs The desired dynamics and returns the subgraph composed of the infected nodes and the edges that connect them
    :param graph:           The Graph on which to perform the dynamics
    :param sources:         Misinformation sources
    :return:
    """
    infected_graph: networkx.DiGraph
    infected_nodes, _ = PerformIndependentCascade(graph, sources)
    nodes = [node for node in infected_nodes]
    return graph.subgraph(nodes)


def ContractNodes(graph: networkx.DiGraph, to_contract: list):
    """
    Contracts a list of nodes into a single node, removing self loops and preserving the graph input parameter
    :param graph:           Graph on where the contraction should take place
    :param to_contract:     List of nodes to be contracted
    :return:                The Contracted Graph
    """
    contracted_graph: networkx.DiGraph = copy.copy(graph)
    if len(to_contract) < 2: return graph
    init_node = to_contract[0]
    for x in range(len(to_contract)):
        networkx.contracted_nodes(contracted_graph, init_node, to_contract[x], self_loops=False, copy=False)
    return contracted_graph






