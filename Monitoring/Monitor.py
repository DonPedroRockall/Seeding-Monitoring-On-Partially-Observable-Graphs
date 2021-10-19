import copy
import csv
import os
import random
import sys
from fractions import Fraction
from timeit import default_timer as timer

import networkx
import numpy
from networkx.algorithms import approximation

# from Monitoring.SourceIdentification.Camerini import Camerini
# from Monitoring.SourceIdentification.imeterOpt import IMeterSort
# from Monitoring.SourceIdentification.independent_cascade_opt import independent_cascade, get_infected_subgraphs


# def PlaceMonitors(graph, k, steps):
    # # n_jobs = 30
    # # processes = 16
    # #
    # intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
    # # queue = Manager().Queue()
    # #
    # # for i in range(n_jobs):
    # #     queue.put(i)
    # #
    # # for k in range(2, 5):
    # #     for interval in range(0, len(intervals), 2):
    # #         pool = Pool(processes)
    # #         pool.map(run, [(steps, graph, k, queue, intervals[interval:interval + 2]) for i in range(n_jobs)])
    #
    # sources = list()
    #
    # for k in range(2, 5):
    #     for interval in range(0, len(intervals), 2):
    #         sources = process(steps, graph, k, intervals[interval:interval + 2], file_version=False)
def MonitorPlacement(graph: networkx.DiGraph, sources: list, targets: list, budget: int):
    """
    Given sources S and targets T to protect, calculates where to place
    :param graph:       Input Graph
    :param sources:     List of graph nodes that are considered Source of misinformation
    :param targets:     List of graph nodes that have to be protected by the spread of misinformation
    :param budget:      Upper limit of the number of monitors to place
    :return:            A list containing the nodes where the monitors should be placed
    """
    contr_graph, contr_source, contr_target = SourceContraction(graph, sources, targets)
    weighted_contr_graph = WeightConversion(contr_graph)

    # Get the maximum weight among all edges
    max_weight = -1
    for u, v, data in weighted_contr_graph.edges(data=True):
        if data["weight"] > max_weight:
            max_weight = data["weight"]

    (L, R, alpha) = UnbalancedCut(weighted_contr_graph, contr_source, contr_target, budget, alpha_start=0, alpha_end=max_weight, alpha_step=max_weight / 100)
    C = BipartiteGraphFromCut(contr_graph, L, R)
    # Remove source and target from the bipartite graph
    # TODO: Check if this reasoning is correct
    if C.has_node(contr_source):
        C.remove_node(contr_source)
    if C.has_node(contr_target):
        C.remove_node(contr_target)
    M = networkx.algorithms.approximation.min_weighted_vertex_cover(C)
    return M


def UnbalancedCut(graph: networkx.DiGraph, source, target, k, alpha_start=0, alpha_end=1, alpha_step=0.01):
    """
    Performs a k-unbalanced cut, by adding edges from every node of "graph" to "target" (except target itself).
    Every such edge will be added at an increasing capacity (read: weight) alpha and a min-cut will be performed on
    the resulting graph. At the end of the process, the algorithm returns the (source, target)-mincut of "graph" such
    that the source partition will have at most k nodes.
    :param graph:       The input graph
    :param source:      The source node the the mincut
    :param target:      The target node for the mincut
    :param k:           The maximum size of the source partition
    :return:            Tuple of three elements, consisting of the source partition, the target partition, the alpha
    """
    alpha = alpha_start
    while alpha <= alpha_end:
        src_part, trg_part, part_size = GetAlphaMinCut(graph, source, target, alpha)
        print(alpha, part_size)
        if part_size <= k:
            return src_part, trg_part, alpha
        alpha += alpha_step
    return None


def GetAlphaMinCut(graph: networkx.DiGraph, source, target, alpha):
    """
    Performs a min-cut on a graph "alpha_graph", obtained from graph and adding an edge of capacity alpha from each node
    to the target, adding parallel edges if necessary
    :param graph:       The input graph
    :param source:      Source of the mincut
    :param target:      Target of the mincut
    :param alpha:       Alpha parameter, described above
    :return:            The partitions and |S|, that is, the number of nodes contained in the -s (source) partition
    """
    alpha_graph = copy.copy(graph)
    for node in alpha_graph.nodes():
        if node is target or node is source:
            print(node is target, node is source)
            continue
        alpha_graph.add_edge(node, target, weight=alpha)

    _, partitions = networkx.algorithms.minimum_cut(alpha_graph, source, target, capacity="weight")
    source_partition, target_partition = partitions
    return source_partition, target_partition, len(source_partition)


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


def SourceContraction(graph: networkx.DiGraph, sources: list, targets: list):
    """
    Performs Graph Contraction by contracting all sources into one node, and all targets into another node
    :param graph:       Graph on which to perform the contraction
    :param sources:     List of graph nodes that are considered the sources of misinformation to contract
    :param targets:     List of graph nodes that have to be protected by the spread of mininformation
    :return:            (Tuple, in order:) A Contracted graph, the source node and the target node
    """
    contracted_graph, contracted_source = ContractNodes(graph, sources)
    contracted_graph, contracted_target = ContractNodes(contracted_graph, targets)
    return contracted_graph, contracted_source, contracted_target


def ContractNodes(graph: networkx.DiGraph, to_contract: list):
    """
    Contracts a list of nodes into a single node, removing self loops and preserving the graph input parameter
    :param graph:           Graph on where the contraction should take place
    :param to_contract:     List of nodes to be contracted
    :return:                The Contracted Graph
    """
    contracted_graph: networkx.DiGraph = copy.copy(graph)
    if len(to_contract) < 2:
        return graph
    init_node = to_contract[0]
    for x in range(len(to_contract)):
        networkx.contracted_nodes(contracted_graph, init_node, to_contract[x], self_loops=False, copy=False)
    return contracted_graph, init_node


def BipartiteGraphFromCut(graph: networkx.Graph, L: list, R: list):
    """
    Computes the graph C = (W, F) induced by the graph "graph" such that:
    The nodes W are:
    W = {u belongs to L so that (u, v) belongs to graph.edges() and v belongs to R} union
        {v belongs to R so that (u, v) belongs to graph.edges() and u belongs to L}
    F = {(u, v) belongs to graph.edges() so that u belongs to L and v belongs to R}

    :param graph:       Original graph on which the cut has been made
    :param L:           List of nodes representing the partition that contains the source
    :param R:           List of nodes representing the partition that contains the target
    :return:            Networkx Graph made as stated above
    """
    bipartite_graph = networkx.DiGraph()
    for edge in graph.edges():
        (u, v) = edge
        if u in L and v in R and not bipartite_graph.has_node(u):
            bipartite_graph.add_node(u)
            if not bipartite_graph.has_edge(u, v):
                bipartite_graph.add_edge(u, v)
        elif v in R and u in L and not bipartite_graph.has_node(v):
            bipartite_graph.add_node(v)
    return bipartite_graph


def WeightConversion(graph: networkx.DiGraph):
    """
    Transforms the weights of the graph "graph" from floating point to integer.
    Moreover the integer weight will be inversely proportional to the floating point value
    :param graph:       The Graph on which to perform the transformation
    :return:            A Graph with the new weights
    """
    new_graph = copy.copy(graph)
    edge_to_weight = dict()
    for edge in graph.edges():
        fract = Fraction.from_float(1 / graph.get_edge_data(edge[0], edge[1])["weight"]).limit_denominator()
        edge_to_weight[edge] = (fract.numerator, fract.denominator)

    denom_list = list()
    for weight in edge_to_weight.values():
        denom_list.append(weight[1])
    least_common_multiple = numpy.lcm.reduce(denom_list)

    for edge in graph.edges():
        new_graph.add_edge(edge[0], edge[1], weight=(least_common_multiple / edge_to_weight[edge][1]) * edge_to_weight[edge][0])

    return new_graph


if __name__ == "__main__":
    g = networkx.DiGraph()
    g = networkx.generators.fast_gnp_random_graph(250, 0.15, directed=True)
    print(g.number_of_edges())

    for edge in g.edges:
        g[edge[0]][edge[1]]["weight"] = random.random()

    g = WeightConversion(g)
    weights = list()
    for edge in g.edges():
        weights.append(g[edge[0]][edge[1]]["weight"])

    min = min(weights)
    max = max(weights)

    astart = min
    aend = max
    astep = (max - min) / 100

    for _ in range(5):

        print("--------------------------------------------------------------------------")

        result = UnbalancedCut(g, random.randint(1, 125), random.randint(126, 250), 3, alpha_start=astart, alpha_end=aend, alpha_step=astep)
        if result is not None:
            src, trg, alpha = result
        else:
            input("Program terminated")
            sys.exit()

        print(alpha, len(src), src, trg)

        astart = alpha - astep
        aend = alpha + astep
        astep /= 10




