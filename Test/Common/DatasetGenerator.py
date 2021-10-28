import copy

import networkx
import networkx as nx

from Test.Common.DistributionFunctions import DegreeDistribution
from Test.Common.HidingFunctions import TotalNodeClosure
from Utilities.GraphGenerator import RandomConnectedDirectedGraph
from OverlappingCommunityDetection.CommunityDetector import InfluentialNodeRecovery


def GenerateRandomGraphTriple(number_of_nodes: int,
                              minimum_num_of_edges: int,
                              num_nodes_to_hide: int,
                              distribution_function=DegreeDistribution,
                              hiding_function=TotalNodeClosure,
                              influential_threshold=0,
                              influential_centrality="deg"):
    """
    Generates a random triple of graphs.
    The first graph is named "full graph" and is the completely observable graph, the ground truth of the experiment;
    The second graph is named "partially observable graph", obtained from the full graph by hiding some edges following
    a given distribution;
    The Third one is named "reconstructed graph", and it is obtained by performing an "Influential Node Recovery
    :param number_of_nodes:             Number of nodes of the full graph
    :param minimum_num_of_edges:        Minimum number of edges of the full graph. The algorithm will continue to add edges
                                        until it is strongly connected
    :param num_nodes_to_hide:           Number of nodes to hide
    :param distribution_function:       Function(graph, int) -> list<nodes> that chooses the nodes to hide
    :param hiding_function:             Function(graph, list<nodes>) -> graph that chooses which edges to hide
    :param influential_centrality:      The centrality measure to use to choose which nodes are influential and have to be
                                        recovered and which ones are to ignore (valid values: "deg" (degree centrality) or
                                        "katz" (katz centrality))
    :param influential_threshold:       The threshold value for the selected centrality measure that a node has to have
                                        to be considered influential. For example, selecting "deg" (degree centrality) and
                                        setting this parameter to 5, means that a reconstructed node has to have a degree
                                        of at least 5 to be considered influential and thus to be recovered
    :return: A triple of graphs as described above
    """

    # Generate a full graph
    full_graph = RandomConnectedDirectedGraph(number_of_nodes, minimum_num_of_edges)

    # Generate a copy and start removing edges
    part_obs_graph = full_graph.copy()
    nodes_to_hide = distribution_function(part_obs_graph, num_nodes_to_hide)

    part_obs_graph = hiding_function(part_obs_graph, nodes_to_hide)

    # Reconstruct the graph
    reconstructed_graph, nodes_recovered = InfluentialNodeRecovery(
        part_obs_graph.copy(), num_nodes_to_hide, N0=2, alpha=None, beta=None,
        epsilon=influential_threshold, centrality=influential_centrality)

    # Print out useful information that is not used in the process (nor returned by this function)
    print("Number of recovered nodes:", nodes_recovered)

    # Return the triple
    return full_graph, part_obs_graph, reconstructed_graph


def SetSameWeightsToOtherGraphs(original_graph: nx.Graph, other_graphs: list[nx.Graph]):
    """
    Copies all the attributes of original_graph to all the other graphs in other_graphs list, without altering the
    structure of the graph(s) itself. (I.E.: it will not create new nodes or new edges, every graph will stay the same)
    :param original_graph:
    :param other_graphs:
    :return:
    """
    for u, v, data in original_graph.edges(data=True):
        for graph in other_graphs:
            if graph.has_edge(u, v):
                for key in data:
                    graph[u][v][key] = data[key]

