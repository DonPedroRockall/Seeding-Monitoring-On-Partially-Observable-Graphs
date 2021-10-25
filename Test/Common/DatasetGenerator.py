import networkx

from Test.Common.DistributionFunctions import DegreeDistribution
from Test.Common.HidingFunctions import TotalNodeClosure
from Utilities.GraphGenerator import RandomConnectedDirectedGraph
from OverlappingCommunityDetection.CommunityDetector import InfluentialNodeRecovery


def GenerateRandomGraphTriple(number_of_nodes: int,
                              minimum_num_of_edges: int,
                              num_nodes_to_hide: int,
                              hiding_function=TotalNodeClosure,
                              distribution_function=DegreeDistribution):

    full_graph = RandomConnectedDirectedGraph(number_of_nodes, minimum_num_of_edges)
    part_obs_graph = full_graph.copy()

    nodes_to_hide = list()

    for _ in range(num_nodes_to_hide):
        nodes_to_hide.append(distribution_function(part_obs_graph))

    hiding_function(part_obs_graph, nodes_to_hide)

    reconstructed_graph = InfluentialNodeRecovery(part_obs_graph, num_nodes_to_hide, N0=2, alpha=None, beta=None, epsilon=5, centrality="deg")

    return full_graph, part_obs_graph, reconstructed_graph

