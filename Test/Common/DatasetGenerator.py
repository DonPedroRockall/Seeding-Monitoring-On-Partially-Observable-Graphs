import copy

import networkx
import networkx as nx

from joblib import Parallel, delayed

from Test.Common.DistributionFunctions import DegreeDistribution
from Test.Common.HidingFunctions import TotalNodeClosure
from Utilities.ColorPrints import *
from Test.Common.GraphGenerator import GNCConnectedDirectedGraph
from OverlappingCommunityDetection.CommunityDetector import InfluentialNodeRecovery
from definitions import ROOT_DIR as ROOT


def CheckPartFull(full: networkx.DiGraph, part: networkx.DiGraph):
    cnt = 0
    for node in part.nodes():
        if node not in full.nodes():
            cprint(bcolors.FAIL, node)
            cnt += 1
    cprint(bcolors.FAIL, "TOTAL FAILED NODES", cnt)


def CheckPartRecv(part: networkx.DiGraph, recv: networkx.DiGraph):
    cnt = 0
    for node in part.nodes():
        if node not in recv.nodes():
            cprint(bcolors.FAIL, node)
            cnt += 1
    cprint(bcolors.FAIL, "TOTAL FAILED NODES", cnt)


def GenerateRandomGraphTriple(number_of_nodes: int,
                              num_nodes_to_hide: int,
                              generation_function=GNCConnectedDirectedGraph,
                              distribution_function=DegreeDistribution,
                              hiding_function=TotalNodeClosure,
                              influential_threshold=0,
                              influential_centrality="deg",
                              verbose=False):
    """
    Generates a random triple of graphs.
    The first graph is named "full graph" and is the completely observable graph, the ground truth of the experiment;
    The second graph is named "partially observable graph", obtained from the full graph by hiding some edges following
    a given distribution;
    The Third one is named "reconstructed graph", and it is obtained by performing an "Influential Node Recovery
    :param number_of_nodes:             Number of nodes of the full graph
    :param num_nodes_to_hide:           Number of nodes to hide
    :param generation_function:         Function to use to generate the random initial full graph
    :param distribution_function:       Function(graph, int) -> list<nodes> that chooses the nodes to hide
    :param hiding_function:             Function(graph, list<nodes>) -> graph that chooses which edges to hide
    :param influential_centrality:      The centrality measure to use to choose which nodes are influential and have to be
                                        recovered and which ones are to ignore (valid values: "deg" (degree centrality) or
                                        "katz" (katz centrality))
    :param influential_threshold:       The threshold value for the selected centrality measure that a node has to have
                                        to be considered influential. For example, selecting "deg" (degree centrality) and
                                        setting this parameter to 5, means that a reconstructed node has to have a degree
                                        of at least 5 to be considered influential and thus to be recovered
    :param verbose:
    :return: A triple of graphs as described above
    """

    # Generate a full graph
    full_graph = generation_function(number_of_nodes)

    # Generate a copy and start removing edges
    part_obs_graph = full_graph.copy()
    nodes_to_hide = distribution_function(part_obs_graph, num_nodes_to_hide)

    if verbose:
        cprint(bcolors.OKBLUE, "Nodes selected for hiding:", nodes_to_hide)

    part_obs_graph = hiding_function(part_obs_graph, nodes_to_hide)

    # Adaptive Influential Treshold
    if influential_threshold is None:
        influential_threshold = sum(deg for node, deg in part_obs_graph.degree()) / float(part_obs_graph.number_of_nodes())
        if verbose:
            cprint(bcolors.OKBLUE, "Influential Treshold was set to None. Setting it to average of degree")

    # Reconstruct the graph
    reconstructed_graph, nodes_recovered = InfluentialNodeRecovery(
        copy.copy(part_obs_graph.copy()), num_nodes_to_hide, N0=2, alpha=None, beta=None,
        epsilon=influential_threshold, centrality=influential_centrality)

    # Print out useful information that is not used in the process (nor returned by this function)
    if verbose:
        cprint(bcolors.OKBLUE, "Number of recovered nodes:", nodes_recovered)

    # Return the triple
    return full_graph, part_obs_graph, reconstructed_graph


def ParallelDatasetGeneration(num_nodes, num_to_hide, gen_func, distr_func, hiding_func, inf_thresh, inf_centr,
                              num_cores=4, num_of_graphs=10, file_path=ROOT):
    # Result storage
    graph_list = Parallel(n_jobs=num_cores)(delayed(GenerateRandomGraphTriple)(num_nodes, num_to_hide, gen_func, distr_func, hiding_func, inf_thresh, inf_centr, True) for _ in range(num_of_graphs))

    # Write to file
    i = 0
    for full, part, recv in graph_list:
        nx.write_weighted_edgelist(full, file_path + str(i) + "_full_hid" + str(num_to_hide) + "_tresh" + str(inf_thresh) + ".txt")
        nx.write_weighted_edgelist(part, file_path + str(i) + "_part_hid" + str(num_to_hide) + "_tresh" + str(inf_thresh) + ".txt")
        nx.write_weighted_edgelist(recv, file_path + str(i) + "_recv_hid" + str(num_to_hide) + "_tresh" + str(inf_thresh) + ".txt")
        i += 1

    # Return if needed
    return graph_list
