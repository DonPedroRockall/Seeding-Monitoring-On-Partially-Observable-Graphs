import random

import networkx
import networkx as nx
from joblib import Parallel, delayed

from Common.GraphUtilities import SetSameWeightsToOtherGraphs
from DiffusionModels.IndependentCascade import AdaptiveCascade, GetInfectedSubgraphs, ParallelAdaptiveCascade, \
    IndependentCascadeWithMonitors
from GraphRecovery.GraphRecoverer import InfluentialNodeRecovery
from Monitoring.MonitorPlacement.Monitor import PlaceMonitors
from Monitoring.MonitorPlacement.MonitorUtility import PrintCascadeResults
from Monitoring.SourceIdentification.SourceIdentificator import IdentifySources
from Test.Common.DistributionFunctions import DegreeDistribution
from Test.Common.GraphGenerator import GNCConnectedDirectedGraph
from Test.Common.HidingFunctions import TotalNodeClosure
from Test.Common.WeightGenerator import EWeightSetterFunction
from definitions import ROOT_DIR


def get_accuracy(graph: networkx.DiGraph, real_sources, est_sources, extract_et_sources=True):
    points = 0
    for est_source in est_sources:
        if est_source in real_sources:
            real_sources.remove(est_source)

    estimated_sources = est_sources
    if extract_et_sources:
        estimated_sources = []
        for source in est_sources:
            estimated_sources.append(source)

    # Remove weight information from the graph, since it is not needed
    unw_dir_graph = nx.Graph()
    # Add all nodes and edges (including isolated nodes)
    unw_dir_graph.add_nodes_from(graph.nodes())
    unw_dir_graph.add_edges_from(graph.edges(data=False))

    # Store information about the shortest distance between nodes
    distance = {}

    for real_source in real_sources:
        for est_source in estimated_sources:
            # Check if a path exists, before computing the distance
            if networkx.has_path(unw_dir_graph, real_source, est_source):
                dist = networkx.shortest_path_length(unw_dir_graph, real_source, est_source)
            else:
                dist = 100
            if real_source not in distance.keys() or dist < distance[real_source][0]:
                distance[real_source] = (est_source, dist)

    # Print the results
    for src in distance.keys():
        points += distance[src][1]
        print(f"Distance from {src} to {distance[src][0]} is {distance[src][1]}")

    print("Total points:", (graph.number_of_nodes() / points))


# Test function, used for parallelization
def single_test(inf_thresh, full, part, sources, targets):
    recv, _ = InfluentialNodeRecovery(part, 500, 2, epsilon=inf_thresh)
    networkx.write_edgelist(recv, f"graph_{inf_thresh}", data=False)
    SetSameWeightsToOtherGraphs(full, [recv])
    EWeightSetterFunction.EInDegreeWeights.value["function"](recv, attribute="weight", force=False)
    # print("Starting adaptive cascade", steps, intervals, max_iter)
    #
    # # infected_recv = AdaptiveCascade(recv, random_sources, steps=10, intervals=intervals)
    # infected_recv = AdaptiveCascade(recv, random_sources, steps=steps, intervals=intervals, max_iterations=max_iter)
    # est_sources = IdentifySources(recv, len(random_sources), GetInfectedSubgraphs(recv, infected_recv))
    #
    # get_accuracy(full, random_sources, est_sources)
    print(f"Monitors on {inf_thresh}", file=open(f"GRAPH_{inf_thresh}", "a+"))
    print("Nodes recovered:", recv.number_of_nodes() - part.number_of_nodes(), file=open(f"GRAPH_{inf_thresh}", "a+"))
    monitors, _ = PlaceMonitors(recv, sources, targets, cascade_iterations=100, verbose=True)
    ic_results = IndependentCascadeWithMonitors(full, sources, monitors, 100)
    PrintCascadeResults(ic_results, full, sources, targets, monitors, file=open(f"GRAPH_{inf_thresh}", "a+"))


def test_inf_thresh():
    # full: networkx.DiGraph = networkx.read_edgelist(ROOT_DIR + "/Datasets/Real/Wiki-Vote.txt", create_using=networkx.DiGraph, nodetype=int)
    full = GNCConnectedDirectedGraph(200)
    nth = DegreeDistribution(full, 50)
    part = TotalNodeClosure(full.copy(), nth)

    print(full.nodes())
    print(part.nodes())

    # Set variables
    # intervals = [50, 1000]
    # max_iter = 2000
    # steps = 50
    sources = random.sample(list(part.nodes()), 30)
    targets = []
    while len(targets) < 30:
        node = random.choice(list(part.nodes))
        if node not in sources and node not in targets:
            targets.append(node)
    print("Ground truth random sources:", sources)
    print("Ground truth random targets:", targets)

    # Computate max, mean and median
    mean = 0
    median_arr = []
    max_indeg = 0
    for node in part.nodes():
        in_deg = part.in_degree(node)
        if in_deg > max_indeg:
            max_indeg = in_deg
        mean += in_deg
        if in_deg != 0:
            median_arr.append(in_deg)

    print("MAX In-Degree:", max_indeg)
    mean /= part.number_of_nodes()
    median_arr.sort()
    if len(median_arr) % 2 == 1:
        median = median_arr[int(len(median_arr) / 2)]
    else:
        median = (median_arr[int(len(median_arr) / 2)] + median_arr[int(len(median_arr) / 2) - 1]) * 0.5

    # Populate the inf_thresh list with all the tests, including the control test with all nodes
    it_list = []
    for x in range(1, 11):
        it_list.append(max_indeg * (x * 0.01))

    it_list.append(mean)
    it_list.append(median)

    # Assign weights to the graphs
    EWeightSetterFunction.EUniformWeights.value["function"](full, attribute="weight", force=True, **{"max_val": 0.1, "min_val": 0})
    SetSameWeightsToOtherGraphs(full, [part])

    # Sanity check
    print(it_list)

    # Execute monitor placement on full and part
    print("Monitors on full")
    monitors_full, _ = PlaceMonitors(full, sources, targets, cascade_iterations=100, verbose=True)
    ic_full = IndependentCascadeWithMonitors(full, sources, monitors_full, 100)
    PrintCascadeResults(ic_full, full, sources, targets, monitors_full)

    print("Monitors on part")
    monitors_part, _ = PlaceMonitors(part, sources, targets, cascade_iterations=100, verbose=True)
    ic_part = IndependentCascadeWithMonitors(part, sources, monitors_part, 100)
    PrintCascadeResults(ic_part, part, sources, targets, monitors_full)

    Parallel(n_jobs=6)(delayed(single_test)(th, full, part, sources, targets) for th in it_list)



    # Run control tests on full and part
    # infected_full = AdaptiveCascade(full, random_sources, steps=steps, intervals=intervals, max_iterations=max_iter)
    # infected_part = AdaptiveCascade(part, random_sources, steps=steps, intervals=intervals, max_iterations=max_iter)
    # full_sources = IdentifySources(full, len(random_sources), GetInfectedSubgraphs(full, infected_full))
    # part_sources = IdentifySources(part, len(random_sources), GetInfectedSubgraphs(part, infected_part))
    #
    # get_accuracy(full, random_sources, full_sources)
    # get_accuracy(full, random_sources, part_sources)

    # # Execute tests
    # print("LENLIST", len(it_list))
    # # Parallel(n_jobs=int(len(it_list) / 2))(delayed(single_test)(thresh, full, part, random_sources, steps, intervals, max_iter) for thresh in it_list)
    # for th in it_list:
    #     single_test(th, full, part, random_sources, steps, intervals, max_iter)


if __name__ == "__main__":
    test_inf_thresh()
