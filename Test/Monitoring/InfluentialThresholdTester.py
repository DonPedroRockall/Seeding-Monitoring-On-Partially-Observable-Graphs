import random
import statistics

import networkx
import networkx as nx
from joblib import Parallel, delayed

from Common.GraphUtilities import SetSameWeightsToOtherGraphs
from DiffusionModels.IndependentCascade import IndependentCascadeWithMonitors
from GraphRecovery.GraphRecoverer import InfluentialNodeRecovery
from Monitoring.MonitorPlacement.Monitor import PlaceMonitors
from Monitoring.MonitorPlacement.MonitorUtility import PrintCascadeResults, GatherCascadeResults
from Test.Common.DistributionFunctions import DegreeDistribution, UniformDistribution
from Test.Common.GraphGenerator import EGraphGenerationFunction
from Test.Common.HidingFunctions import TotalNodeClosure
from Test.Common.WeightGenerator import EWeightSetterFunction


# Test function, used for parallelization
def single_test(inf_thresh, full, part, sources, targets):
    num_hidden_nodes = full.number_of_nodes() - part.number_of_nodes()
    recv, _ = InfluentialNodeRecovery(part, num_hidden_nodes, 2, epsilon=inf_thresh)
    networkx.write_edgelist(recv, f"graph_{inf_thresh}", data=False)
    SetSameWeightsToOtherGraphs(full, [recv])
    EWeightSetterFunction.EInDegreeWeights.value["function"](recv, attribute="weight", force=False)

    # print(f"Monitors on {inf_thresh}", file=open(f"GRAPH_{inf_thresh}", "a+"))
    # print("Nodes recovered:", recv.number_of_nodes() - part.number_of_nodes(), file=open(f"GRAPH_{inf_thresh}", "a+"))
    # monitors, _ = PlaceMonitors(recv, sources, targets, cascade_iterations=100, verbose=True)
    # ic_results = IndependentCascadeWithMonitors(full, sources, monitors, 100)
    # results = GatherCascadeResults(ic_results, full, sources, targets, monitors)
    # PrintCascadeResults(results["num_of_nodes"], results["num_of_infected"], results["num_of_non_source_infected"],
    #                     results["num_of_infected_targets"], results["num_targets"], results["num_of_monitors"],
    #                     results["num_of_iterations"], file=open(f"GRAPH_{inf_thresh}", "a+"))

    print(f"Monitors on {inf_thresh}", file=open(f"GRAPH_{inf_thresh}", "a+"))
    print("Nodes recovered:", recv.number_of_nodes() - part.number_of_nodes(), file=open(f"GRAPH_{inf_thresh}", "a+"))
    stat_dict = {"NUM_INFECTED": [],
                 "NUM_NS_INFECTED": [],
                 "NUM_TG_INFECTED": [],
                 "NUM_MONITORS": []}
    for i in range(40):
        monitors, _ = PlaceMonitors(recv, sources, targets, cascade_iterations=100, verbose=True)
        ic_results = IndependentCascadeWithMonitors(full, sources, monitors, 100)
        results = GatherCascadeResults(ic_results, full, sources, targets, monitors)
        stat_dict["NUM_INFECTED"].append(results["num_of_infected"])
        stat_dict["NUM_NS_INFECTED"].append(results["num_of_non_source_infected"])
        stat_dict["NUM_TG_INFECTED"].append(results["num_of_infected_targets"])
        stat_dict["NUM_MONITORS"].append(results["num_of_monitors"])

    for key in stat_dict:
        print(key, " - mean:", statistics.mean(stat_dict[key]), " - std:", statistics.stdev(stat_dict[key]),
              file=open(f"GRAPH_{inf_thresh}", "a+"))


def test_inf_thresh():
    # full: networkx.DiGraph = networkx.read_edgelist(ROOT_DIR + "/Datasets/Real/Wiki-Vote.txt", create_using=networkx.DiGraph, nodetype=int)
    # TEST 1: full = GNCConnectedDirectedGraph(200)
    full = EGraphGenerationFunction.EGNCConnectedDirectedGraph.value["function"](200)

    # TEST PARAM 1:
    nth = UniformDistribution(full, 50, min_val=0, max_val=0.1)
    # TEST PARAM 2:
    # nth = UniformDistribution(full, 50)
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
    min_indeg = 400  # like float('inf'), but since number of nodes is 200, 400 is enough
    for node in part.nodes():
        in_deg = part.in_degree(node)
        if in_deg > max_indeg:
            max_indeg = in_deg
        elif in_deg < min_indeg:
            min_indeg = in_deg
        mean += in_deg
        # Ignore isolated nodes
        if in_deg != 0:
            median_arr.append(in_deg)
    mean /= part.number_of_nodes()
    median_arr.sort()
    if len(median_arr) % 2 == 1:
        median = median_arr[int(len(median_arr) / 2)]
    else:
        median = (median_arr[int(len(median_arr) / 2)] + median_arr[int(len(median_arr) / 2) - 1]) * 0.5

    print("MAX In-Degree:", max_indeg)
    print("MIN In-Degree:", min_indeg)
    print("MEAN:", mean)
    print("MEDIAN:", median)

    # Populate the inf_thresh list with all the tests, including the control test with all nodes
    it_list = [mean, median]
    num_samples_per_side = 5
    a = min(mean, median)
    b = max(mean, median)
    step_min = (a - min_indeg) / int(num_samples_per_side)
    step_max = (max_indeg - b) / int(num_samples_per_side)
    for i in range(num_samples_per_side):
        it_list.append(min_indeg + step_min * i)
        it_list.append(b + step_max * (i + 1))

    # Assign weights to the graphs
    EWeightSetterFunction.EUniformWeights.value["function"](full, attribute="weight", force=True, **{"max_val": 0.1, "min_val": 0})
    SetSameWeightsToOtherGraphs(full, [part])

    # Sanity check
    print("IT_LIST", it_list)

    # Execute monitor placement on full and part
    print("Monitors on full")
    stat_dict = {"NUM_INFECTED": [],
                 "NUM_NS_INFECTED": [],
                 "NUM_TG_INFECTED": [],
                 "NUM_MONITORS": []}
    for i in range(40):
        monitors, _ = PlaceMonitors(full, sources, targets, cascade_iterations=100, verbose=True)
        ic_results = IndependentCascadeWithMonitors(full, sources, monitors, 100)
        results = GatherCascadeResults(ic_results, full, sources, targets, monitors)
        stat_dict["NUM_INFECTED"].append(results["num_of_infected"])
        stat_dict["NUM_NS_INFECTED"].append(results["num_of_non_source_infected"])
        stat_dict["NUM_TG_INFECTED"].append(results["num_of_infected_targets"])
        stat_dict["NUM_MONITORS"].append(results["num_of_monitors"])

    for key in stat_dict:
        print(key, " - mean:", statistics.mean(stat_dict[key]), " - std:", statistics.stdev(stat_dict[key]))

    # monitors_full, _ = PlaceMonitors(full, sources, targets, cascade_iterations=100, verbose=True)
    # ic_full = IndependentCascadeWithMonitors(full, sources, monitors_full, 100)
    # res_full = GatherCascadeResults(ic_full, full, sources, targets, monitors_full)
    # PrintCascadeResults(res_full["num_of_nodes"], res_full["num_of_infected"], res_full["num_of_non_source_infected"],
    #                     res_full["num_of_infected_targets"], res_full["num_targets"], res_full["num_of_monitors"],
    #                     res_full["num_of_iterations"])

    print("Monitors on part")
    stat_dict = {"NUM_INFECTED": [],
                 "NUM_NS_INFECTED": [],
                 "NUM_TG_INFECTED": [],
                 "NUM_MONITORS": []}
    for _ in range(40):
        monitors, _ = PlaceMonitors(part, sources, targets, cascade_iterations=100, verbose=True)
        ic_results = IndependentCascadeWithMonitors(full, sources, monitors, 100)
        results = GatherCascadeResults(ic_results, full, sources, targets, monitors)
        stat_dict["NUM_INFECTED"].append(results["num_of_infected"])
        stat_dict["NUM_NS_INFECTED"].append(results["num_of_non_source_infected"])
        stat_dict["NUM_TG_INFECTED"].append(results["num_of_infected_targets"])
        stat_dict["NUM_MONITORS"].append(results["num_of_monitors"])

    for key in stat_dict:
        print(key, " - mean:", statistics.mean(stat_dict[key]), " - std:", statistics.stdev(stat_dict[key]))

    # print("Monitors on part")
    # monitors_part, _ = PlaceMonitors(part, sources, targets, cascade_iterations=100, verbose=True)
    # ic_part = IndependentCascadeWithMonitors(part, sources, monitors_part, 100)
    # res_part = GatherCascadeResults(ic_part, full, sources, targets, monitors_full)
    # PrintCascadeResults(res_part["num_of_nodes"], res_part["num_of_infected"], res_part["num_of_non_source_infected"],
    #                     res_part["num_of_infected_targets"], res_part["num_targets"], res_part["num_of_monitors"],
    #                     res_part["num_of_iterations"])

    Parallel(n_jobs=12)(delayed(single_test)(th, full, part, sources, targets) for th in it_list)



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
