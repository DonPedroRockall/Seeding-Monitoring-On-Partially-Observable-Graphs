import random

import networkx
import networkx as nx

from Common.ColorPrints import cprint, bcolors
from Common.GraphUtilities import SetSameWeightsToOtherGraphs
from DiffusionModels.IndependentCascade import GetInfectedSubgraphs, AdaptiveCascade, AdaptiveIntervalCascade
from GraphRecovery.GraphRecoverer import InfluentialNodeRecovery
from Monitoring.SourceIdentification.Camerini import CameriniAlgorithm
from Test.Common.DistributionFunctions import ENodeHidingSelectionFunction
from Test.Common.GraphGenerator import EGraphGenerationFunction
from Test.Common.HidingFunctions import EClosureFunction
from Test.Common.WeightGenerator import EWeightSetterFunction
from definitions import ROOT_DIR


def process(graph, steps, k, intervals):

    while True:
        random_sources = list(random.sample(list(graph.nodes()), k))
        infected_nodes = AdaptiveCascade(graph, random_sources, steps, intervals)
        if infected_nodes is not None:
            break
    infected_subgraphs = GetInfectedSubgraphs(graph, infected_nodes)

    camerini = CameriniAlgorithm(graph, attr='weight')

    # edges = 0
    solutions = camerini.find_roots_branching(k, scores=True, subgraphs=infected_subgraphs)
    sources = []
    for solution in solutions:
        sources.append(solution[0])

    return solutions

    # for subgraph in infected_subgraphs:
    #     edges += len(subgraph.edges())
    #
    # sources = []
    # for element in solutions:
    #     sources.append(element[0])
    #
    # return sources





def source_test():
    k = 20
    steps = 2

    pathname_graph = ROOT_DIR + "/Datasets/Real/Wiki-Vote.txt"
    graph: nx.DiGraph = nx.read_edgelist(pathname_graph, create_using=nx.DiGraph(), nodetype=int, comments='#')

    for u, v, data in graph.edges(data=True):
        data["weight"] = random.random() * 0.1

    intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
    process(graph, steps, k, intervals=intervals)


def full_test():
    # Test setup

    # -------------------------- START PART EQUAL FOR ALL CASES -----------------------------------
    full: nx.DiGraph
    part: nx.DiGraph
    recv: nx.DiGraph
    # path = ROOT_DIR + "/Datasets/Real/Wiki-Vote.txt"
    # full = networkx.read_edgelist(path, create_using=nx.DiGraph, nodetype=int)
    full = EGraphGenerationFunction.ERandomSparseDirectedGraph.value["function"](200)
    hiding_func = ENodeHidingSelectionFunction.EDegreeDistribution.value
    closure_func = EClosureFunction.ECrawlerClosure.value
    weight_func = EWeightSetterFunction.EInDegreeWeights.value
    nodes_to_hide = hiding_func["function"](full, 100)
    part = closure_func["function"](full, nodes_to_hide)
    recv, _ = InfluentialNodeRecovery(part, len(nodes_to_hide), 2)
    intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
    num_random_sources = 10
    steps = 10

    # Set weights
    weight_func["function"](full, attribute="weight", force=True)
    SetSameWeightsToOtherGraphs(full, [part, recv])
    weight_func["function"](recv, attribute="weight", force=False)
    # -------------------------- END PART EQUAL FOR ALL CASES -------------------------------------

    # -------------------------- START PART DIFFERENT FOR ALL CASES -------------------------------

    scores_full = [{"true positives": [], "false_positives": [], "false_negatives": []}] * 40
    scores_part = [{"true positives": [], "false_positives": [], "false_negatives": []}] * 40
    scores_recv = [{"true positives": [], "false_positives": [], "false_negatives": []}] * 40

    stats = {"full": {"TP": {"avg": 0, "std": 0}, "FP":  {"avg": 0, "std": 0}, "FN":  {"avg": 0, "std": 0}},
             "part": {"TP": {"avg": 0, "std": 0}, "FP":  {"avg": 0, "std": 0}, "FN":  {"avg": 0, "std": 0}},
             "recv": {"TP": {"avg": 0, "std": 0}, "FP":  {"avg": 0, "std": 0}, "FN":  {"avg": 0, "std": 0}}
    }

    for j in range(40):

        # Random Source selections
        random_sources = random.sample(list(part.nodes()), num_random_sources)

        # Run adaptive cascade on all the graphs
        infected_full = AdaptiveIntervalCascade(full, random_sources, steps=steps, full_intervals=intervals, return_interval=False)
        infected_part = AdaptiveIntervalCascade(part, random_sources, steps=steps, full_intervals=intervals, return_interval=False)
        infected_recv = AdaptiveIntervalCascade(recv, random_sources, steps=steps, full_intervals=intervals, return_interval=False)

        inf_subgraph_full = GetInfectedSubgraphs(full, infected_full)
        inf_subgraph_part = GetInfectedSubgraphs(part, infected_part)
        inf_subgraph_recv = GetInfectedSubgraphs(recv, infected_recv)

        # Compute solutions
        camerini = CameriniAlgorithm(full, attr="weight")
        solutions_full = camerini.find_roots_branching(num_random_sources, scores=False, subgraphs=inf_subgraph_full)
        sources_full = []
        for solution in solutions_full:
            if isinstance(solution, int):
                sources_full.append(solution)
            else:
                sources_full.append(solution[0])

        camerini = CameriniAlgorithm(part, attr="weight")
        solutions_part = camerini.find_roots_branching(num_random_sources, scores=False, subgraphs=inf_subgraph_part)
        sources_part = []
        for solution in solutions_part:
            if isinstance(solution, int):
                sources_part.append(solution)
            else:
                sources_part.append(solution[0])

        camerini = CameriniAlgorithm(recv, attr="weight")
        solutions_recv = camerini.find_roots_branching(num_random_sources, scores=False, subgraphs=inf_subgraph_recv)
        sources_recv = []
        for solution in solutions_recv:
            if isinstance(solution, int):
                sources_recv.append(solution)
            else:
                sources_recv.append(solution[0])

        cprint(bcolors.OKCYAN, "True Sources:", random_sources)
        for graph, srcs in zip([full, part, recv], [sources_full, sources_part, sources_recv]):
            for src in srcs:
                if src in random_sources:
                    cprint(bcolors.OKGREEN, src, end=" ")
                else:
                    cprint(bcolors.FAIL, src, end=" ")
            print("")

        for source in sources_full:
            if source in random_sources:
                scores_full[j]["true_positives"].append(source)
            else:
                scores_full[j]["false_positives"].append(source)


        for source in sources_part:
            if source in random_sources:
                scores_part[j]["true_positives"].append(source)
            else:
                scores_part[j]["false_positives"].append(source)

        for source in sources_recv:
            if source in random_sources:
                scores_recv[j]["true_positives"].append(source)
            else:
                scores_recv[j]["false_positives"].append(source)

        for source in random_sources:
            if source not in sources_full:
                scores_full[j]["false_negatives"].append(source)
            if source not in sources_part:
                scores_part[j]["false_negatives"].append(source)
            if source not in sources_recv:
                scores_recv[j]["false_negatives"].append(source)

    for j in range(40):
        stats["full"]["TP"]["avg"] += len(scores_full[j]["true_positives"])
        stats["full"]["FP"]["avg"] += len(scores_full[j]["false_positives"])
        stats["full"]["FN"]["avg"] += len(scores_full[j]["false_negatives"])

        stats["part"]["TP"]["avg"] += len(scores_part[j]["true_positives"])
        stats["part"]["FP"]["avg"] += len(scores_part[j]["false_positives"])
        stats["part"]["FN"]["avg"] += len(scores_part[j]["false_negatives"])

        stats["recv"]["TP"]["avg"] += len(scores_recv[j]["true_positives"])
        stats["recv"]["FP"]["avg"] += len(scores_recv[j]["false_positives"])
        stats["recv"]["FN"]["avg"] += len(scores_recv[j]["false_negatives"])

    stats["full"]["TP"]["avg"] /= 40
    stats["full"]["FP"]["avg"] /= 40
    stats["full"]["FN"]["avg"] /= 40

    stats["part"]["TP"]["avg"] /= 40
    stats["part"]["FP"]["avg"] /= 40
    stats["part"]["FN"]["avg"] /= 40

    stats["recv"]["TP"]["avg"] /= 40
    stats["recv"]["FP"]["avg"] /= 40
    stats["recv"]["FN"]["avg"] /= 40

    print("FULL stats:")
    print("True Positives (average):",  stats["full"]["TP"]["avg"])
    print("False Positives (average):", stats["full"]["FP"]["avg"])
    print("False Negatives (average):", stats["full"]["FN"]["avg"])

    print("PART stats:")
    print("True Positives (average):",  stats["part"]["TP"]["avg"])
    print("False Positives (average):", stats["part"]["FP"]["avg"])
    print("False Negatives (average):", stats["part"]["FN"]["avg"])

    print("RECV stats:")
    print("True Positives (average):",  stats["recv"]["TP"]["avg"])
    print("False Positives (average):", stats["recv"]["FP"]["avg"])
    print("False Negatives (average):", stats["recv"]["FN"]["avg"])


    # -------------------------- END PART DIFFERENT FOR ALL CASES ---------------------------------


if __name__ == "__main__":
    full_test()



# n_jobs = 30
# processes = 16
#
# intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
# queue = Manager().Queue()
# for i in range(n_jobs):
#     queue.put(i)
# for k in range(2, 5):
#     for interval in range(0, len(intervals), 2):
#         pool = Pool(processes)
#         pool.map(run, [(steps, graph, k, queue, intervals[interval:interval + 2]) for i in range(n_jobs)])
# -------------------------------------------------------
    # for process in processes:
    #   process.start()

    # for process in processes:
    #   process.join()
    # # Parallel(n_jobs=-1)(delayed(run)(steps, graph, k, i) for i in range(n_jobs) )

    # process(steps, graph, random_sources, k, writer=writer)
