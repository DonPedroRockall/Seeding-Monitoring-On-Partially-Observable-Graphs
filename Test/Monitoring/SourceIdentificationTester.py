import random
import sys

import networkx
import networkx as nx

from Common.ColorPrints import cprint, bcolors
from Common.GraphUtilities import SetSameWeightsToOtherGraphs
from GraphRecovery.GraphRecoverer import InfluentialNodeRecovery
from Monitoring.SourceIdentification.Camerini import CameriniAlgorithm
from DiffusionModels.IndependentCascade import GetInfectedSubgraphs, AdaptiveCascade
from Test.Common.DistributionFunctions import ENodeHidingSelectionFunction
from Test.Common.GraphGenerator import GNCConnectedDirectedGraph
from Test.Common.HidingFunctions import EClosureFunction
from Test.Common.WeightGenerator import EWeightSetterFunction
from definitions import ROOT_DIR


def process(graph, steps, k, intervals=None):
    if intervals is None:
        intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]

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
    full: nx.DiGraph
    part: nx.DiGraph
    recv: nx.DiGraph
    path = ROOT_DIR + "/Datasets/Real/Wiki-Vote.txt"
    full = networkx.read_edgelist(path, create_using=nx.DiGraph, nodetype=int)
    hiding_func = ENodeHidingSelectionFunction.EDegreeDistribution.value
    closure_func = EClosureFunction.ECrawlerClosure.value
    weight_func = EWeightSetterFunction.EInDegreeWeights.value
    nodes_to_hide = hiding_func["function"](full, 100)
    part = closure_func["function"](full, nodes_to_hide)
    recv, _ = InfluentialNodeRecovery(part, len(nodes_to_hide), 2)
    intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
    num_random_sources = 10
    steps = 10

    # Random Source selections
    random_sources = random.sample(list(part.nodes()), num_random_sources)

    # Set weights
    weight_func["function"](full, attribute="weight", force=True)
    SetSameWeightsToOtherGraphs(full, [part, recv])
    weight_func["function"](recv, attribute="weight", force=False)

    # Run adaptive cascade on all the graphs
    infected_full = AdaptiveCascade(full, random_sources, steps=steps, intervals=intervals)
    infected_part = AdaptiveCascade(part, random_sources, steps=steps, intervals=intervals)
    infected_recv = AdaptiveCascade(recv, random_sources, steps=steps, intervals=intervals)

    inf_subgraph_full = GetInfectedSubgraphs(full, infected_full)
    inf_subgraph_part = GetInfectedSubgraphs(part, infected_part)
    inf_subgraph_recv = GetInfectedSubgraphs(recv, infected_recv)


    # Compute solutions
    camerini = CameriniAlgorithm(full, attr="weight")
    solutions_full = camerini.find_roots_branching(num_random_sources, scores=False, subgraphs=inf_subgraph_full)
    sources_full = []
    for solution in solutions_full:
        sources_full.append(solution[0])

    camerini = CameriniAlgorithm(part, attr="weight")
    solutions_part = camerini.find_roots_branching(num_random_sources, scores=False, subgraphs=inf_subgraph_part)
    sources_part = []
    for solution in solutions_part:
        sources_part.append(solution[0])

    camerini = CameriniAlgorithm(recv, attr="weight")
    solutions_recv = camerini.find_roots_branching(num_random_sources, scores=False, subgraphs=inf_subgraph_recv)
    sources_recv = []
    for solution in solutions_recv:
        sources_recv.append(solution[0])

    cprint(bcolors.OKCYAN, "True Sources:", random_sources)
    for graph, srcs in zip([full, part, recv], [sources_full, sources_part, sources_recv]):
        for src in srcs:
            if src in random_sources:
                cprint(bcolors.OKGREEN, src, end=" ")
            else:
                cprint(bcolors.FAIL, src, end=" ")
        print("")


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
