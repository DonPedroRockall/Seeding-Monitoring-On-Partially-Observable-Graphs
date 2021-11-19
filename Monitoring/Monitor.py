import random

import networkx

from Monitoring.DiffusionModels import independent_cascade
from Monitoring.EfficientAlgorithm import eAlgorithm
from Monitoring.MinimumMonitorSetConstruction import MMSC, N_delta
from Monitoring.MonitorUtility import SourceContraction
from Utilities.ColorPrints import *
from Utilities.DrawGraph import DrawGraph
from Test.Common.GraphGenerator import RandomConnectedDirectedGraph


def PlaceMonitors(graph, sources, targets, delta=1, tau=0.1, cascade_iterations=100, virtual_set=[], verbose=False):
    if verbose:
        cprint(bcolors.OKGREEN, "Number of Sources =", len(sources))
        cprint(bcolors.OKGREEN, "SOURCES =", sources)
        cprint(bcolors.OKGREEN, "TARGET =", targets)

    G_contracted, c_source, c_target = SourceContraction(graph.copy(), sources, targets)

    G_m_test_contracted = G_contracted.copy()

    n_d = N_delta(G_contracted, c_source)

    B = set(n_d)
    m1 = MMSC(G_contracted, B, delta, tau, c_source, c_target)
    if verbose:
        cprint(bcolors.OKGREEN, "\nMMSC\nMonitor set =", m1, "\nNumber of monitors =", len(m1))
        cprint(bcolors.OKGREEN, "Num delta neighbors =", delta, ",", len(n_d))

    G_cascade = G_m_test_contracted.copy()
    total_nodes = 0
    for x in range(cascade_iterations):
        ic = independent_cascade(G_cascade, [c_source], m1)
        Infected_set = set()
        for sublist in ic:
            Infected_set = Infected_set.union(sublist)
        total_nodes += len(Infected_set)
    c_nodes = total_nodes / cascade_iterations

    if verbose:
        cprint(bcolors.OKGREEN, "\nAVG " + str(cascade_iterations) + " CASCADE: Number of Infected = [", c_nodes, "]\n")

    m2, inf = eAlgorithm(G_m_test_contracted, c_target, c_nodes, c_source, virtual_set=virtual_set, verbose=verbose)

    if verbose:
        cprint(bcolors.OKGREEN, "\n== MONITOR PLACEMENT REPORT ==\nMonitor set =", m2, "\nNumber of monitors =", len(m2))
        cprint(bcolors.OKGREEN, "Max number of infected nodes =", inf)

    return m2, inf


def TestMonitorsSynthetic(full, part, recv, monitors, cascade_iterations, virtual_set, verbose=False):
    if verbose:
        cprint(bcolors.OKGREEN, "Number of Sources =", len(sources))


if __name__ == "__main__":
    g = networkx.DiGraph()

    # while g.size() == 0 or not networkx.is_strongly_connected:
    #     g = networkx.generators.gn_graph(100, create_using=networkx.DiGraph)
    g = RandomConnectedDirectedGraph(30, 40)
    # g.add_nodes_from(list(range(1, 11)))
    # g.add_edges_from([(1, 2), (3, 2), (1, 3), (3, 1), (4, 3), (2, 4), (2, 5), (6, 4), (4, 6), (5, 8), (4, 8), (8, 6), (6, 7), (7, 6), (7, 10), (10, 7), (7, 9), (10, 6), (10, 8), (8, 9), (9, 10)])

    sources = random.sample(list(g.nodes()), 3)
    targets = list()
    while len(targets) < 1:
        node = random.choice(list(g.nodes()))
        if node not in sources:
            targets.append(node)

    color_dict = dict()
    for node in sources:
        color_dict[node] = "red"
    for node in targets:
        color_dict[node] = "green"

    # DrawGraph(g, color_dict)

    for u, v, data in g.edges(data=True):
        val = round(max(random.random(), 0.001), 3)
        data["weight"] = val
        data["capacity"] = val

    monitors, num_of_infected = PlaceMonitors(g.copy(), sources, targets, verbose=True)
    infected = list()

    color_dict = dict()
    for node in sources:
        color_dict[node] = "red"
    for node in targets:
        color_dict[node] = "green"
    for node in monitors:
        color_dict[node] = "blue"

    for node in list(color_dict.keys()):
        if node not in g:
            del color_dict[node]

    DrawGraph(g, color_dict)



