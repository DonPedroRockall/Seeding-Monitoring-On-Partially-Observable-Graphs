import random
import networkx

from DiffusionModels.IndependentCascade import IndependentCascadeWithMonitors
from Monitoring.MonitorPlacement.EfficientAlgorithm import eAlgorithm
from Monitoring.MonitorPlacement.MinimumMonitorSetConstruction import MMSC, N_delta
from Monitoring.MonitorPlacement.MonitorUtility import ContractGraph
from Common.ColorPrints import *
from Common.DrawGraph import DrawGraph
from Test.Common.GraphGenerator import RandomConnectedDirectedGraph


def PlaceMonitors(graph, sources, targets, c_nodes, virtual_set=[], verbose=False):
    """
    Main Monitor Placement Algorithm
    :param graph:           The graph on which to perform the Monitor Placement algorithm
    :param sources:         The identified sources
    :param targets:         The selected targets
    :param c_nodes:         The maximum number of admissible infected nodes
    :param virtual_set:     The virtual nodes, that is, the list of nodes present in RECV but not in PART
    :param verbose:         Whether or not to print debug information
    :return:
    """
    if verbose:
        cprint(bcolors.OKGREEN, "Number of Sources =", len(sources))
        cprint(bcolors.OKGREEN, "SOURCES =", sources)
        cprint(bcolors.OKGREEN, "TARGET =", targets)

    # G_contracted, c_source, c_target = SourceContraction(graph.copy(), sources, targets)
    if graph.is_directed():
        new_graph = graph.copy()
    else:
        new_graph = graph.to_directed()

    G_contracted, c_source = ContractGraph(new_graph, sources)
    G_contracted, c_target = ContractGraph(G_contracted.copy(), targets)

    G_m_test_contracted = G_contracted.copy()

    m2, inf = eAlgorithm(G_m_test_contracted, c_target, c_nodes, c_source, virtual_set=virtual_set, verbose=verbose)

    if verbose:
        cprint(bcolors.OKGREEN, "\n== MONITOR PLACEMENT REPORT ==\nMonitor set =", m2, "\nNumber of monitors =", len(m2))
        cprint(bcolors.OKGREEN, "Max number of infected nodes =", inf)

    return m2, inf


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

    monitors, num_of_infected = PlaceMonitors(g.copy(), sources, targets, verbose=False)
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



