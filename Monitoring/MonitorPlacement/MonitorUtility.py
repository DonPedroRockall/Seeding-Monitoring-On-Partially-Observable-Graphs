import copy
import operator
import random
import networkx

from Common.ColorPrints import fprint


def contractGraph(graph: networkx.DiGraph, nodes: list, new_label="super"):
    super_out = dict()
    super_in = dict()
    in_edges = []
    out_edges = []

    for x in nodes:
        for out_node in graph.neighbors(x):
            if out_node not in nodes:
                out_edges.append((out_node, graph[x][out_node]['weight']))

        for in_node in graph.predecessors(x):
            if in_node not in nodes:
                in_edges.append((in_node, graph[in_node][x]['weight']))

    for rem_node in nodes:
        graph.remove_node(rem_node)

    for (node, cap) in out_edges:
        if node in super_out.keys():
            super_out[node].append((node, cap))
        else:
            super_out[node] = [(node, cap)]

    for (node, cap) in in_edges:
        if node in super_in.keys():
            super_in[node].append((node, cap))
        else:
            super_in[node] = [(node, cap)]

    for key in super_out.keys():
        if len(super_out[key]) > 1:
            prod = 1
            for (u, c) in super_out[key]:
                prod = prod * (1 - c)
            new_cap = 1 - prod
            graph.add_edge(new_label, key, weight=new_cap)
        else:
            graph.add_edge(new_label, key, weight=super_out[key][0][1])

    for key in super_in.keys():
        if len(super_in[key]) > 1:
            prod = 1
            for (u, c) in super_in[key]:
                prod = prod * (1 - c)
            new_cap = 1 - prod
            graph.add_edge(key, new_label, weight=new_cap)
        else:
            graph.add_edge(key, new_label, weight=super_in[key][0][1])

    return graph


def SourceContraction(graph: networkx.DiGraph, sources: list, targets: list):
    """
    Performs Graph Contraction by contracting all sources into one node, and all targets into another node
    :param graph:       Graph on which to perform the contraction
    :param sources:     List of graph nodes that are considered the sources of misinformation to contract
    :param targets:     List of graph nodes that have to be protected by the spread of mininformation
    :return:            (Tuple, in order:) A Contracted graph, the source node and the target node
    """
    contracted_graph, contracted_source, contracted_target = graph, sources[0], targets[0]
    if len(sources) > 1:
        contracted_graph, contracted_source = ContractNodes(graph, sources, target_label="super_source")
    if len(targets) > 1:
        contracted_graph, contracted_target = ContractNodes(contracted_graph, targets, target_label="super_target")
    return contracted_graph, contracted_source, contracted_target


def ContractNodes(graph: networkx.DiGraph, to_contract: list, target_label="super"):
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
        if init_node != to_contract[x]:
            networkx.contracted_nodes(contracted_graph, init_node, to_contract[x], self_loops=False, copy=False)
    networkx.relabel_nodes(graph, {init_node: target_label})
    return contracted_graph, init_node


def select_ss_r(G: networkx.DiGraph, ns):
    # Compute the in-degree for all nodes
    in_degree = dict()
    # for x in G.nodes():
    # 	in_degree[x] = len(G.predecessors(x))
    for node, deg in G.in_degree():
        in_degree[node] = deg

    # Compute average in-degree
    avg_in_l = list(in_degree.values())
    avg_in_value = sum(avg_in_l) / len(avg_in_l)

    # Compute out-degree for all nodes
    out_degree = dict()
    # for x in G.nodes():
    # 	out_degree[x] = len(G.neighbors(x))
    for node, deg in G.out_degree():
        out_degree[node] = deg

    # Compute averate out-degree
    avg_out_l = list(out_degree.values())
    avg_out_value = sum(avg_out_l) / len(avg_out_l)

    # Compute degree for all nodes
    deg = dict()
    for x in G.nodes():
        deg[x] = in_degree[x] + out_degree[x]

    # Sort nodes by degree in descending order
    sorted_in = sorted(deg.items(), key=operator.itemgetter(1), reverse=True)

    PS = set()
    si = sorted_in[:ns]
    for c in si:
        for pred in G.predecessors(c[0]):
            if out_degree[pred] < avg_out_value:
                PS.add(pred)

    PLID = set()
    for key in in_degree.keys():
        if 0 < in_degree[key] < avg_in_value:
            PLID.add(key)

    sources = random.sample(PS, ns)
    target = random.choice(list(PLID))

    return sources, target


def InterpretCascadeResults(ic_results, graph, source, targets, monitors, path=None):
    """
    Transforms the Independent Cascade results in more readable metrics. Can print on std output or on file
    :param monitors:            The set of monitors for a specific graph
    :param targets:             The set of targets
    :param source:              The set of sources
    :param graph:               The graph itself
    :param ic_results:          The results of the independent cascade
    :param path:                Where to write the results. If path is None, then this will print in std out
    """
    num_of_infected = 0
    num_of_non_source_infected = 0
    num_of_infected_targets = 0
    num_of_iterations = len(ic_results) - 1 if ic_results[-1] == [] else len(ic_results)
    num_of_monitors = len(monitors)
    num_of_nodes = graph.number_of_nodes()

    for iteration in ic_results:
        for node in iteration:
            if node not in source:
                num_of_non_source_infected += 1
            if node in targets:
                num_of_infected_targets += 1
            num_of_infected += 1

    if path is not None:
        fprint(path, "Number of Total infected nodes:", num_of_infected, "(", num_of_infected / num_of_nodes * 100,
               "% of total nodes)")
        fprint(path, "Number of Non-Source nodes infected:", num_of_non_source_infected, "(",
               num_of_non_source_infected / num_of_nodes * 100, "% of total nodes)")
        fprint(path, "Number of Infected Targets:", num_of_infected_targets, "(",
               num_of_infected_targets / num_of_nodes * 100,
               "% of total nodes)",
               num_of_infected_targets / len(targets) * 100, "% of total targets)")
        fprint(path, "Number of monitors:", num_of_monitors, "(", num_of_monitors / num_of_nodes * 100,
               "% of total nodes)")
        fprint(path, "Independent Cascade ran for", num_of_iterations, "iterations\n")

    else:
        print("Number of Total infected nodes:", num_of_infected, "(", num_of_infected / num_of_nodes * 100,
              "% of total nodes)")
        print("Number of Non-Source nodes infected:", num_of_non_source_infected, "(",
              num_of_non_source_infected / num_of_nodes * 100, "% of total nodes)")
        print("Number of Infected Targets:", num_of_infected_targets, "(", num_of_infected_targets / num_of_nodes * 100,
              "% of total nodes)",
              num_of_infected_targets / len(targets) * 100, "% of total targets)")
        print("Number of monitors:", num_of_monitors, "(", num_of_monitors / num_of_nodes * 100, "% of total nodes)")
        print("Independent Cascade ran for", num_of_iterations, "iterations\n")