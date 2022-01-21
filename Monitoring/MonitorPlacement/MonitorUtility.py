import operator
import random
import sys
import networkx


def ContractGraph(graph: networkx.DiGraph, nodes: list):
    """
    Contracts a weighted graph in the following way:
    - For every edge not involved in the contraction, the edge weight stays the same
    - For every edge from contracted -> other node,

    - For edges between nodes to contract, they are removed
    Reference:
    Amoruso M., Anello D., Auletta V., Cerulli R., Ferraioli D., Raiconi A., "Contrasting the Spread of Misinformation
    in Online Social Networks", p9-10
    :param graph:       Input graph on which to perform the contraction
    :param nodes:       The nodes to contract
    :return:            The contracted graph and the resulting contracted node
    """
    super_out = dict()
    super_in = dict()
    in_edges = []
    out_edges = []
    new_label = nodes[0]

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

    return graph, new_label


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


def GatherCascadeResults(ic_results, graph, sources, targets, monitors):
    """
    Transforms the Independent Cascade results in more readable metrics
    :param monitors:            The set of monitors for a specific graph
    :param targets:             The set of targets
    :param sources:             The set of sources
    :param graph:               The graph itself
    :param ic_results:          The results of the independent cascade
    :return                     A dict of string -> result for the cascade
    """

    results = {"num_of_infected": 0,
               "num_of_non_source_infected": 0,
               "num_of_infected_targets": 0,
               "num_of_iterations": len(ic_results) - 1 if ic_results[-1] == [] else len(ic_results),
               "num_of_monitors": len(monitors),
               "num_of_nodes": graph.number_of_nodes(),
               "num_targets": len(targets),
               "num_sources": len(sources)
               }

    for iteration in ic_results:
        for node in iteration:
            if node not in sources:
                results["num_of_non_source_infected"] += 1
            if node in targets:
                results["num_of_infected_targets"] += 1
            results["num_of_infected"] += 1

    return results


def PrintCascadeResults(n_nodes, n_inf, ns_inf, n_inf_t, n_t, n_mon, c_iter, file=sys.stdout):
    """
    Prints the results of an independent cascade execution in a fancy way, to stdout or on file
    :param n_nodes:             Number of nodes of the graph
    :param n_inf:               Number of infected nodes
    :param ns_inf:              Number of non-source infected
    :param n_inf_t:             Number of infected targets
    :param n_t:                 Number of targets
    :param n_mon:               Number of monitors
    :param c_iter:              Number of cascade iterations
    :param file:                Where to write the results. If path is None, then this will print in std out
    """

    print(f"Number of Total infected nodes: {n_inf} ({n_inf / n_nodes * 100} % of total nodes)", file=file)
    print(f"Number of Non-Source nodes infected: {ns_inf} ({ns_inf / n_nodes * 100} % of total nodes)", file=file)
    print(f"Number of Infected Targets: {n_inf_t} ({n_inf_t / n_nodes * 100} % of total nodes; {n_inf_t / n_t * 100} % of total targets)", file=file)
    print(f"Number of monitors: {n_mon} ({n_mon / n_nodes * 100} % of total nodes)", file=file)
    print(f"Independent Cascade ran for {c_iter} iterations\n", file=file)
