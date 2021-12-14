import copy

import networkx as nx
from networkx.algorithms.flow import *
from collections import deque

from Common.ColorPrints import bcolors, cprint


def bfs_res(ResGraph, src):
    heights = dict()

    heights[src] = 0

    q = deque([(src, 0)])
    while q:
        u, height = q.popleft()
        height += 1
        for v, attr in ResGraph.succ[u].items():
            if v not in heights and attr['flow'] < attr['capacity']:
                heights[v] = height
                q.append((v, height))

    for node in set(ResGraph.nodes()) - set(heights.keys()):
        heights[node] = float("Inf")
    return heights


def computePartitions(graph, src, trg, verbose=False):
    n = len(graph.nodes())
    R = preflow_push(graph, src, trg, capacity="weight")

    dists_source = bfs_res(R, src)
    dists_target = bfs_res(R, trg)

    d = dict()
    for v in R.nodes():
        d[v] = min(dists_source[v], dists_target[v] + n)

    S = set()
    for v in R.nodes():
        if d[v] < n:
            S.add(v)

    T = set(graph.nodes()) - S
    return S, T


def createGAlpha(graph, alfa, target):
    G_alfa = graph.copy()
    for node in G_alfa.nodes():
        if node != target:
            if target in G_alfa[node]:
                cap = G_alfa[node][target]['weight']
                G_alfa.add_edge(node, target, weight=cap + alfa)
            else:
                G_alfa.add_edge(node, target, weight=alfa)
    return G_alfa


def reachable_residual_graph(ResGraph, src):
    rc = set()
    heights = {src: 0}
    q = deque([(src, 0)])
    rc.add(src)
    while q:
        u, height = q.popleft()
        height += 1
        for v, attr in ResGraph.succ[u].items():
            if v not in heights and ResGraph.succ[u][v]['flow'] < ResGraph.succ[u][v]['capacity']:
                heights[v] = height
                q.append((v, height))
                rc.add(v)
    return rc


def cutGraph(cutset, source, target):
    cg = nx.DiGraph()
    if (source, target) in cutset:
        c_st = True
    else:
        c_st = False

    for (u, v) in cutset:
        if u != source:
            if v != target or (v == target and not c_st):
                cg.add_edge(u, v, capacity=1)
    return cg


def chooseMs(cutset, source, target):
    cut_graph = cutGraph(cutset, source, target)
    m = set()

    for (u, v) in cutset:
        if u == source and v == target:
            m.add(v)
        else:
            if u == source:
                m.add(v)

    cut_graph_edges = cut_graph.edges()

    for (u, v) in list(cut_graph_edges):
        cut_graph.add_edge('s', u, weight=1)
        cut_graph.add_edge(v, 't', weight=1)

    if len(cut_graph.edges()) != 0:

        R = edmonds_karp(cut_graph, 's', 't')
        A = reachable_residual_graph(R, 's')
        B = set(cut_graph.nodes()) - A
        cut_graph_cutset = computeCutset(cut_graph, A, B)

        for (u, v) in cut_graph_cutset:
            if u == 's':
                m.add(v)
            else:
                m.add(u)
    return m


def computeCutset(Gr, S, T):
    c = set()
    for u, nbrs in ((n, Gr[n]) for n in S):
        c.update((u, v) for v in nbrs if v in T)
    return c


def setCapacities(graph):
    retGraph = nx.DiGraph()
    for (u, v) in graph.edges():
        retGraph.add_edge(u, v, weight=int(round((1 / graph[u][v]['weight']) * 100)))
    return retGraph


def virtualSubgraph(graph: nx.DiGraph, source_edge, virtual_set):
    """
    Returns the set of nodes reachable from at least one of the endpoints of source_edge by a path that is entirely
    virtual
    :param graph:			The recovered graph
    :param source_edge:		The source edge
    :param virtual_set:		The set containing all and only the virtual nodes of the graph
    :return:
    """
    queue = list()
    visited = set()
    real_set = set()
    u, v = source_edge
    # Add both endpoints to the queue
    queue.append(u)
    queue.append(v)
    # Mark the endpoints as visited
    visited.add(u)
    visited.add(v)

    # Repeat until there are no more nodes to visit
    while len(queue) > 0:
        cur_node = queue.pop(0)  # Get the first node of the queue (should be already marked as visited)

        # For each node reachable from cur_node
        for neighbor in graph[cur_node]:

            # If visited ignore. Note: real nodes are never ignored, but sets do not allow duplicate elements
            if neighbor not in visited:

                # If the reachable node is virtual, then add the node to the queue and mark it as visited. In the next
                # iterations it will be searched from it
                if neighbor in virtual_set:
                    queue.append(neighbor)
                    visited.add(neighbor)

                # Otherwise, the neighbor is reachable by a virtual path
                else:
                    real_set.add(neighbor)
    return real_set


def eAlgorithm(G, target, k, source_node, virtual_set=[], verbose=False):
    Gc = setCapacities(G)
    n = len(Gc.nodes())

    step = 1
    alfa = 0
    source = source_node

    old_S_len = -1

    while True:
        G_alfa = createGAlpha(Gc, alfa, target)

        S, T = computePartitions(G_alfa, source, target)

        if len(S) != old_S_len:
            if verbose:
                print("\nAlpha ", alfa)
                print("|S| =", len(S), "|T| =", len(T))
                print("Alpha ", alfa)

            old_S_len = len(S)

        if len(S) <= k:
            break
        alfa += step

    cuts = computeCutset(Gc, S, T)
    starting_cut = copy.copy(cuts)

    # Create a set with all the monitors that have been chosed because of "virtual reasons"
    virtual_monitors = set()

    # If there are some nodes that are virtual (= recovered from InfluentialNodeRecovery, thus they might not exist.
    # Placing monitors in those nodes is not an allowed solution, and we should seek alternative solutions)
    if len(virtual_set) > 0:

        # This function is only needed if there are virtual nodes
        def remove_nodes_from_cutset(*args):
            for node in args:
                for u, v in list(cuts):
                    # Remove both in-edges and out-edges, except if they are both virtual
                    if u == node or v == node:
                        removed_edges.add((u, v))
                        cuts.remove((u, v))

        # Since we cannot modify the list(cuts) collection directly, we store here the removed edges
        removed_edges = set()

        # The variable "cuts" is a set containing all the edges that belong in the cut For each edge (u, v),
        # if u is virtual and v is not or v is virtual and u is not, then set the non-virtual node to be part of the
        # monitors, and remove all the nodes that are connected from out-edges from the non-virtual node

        # list(cuts) is used to make a copy, since the collection will be modified during iteration
        for u, v in list(cuts):

            # Cannot update the list(cuts), so add the removed edges in removed_edges and skip if present
            if (u, v) in removed_edges:
                continue

            # CASE 1: u is virtual, v is not
            if u in virtual_set and v not in virtual_set:
                virtual_monitors.add(v)
                # Remove all nodes that can be reached from the v node (because it has been selected as a monitor)
                remove_nodes_from_cutset(u, v)

            # CASE 2: v is virtual, u is not
            elif u not in virtual_set and v in virtual_set:
                virtual_monitors.add(u)
                # Remove all nodes that can be reached from the v node (because it has been selected as a monitor)
                remove_nodes_from_cutset(u, v)

            # CASE 3: both u and v are virtual
            elif u in virtual_set and v in virtual_set:
                additional_monitors = virtualSubgraph(G, (u, v), virtual_set)
                remove_nodes_from_cutset(node for node in additional_monitors)
                virtual_monitors = virtual_monitors.union(additional_monitors)



    # Perform the normal vertex cover on the remaining parts
    monitor_set = chooseMs(cuts, source, target)
    if verbose:
        for node in virtual_set:
            if not str(node).startswith("RECV"):
                cprint(bcolors.FAIL, "CUTS:", starting_cut)
                cprint(bcolors.FAIL, "Virtual set:", virtual_set)
                break

        cprint(bcolors.OKCYAN, len(virtual_monitors), "Virtual Monitors:", virtual_monitors)
        cprint(bcolors.OKCYAN, len(monitor_set), "Non-virtual monitors:", monitor_set)
    return monitor_set.union(virtual_monitors), len(S)


if __name__ == "__main__":
    graph = nx.DiGraph()
    graph.add_edges_from(
        [(1, 2), (2, 5), (1, 4), (4, 9), (5, 7), (5, 6), (6, 8), (7, 8), (7, 13), (13, 14), (14, 7), (9, 7), (1, 3),
         (3, 10), (10, 3), (12, 11), (11, 1), (16, 14), (15, 16), (14, 15), (15, 14), (2, 17), (17, 18), (18, 11)])

    _virtual_set = [1, 2, 3, 5, 7, 15, 16, 17]
    final_set = set()

    for u, v in graph.edges():
        if u in _virtual_set and v in _virtual_set:
            final_set = final_set.union(virtualSubgraph(graph, (u, v), _virtual_set))

    print(final_set)
