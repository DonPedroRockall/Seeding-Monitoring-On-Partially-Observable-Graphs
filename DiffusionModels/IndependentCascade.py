import copy
import random
import networkx as nx


def IndependentCascadeWithMonitors(G: nx.DiGraph, seeds, monitors=[], steps=0):
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("independent_cascade() is not defined for graphs with multiedges.")

    for s in seeds:
        if s not in G.nodes():
            raise Exception("seed", s, "is not in graph")

    # change to directed graph
    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)

    # perform diffusion
    A = copy.deepcopy(seeds)  # prevent side effect
    if steps <= 0:
        # perform diffusion until no more nodes can be activated
        return _DiffuseAll(DG, A, monitors)
    # perform diffusion for at most "steps" rounds
    return _DiffuseKRounds(DG, A, steps, monitors)


def _DiffuseAll(G, A, monitors=[]):
    tried_edges = set()
    layer_i_nodes = [[i for i in A]]
    while True:
        len_old = len(A)
        (A, activated_nodes_of_this_round, cur_tried_edges) = \
            _DiffuseOneRound(G, A, tried_edges, monitors)
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(A) == len_old:
            break
    return layer_i_nodes


def _DiffuseKRounds(G, A, steps, monitors=[]):
    tried_edges = set()
    layer_i_nodes = [[i for i in A]]
    while steps > 0 and len(A) < len(G):
        len_old = len(A)
        (A, activated_nodes_of_this_round, cur_tried_edges) = _DiffuseOneRound(G, A, tried_edges, monitors)
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(A) == len_old:
            # print "Step n°",steps
            break
        # print "STEP ",steps
        steps -= 1
    return layer_i_nodes


def _DiffuseOneRound(G, A, tried_edges, monitors=[]):
    activated_nodes_of_this_round = set()
    cur_tried_edges = set()
    for s in A:
        for nb in G.successors(s):
            if nb in A or (s, nb) in tried_edges or (s, nb) in cur_tried_edges or nb in monitors:
                continue
            if _PropSuccess(G, s, nb):
                activated_nodes_of_this_round.add(nb)
            cur_tried_edges.add((s, nb))
    activated_nodes_of_this_round = list(activated_nodes_of_this_round)
    A.extend(activated_nodes_of_this_round)
    return A, activated_nodes_of_this_round, cur_tried_edges


def _PropSuccess(G, src, dest):
    return random.random() <= G[src][dest]['weight']


# -------------------------------------------------------------------------------

def GetInfectedSubgraph(graph, infected_nodes):
    subgraph = nx.DiGraph()
    for node in infected_nodes:
        singleton = True
        for out_node in graph.out_edges([node]):
            if out_node[1] in infected_nodes:
                subgraph.add_edge(node, out_node[1], weight=graph[node][out_node[1]]['weight'])
                singleton = False
        if singleton:
            subgraph.add_node(node)
    return subgraph


def GetInfectedSubgraphs(graph: nx.DiGraph, infected_nodes):
    infected_graph = GetInfectedSubgraph(graph, infected_nodes)
    subgraphs = []
    components = list(nx.components.weakly_connected_components(infected_graph))
    if len(components) == 1:
        subgraphs.append(infected_graph)
        return subgraphs
    for comp in components:
        subgraph = nx.DiGraph()
        for node in comp:
            singleton = node
            for out_node in infected_graph.out_edges([node]):
                if out_node[1] in comp:
                    subgraph.add_edge(node, out_node[1], weight=infected_graph[node][out_node[1]]['weight'])
        if subgraph.size() == 0:
            subgraph.add_node(singleton)
        subgraphs.append(subgraph)
    return subgraphs


def AdaptiveCascade(graph, random_sources, steps, interval):
    i_low = 0
    i_high = 0
    no_inf = 0
    i = 0
    while True:
        if i > 35:
            return None
        if steps == 0:
            return None

        ic = IndependentCascadeWithMonitors(G=graph, seeds=random_sources, monitors=[], steps=steps)
        infected_nodes = set()
        for sublist in ic:
            infected_nodes = infected_nodes.union(sublist)

        if no_inf > 2:
            return None

        if infected_nodes is None:
            no_inf += 1
            continue
            # print ('Len infected nodes',len(infected_nodes),i_low, i_high, steps,interval)
        if len(infected_nodes) < interval[0]:
            i_low += 1
        else:
            if len(infected_nodes) > interval[1]:
                i_high += 1
            else:
                return infected_nodes
        i += 1

        if infected_nodes is not None:
            if i_low > 4:
                steps += 1
                i_low = 0
                i_high = 0
            if i_high > 4:
                steps -= 1
                i_low = 0
                i_high = 0

# -------------------------------------------------------------------------------
