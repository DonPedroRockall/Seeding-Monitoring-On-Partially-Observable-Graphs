import copy
import random

import networkx as nx


def independent_cascade(G, seeds, monitors=[], steps=0):
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
        return _diffuse_all(DG, A, monitors)
    # perform diffusion for at most "steps" rounds
    return _diffuse_k_rounds(DG, A, steps, monitors)


def _diffuse_all(G, A, monitors=[]):
    tried_edges = set()
    layer_i_nodes = [[i for i in A]]
    while True:
        len_old = len(A)
        (A, activated_nodes_of_this_round, cur_tried_edges) = \
            _diffuse_one_round(G, A, tried_edges, monitors)
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(A) == len_old:
            break
    return layer_i_nodes


def _diffuse_k_rounds(G, A, steps, monitors=[]):
    tried_edges = set()
    layer_i_nodes = [[i for i in A]]
    while steps > 0 and len(A) < len(G):
        len_old = len(A)
        (A, activated_nodes_of_this_round, cur_tried_edges) = _diffuse_one_round(G, A, tried_edges, monitors)
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(A) == len_old:
            # print "Step n°",steps
            break
        # print "STEP ",steps
        steps -= 1
    return layer_i_nodes


def _diffuse_one_round(G, A, tried_edges, monitors=[]):
    activated_nodes_of_this_round = set()
    cur_tried_edges = set()
    for s in A:
        for nb in G.successors(s):
            if nb in A or (s, nb) in tried_edges or (s, nb) in cur_tried_edges or nb in monitors:
                continue
            if _prop_success(G, s, nb):
                activated_nodes_of_this_round.add(nb)
            cur_tried_edges.add((s, nb))
    activated_nodes_of_this_round = list(activated_nodes_of_this_round)
    A.extend(activated_nodes_of_this_round)
    return A, activated_nodes_of_this_round, cur_tried_edges


def _prop_success(G, src, dest):
    return random.random() <= G[src][dest]['weight']
