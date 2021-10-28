import math

import networkx as nx
import numpy
import random
from Utilities.DrawGraph import DrawGraph

# for the IC model, we only need the edge weights (assigned randomly in the range [0, 1]);
# in the following, the adjectives 'influenced' and 'activated' are used interchangeably

influence = {}


def InitGraphParametersIC(G: nx.DiGraph):
    for u, v, data in G.edges(data=True):
        data['weight'] = random.uniform(0, 1)
    for node in G.nodes():
        influence[node] = 0


def RunIC(G: nx.DiGraph, seeds: list):
    active_nodes = []

    # set all the seeds' influence to 1 and add them to active_nodes
    for s in seeds:
        influence[s] = 1
        active_nodes.append(s)

    is_there_new_influenced = True
    # continue while no new node has been influenced in the previous iteration
    while is_there_new_influenced:
        is_there_new_influenced = False
        # for each active node, influence its inactive neighbors with a certain probability
        for node in active_nodes:
            for u, v, data in G.out_edges(node, data=True):
                if random.uniform(0, 1) < data['weight'] and influence[v] == 0:
                    influence[v] = 1
                    active_nodes.append(v)
                    is_there_new_influenced = True

    # reset influence to 0 for all nodes for a later use of the algorithm
    for node in G.nodes():
        influence[node] = 0

    return active_nodes


###################### SEED SELECTION ALGORITHMS ######################

# [PROVABLE GUARANTEE] computes the best seed set of size at most k: at each iteration, it runs several IC processes
# and selects the node which has the best gain in terms of newly influenced nodes, and then adds it to the seed set.
# The algorithm stops either at the end of the for-loop or when no new node has been infected
def SIMBasicGreedy(G: nx.DiGraph, k):
    seed_set = []
    active_nodes = 0

    for i in range(k):
        max_gain = 0
        max_node = None
        is_there_new_activated = False
        for node in G.nodes():
            if node not in seed_set:
                seed_set.append(node)
                gain = len(RunIC(G, seed_set)) - active_nodes
                seed_set.pop()
                if gain > max_gain:
                    max_gain = gain
                    max_node = node
                    is_there_new_activated = True
        if not is_there_new_activated:
            break
        seed_set.append(max_node)
        active_nodes += max_gain

    return RunIC(G, seed_set)

'''
    print("############### BASIC GREEDY ###############")
    print("Basic Greedy seed set: ", seed_set)
    activated = RunIC(G, seed_set)
    print("Activated nodes: ", activated)
    print("Num. of activated nodes: ", len(activated))
    print("############################################")'''


# [HEURISTIC] computes the best nodes based on vote-rank centrality, with an upper limit given by the value k;
# then, IC is run and the activated nodes at the end are shown
def SIMVoterank(G: nx.DiGraph, k):
    vote_rank = nx.voterank(G)
    if len(vote_rank) > k:
        temp = []
        for i in range(k):
            temp.append(vote_rank[i])
        vote_rank = temp
    return RunIC(G, vote_rank)

'''
    print("############### VOTERANK ###############")
    print("Vote-rank seed set: ", vote_rank)
    activated = RunIC(G, vote_rank)
    print("Activated nodes: ", activated)
    print("Num. of activated nodes: ", len(activated))
    print("########################################")'''
