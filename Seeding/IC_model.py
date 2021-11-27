from joblib import Parallel, delayed

import networkx as nx
import random
from Utilities.PrintResultsSeeding import Avg
from Test.Common.Utility import SetRandomEdgeWeights, IsVirtualNode


# for the IC model, we only need the edge weights (assigned randomly in the range [0, 0.1]);
# in the following, the adjectives 'influenced' and 'activated' are used interchangeably


def InitGraphParametersIC(G: nx.DiGraph):
    # SetRandomEdgeWeights(G, attribute="weight", distribution="smallrand")
    for u, v, data in G.edges(data=True):
        data['weight'] = random.uniform(0, 0.1)  # as seen in the literature


def RunIC(G: nx.DiGraph, seeds: list):
    influence = {}

    for node in G.nodes():
        influence[node] = 0

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

    return len(active_nodes)


# PARALLEL VERSION: it is used as part of the Voterank parallelization, for which we run the Voterank itself just
# one time, and then evaluate the mean over num_run executions of IC with the seed set returned by the algorithm
def ParallelRunIC(G: nx.DiGraph, seeds: list, num_cores=4, num_run=40):
    active_per_run = Parallel(n_jobs=num_cores)(delayed(RunIC)(G, seeds)
                                                for _ in range(num_run))
    return Avg(active_per_run)


###################### SEED SELECTION ALGORITHMS ######################

# [PROVABLE GUARANTEE] computes the best seed set of size at most k: at each iteration, it runs several IC processes
# and selects the node which has the best gain in terms of newly influenced nodes, and then adds it to the seed set.
# The algorithm stops either at the end of the for-loop or when no new node has been infected
def SIMBasicGreedy(G: nx.DiGraph, k, restricted_set=None):
    seed_set = []
    active_nodes = 0

    for i in range(k):
        max_gain = 0
        max_node = None
        is_there_new_activated = False

        for node in G.nodes():
            if node not in seed_set and ((restricted_set is None) or (node not in restricted_set)):
                seed_set.append(node)
                gain = RunIC(G, seed_set) - active_nodes
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


# PARALLEL VERSION: we can't parallelize any code snippet in the function itself, given its nature: RunIC has to be
# executed sequentially, i.e. it depends on the previous RunIC executions in the same algorithm.
# For that reason, we run the whole algorithm in parallel num_run times, and evaluated the mean of the activated nodes
# over all runs
def ParallelSIMBasicGreedy(G: nx.DiGraph, k, num_cores=4, num_run=40, restricted_set=None):
    active_per_run = Parallel(n_jobs=num_cores)(delayed(SIMBasicGreedy)(G, k, restricted_set)
                                                for _ in range(num_run))
    return Avg(active_per_run)


# [HEURISTIC] computes the best nodes based on vote-rank centrality, with an upper limit given by the value k;
# then, IC is run and the activated nodes at the end are shown
def SIMVoterank(G: nx.DiGraph, k, restricted_nodes=None):
    vote_rank = nx.voterank(G)

    if restricted_nodes is not None:
        for node in vote_rank:
            if IsVirtualNode(node, restricted_nodes):
                vote_rank.remove(node)

    if len(vote_rank) > k:
        temp = []
        for i in range(k):
            temp.append(vote_rank[i])
        vote_rank = temp

    return vote_rank
