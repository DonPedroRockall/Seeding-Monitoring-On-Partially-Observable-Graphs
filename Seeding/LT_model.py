import math

import networkx as nx
import numpy
import random


# For the LT model, we need node thresholds (assigned randomly in the range [0, 1]), and also need the sum of the
# in-edge weights for each node to be less than 1

def InitGraphParametersLT(G: nx.DiGraph):
    node_thresh = {}
    node_influence = {} # da mettere in RunLT

    for node in G.nodes():
        node_thresh[node] = random.uniform(0, 1)
        node_influence[node] = 0
        for u, v, data in G.in_edges(node, data=True):
            data['weight'] = random.uniform(0, 1)

        e_sum = sum(data['weight'] for u, v, data in G.in_edges(node, data=True))
        # if the sum of in_edges is > 1, the weights are normalized
        if e_sum > 1:
            for u, v, data in G.in_edges(node, data=True):
                data['weight'] /= e_sum
