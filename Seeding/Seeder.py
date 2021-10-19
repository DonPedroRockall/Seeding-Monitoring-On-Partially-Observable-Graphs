import math

import networkx
import numpy


def BasicGreedy(Graph: networkx.Graph, k):
    """

    :param Graph:
    :param k:
    :return:
    """

    # Initialization step
    S = set()
    for i in range(1, k):
        max = -math.inf
        argmax = None
        for u in set(Graph.nodes()).difference(S):
            diff = SocialInfluenceFunction() - SocialInfluenceFunction()
            if diff > max:
                max = diff
                argmax = u
        S = S.union(argmax)
    return S


def BudgetedInfluenceMaximization(Graph: networkx.Graph, cost_function, B):
    """

    :param Graph:
    :param cost_function:
    :param B:
    :return:
    """

    S1 = BasicGreedy(Graph, B)  # TODO: check if B has to be sent as parameter to BasicGreedy



def SocialInfluenceFunction():
    """
    Also called sigma function, it returns the social influence of a node. For any given S, sigma(S) returns the set of
    influenced nodes, when the diffusion process is over
    :return:
    """
    pass