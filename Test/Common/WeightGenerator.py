import random
import networkx as nx
from enum import Enum


def UniformWeights(graph: nx.DiGraph, attribute="weight", force=False, **kwargs):
    """Sets weight as Uniform Distribution Realizations. Pass 'min_val' and 'max_val' in kwargs to define boundaries"""
    SetRandomEdgeWeightsByLambda(graph,
                                 lambda arg: random.random() * (arg["max_val"] - arg["min_val"]) + arg["min_val"],
                                 attribute=attribute, force=force, **kwargs)


def InDegreeWeights(graph: nx.DiGraph, attribute="weight", force=False, **kwargs):
    """Sets weight as Gauss Distribution Realizations. No kwargs needed"""
    max_in_degree = max(graph.in_degree[node] for node in graph.nodes())
    for u, v, data in graph.edges(data=True):
        if not force and attribute in data:
            continue
        data[attribute] = graph.in_degree[v] / max_in_degree


def GaussWeights(graph: nx.DiGraph, attribute="weight", force=False, **kwargs):
    """Sets weight as Gauss Distribution Realizations. Pass 'mu' and 'sigma' in kwargs to define the distribution"""
    SetRandomEdgeWeightsByLambda(graph, lambda arg: random.gauss(arg["mu"], arg["sigma"]), attribute=attribute, force=force, **kwargs)


def SetRandomEdgeWeightsByLambda(graph: nx.DiGraph, Lambda, attribute="weight", force=False, **kwargs):
    """
     Sets random weights for the input graph
    :param graph:           The graph whose weights have to be randomized
    :param attribute        The weight attribute. Can be any string
    :param Lambda           The lambda function to use to sample random values
                            A function or a lambda can be passed as argument
    :param force            If True, then all edge weights are assigned, otherwise, only the edges that do not have the
                            attribute set will have their weights set
    :param kwargs           Args for the lambda function
    :return:                A Graph in which the attribute "attribute" is randomized
    """
    for u, v, data in graph.edges(data=True):
        if not force and attribute in data:
            continue
        data[attribute] = Lambda(kwargs)


class EWeightSetterFunction(Enum):
    EUniformWeights = {"name": "Uniform", "function": UniformWeights, "short_name": "UNIF"}
    EInDegreeWeights = {"name": "In-Degree", "function": InDegreeWeights, "short_name": "INDEG"}
    EGaussWeights = {"name": "Gauss", "function": GaussWeights, "short_name": "GAUSS"}
    ECustom = {"name": "Custom", "function": SetRandomEdgeWeightsByLambda, "short_name": "CUSTOM"}
