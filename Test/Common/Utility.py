import random
import networkx as nx


def GetVirtualNodesByLabel(part: nx.DiGraph, recv: nx.DiGraph):
    """
    Returns the set of virtual nodes. Virtual nodes are defined as the nodes that are present in recv graph but not in
    part graph. A node is present if it has the same label
    :param part:        The Partial graph
    :param recv:        The Recovered graph
    :return:
    """
    virtuals = set()
    for node in recv.nodes():
        if node not in part.nodes():
            virtuals.add(node)
    return virtuals


def SetSameWeightsToOtherGraphs(original_graph: nx.Graph, other_graphs: list):
    """
    Copies all the attributes of original_graph to all the other graphs in other_graphs list, without altering the
    structure of the graph(s) itself. (I.E.: it will not create new nodes or new edges, every graph will stay the same)
    :param original_graph:
    :param other_graphs:
    :return:
    """
    for u, v, data in original_graph.edges(data=True):
        for graph in other_graphs:
            if graph.has_edge(u, v):
                for key in data:
                    graph[u][v][key] = data[key]


def SetRandomEdgeWeights(graph: nx.DiGraph, attribute="weight", distribution="uniform", force=False, *args):
    """
     Sets random weights for the input graph
    :param graph:           The graph whose weights have to be randomized
    :param attribute        The weight attribute. Can be any string
    :param distribution     The distribution to use to sample random values
                            Valid values are: "uniform", "gauss", "betavariate", "expovariate", "lognormal".
                            Any value not specified here will result in uniform distribution
    :param force            If True, then all edge weights are assigned, otherwise, only the edges that do not have the
                            attribute set will have their weights set
    :param args             An array of parameters that describe the distribution
                            -uniform:
                            --(minimum_value, maximum_value)
                            -gaussian:
                            --(mean, standard_deviation)
                            -betavariate
                            --(alpha, beta)
                            -expovariate
                            --(lambda)
                            -indegree
                            --Widely used in Literature for Independent Cascade, sets the weight as the normalized
                                in-degree value of the node the edge is pointing to
                            -smallrand
                            --Also used in literature, uses a random uniform from 0 to 0.1 instead from 0 to 1
    :return:                A Graph in which the attribute "attribute" is randomized
    """

    if distribution == "gauss":
        for u, v, data in graph.edges(data=True):
            if not force and attribute not in data:
                data[attribute] = random.gauss(args[0], args[1])

    elif distribution == "betavariate":
        for u, v, data in graph.edges(data=True):
            if not force and attribute not in data:
                data[attribute] = random.betavariate(args[0], args[1])

    elif distribution == "gammavariate":
        for u, v, data in graph.edges(data=True):
            if not force and attribute not in data:
                data[attribute] = random.gammavariate(args[0], args[1])

    elif distribution == "expovariate":
        for u, v, data in graph.edges(data=True):
            if not force and attribute not in data:
                data[attribute] = random.expovariate(args[0])

    elif distribution == "lognormal":
        for u, v, data in graph.edges(data=True):
            if not force and attribute not in data:
                data[attribute] = random.lognormvariate(args[0], args[1])

    elif distribution == "indegree":
        max_in_degree = max(graph.in_degree[node] for node in graph.nodes())
        for u, v, data in graph.edges(data=True):
            if not force and attribute not in data:
                data[attribute] = graph.in_degree[v] / max_in_degree

    elif distribution == "smallrand":
        for u, v, data in graph.edges(data=True):
            if not force and attribute not in data:
                data[attribute] = float(random.random() * 0.1)

    else:
        for u, v, data in graph.edges(data=True):
            if not force and attribute not in data:
                data[attribute] = random.uniform(args[0], args[1])


def SetRandomEdgeWeightsByDistribution(graph: nx.DiGraph, distribution, attribute="weight", force=False):
    """
     Sets random weights for the input graph
    :param graph:           The graph whose weights have to be randomized
    :param attribute        The weight attribute. Can be any string
    :param distribution     The distribution to use to sample random values
                            A function or a lambda can be passed as argument
    :param force            If True, then all edge weights are assigned, otherwise, only the edges that do not have the
                            attribute set will have their weights set

    :return:                A Graph in which the attribute "attribute" is randomized
    """

    for u, v, data in graph.edges(data=True):
        if not force and attribute in data:
            continue
        data[attribute] = distribution()
