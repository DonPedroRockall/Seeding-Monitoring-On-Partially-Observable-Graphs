import numpy
import networkx
from KronFit.KroneckerFit import *


def InfluentialNodeRecovery(graph: networkx.DiGraph, M, N0, alpha=None, beta=None, epsilon=0, centrality="deg"):
    """
    Estimates and recovers a number of influential nodes from a partially observable graph
    :param graph:           The original observable graph
    :param M:               The maximum number of influential nodes to recover
    :param N0:              Dimension of the Kronecker initiator matrix
    :param alpha:           Parameter used for Katz centrality
    :param beta:            Parameter used for Katz centrality
    :param epsilon:         Minimum value of Katz centrality that a node has to have to be considered influential
    :param centrality:      The Centrality Measure to use. Options: "katz", "deg". Defaults to "deg" (Degree centrality)
    :return:                A Graph with the recovered nodes connected to it and how many nodes have been connected
    """
    # Transform graph to directed if necessary
    graph_was_directed = True
    if not networkx.is_directed(graph):
        graph = graph.to_directed()
        graph_was_directed = False

    # Remap node labels to be integers
    observable_graph: DiGraph = networkx.relabel.convert_node_labels_to_integers(graph)
    # Count the number of observable nodes
    N = observable_graph.number_of_nodes()
    # Transforms the (partially observable) graph into adjacency matrix
    A = networkx.convert_matrix.to_numpy_matrix(observable_graph)
    # Performs Graph recovery to recover the graph
    Ar = GraphRecv(A, N0, M)

    # Performs node selection to remove non-influential nodes
    H, r = NodeSelect(Ar, N, M, alpha, beta, epsilon, centrality=centrality)
    # Estimates the new graph by connecting the nodes
    estimated_graph = ConnectNodes(graph, Ar, r)

    # If necessary, transform back the graph to undirected
    if not graph_was_directed:
        estimated_graph = networkx.to_undirected(estimated_graph)

    return estimated_graph, H


def ConnectNodes(graph: networkx.DiGraph, Ar: numpy.ndarray, r):
    """
    Connects Recovered nodes to the Original observable graph by using the recovered adjacency matrix Ar
    :param graph:   The original observable graph
    :param Ar:      The Adjacency Matrix of the recovered graph
    :param r:       Ranking vector of the recovered influential nodes
    :return:        A Graph with the recovered nodes connected to it
    """
    N = graph.number_of_nodes()
    for column in range(len(Ar[0]) - 1, N - 1, -1):
        if column not in r.keys():
            Ar = numpy.delete(Ar, column, axis=1)
            Ar = numpy.delete(Ar, column, axis=0)
    recovered_graph = networkx.convert_matrix.from_numpy_matrix(Ar, create_using=networkx.DiGraph)
    return recovered_graph


def GraphRecv(A, N0, M):
    """
    Performs the Graph Recovery function by the means of KronFit algorithm.
    source: https://deepai.org/publication/community-detection-in-partially-observable-social-networks

    :param A:               Adjacency matrix of the observable graph
    :param N0:              Number of Initiator Matrix Nodes (Initiator Matrix will be N0xN0)
    :param M:               Number of missing nodes
    :return:                The Adjacency Matrix of the Full Recovered Graph
    """

    # Get the graph and the number of nodes from the adjacency matrix
    graph: networkx.DiGraph = networkx.from_numpy_matrix(A, create_using=networkx.DiGraph)
    N = graph.number_of_nodes()

    # Compute the necessary power of the kronecker product K
    K = 0
    while N0 ** K < N + M:
        K += 1

    # Perform KronFit on graph and generate a Stochastic Kronecker Graph
    Theta = KronFit(graph, N0)
    P = GenerateSKG(Theta, K)

    # Instantiate the graph by performing Bernoulli realizations
    for u in range(0, len(P)):
        for v in range(0, len(P)):
            if u < N and v < N:  # Copy the original elements
                P[u, v] = A[u, v]
            elif v >= u:
                P[u, v] = Bernoulli(P[u, v])  # Perform realizations for the new elements
                P[v, u] = P[u, v]  # The matrix is symmetrical, so mirror it

    return P


def NodeSelect(Ar: numpy.ndarray, N, M, alpha=None, beta=None, epsilon=0, centrality="deg"):
    """
    Selects the nodes of the recovered adjacency matrix, since not all of them will be influential

    :param Ar:          Recovered Adjacency Matrix
    :param N:           Number of observable nodes
    :param M:           Number of missing nodes
    :param alpha:       Parameter used for Katz centrality, set as None for default
    :param beta:        Parameter used for Katz centrality, set as None for default
    :param epsilon:     Minimum value of Katz centrality that a node has to have to be considered influential
    :param centrality:  The centrality measure to use. Valid values: "deg" and "katz"
    :return H:          Maximum number of influential nodes to be selected
    :return r:          Ranking vector node -> centrality (ordered on keys)
    """

    # Graph now contains the full recovered graph
    graph: DiGraph = networkx.from_numpy_matrix(Ar, create_using=networkx.DiGraph)

    # Default parameter configuration
    if alpha is None:
        eigen = networkx.adjacency_spectrum(graph)
        # Default alpha to the largest eigenvalue of the adj matrix of the graph
        alpha = max(eigen)
    if beta is None:
        # Default beta to 1
        beta = 1

    # Initialization step
    Cen = dict()
    if centrality == "katz":
        Centrality = networkx.algorithms.centrality.katz_centrality(graph, alpha, beta, normalized=True)
    else:
        # Compute un-normalized degree centrality
        Centrality = dict()
        for node in graph.nodes():
            Centrality[node] = graph.degree[node]

    # Build the Cen dictionary
    rank_index = 0
    for node in dict(sorted(Centrality.items(), key=lambda item: -item[1])):  # The minus sign is for descending order
        Cen[rank_index] = node
        rank_index += 1

    # while abs(Cen_temp - Cen) > Ni_select:    |
    #     Cen_temp = Cen                        | -> TODO: not fully understood, might not be useful
    #     Cen = alpha * Cen * Ar + beta         |

    # Recover at most M influential nodes
    H = 0
    ranking = dict()
    search_index = 0
    for _ in range(M):
        while Cen[search_index] < N:
            search_index += 1
        node_highest_ranking = Cen[search_index]
        if Centrality[node_highest_ranking] < epsilon:
            break
        H += 1
        ranking[node_highest_ranking] = Centrality[node_highest_ranking]
        search_index += 1

    return H, ranking


def Bernoulli(float_value):
    return 1 if random.random() < float_value else 0




