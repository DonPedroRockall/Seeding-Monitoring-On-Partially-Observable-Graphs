import random
import numpy
import os
import networkx as nx
from networkx import DiGraph
from definitions import SNAP_EXE_PATH


def KronFit(graph: DiGraph, n0, theta: numpy.array = None, gd_iterations=None, lr=None, min_gd_step=None,
            max_gd_step=None, n_warmup=None, n_samples=None, swap_prob=None):
    """
    Performs a KronFit on a given graph
    :param graph:               The graph to be fitted
    :param n0:                  Linear dimension of parameter matrix (will be of size n0 x n0)
    :param theta:               Initial values for parameter matrix. None for random
    :param gd_iterations:       Gradient Descent iterations. None for default (50)
    :param lr:                  Learning Rate. None for default (1e-5)
    :param min_gd_step:         Minimum Gradient Descent Step. None for default
    :param max_gd_step:         Maximum Gradient Descent Step. None for default
    :param n_warmup:            Number of Warm-up samples. None for default (10k)
    :param n_samples:           Number of Samples per iteration. None for default (10k)
    :param swap_prob:           Probability of Swapping Nodes instead of Swapping Edges' endpoints. None for default (0.6)
    :return:                    The estimated parameter matrix Theta
    """
    # Relabel nodes
    graph = nx.relabel.convert_node_labels_to_integers(graph)

    # Change dir
    os.chdir(SNAP_EXE_PATH)

    # Write edge list to file
    nx.write_edgelist(graph, "graph.txt", data=False)



    # Issue cmd command
    os.system("cmd /c kronfit.exe -i:graph.txt -n0:{0} -m:{1} {2} {3} {4} {5} {6} {7} {8}"
              .format(n0,
                      "R" if theta is None else TransformTheta(theta),
                      "" if gd_iterations is None else "-gd:" + str(gd_iterations),
                      "" if lr is None else "-l:" + str(lr),
                      "" if min_gd_step is None else "-mns:" + str(min_gd_step),
                      "" if max_gd_step is None else "-mxs:" + str(max_gd_step),
                      "" if n_warmup is None else "-w:" + str(n_warmup),
                      "" if n_samples is None else "-s:" + str(n_samples),
                      "" if swap_prob is None else "-nsp:" + str(swap_prob)))

    theta_final = numpy.zeros(shape=(n0, n0))

    # Open the file that contains the Theta and read it
    with open('graph-fit' + str(n0)) as f:
        last_line = f.readlines()[-1]
    result = (last_line.split("[")[-1])[:-2]
    params = result.split("; ")

    # Build the Theta matrix
    for i, p in enumerate(params):
        temp = p.split(", ")
        for j, t in enumerate(temp):
            theta_final[i, j] = float(t)

    # Remove leftover files
    os.remove("graph-fit" + str(n0))
    os.remove("graph.txt")

    return theta_final


def TransformTheta(theta):
    """
    Transforms a numpy array into a string format compatible for the KronFit algorithm
    :param theta:       Numpy array to be transformed
    :return:            String representation of the numpy array
    """
    string = ""
    for i in range(len(theta)):
        for j in range(len(theta[0])):
            string += str(theta[i, j]) + (" " if j != len(theta[0]) - 1 else "")
        string += "; " if i != len(theta) - 1 else ""

    return string


def GenerateSKG(Theta: numpy.ndarray, K):
    """
    Generates a Stochastic Adjacency Matrix by Kronecker Product performed recursively K times on the matrix Theta
    :param Theta:       Generates a Stochastic Kronecker Matrix from a parameter matrix
    :param K:           Power of Kronecker product
    :return:            The Stochastic Kronecker Matrix, to be interpreted as a Stochastic Adjacency Matrix
    """
    P = Theta
    while K > 1:
        P = numpy.kron(P, Theta)
        K -= 1
    return P


def InstantiateGraph(P):
    """Given a Stochastic Adjacency Matrix returns an instance of such matrix by performing a Bernoulli draw for each
    of the elements and returning the resulting instance of the adjacency matrix
    :param P:           The Stochastic Kronecker Matrix to be instantiated
    """
    H = len(P)
    AdjMat = numpy.zeros(shape=(H, H))
    for i in range(H):
        for j in range(H):
            if random.random() < P[i, j]:
                AdjMat[i, j] = 1
    return AdjMat
