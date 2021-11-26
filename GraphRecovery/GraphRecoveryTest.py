import copy
import random

import networkx
import numpy
from joblib import Parallel, delayed

from GraphRecovery.GraphRecovery import *
from Test.Common.DatasetGenerator import GenerateRandomGraphTriple, CheckPartFull, CheckPartRecv
from Test.Common.DistributionFunctions import DegreeDistribution, UniformDistribution
from Test.Common.GraphGenerator import GNCConnectedDirectedGraph
from Test.Common.HidingFunctions import TotalNodeClosure
from Test.Common.Utility import SetRandomEdgeWeightsByDistribution, SetSameWeightsToOtherGraphs
from Utilities.DrawGraph import DrawGraph

# full = networkx.DiGraph()
# full.add_edge(0, 1)
# full.add_edge(0, 12)
# full.add_edge(0, 2)
# full.add_edge(0, 4)
# full.add_edge(1, 2)
# full.add_edge(1, 4)
# full.add_edge(1, 12)
# full.add_edge(2, 3)
# full.add_edge(2, 5)
# full.add_edge(3, 4)
# full.add_edge(4, 5)
# full.add_edge(4, 12)
# full.add_edge(5, 6)
# full.add_edge(6, 8)
# full.add_edge(6, 10)
# full.add_edge(6, 12)
# full.add_edge(7, 10)
# full.add_edge(7, 9)
# full.add_edge(8, 10)
# full.add_edge(9, 11)
# full.add_edge(9, 12)
# full.add_edge(10, 11)
# full.add_edge(10, 12)
# full.add_edge(11, 12)
# full.add_edges_from([
#     (12,13),
#     (15,12),
#     (13,14),
#     (14,7),
#     (4,13),
#     (15,9),
# ])

# full = networkx.generators.gnc_graph(100, create_using=networkx.DiGraph)
#
# part = copy.copy(full.copy())
# part.remove_node(25)
# part.remove_node(50)
# part.remove_node(75)
# part.remove_node(1)
# part.remove_node(99)
# part.remove_node(10)
# part.remove_node(90)
#
# print(full.number_of_nodes())
# print(part.number_of_nodes())
#
# inf_tresh = sum(deg for node, deg in part.degree() if deg > 0) / float(part.number_of_nodes())
#
# recv, H = InfluentialNodeRecovery(copy.copy(part.copy()), 7, 7, None, None, inf_tresh, "deg")
#
# print("____________________________")
# print(full.number_of_nodes())
# print(part.number_of_nodes())
# print(recv.number_of_nodes())
#
# print("HIDDEN NODES")
# for node in full.nodes():
#     if node not in part.nodes():
#         print(node)
#
# print("RECOVERED NODES")
# for node in recv.nodes():
#     if node not in part.nodes():
#         print(node)
#
# print("Recovered nodes:", H)
# DrawGraph(full, graph_name="full")
# DrawGraph(part, graph_name="part")
# DrawGraph(recv, graph_name="recv")


# full = networkx.read_edgelist(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote.txt")
# full = networkx.read_edgelist(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote-PART.txt")
# full = networkx.read_edgelist(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote-RECV.txt")

# part = TotalNodeClosure(full.copy(), [1, 2, 3, 4, 5])


def CheckMatrix(A, Ar):
    N = len(A)
    M = len(A[0])
    for x in range(N):
        for y in range(M):
            if A[x, y] != Ar[x, y]:
                return False
    return True


# full: networkx.DiGraph = networkx.generators.gnc_graph(1500, create_using=networkx.DiGraph)
# part = full.copy()
# to_remove = random.sample(list(full.nodes()), 500)
# for node in to_remove:
#     part.remove_node(node)
#
#
# recv, _ = InfluentialNodeRecovery(part, 300, 2)

def test():

    full, part, recv = GenerateRandomGraphTriple(1500, 300, GNCConnectedDirectedGraph, UniformDistribution, TotalNodeClosure, None)

    CheckPartFull(full, part)
    CheckPartRecv(recv, part)

    # # SetRandomEdgeWeights(full, "weight", self.WEIGHT, True, *[0, 1])
    # SetRandomEdgeWeightsByDistribution(full, lambda: random.random() * 0.1, attribute="weight", force=True)
    # # Copy the weights to the other two graphs
    # SetSameWeightsToOtherGraphs(full, [part, recv])
    # # Assign random edges to the newly reconstructed edges
    # # SetRandomEdgeWeights(recv, "weight", self.WEIGHT, False, *[0, 1])
    # SetRandomEdgeWeightsByDistribution(recv, lambda: random.random() * 0.1, attribute="weight", force=False)

    number_of_recovered_nodes = recv.number_of_nodes() - part.number_of_nodes()
    number_of_hidden_nodes = full.number_of_nodes() - part.number_of_nodes()

    print("NUMBER OF RECOVERED NODES:", number_of_recovered_nodes)
    print("NUMBER OF HIDDEN NODES:", number_of_hidden_nodes)
    print("HIDDEN NODES", sorted(set(full.nodes()).difference(set(part.nodes()))))
    print("RECOVERED NODES", sorted(set(recv.nodes()).difference(set(part.nodes()))))
    print("MISSING NODES", sorted(set(part.nodes()).difference(set(recv.nodes()))))

    print(part.number_of_nodes())
    print(full.number_of_nodes())
    print(recv.number_of_nodes())

    print(full.nodes())
    print(part.nodes())
    print(recv.nodes())

    Afull = networkx.convert_matrix.to_numpy_matrix(full)
    Apart = networkx.convert_matrix.to_numpy_matrix(part)
    Arecv = networkx.convert_matrix.to_numpy_matrix(recv)

    print(CheckMatrix(Apart, Afull))
    print(CheckMatrix(Apart, Arecv))


if __name__ == "__main__":
    Parallel(n_jobs=1)(delayed(test)() for _ in range(1))
