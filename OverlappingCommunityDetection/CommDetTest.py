import copy

import networkx
import numpy
from OverlappingCommunityDetection.CommunityDetector import *
from Test.Common.HidingFunctions import TotalNodeClosure
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
from definitions import ROOT_DIR

full = networkx.read_edgelist(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote.txt")
# full = networkx.read_edgelist(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote-PART.txt")
# full = networkx.read_edgelist(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote-RECV.txt")

part = TotalNodeClosure(full.copy(), [1, 2, 3, 4, 5])

