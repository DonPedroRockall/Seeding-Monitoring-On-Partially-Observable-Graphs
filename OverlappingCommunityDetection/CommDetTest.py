import copy

import networkx
import numpy
from OverlappingCommunityDetection.CommunityDetector import *
from Utilities.DrawGraph import DrawGraph

graph_real = Graph()
graph_real.add_edge("A", "B")
graph_real.add_edge("A", "X")
graph_real.add_edge("A", "C")
graph_real.add_edge("A", "E")
graph_real.add_edge("B", "C")
graph_real.add_edge("B", "E")
graph_real.add_edge("B", "X")
graph_real.add_edge("C", "D")
graph_real.add_edge("C", "F")
graph_real.add_edge("D", "E")
graph_real.add_edge("E", "F")
graph_real.add_edge("E", "X")
graph_real.add_edge("F", "G")
graph_real.add_edge("G", "I")
graph_real.add_edge("G", "K")
graph_real.add_edge("G", "X")
graph_real.add_edge("H", "K")
graph_real.add_edge("H", "J")
graph_real.add_edge("I", "K")
graph_real.add_edge("J", "L")
graph_real.add_edge("J", "X")
graph_real.add_edge("K", "L")
graph_real.add_edge("K", "X")
graph_real.add_edge("L", "X")

graph = copy.copy(graph_real)
graph.remove_node("X")

graph_estimate, H = InfluentialNodeRecovery(graph, 4, 2, None, None, 1, "deg")

print("Recovered nodes:", H)
DrawGraph(graph_real)
DrawGraph(graph_estimate)

