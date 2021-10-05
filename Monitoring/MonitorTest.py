import networkx as nx
from Monitor import *
from Utilities.DrawGraph import DrawGraph

graph = nx.DiGraph()
graph.add_edge(1, 2, weight=1)
graph.add_edge(2, 3, weight=1)
graph.add_edge(3, 4, weight=1)
graph.add_edge(4, 5, weight=0.9)
graph.add_edge(5, 6, weight=0)
graph.add_edge(6, 1, weight=0)

cGraph = GetInfectedSubgraph(graph, [1, 2])
DrawGraph(cGraph)

