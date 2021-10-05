from KronFit.KroneckerFit import *
import networkx as nx

graph: nx.Graph
graph = nx.read_edgelist("../Datasets/gnutella_30.txt")
theta = KronFit(graph, 2)
print(theta)