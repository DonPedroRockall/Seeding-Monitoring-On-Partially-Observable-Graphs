import networkx as nx
from pyvis.network import Network
import random

def ShowGraph(G: nx.DiGraph):
    net = Network(notebook=True)
    net.from_nx(G)
    for node in net.nodes:
        node["physics"] = False
    for edge in net.edges:
        edge["physics"] = False
    net.show("example.html")

# randomly selects a value in the range [0, 1] for each edge of graph G. This is used e.g. in the Basic Greedy
# algorithm to resolve an important problem related to the social influence function: each time the IC is run,
# the values to compare to the edge weights are chosen randomly, so they always change. This renders the social
# influence function not monotone (which is an assumption made for the algorithm), so the scope of this function is
# to select the values a priori before running the Basic Greedy, and then to use them to run the IC at each iteration
def FixEdgeProb(G: nx.DiGraph):
    edge_values = {}
    for edge in G.edges():
        edge_values[edge] = random.uniform(0, 1)
    return edge_values