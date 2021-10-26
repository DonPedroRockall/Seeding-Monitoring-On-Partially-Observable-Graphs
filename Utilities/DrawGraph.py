import networkx as nx
import pyvis.options
from pyvis.network import Network


def DrawGraph(graph, color_dict=None, graph_name="graph", physics=True):

    if color_dict is not None:
        for node in color_dict:
            graph.nodes[node]["color"] = color_dict[node]

    nt = Network("1080px", "1920px")
    nt.from_nx(graph)
    if not physics:
        for node in nt.nodes:
            node["physics"] = False
        for edge in nt.edges:
            edge["physics"] = False
    nt.show(graph_name + ".html")
