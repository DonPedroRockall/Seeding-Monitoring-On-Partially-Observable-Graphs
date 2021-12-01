import networkx as nx
import pyvis.options
from pyvis.network import Network


def DrawGraph(graph, color_dict=None, graph_name="graph", physics=False, directed=True):

    # Set color attributes
    if color_dict is not None:
        for node in color_dict:
            graph.nodes[node]["color"] = color_dict[node]

    # Construct network from networkx
    nt = Network("1080px", "1920px", directed=directed)
    nt.from_nx(graph)

    # Disable Physics
    if not physics:
        for node in nt.nodes:
            node["physics"] = False
        for edge in nt.edges:
            edge["physics"] = False

    # Show the graph
    nt.show(graph_name + ".html")
