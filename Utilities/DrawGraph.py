import networkx as nx
import pylab


def DrawGraph(graph):
    nx.draw(graph, with_labels=True)
    nx.spring_layout(graph)
    pylab.show()