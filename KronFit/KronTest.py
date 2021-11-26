# import networkx
#
# from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
# from Test.Common.Utility import GetVirtualNodesByLabel
# from Utilities.DrawGraph import DrawGraph
#
#
# full, part, recv = GenerateRandomGraphTriple(20, 5, verbose=True)
# DrawGraph(full)
# DrawGraph(part)
# DrawGraph(recv)
# GetVirtualNodesByLabel(part, recv)
#
from Utilities.DrawGraph import DrawGraph
import networkx as nx


graph = nx.DiGraph()
graph.add_edges_from([
    (0, 1),

    (1, 4),
    (4, 0),
])

DrawGraph(graph)
