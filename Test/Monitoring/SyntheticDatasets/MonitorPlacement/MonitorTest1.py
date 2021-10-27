import networkx as nx

from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Utilities.DrawGraph import DrawGraph


if __name__ == "__main__":
    full, part, recv = GenerateRandomGraphTriple(10, 25, 3, UniformDistribution, TotalNodeClosure)

    print(list(full.edges()))
    print(list(part.edges()))
    print(list(recv.edges()))

    nx.draw_random(full)
    nx.draw_random(part)

    DrawGraph(full, physics=False, directed=True, graph_name="Full")
    DrawGraph(part, physics=False, directed=True, graph_name="Part")
    DrawGraph(recv, physics=False, directed=True, graph_name="Recv")
