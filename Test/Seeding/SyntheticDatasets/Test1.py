from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Utilities.DrawGraph import DrawGraph


full, part, rec = GenerateRandomGraphTriple(50, 20, 5, distribution_function=UniformDistribution)

print(list(full.edges()))
print(list(part.edges()))
print(list(rec.edges()))


DrawGraph(full, graph_name="full", physics=False)
DrawGraph(part, graph_name="part", physics=False)
DrawGraph(rec, graph_name="rec", physics=False)

