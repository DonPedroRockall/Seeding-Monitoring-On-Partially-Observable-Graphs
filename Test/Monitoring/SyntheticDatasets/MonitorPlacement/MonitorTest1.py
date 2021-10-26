from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Utilities.DrawGraph import DrawGraph


full, part, recv = GenerateRandomGraphTriple(10, 25, 3, UniformDistribution, TotalNodeClosure)

print(list(full.edges()))
print(list(part.edges()))
print(list(recv.edges()))

DrawGraph(full)
DrawGraph(part)
DrawGraph(recv)
