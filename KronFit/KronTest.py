from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Common.GraphUtilities import GetVirtualNodesNodeDifference
from Common.DrawGraph import DrawGraph


full, part, recv = GenerateRandomGraphTriple(20, 5, verbose=True)
DrawGraph(full)
DrawGraph(part)
DrawGraph(recv)
GetVirtualNodesNodeDifference(part, recv)

