import networkx

from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.Utility import GetVirtualNodesByLabel
from Utilities.DrawGraph import DrawGraph


full, part, recv = GenerateRandomGraphTriple(20, 5, verbose=True)
DrawGraph(full)
DrawGraph(part)
DrawGraph(recv)
GetVirtualNodesByLabel(part, recv)

