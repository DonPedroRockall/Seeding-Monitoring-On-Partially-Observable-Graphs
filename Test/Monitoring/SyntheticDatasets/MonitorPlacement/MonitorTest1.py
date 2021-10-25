from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Utilities.DrawGraph import DrawGraph

import scipy

full, part, rec = GenerateRandomGraphTriple(10, 25, 3, TotalNodeClosure, UniformDistribution)

DrawGraph(full)
DrawGraph(part)
DrawGraph(rec)