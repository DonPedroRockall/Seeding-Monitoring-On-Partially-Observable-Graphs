import os

from Test.Common.DistributionFunctions import ENodeHidingSelectionFunction
from Test.Common.GraphGenerator import EGraphGenerationFunction
from Test.Common.HidingFunctions import EClosureFunction
from Test.Common.WeightGenerator import EWeightSetterFunction
from definitions import ROOT_DIR


def rename_files():

    for filename in os.listdir(ROOT_DIR + "/Datasets/Synthetic"):
        print(filename)

        os.chdir(ROOT_DIR + "/Datasets/Synthetic")
        os.rename(filename, filename
                  .replace("EGraphGenerationFunction.EGNCConnectedDirectedGraph", EGraphGenerationFunction.EGNCConnectedDirectedGraph.value["short_name"])
                  .replace("ENodeHidingSelectionFunction.EDegreeDistribution", ENodeHidingSelectionFunction.EDegreeDistribution.value["short_name"])
                  .replace("EClosureFunction.ETotalClosure", EClosureFunction.ETotalClosure.value["short_name"])
                  .replace("EWeightSetterFunction.EInDegreeWeights", EWeightSetterFunction.EInDegreeWeights.value["short_name"])
                  .replace("src10", "")
                  .replace("src20", "")
                  .replace("src50", "")
                  .replace("trg10", "")
                  .replace("trg20", "")
                  .replace("trg50", "")
                  .replace("___", "_")
                  )

if __name__ == "__main__":
    rename_files()