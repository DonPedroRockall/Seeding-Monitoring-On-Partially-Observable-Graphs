import random

import networkx as nx

from Monitoring.Monitor import PlaceMonitors
from Test.Common.DatasetGenerator import GenerateRandomGraphTriple, SetSameWeightsToOtherGraphs
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Test.Common.WeightRandomizer import SetRandomEdgeWeights
from Utilities.DrawGraph import DrawGraph


if __name__ == "__main__":

    # Common test config
    NUM_NODES = 150
    NUM_SOURCES = 5
    NUM_TARGETS = 1

    for i in range(50):

        # Generate the graph triplet
        full, part, recv = GenerateRandomGraphTriple(NUM_NODES, 0, 20, UniformDistribution, TotalNodeClosure)

        # Generate the weights for the full graph
        SetRandomEdgeWeights(full, "weight", "uniform", True, *[0, 1])
        # Copy the weights to the other two graphs
        SetSameWeightsToOtherGraphs(full, [part])
        SetSameWeightsToOtherGraphs(part, [recv])

        # Assign random edges to the newly reconstructed edges
        SetRandomEdgeWeights(recv, "weight", "uniform", False, *[0, 1])

        # Choose sources and targets (they have to be in all 3 graphs)
        sources = list()
        targets = list()
        while len(sources) < NUM_SOURCES:
            node = random.choice(list(part.nodes()))
            if node not in sources and node in full.nodes() and node in recv.nodes():
                sources.append(node)

        while len(targets) < NUM_TARGETS:
            node = random.choice(list(part.nodes()))
            if node not in sources and node not in targets and node in full.nodes() and node in recv.nodes():
                targets.append(node)

        # Place the monitors on the 3 graphs
        monitors_full = PlaceMonitors(full, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_part = PlaceMonitors(part, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_recv = PlaceMonitors(recv, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)

        # Run the independent cascade on the three graphs





    

