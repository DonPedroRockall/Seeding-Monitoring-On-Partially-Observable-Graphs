import random

import networkx
import networkx as nx
from joblib import Parallel, delayed
from networkx.algorithms.flow import edmonds_karp

from Monitoring.Monitor import PlaceMonitors
from OverlappingCommunityDetection.CommunityDetector import InfluentialNodeRecovery
from Test.Common.DatasetGenerator import GenerateRandomGraphTriple, SetSameWeightsToOtherGraphs, \
    ParallelDatasetGeneration
from Test.Common.DatasetReader import WriteGraphTriple, ReadGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Test.Common.WeightRandomizer import SetRandomEdgeWeights
from Utilities.ColorPrints import bcolors, cprint
from Utilities.GraphGenerator import GNCConnectedDirectedGraph
from definitions import ROOT_DIR


class MonitorTester:

    # Common test config
    def __init__(self):
        self.NUM_NODES = 1500
        self.NUM_TO_HIDE = 300
        self.NUM_SOURCES = 10
        self.NUM_TARGETS = 4
        self.DISTRIBUTION = "deg"
        self.generate = False
        self.full = networkx.DiGraph()
        self.part = networkx.DiGraph()
        self.recv = networkx.DiGraph()


    def test_1(self, to_hide=None):
        """
        Tests with a single graph, but with different percent of hidden nodes. Performs KRONFIT multiple times
        :return:
        """

        # Default percent
        if to_hide is None:
            to_hide = [.8]

        # Adjust from percent to
        for i in range(len(to_hide)):
            to_hide[i] = int(to_hide[i] * self.NUM_NODES)

        # Or read the full graph and re-hide nodes and re-perform KRONFIT
        full: networkx.DiGraph = networkx.read_edgelist(ROOT_DIR + "/Datasets/Synthetic" + "/DegreeDist/Medium_1500/0_full_hid300_deg.txt", create_using=networkx.DiGraph, nodetype=int)

        for nodes_to_hide in to_hide:
            # Hide a part of the nodes
            nth = UniformDistribution(full, nodes_to_hide)
            part = TotalNodeClosure(full, nth)

            # Set influential treshold
            influential_threshold = sum(deg for node, deg in part.degree() if deg > 0) / float(
                part.number_of_nodes())

            # Reconstruct the graph
            recv, nodes_recovered = InfluentialNodeRecovery(
                part.copy(), nodes_to_hide, N0=2, alpha=None, beta=None,
                epsilon=influential_threshold, centrality="deg")

            cprint(bcolors.OKGREEN, "Nodes recovered:", nodes_recovered)

            self.perform_test(full, part, recv)


    def test_2(self, i, verbose=False, generate=False):
        if generate:
            # Generate the graph triplet
            full, part, recv = GenerateRandomGraphTriple(self.NUM_NODES,
                                                         self.NUM_TO_HIDE,
                                                         GNCConnectedDirectedGraph,
                                                         UniformDistribution,
                                                         TotalNodeClosure,
                                                         None, self.DISTRIBUTION, True)

            # Write the graph to path
            cprint(bcolors.OKGREEN, "Writing generated graph to file")
            WriteGraphTriple(ROOT_DIR + "/Datasets/Synthetic/DegreeDist/Medium_1500/", full, part, recv, self.NUM_TO_HIDE)

        else:
            # Or read graph triple
            full, part, recv = ReadGraphTriple(ROOT_DIR + "/Datasets/Synthetic/DegreeDist/Medium_1500/", index=0, hid=150, dist=self.DISTRIBUTION)

            return self.perform_test(full, part, recv)

    def perform_test(self, full, part, recv):
        # Generate the weights for the full graph
        cprint(bcolors.OKGREEN, "Setting weights...")
        SetRandomEdgeWeights(full, "weight", "uniform", True, *[0, 1])
        # Copy the weights to the other two graphs
        SetSameWeightsToOtherGraphs(full, [part])
        SetSameWeightsToOtherGraphs(part, [recv])
        # Assign random edges to the newly reconstructed edges
        SetRandomEdgeWeights(recv, "weight", "uniform", False, *[0, 1])

        # Choose sources and targets (they have to be in all 3 graphs)
        cprint(bcolors.OKGREEN, "Choosing sources...")
        sources = list()
        targets = list()

        while len(sources) < self.NUM_SOURCES:
            node = random.choice(list(part.nodes()))
            if node not in sources and node in full.nodes() and node in recv.nodes():
                sources.append(node)

        cprint(bcolors.OKGREEN, "Choosing targets...")
        while len(targets) < self.NUM_TARGETS:
            node = random.choice(list(part.nodes()))
            if node not in sources and node not in targets and node in full.nodes() and node in recv.nodes():
                targets.append(node)

        # Place the monitors on the 3 graphs
        cprint(bcolors.OKGREEN, "Running monitor placement...")
        monitors_full = PlaceMonitors(full, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_part = PlaceMonitors(part, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_recv = PlaceMonitors(recv, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)

        cprint(bcolors.OKBLUE, "==============================================================")

        return monitors_full, monitors_part, monitors_recv


if __name__ == "__main__":

    # ParallelDatasetGeneration(num_nodes=NUM_NODES,
    #                           num_to_hide=NUM_TO_HIDE,
    #                           gen_func=GNCConnectedDirectedGraph,
    #                           distr_func=DegreeDistribution,
    #                           hiding_func=TotalNodeClosure,
    #                           inf_thresh=None,
    #                           inf_centr="deg",
    #                           num_cores=1,
    #                           num_of_graphs=1,
    #                           file_path=ROOT_DIR + "/Datasets/Synthetic/DegreeDist/Big_10k/")

    # monitors_list = Parallel(n_jobs=1)(delayed(run_test)(i, True) for i in range(1))

    mt = MonitorTester()
    mt.NUM_NODES = 1500
    mt.NUM_TO_HIDE = 300
    mt.NUM_SOURCES = 10
    mt.NUM_TARGETS = 4
    mt.DISTRIBUTION = "uniform"
    mt.test_1()





