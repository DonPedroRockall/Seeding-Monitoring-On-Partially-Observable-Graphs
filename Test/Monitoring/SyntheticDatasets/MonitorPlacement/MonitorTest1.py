from Monitoring.DiffusionModels import independent_cascade
from Monitoring.Monitor import PlaceMonitors
from Monitoring.MonitorUtility import InterpretCascadeResults
from OverlappingCommunityDetection.CommunityDetector import InfluentialNodeRecovery
from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DatasetReader import WriteGraphTriple, ReadGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Test.Common.Utility import *
from Utilities.ColorPrints import bcolors, cprint
from Test.Common.GraphGenerator import GNCConnectedDirectedGraph
from definitions import ROOT_DIR


class MonitorTester:

    # Common test config
    def __init__(self):
        self.NUM_NODES = 1500
        self.NUM_TO_HIDE = 300
        self.NUM_SOURCES = 10
        self.NUM_TARGETS = 4
        self.GENERATION = "Random Graph"
        self.WEIGHT = "indegree"
        self.CLOSURE = "Total Closure"
        self.DISTRIBUTION = "deg"
        self.FOLDER = ""
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


    def test_2(self, generate=False, verbose=False):
        if generate:
            # Generate the graph triplet
            full, part, recv = GenerateRandomGraphTriple(self.NUM_NODES,
                                                         self.NUM_TO_HIDE,
                                                         GNCConnectedDirectedGraph,
                                                         DegreeDistribution if self.DISTRIBUTION == "deg" else UniformDistribution,
                                                         TotalNodeClosure if self.CLOSURE == "Total Closure" else PartialNodeClosure,
                                                         None, "deg", True)

            # Write the graph to path
            if verbose:
                cprint(bcolors.OKGREEN, "Writing generated graph to file")
            WriteGraphTriple(ROOT_DIR + "/Datasets/Synthetic/DegreeDist/", self.FOLDER, full, part, recv, self.NUM_TO_HIDE)

            return self.perform_test(full, part, recv)

        else:
            # Or read graph triple
            full, part, recv = ReadGraphTriple(ROOT_DIR + "/Datasets/Synthetic/DegreeDist/", folder=self.FOLDER, index=0, hid=self.NUM_TO_HIDE, dist=self.DISTRIBUTION)

            return self.perform_test(full, part, recv)

    def perform_test(self, full, part, recv):
        # Generate the weights for the full graph
        cprint(bcolors.OKGREEN, "Setting weights...")
        SetRandomEdgeWeights(full, "weight", self.WEIGHT, True, *[0, 1])
        # Copy the weights to the other two graphs
        SetSameWeightsToOtherGraphs(full, [part])
        SetSameWeightsToOtherGraphs(part, [recv])
        # Assign random edges to the newly reconstructed edges
        SetRandomEdgeWeights(recv, "weight", self.WEIGHT, False, *[0, 1])

        number_of_recovered_nodes = recv.number_of_nodes() - part.number_of_nodes()
        number_of_hidden_nodes = full.number_of_nodes() - part.number_of_nodes()

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

        # Compute the set of virtual nodes
        virtual_set = GetVirtualNodesByLabel(part, recv)

        # Place the monitors on the 3 graphs
        cprint(bcolors.OKGREEN, "Running monitor placement...")

        monitors_full = PlaceMonitors(full, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_part = PlaceMonitors(part, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_recv = PlaceMonitors(recv, sources, targets, delta=1, tau=0.1, cascade_iterations=100, virtual_set=virtual_set, verbose=True)

        # Test the monitor placement by running the independent cascade on all three
        ic_full_full = independent_cascade(full, sources, monitors_full, 100)

        # This is the comparison test, where can assess our improvements over doing the procedure without trying to
        # recover a part of the hidden information in the part graph
        ic_full_part = independent_cascade(full, sources, monitors_part, 100)

        # This is the important test, where we run the monitor test on the full graph with the recv graph monitors
        ic_recv_recv = independent_cascade(recv, sources, monitors_recv, 100)
        ic_full_recv = independent_cascade(full, sources, monitors_recv, 100)

        # Print the results in a nice way
        cprint(bcolors.HEADER, "== MONITOR TEST REPORT ==\n")

        cprint(bcolors.BOLD, "-- General Information --")
        print(len(sources), "sources:", sources)
        print(len(targets), "targets:", targets)
        print("Hidden Nodes:", number_of_hidden_nodes)
        print("Recovered Nodes:", number_of_recovered_nodes, "\n")

        cprint(bcolors.BOLD, "-- General Graph Information --")
        print("Full graph:\n", full.number_of_nodes(), "nodes\n", full.number_of_edges(), "edges\n")
        print("Generation Function:", self.GENERATION)
        print("Hiding Function:", self.DISTRIBUTION)
        print("Weight Function:", self.WEIGHT)
        print("Closure Function:", self.CLOSURE, "\n")

        cprint(bcolors.BOLD, "-- Full Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_full, full, sources, targets, monitors_full)

        cprint(bcolors.BOLD, "-- Part Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_part, full, sources, targets, monitors_part)

        cprint(bcolors.BOLD, "-- Recv Monitors on Recv Graph --")
        InterpretCascadeResults(ic_recv_recv, recv, sources, targets, monitors_recv)

        cprint(bcolors.BOLD, "-- Recv Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_recv, full, sources, targets, monitors_recv)

        cprint(bcolors.OKBLUE, "==============================================================")

        # Return the monitors for eventual further processing
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
    mt.NUM_TARGETS = 10
    mt.DISTRIBUTION = "deg"
    mt.FOLDER = "MEDIUM_1500/"
    mt.GENERATION = "Random Graph"
    mt.WEIGHT = "smallrand"  # Should set this to "smallrand"
    mt.CLOSURE = "Total Closure"
    mt.DISTRIBUTION = "deg"
    # mt.test_1()
    mt.test_2(generate=True, verbose=True)






