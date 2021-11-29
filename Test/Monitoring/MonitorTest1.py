import sys

from joblib import delayed, Parallel
from DiffusionModels.IndependentCascade import IndependentCascadeWithMonitors
from Monitoring.MonitorPlacement.Monitor import PlaceMonitors
from Monitoring.MonitorPlacement.MonitorUtility import InterpretCascadeResults
from GraphRecovery.GraphRecoverer import InfluentialNodeRecovery
from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DatasetReader import WriteGraphTriple, ReadGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Test.Common.Utility import *
from Test.Common.WeightGenerator import EWeightSetterFunction
from Utilities.ColorPrints import bcolors, cprint, fprint
from Test.Common.GraphGenerator import EGraphGenerationFunction
from definitions import ROOT_DIR


class MonitorTester:

    # Common test config
    def __init__(self):
        self.NUM_NODES = 1500
        self.NUM_TO_HIDE = 300
        self.NUM_SOURCES = 10
        self.NUM_TARGETS = 4
        self.GENERATION = EGraphGenerationFunction.EGNCConnectedDirectedGraph.value
        self.CLOSURE = EClosureFunction.ETotalClosure.value
        self.DISTRIBUTION = ENodeHidingSelectionFunction.EDegreeDistribution.value
        self.WEIGHT = EWeightSetterFunction.EInDegreeWeights.value
        self.WEIGHT_KWARGS = {}
        self.DATASET_PATH = ROOT_DIR + "/Datasets"
        self.FOLDER = ""
        self.PRINT_TO_FILE = None
        self.TEST_PARAMS = ""
        self.generate = False
        self.full = networkx.DiGraph()
        self.part = networkx.DiGraph()
        self.recv = networkx.DiGraph()

    def test_1(self, folder, num_nodes, verbose=False):
        """Reads a graph triple from file(s) and performs the monitoring algorithm on all of them"""
        full, part, recv = ReadGraphTriple()

        self.perform_test(full, part, recv)

    def test_2(self, verbose=False):
        """Performs a test by generating a graph triple and executing the monitor placement on all of them"""
        # Generate the graph triplet
        # full, part, recv = GenerateRandomGraphTriple(self.NUM_NODES,
        #                                              self.NUM_TO_HIDE,
        #                                              self.GENERATION.value["function"],
        #                                              self.DISTRIBUTION.value["function"],
        #                                              self.CLOSURE.value["function"],
        #                                              None, "deg", True)
        full = networkx.gnc_graph(100)
        part = full.copy()
        recv = full.copy()

        # Write the graph to path
        if verbose:
            cprint(bcolors.OKGREEN, "Writing generated graph to file...")
        # WriteGraphTriple(self.DATASET_PATH, self.FOLDER, GenerateGraphFilename(self.NUM_NODES, self.NUM_TO_HIDE,
        #                  self.NUM_SOURCES, self.NUM_TARGETS, self.GENERATION, self.DISTRIBUTION, self.CLOSURE, self.WEIGHT),
        #                  full, part, recv)

        return self.perform_test(full, part, recv)

    def test_real_dataset(self, path, directed, generate):
        full = networkx.read_edgelist(path, create_using=networkx.DiGraph if directed else networkx.Graph)

        if not generate:
            part = networkx.read_edgelist(path[:-4] + "-PART.txt",
                                          create_using=networkx.DiGraph if directed else networkx.Graph)
            recv = networkx.read_edgelist(path[:-4] + "-RECV.txt",
                                          create_using=networkx.DiGraph if directed else networkx.Graph)
            self.perform_test(full, part, recv)
            return

        hiding_distr = UniformDistribution
        if self.DISTRIBUTION == "deg":
            hiding_distr = DegreeDistribution

        closure_func = TotalNodeClosure
        if self.CLOSURE == "Partial Closure":
            closure_func = PartialNodeClosure

        nth = hiding_distr(full, self.NUM_TO_HIDE)
        # Hide a part of the nodes
        part = closure_func(full, nth)

        # Set influential treshold
        influential_threshold = sum(deg for node, deg in part.degree() if deg > 0) / float(part.number_of_nodes())

        # Reconstruct the graph
        recv, nodes_recovered = InfluentialNodeRecovery(
            part.copy(), self.NUM_TO_HIDE, N0=2, alpha=None, beta=None,
            epsilon=influential_threshold, centrality="deg")

        networkx.write_edgelist(part, ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote-PART.txt", data=False)
        networkx.write_edgelist(recv, ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote-RECV.txt", data=False)

        self.perform_test(full, part, recv)

    def perform_test(self, full, part, recv):

        # Generate the weights for the full graph
        cprint(bcolors.OKGREEN, "Setting weights...")
        self.WEIGHT.value["function"](full, attribute="weight", force=True, **self.WEIGHT_KWARGS)
        # Copy the weights to the other two graphs
        SetSameWeightsToOtherGraphs(full, [part, recv])

        # Assign random edges to the newly reconstructed edges
        self.WEIGHT.value["function"](recv, attribute="weight", force=False, **self.WEIGHT_KWARGS)

        # Choose sources and targets (they have to be in all 3 graphs)
        cprint(bcolors.OKGREEN, "Choosing targets...")
        valid_nodes = set(part.nodes())

        if len(valid_nodes) < self.NUM_SOURCES + self.NUM_TARGETS:
            raise ValueError("Cannot continue with the algorithm, as there are not enough nodes in partial graph to "
                             "select {0} sources and {1} targets".format(self.NUM_SOURCES, self.NUM_TARGETS))

        sources = list(random.sample(valid_nodes, self.NUM_SOURCES))
        for src in sources:
            valid_nodes.remove(src)
        targets = list(random.sample(valid_nodes, self.NUM_TARGETS))
        cprint(bcolors.OKGREEN, "Set sources and targets. Computing the virtual nodes set...")

        # Compute the set of virtual nodes
        virtual_set = GetVirtualNodesByLabel(part, recv)

        # Place the monitors on the 3 graphs
        cprint(bcolors.OKGREEN, "Running monitor placement...")

        # Run the monitor placement algorithm on all the 3 graphs
        monitors_full = PlaceMonitors(full, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_part = PlaceMonitors(part, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_recv = PlaceMonitors(recv, sources, targets, delta=1, tau=0.1, cascade_iterations=100,
                                      virtual_set=virtual_set, verbose=True)

        # Test the monitor placement by running the independent cascade on all three
        ic_full_full = IndependentCascadeWithMonitors(full, sources, monitors_full, 100)

        # This is the comparison test, where can assess our improvements over doing the procedure without trying to
        # recover a part of the hidden information in the part graph
        ic_full_part = IndependentCascadeWithMonitors(full, sources, monitors_part, 100)

        # This is the important test, where we run the monitor test on the full graph with the recv graph monitors
        ic_recv_recv = IndependentCascadeWithMonitors(recv, sources, monitors_recv, 100)
        ic_full_recv = IndependentCascadeWithMonitors(full, sources, monitors_recv, 100)

        # Print either to file or to stdout
        if self.PRINT_TO_FILE is None:
            self.print_to_stdout(full, part, recv, sources, targets, monitors_full, monitors_part, monitors_recv,
                                 ic_full_full, ic_full_part, ic_recv_recv, ic_full_recv)
        else:
            self.print_to_file(self.PRINT_TO_FILE, full, part, recv, sources, targets, monitors_full, monitors_part,
                               monitors_recv, ic_full_full, ic_full_part, ic_recv_recv, ic_full_recv)

        # Return the monitors for eventual further processing
        return monitors_full, monitors_part, monitors_recv

    def print_to_stdout(self, full, part, recv, sources, targets, monitors_full, monitors_part, monitors_recv,
                        ic_full_full, ic_full_part, ic_recv_recv, ic_full_recv):

        number_of_recovered_nodes = recv.number_of_nodes() - part.number_of_nodes()
        number_of_hidden_nodes = full.number_of_nodes() - part.number_of_nodes()

        # Print the results in a nice way
        cprint(bcolors.HEADER, "== MONITOR TEST REPORT ==\n")

        cprint(bcolors.BOLD, "-- General Information --")
        print(len(sources), "sources:", sources)
        print(len(targets), "targets:", targets)
        print("Hidden Nodes:", number_of_hidden_nodes)
        print("Recovered Nodes:", number_of_recovered_nodes, "\n")

        cprint(bcolors.BOLD, "-- General Graph Information --")
        print("Full graph:\n", full.number_of_nodes(), "nodes\n", full.number_of_edges(), "edges\n")

        cprint(bcolors.BOLD, "-- General Graph Information --")
        print("Part graph:\n", part.number_of_nodes(), "nodes\n", part.number_of_edges(), "edges\n")

        cprint(bcolors.BOLD, "-- General Graph Information --")
        print("Recv graph:\n", recv.number_of_nodes(), "nodes\n", recv.number_of_edges(), "edges\n")

        print("Generation Function:", self.GENERATION.value["name"])
        print("Hiding Function:", self.DISTRIBUTION.value["name"])
        print("Weight Function:", self.WEIGHT.value["name"], "with parameters:", str(self.WEIGHT_KWARGS))
        print("Closure Function:", self.CLOSURE.value["name"], "\n")

        cprint(bcolors.BOLD, "-- Full Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_full, full, sources, targets, monitors_full, path=None)

        cprint(bcolors.BOLD, "-- Part Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_part, full, sources, targets, monitors_part, path=None)

        cprint(bcolors.BOLD, "-- Recv Monitors on Recv Graph --")
        InterpretCascadeResults(ic_recv_recv, recv, sources, targets, monitors_recv, path=None)

        cprint(bcolors.BOLD, "-- Recv Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_recv, full, sources, targets, monitors_recv, path=None)

        cprint(bcolors.OKBLUE, "==============================================================")

    def print_to_file(self, path, full, part, recv, sources, targets, monitors_full, monitors_part, monitors_recv,
                      ic_full_full, ic_full_part, ic_recv_recv, ic_full_recv):

        number_of_recovered_nodes = recv.number_of_nodes() - part.number_of_nodes()
        number_of_hidden_nodes = full.number_of_nodes() - part.number_of_nodes()

        # Print the results in a nice way
        fprint(path, "== MONITOR TEST REPORT ==\n")

        fprint(path, "-- General Information --")
        fprint(path, len(sources), "sources:", sources)
        fprint(path, len(targets), "targets:", targets)
        fprint(path, "Hidden Nodes:", number_of_hidden_nodes)
        fprint(path, "Recovered Nodes:", number_of_recovered_nodes, "\n")

        fprint(path, "-- General Graph Information --")
        fprint(path, "Full graph:\n", full.number_of_nodes(), "nodes\n", full.number_of_edges(), "edges\n")

        fprint(path, "-- General Graph Information --")
        fprint(path, "Part graph:\n", part.number_of_nodes(), "nodes\n", part.number_of_edges(), "edges\n")

        fprint(path, "-- General Graph Information --")
        fprint(path, "Recv graph:\n", recv.number_of_nodes(), "nodes\n", recv.number_of_edges(), "edges\n")

        fprint(path, "Generation Function:", self.GENERATION.value["name"])
        fprint(path, "Hiding Function:", self.DISTRIBUTION.value["name"])
        fprint(path, "Weight Function:", self.WEIGHT.value["name"], "with parameters:", str(self.WEIGHT_KWARGS))
        fprint(path, "Closure Function:", self.CLOSURE.value["name"], "\n")

        fprint(path, "-- Full Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_full, full, sources, targets, monitors_full, path=path)

        fprint(path, "-- Part Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_part, full, sources, targets, monitors_part, path=path)

        fprint(path, "-- Recv Monitors on Recv Graph --")
        InterpretCascadeResults(ic_recv_recv, recv, sources, targets, monitors_recv, path=path)

        fprint(path, "-- Recv Monitors on Full Graph --")
        InterpretCascadeResults(ic_full_recv, full, sources, targets, monitors_recv, path=path)

        fprint(path, "==============================================================")


########################################################################################################################
# --- TEST STARTING CODE ---
########################################################################################################################

def single_test(**kwargs):
    mt = MonitorTester()
    mt.NUM_NODES = kwargs["NUM_NODES"]
    mt.NUM_TO_HIDE = kwargs["NUM_TO_HIDE"]
    mt.NUM_SOURCES = kwargs["NUM_SOURCES"]
    mt.NUM_TARGETS = kwargs["NUM_TARGETS"]
    mt.DISTRIBUTION = kwargs["DISTRIBUTION"]
    mt.FOLDER = kwargs["FOLDER"]
    mt.GENERATION = kwargs["GENERATION"]
    mt.CLOSURE = kwargs["CLOSURE"]
    mt.WEIGHT = kwargs["WEIGHT"]
    mt.WEIGHT_KWARGS = kwargs["WEIGHT_KWARGS"]

    mt.PRINT_TO_FILE = "Results/SyntheticDatasets/" + kwargs["FILENAME"]
    # mt.test_1()
    mt.test_2(verbose=True)
    # mt.test_real_dataset(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote.txt", directed=True, generate=True)


if __name__ == "__main__":
    """
    NUM_NODES = [1500]
    
    NUM_SOURCES = [10, 20, 50, 100]
    NUM_TARGETS = [10, 20, 50, 100]
    FOLDER = ["Medium_1500"]
    DISTRIBUTION = [EDistributionFunctions.EDegreeDistribution, EDistributionFunctions.EUniformDistribution]
    GENERATION = [EGraphGenerator.EGNCConnectedDirectedGraph, EGraphGenerator.EGNCConnectedDirectedGraph]
    WEIGHT = [EWeightFunctions.ESmallRandWeights, EWeightFunctions.EInDegreeWeights]
    CLOSURE = [EHidingFunctions.ETotalClosure]
    """

    NUM_TO_HIDE = [50, 100, 200, 250, 350, 500, 650, 750, 800, 950]

    Parallel(n_jobs=10)(delayed(single_test)(**{
        "NUM_NODES": 1000,
        "NUM_TO_HIDE": NUM_TO_HIDE[i],
        "NUM_SOURCES": 10,
        "NUM_TARGETS": 10,
        "DISTRIBUTION": ENodeHidingSelectionFunction.EDegreeDistribution,
        "GENERATION": EGraphGenerationFunction.EGNCConnectedDirectedGraph,
        "CLOSURE": EClosureFunction.ETotalClosure,
        "WEIGHT": EWeightSetterFunction.EInDegreeWeights,
        "WEIGHT_KWARGS": {},
        "FOLDER": "Synthetic",
        "FILENAME": GenerateGraphFilename(
            1000, NUM_TO_HIDE[i], 10, 10,
            EGraphGenerationFunction.EGNCConnectedDirectedGraph.value["short_name"],
            ENodeHidingSelectionFunction.EDegreeDistribution.value["short_name"],
            EClosureFunction.ETotalClosure.value["short_name"],
            EWeightSetterFunction.EInDegreeWeights.value["short_name"],
        )}) for i in range(10))
