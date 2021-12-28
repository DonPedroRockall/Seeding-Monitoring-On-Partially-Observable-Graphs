import os
import statistics
import sys

import networkx
import numpy
from joblib import delayed, Parallel

from Common.DrawGraph import DrawGraph
from DiffusionModels.IndependentCascade import IndependentCascadeWithMonitors
from Monitoring.MonitorPlacement.Monitor import PlaceMonitors
from Monitoring.MonitorPlacement.MonitorUtility import PrintCascadeResults, GatherCascadeResults
from GraphRecovery.GraphRecoverer import InfluentialNodeRecovery
from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DatasetReader import WriteGraphTriple, ReadGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Common.GraphUtilities import *
from Test.Common.WeightGenerator import EWeightSetterFunction
from Common.ColorPrints import bcolors, cprint, fprint
from Test.Common.GraphGenerator import EGraphGenerationFunction
from definitions import ROOT_DIR


class MonitorTestReport:
    def __init__(self):
        # nodes and edges
        self.NUM_NODES_FULL = 0
        self.NUM_EDGES_FULL = 0
        self.NUM_NODES_PART = 0
        self.NUM_EDGES_PART = 0
        self.NUM_NODES_RECV = 0
        self.NUM_EDGES_RECV = 0
        # hidden and recovered
        self.NUM_HIDDEN = 0
        self.PERC_HIDDEN = 0
        self.NUM_RECOVERED = 0
        self.PERC_RECOVERED_FULL = 0
        self.PERC_RECOVERED_HIDDEN = 0
        # sources and targets
        self.SOURCES = []
        self.NUM_SOURCES = 0
        self.PERC_SOURCES = 0
        self.TARGETS = []
        self.NUM_TARGETS = 0
        self.PERC_TARGETS = 0
        # infected
        self.NUM_INFECTED_FF = 0
        self.PERC_INFECTED_FF = 0
        self.NUM_INFECTED_FP = 0
        self.PERC_INFECTED_FP = 0
        self.NUM_INFECTED_RR = 0
        self.PERC_INFECTED_RR = 0
        self.NUM_INFECTED_FR = 0
        self.PERC_INFECTED_FR = 0
        # non-source infected
        self.NON_SOURCE_INF_FF = 0
        self.PERC_NS_INFECTED_FF = 0
        self.NON_SOURCE_INF_FP = 0
        self.PERC_NS_INFECTED_FP = 0
        self.NON_SOURCE_INF_RR = 0
        self.PERC_NS_INFECTED_RR = 0
        self.NON_SOURCE_INF_FR = 0
        self.PERC_NS_INFECTED_FR = 0
        # infected targets
        self.INFECTED_TARGETS_FF = 0
        self.PERC_INFECTED_TARGETS_FF = 0
        self.INFECTED_TARGETS_FP = 0
        self.PERC_INFECTED_TARGETS_FP = 0
        self.INFECTED_TARGETS_RR = 0
        self.PERC_INFECTED_TARGETS_RR = 0
        self.INFECTED_TARGETS_FR = 0
        self.PERC_INFECTED_TARGETS_FR = 0
        # monitors
        self.MONITORS_FULL = 0
        self.NUM_MONITORS_FULL = 0
        self.PERC_MONITORS_FULL = 0
        self.MONITORS_PART = 0
        self.NUM_MONITORS_PART = 0
        self.PERC_MONITORS_PART = 0
        self.MONITORS_RECV = []
        self.NUM_MONITORS_RECV = 0
        self.PERC_MONITORS_RECV = 0
        # cascade iterations
        self.CASCADE_ITERATIONS_FF = 0
        self.CASCADE_ITERATIONS_FP = 0
        self.CASCADE_ITERATIONS_RR = 0
        self.CASCADE_ITERATIONS_FR = 0

    @staticmethod
    def aggregate_results(results: list):
        agg_res = {}

        for key in vars(results[0]).keys():
            if type(vars(results[0])[key]) is not int and type(vars(results[0])[key]) is not float:
                continue
            values = []
            for res in results:
                values.append(vars(res)[key])
            if len(values) > 1:
                agg_res[key] = (statistics.mean(values), statistics.stdev(values))
            else:
                agg_res[key] = (values[0], 0)
        return agg_res

    def print_results(self, file=sys.stdout):
        raise NotImplementedError


class MonitorTester:

    # Common test config
    def __init__(self):
        self.NUM_NODES = 150
        self.NUM_TO_HIDE = 10
        self.NUM_SOURCES = 10
        self.NUM_TARGETS = 10
        self.GENERATION = EGraphGenerationFunction.EGNCConnectedDirectedGraph.value
        self.GENERATION_KWARGS = {}
        self.CLOSURE = EClosureFunction.ETotalClosure.value
        self.CLOSURE_KWARGS = {}
        self.DISTRIBUTION = ENodeHidingSelectionFunction.EDegreeDistribution.value
        self.DISTRIBUTION_KWARGS = {}
        self.WEIGHT = EWeightSetterFunction.EInDegreeWeights.value
        self.WEIGHT_KWARGS = {}
        self.DATASET_PATH = ROOT_DIR + "/Datasets"
        self.FOLDER = ""
        self.PRINT_TO_FILE = None
        self.TEST_PARAMS = ""
        self.TRIPLE_INDEX = -1
        self.generate = False
        self.full = networkx.DiGraph()
        self.part = networkx.DiGraph()
        self.recv = networkx.DiGraph()

    def read_synthetic_dataset(self, path, folder, index):
        """Reads a graph triple from file(s) and returns the triple"""
        return ReadGraphTriple(path, folder, index=index)

    def generate_synthetic_dataset(self, verbose=False):
        """Performs a test by generating a graph triple and executing the monitor placement on all of them"""
        # Generate the graph triplet
        full, part, recv = GenerateRandomGraphTriple(self.NUM_NODES,
                                                     self.NUM_TO_HIDE,
                                                     self.GENERATION.value["function"],
                                                     self.GENERATION_KWARGS,
                                                     self.DISTRIBUTION.value["function"],
                                                     self.DISTRIBUTION_KWARGS,
                                                     self.CLOSURE.value["function"],
                                                     self.CLOSURE_KWARGS,
                                                     None, "deg", True)

        if verbose:
            cprint(bcolors.OKGREEN, "Writing generated graph to file...")

        # Write the graph to path
        self.TRIPLE_INDEX = WriteGraphTriple(self.DATASET_PATH, self.FOLDER, GenerateGraphFilename(
            self.NUM_NODES, self.NUM_TO_HIDE, self.GENERATION.value["short_name"],
            self.DISTRIBUTION.value["short_name"], self.CLOSURE.value["short_name"],
            self.WEIGHT.value["short_name"]), full, part, recv)

        return full, part, recv

    def test_real_dataset(self, path):
        full = networkx.read_edgelist(path, create_using=networkx.DiGraph, nodetype=int)
        self.NUM_NODES = full.number_of_nodes()
        nth = self.DISTRIBUTION.value["function"](full, self.NUM_TO_HIDE)
        part = self.CLOSURE.value["function"](full.copy(), nth)
        recv, _ = InfluentialNodeRecovery(part, self.NUM_TO_HIDE, 2)
        return self.perform_test(full, part, recv)

    # TODO: remove nth
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
            print(f"Cannot continue with the algorithm, as there are not enough nodes in partial graph to "
                  f"select {self.NUM_SOURCES} sources and {self.NUM_TARGETS} targets")
            return

            # raise ValueError(f"Cannot continue with the algorithm, as there are not enough nodes in partial graph to "
            #                  f"select {self.NUM_SOURCES} sources and {self.NUM_TARGETS} targets")

        sources = list(random.sample(list(valid_nodes), self.NUM_SOURCES))
        for src in sources:
            valid_nodes.remove(src)
        targets = list(random.sample(list(valid_nodes), self.NUM_TARGETS))
        cprint(bcolors.OKGREEN, "Set sources and targets. Computing the virtual nodes set...")

        # Compute the set of virtual nodes
        # virtual_set = GetVirtualNodesByDifference(part, recv)
        virtual_set = GetVirtualNodesByNodeLabel(recv, "RECV")

        # Place the monitors on the 3 graphs
        cprint(bcolors.OKGREEN, "Running monitor placement...")

        # Run the monitor placement algorithm on all the 3 graphs
        monitors_full, _ = PlaceMonitors(full, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_part, _ = PlaceMonitors(part, sources, targets, delta=1, tau=0.1, cascade_iterations=100, verbose=True)
        monitors_recv, _ = PlaceMonitors(recv, sources, targets, delta=1, tau=0.1, cascade_iterations=100,
                                      virtual_set=virtual_set, verbose=True)

        # Test the monitor placement by running the independent cascade on all three
        ic_full_full = IndependentCascadeWithMonitors(full, sources, monitors_full, 100)

        # This is the comparison test, where can assess our improvements over doing the procedure without trying to
        # recover a part of the hidden information in the part graph
        ic_full_part = IndependentCascadeWithMonitors(full, sources, monitors_part, 100)

        # This is the important test, where we run the monitor test on the full graph with the recv graph monitors
        ic_recv_recv = IndependentCascadeWithMonitors(recv, sources, monitors_recv, 100)
        ic_full_recv = IndependentCascadeWithMonitors(full, sources, monitors_recv, 100)

        # Gather results for all ICs
        res_full_full = GatherCascadeResults(ic_full_full, full, sources, targets, monitors_full)
        res_full_part = GatherCascadeResults(ic_full_part, full, sources, targets, monitors_part)
        res_recv_recv = GatherCascadeResults(ic_recv_recv, recv, sources, targets, monitors_recv)
        res_full_recv = GatherCascadeResults(ic_full_recv, full, sources, targets, monitors_recv)

        results = self.gather_results(full, part, recv, monitors_full, monitors_part, monitors_recv,
                                      res_full_full, res_full_part, res_recv_recv, res_full_recv, sources, targets)

        # Print either to file or to stdout
        self.print_to_file(sys.stdout if self.PRINT_TO_FILE is None else open(self.PRINT_TO_FILE, "a+"), results)

        # Return the monitors for eventual further processing
        return results

    @staticmethod
    def gather_results(full, part, recv, mfull, mpart, mrecv, res_ff, res_fp, res_rr, res_fr, sources, targets):
        """Organizes results into a dict so to manage them in different ways."""
        # Cache for commodity
        n = full.number_of_nodes()
        n_part = part.number_of_nodes()
        n_recv = recv.number_of_nodes()

        mtr = MonitorTestReport()
        # nodes and edges
        mtr.NUM_NODES_FULL = n
        mtr.NUM_EDGES_FULL = full.number_of_edges()
        mtr.NUM_NODES_PART = n_part
        mtr.NUM_EDGES_PART = part.number_of_edges()
        mtr.NUM_NODES_RECV = n_recv
        mtr.NUM_EDGES_RECV = recv.number_of_edges()
        # hidden and recovered
        mtr.NUM_HIDDEN = n - n_part
        mtr.PERC_HIDDEN = mtr.NUM_HIDDEN / n
        mtr.NUM_RECOVERED = n_recv - n_part
        mtr.PERC_RECOVERED_FULL = mtr.NUM_RECOVERED / n
        mtr.PERC_RECOVERED_HIDDEN = mtr.NUM_RECOVERED / mtr.NUM_HIDDEN
        # source and targets
        mtr.SOURCES = sources
        mtr.NUM_SOURCES = len(sources)
        mtr.PERC_SOURCES = mtr.NUM_SOURCES / n
        mtr.TARGETS = targets
        mtr.NUM_TARGETS = len(targets)
        mtr.PERC_TARGETS = mtr.NUM_TARGETS / n
        # infected
        mtr.NUM_INFECTED_FF = res_ff["num_of_infected"]
        mtr.PERC_INFECTED_FF = mtr.NUM_INFECTED_FF / n
        mtr.NUM_INFECTED_FP = res_fp["num_of_infected"]
        mtr.PERC_INFECTED_FP = mtr.NUM_INFECTED_FP / n
        mtr.NUM_INFECTED_RR = res_rr["num_of_infected"]
        mtr.PERC_INFECTED_RR = mtr.NUM_INFECTED_RR / n_recv
        mtr.NUM_INFECTED_FR = res_fr["num_of_infected"]
        mtr.PERC_INFECTED_FR = mtr.NUM_INFECTED_FR / n
        # non-source infected
        mtr.NON_SOURCE_INF_FF = res_ff["num_of_non_source_infected"]
        mtr.PERC_NS_INFECTED_FF = mtr.NON_SOURCE_INF_FF / n
        mtr.NON_SOURCE_INF_FP = res_fp["num_of_non_source_infected"]
        mtr.PERC_NS_INFECTED_FP = mtr.NON_SOURCE_INF_FP / n
        mtr.NON_SOURCE_INF_RR = res_rr["num_of_non_source_infected"]
        mtr.PERC_NS_INFECTED_RR = mtr.NON_SOURCE_INF_RR / n_recv
        mtr.NON_SOURCE_INF_FR = res_fr["num_of_non_source_infected"]
        mtr.PERC_NS_INFECTED_FR = mtr.NON_SOURCE_INF_FR / n
        # infected targets
        mtr.INFECTED_TARGETS_FF = res_ff["num_of_infected_targets"]
        mtr.PERC_INFECTED_TARGETS_FF = mtr.INFECTED_TARGETS_FF / n
        mtr.INFECTED_TARGETS_FP = res_fp["num_of_infected_targets"]
        mtr.PERC_INFECTED_TARGETS_FP = mtr.INFECTED_TARGETS_FP / n
        mtr.INFECTED_TARGETS_RR = res_rr["num_of_infected_targets"]
        mtr.PERC_INFECTED_TARGETS_RR = mtr.INFECTED_TARGETS_RR / n_recv
        mtr.INFECTED_TARGETS_FR = res_fr["num_of_infected_targets"]
        mtr.PERC_INFECTED_TARGETS_FR = mtr.INFECTED_TARGETS_FR / n
        # monitors
        mtr.MONITORS_FULL = mfull
        mtr.NUM_MONITORS_FULL = len(mfull)
        mtr.PERC_MONITORS_FULL = mtr.NUM_MONITORS_FULL / n
        mtr.MONITORS_PART = mpart
        mtr.NUM_MONITORS_PART = len(mpart)
        mtr.PERC_MONITORS_PART = mtr.NUM_MONITORS_PART / n_part
        mtr.MONITORS_RECV = mrecv
        mtr.NUM_MONITORS_RECV = len(mrecv)
        mtr.PERC_MONITORS_RECV = mtr.NUM_MONITORS_RECV / n_recv
        # cascade iterations
        mtr.CASCADE_ITERATIONS_FF = res_ff["num_of_iterations"]
        mtr.CASCADE_ITERATIONS_FP = res_fp["num_of_iterations"]
        mtr.CASCADE_ITERATIONS_RR = res_rr["num_of_iterations"]
        mtr.CASCADE_ITERATIONS_FR = res_fr["num_of_iterations"]

        return mtr

    def print_to_file(self, file, results: MonitorTestReport):

        # Print the results in a nice way
        print("== MONITOR TEST REPORT ==\n", file=file)

        print("-- General Information --", file=file)
        print(results.NUM_SOURCES, "sources:", results.SOURCES, file=file)
        print(results.NUM_TARGETS, "targets:", results.TARGETS, file=file)
        print("Hidden Nodes:", results.NUM_HIDDEN, file=file)
        print("Recovered Nodes:", results.NUM_RECOVERED, "\n", file=file)

        print("-- General Graph Information --", file=file)
        print("Full graph:\n", results.NUM_NODES_FULL, "nodes\n", results.NUM_EDGES_FULL, "edges\n", file=file)
        print("Part graph:\n", results.NUM_NODES_PART, "nodes\n", results.NUM_EDGES_PART, "edges\n", file=file)
        print("Recv graph:\n", results.NUM_NODES_RECV, "nodes\n", results.NUM_EDGES_RECV, "edges\n", file=file)

        print("Generation Function:", self.GENERATION.value["name"], file=file)
        print("Hiding Function:", self.DISTRIBUTION.value["name"], file=file)
        print("Weight Function:", self.WEIGHT.value["name"], "with parameters:", str(self.WEIGHT_KWARGS), file=file)
        print("Closure Function:", self.CLOSURE.value["name"], "\n", file=file)

        print("-- Full Monitors on Full Graph --", file=file)
        PrintCascadeResults(results.NUM_NODES_FULL, results.NUM_INFECTED_FF, results.NON_SOURCE_INF_FF, results.INFECTED_TARGETS_FF,
                            results.NUM_TARGETS, results.NUM_MONITORS_FULL, results.CASCADE_ITERATIONS_FF, file=file)

        print("-- Part Monitors on Full Graph --", file=file)
        PrintCascadeResults(results.NUM_NODES_FULL, results.NUM_INFECTED_FP, results.NON_SOURCE_INF_FP, results.INFECTED_TARGETS_FP,
                            results.NUM_TARGETS, results.NUM_MONITORS_PART, results.CASCADE_ITERATIONS_FP, file=file)

        print("-- Recv Monitors on Recv Graph --", file=file)
        PrintCascadeResults(results.NUM_NODES_RECV, results.NUM_INFECTED_RR, results.NON_SOURCE_INF_RR, results.INFECTED_TARGETS_RR,
                            results.NUM_TARGETS, results.NUM_MONITORS_RECV, results.CASCADE_ITERATIONS_RR, file=file)

        print("-- Recv Monitors on Full Graph --", file=file)
        PrintCascadeResults(results.NUM_NODES_FULL, results.NUM_INFECTED_FR, results.NON_SOURCE_INF_FR, results.INFECTED_TARGETS_FR,
                            results.NUM_TARGETS, results.NUM_MONITORS_RECV, results.CASCADE_ITERATIONS_FR, file=file)

        print("-- Monitor Placement --", file=file)
        print("Monitors on full:", results.MONITORS_FULL, file=file)
        print("Monitors on part:", results.MONITORS_PART, file=file)
        print("Monitors on recv:", results.MONITORS_RECV, file=file)

        print("==============================================================", file=file)
        file.close()


########################################################################################################################
# --- GENERATION TEST STARTING CODE ---
########################################################################################################################

def single_test_generation(**kwargs):
    mt = MonitorTester()
    mt.NUM_NODES = kwargs["NUM_NODES"]
    mt.NUM_TO_HIDE = kwargs["NUM_TO_HIDE"]
    mt.NUM_SOURCES = kwargs["NUM_SOURCES"]
    mt.NUM_TARGETS = kwargs["NUM_TARGETS"]
    mt.FOLDER = kwargs["FOLDER"]
    mt.DISTRIBUTION = kwargs["DISTRIBUTION"]
    mt.DISTRIBUTION_KWARGS = kwargs["DISTRIBUTION_KWARGS"]
    mt.GENERATION = kwargs["GENERATION"]
    mt.GENERATION_KWARGS = kwargs["GENERATION_KWARGS"]
    mt.CLOSURE = kwargs["CLOSURE"]
    mt.CLOSURE_KWARGS = kwargs["CLOSURE_KWARGS"]
    mt.WEIGHT = kwargs["WEIGHT"]
    mt.WEIGHT_KWARGS = kwargs["WEIGHT_KWARGS"]

    mt.PRINT_TO_FILE = ROOT_DIR + "/Test/Monitoring/Results/SyntheticDatasets/" + kwargs["FILENAME"]
    full, part, recv = mt.generate_synthetic_dataset(verbose=True)
    return mt.perform_test(full, part, recv)
    # mt.test_real_dataset(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote.txt", directed=True, generate=True)


def parallel_test_generation():
    # This is the variable hyperparameter. The test will parallelize N runs of the algorithm, once for each
    # value in the list below. This helps finding a "threshold" where more that X% of hidden will make the
    # Graph Recovery meaningless (e.g. too few nodes to perform inference from)
    # NUM_TO_HIDE = [50, 100, 200, 250, 350, 500, 650, 750, 800, 900]
    NUM_TO_HIDE = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Control this dictionary to set the hyperparameters of the algorithm for testing. Do not change the code
    # within the Parallel statement below, as it ensures consistency between graph analysis and report filenames
    fixed_hyperparameters = {
        "num_nodes": 150,
        "num_sources": 10,
        "num_targets": 10,
        "generation": EGraphGenerationFunction.ERandomSparseDirectedGraph,
        "generation_kwargs": {},
        "distribution": ENodeHidingSelectionFunction.EDegreeDistribution,
        "distribution_kwargs": {},
        "closure": EClosureFunction.ETotalClosure,
        "closure_kwargs": {},
        "weight": EWeightSetterFunction.EUniformWeights,
        "weight_kwargs": {"min_val": 0, "max_val": 0.1}
    }

    # Main call for parallelization. Do not change code below this line
    Parallel(n_jobs=len(NUM_TO_HIDE))(delayed(single_test_generation)(**{
        "NUM_NODES": fixed_hyperparameters["num_nodes"],
        "NUM_TO_HIDE": NUM_TO_HIDE[i],
        "NUM_SOURCES": fixed_hyperparameters["num_sources"],
        "NUM_TARGETS": fixed_hyperparameters["num_targets"],
        "GENERATION": fixed_hyperparameters["generation"],
        "GENERATION_KWARGS": fixed_hyperparameters["generation_kwargs"],
        "DISTRIBUTION": fixed_hyperparameters["distribution"],
        "DISTRIBUTION_KWARGS": fixed_hyperparameters["distribution_kwargs"],
        "CLOSURE": fixed_hyperparameters["closure"],
        "CLOSURE_KWARGS": fixed_hyperparameters["closure_kwargs"],
        "WEIGHT": fixed_hyperparameters["weight"],
        "WEIGHT_KWARGS": fixed_hyperparameters["weight_kwargs"],
        "FOLDER": "Synthetic",
        "FILENAME": GenerateReportFilename(
            fixed_hyperparameters["num_nodes"], NUM_TO_HIDE[i],
            fixed_hyperparameters["num_sources"], fixed_hyperparameters["num_targets"],
            fixed_hyperparameters["generation"].value["short_name"],
            fixed_hyperparameters["distribution"].value["short_name"],
            fixed_hyperparameters["closure"].value["short_name"],
            fixed_hyperparameters["weight"].value["short_name"],
        )}) for i in range(len(NUM_TO_HIDE)))


########################################################################################################################
# --- GENERATION TEST STARTING CODE ---
########################################################################################################################


def single_test_repeat(**kwargs):
    mt = MonitorTester()
    mt.NUM_TO_HIDE = kwargs["NUM_TO_HIDE"]
    mt.NUM_SOURCES = kwargs["NUM_SOURCES"]
    mt.NUM_TARGETS = kwargs["NUM_TARGETS"]
    mt.FOLDER = kwargs["FOLDER"]
    mt.DISTRIBUTION = kwargs["DISTRIBUTION"]
    mt.DISTRIBUTION_KWARGS = kwargs["DISTRIBUTION_KWARGS"]
    mt.GENERATION = kwargs["GENERATION"]
    mt.GENERATION_KWARGS = kwargs["GENERATION_KWARGS"]
    mt.CLOSURE = kwargs["CLOSURE"]
    mt.CLOSURE_KWARGS = kwargs["CLOSURE_KWARGS"]
    mt.WEIGHT = kwargs["WEIGHT"]
    mt.WEIGHT_KWARGS = kwargs["WEIGHT_KWARGS"]
    mt.PRINT_TO_FILE = ROOT_DIR + "/Test/Monitoring/Results/SyntheticDatasets/" + kwargs["FILENAME"]
    full, part, recv = mt.read_synthetic_dataset(path=kwargs["PATH"], folder=kwargs["FOLDER"], index=kwargs["INDEX"])
    return mt.perform_test(full, part, recv)
    # mt.test_real_dataset(ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote.txt", directed=True, generate=True)


def parallel_test_repeat(mt: MonitorTester):

    # Main call for parallelization. Do not change code below this line
    results = Parallel(n_jobs=1)(delayed(single_test_repeat)(**{
        "NUM_TO_HIDE": mt.NUM_TO_HIDE,
        "INDEX": mt.TRIPLE_INDEX,
        "NUM_SOURCES": mt.NUM_SOURCES,
        "NUM_TARGETS": mt.NUM_TARGETS,
        "GENERATION": mt.GENERATION,
        "GENERATION_KWARGS": mt.GENERATION_KWARGS,
        "DISTRIBUTION": mt.DISTRIBUTION,
        "DISTRIBUTION_KWARGS": mt.DISTRIBUTION_KWARGS,
        "CLOSURE": mt.CLOSURE,
        "CLOSURE_KWARGS": mt.CLOSURE_KWARGS,
        "WEIGHT": mt.WEIGHT,
        "WEIGHT_KWARGS": mt.WEIGHT_KWARGS,
        "PATH": mt.DATASET_PATH,
        "FOLDER": mt.FOLDER,
        "FILENAME": f"test_{i}_on_index_{mt.TRIPLE_INDEX}.txt"
    }) for i in range(50))

    agg_results = MonitorTestReport.aggregate_results(results)
    for key in agg_results:
        print(key, agg_results[key])


########################################################################################################################
# --- REAL DATASET TESTING ---
########################################################################################################################


def real_dataset_test():
    mt = MonitorTester()
    mt.TRIPLE_INDEX = -1
    mt.NUM_TO_HIDE = int(mt.NUM_NODES * 0.1)
    mt.NUM_SOURCES = 20
    mt.NUM_TARGETS = 20
    mt.GENERATION = EGraphGenerationFunction.ERealGraph
    mt.GENERATION_KWARGS = {}
    mt.CLOSURE = EClosureFunction.ETotalClosure
    mt.CLOSURE_KWARGS = {}
    mt.DISTRIBUTION = ENodeHidingSelectionFunction.EUniformDistribution
    mt.DISTRIBUTION_KWARGS = {}
    mt.WEIGHT = EWeightSetterFunction.EUniformWeights
    mt.WEIGHT_KWARGS = {"min_val": 0, "max_val": 0.1}
    mt.DATASET_PATH = ROOT_DIR + "/Datasets"
    mt.FOLDER = "Real"
    mt.PRINT_TO_FILE = None
    mt.TEST_PARAMS = ""
    mt.test_real_dataset(os.path.join(mt.DATASET_PATH, "Real", "Wiki-Vote.txt"))


if __name__ == "__main__":
    # real_dataset_test()

    mt = MonitorTester()
    mt.TRIPLE_INDEX = -1
    mt.NUM_NODES = 150
    mt.NUM_TO_HIDE = 100
    mt.NUM_SOURCES = 10
    mt.NUM_TARGETS = 10
    mt.GENERATION = EGraphGenerationFunction.ERandomSparseDirectedGraph
    mt.GENERATION_KWARGS = {}
    mt.CLOSURE = EClosureFunction.ETotalClosure
    mt.CLOSURE_KWARGS = {}
    mt.DISTRIBUTION = ENodeHidingSelectionFunction.EUniformDistribution
    mt.DISTRIBUTION_KWARGS = {}
    mt.WEIGHT = EWeightSetterFunction.EInDegreeWeights
    mt.WEIGHT_KWARGS = {}
    mt.DATASET_PATH = ROOT_DIR + "/Datasets"
    mt.FOLDER = "Synthetic"
    mt.PRINT_TO_FILE = None
    mt.TEST_PARAMS = ""

    full, part, recv = GenerateRandomGraphTriple(
        mt.NUM_NODES, mt.NUM_TO_HIDE, mt.GENERATION.value["function"], mt.GENERATION_KWARGS, mt.DISTRIBUTION.value["function"],
        mt.DISTRIBUTION_KWARGS, mt.CLOSURE.value["function"], mt.CLOSURE_KWARGS, None, "deg", False)

    index = WriteGraphTriple(mt.DATASET_PATH, mt.FOLDER, GenerateGraphFilename(
            mt.NUM_NODES, mt.NUM_TO_HIDE, mt.GENERATION.value["short_name"],
            mt.DISTRIBUTION.value["short_name"], mt.CLOSURE.value["short_name"],
            mt.WEIGHT.value["short_name"]), full, part, recv)

    mt.TRIPLE_INDEX = index

    parallel_test_repeat(mt)


