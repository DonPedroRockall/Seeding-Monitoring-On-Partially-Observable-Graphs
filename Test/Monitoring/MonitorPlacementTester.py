import operator
import os
import statistics
import sys

from joblib import delayed, Parallel
from networkx import single_source_dijkstra_path_length

from Common.DrawGraph import DrawGraph
from Common.GraphUtilities import *
from DiffusionModels.IndependentCascade import IndependentCascadeWithMonitors, GetInfectedSubgraphs
from GraphRecovery.GraphRecoverer import InfluentialNodeRecovery, ExpandGraph
from Monitoring.MonitorPlacement.Monitor import PlaceMonitors
from Monitoring.MonitorPlacement.MonitorUtility import PrintCascadeResults, GatherCascadeResults
from Monitoring.SourceIdentification.SourceIdentificator import IdentifySources
from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DatasetReader import WriteGraphTriple, ReadGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.GraphGenerator import EGraphGenerationFunction
from Test.Common.HidingFunctions import *
from Test.Common.WeightGenerator import EWeightSetterFunction
from definitions import ROOT_DIR


def all_pairs_dijkstra_path_length(Graph, cutoff=None, weight="weight"):
    length = single_source_dijkstra_path_length
    return {n: length(Graph, n, cutoff=cutoff, weight=weight) for n in Graph}


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
        # Nodes saved
        self.FULL_NODES_SAVED_PER_MONITOR = 0
        self.PART_NODES_SAVED_PER_MONITOR = 0
        self.RECV_NODES_SAVED_PER_MONITOR = 0
        # Targets saved
        self.FULL_TARGETS_SAVED_PER_MONITOR = 0
        self.PART_TARGETS_SAVED_PER_MONITOR = 0
        self.RECV_TARGETS_SAVED_PER_MONITOR = 0

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


class SourceIdentificationTestReport:
    def __init__(self):
        self.VIRTUAL_SOURCES = 0
        self.NUM_EST_SOURCES_FULL = 0
        self.NUM_EST_SOURCES_PART = 0
        self.NUM_EST_SOURCES_RECV = 0
        self.TP_FULL = 0
        self.TP_PART = 0
        self.TP_RECV = 0
        self.FP_FULL = 0
        self.FP_PART = 0
        self.FP_RECV = 0
        self.FN_FULL = 0
        self.FN_PART = 0
        self.FN_RECV = 0
        self.FULL_DISTANCE_0 = 0
        self.PART_DISTANCE_0 = 0
        self.RECV_DISTANCE_0 = 0
        self.FULL_DISTANCE_1 = 0
        self.PART_DISTANCE_1 = 0
        self.RECV_DISTANCE_1 = 0
        self.FULL_DISTANCE_2 = 0
        self.PART_DISTANCE_2 = 0
        self.RECV_DISTANCE_2 = 0
        self.FULL_DISTANCE_3 = 0
        self.PART_DISTANCE_3 = 0
        self.RECV_DISTANCE_3 = 0
        self.FULL_DISTANCE_4 = 0
        self.PART_DISTANCE_4 = 0
        self.RECV_DISTANCE_4 = 0
        self.FULL_DISTANCE_5 = 0
        self.PART_DISTANCE_5 = 0
        self.RECV_DISTANCE_5 = 0
        self.FULL_DISTANCE_6 = 0
        self.PART_DISTANCE_6 = 0
        self.RECV_DISTANCE_6 = 0

    @staticmethod
    def aggregate_results(results: list):

        # Purge None values from the list
        new_results = []
        for result in results:
            if result is not None:
                new_results.append(result)

        agg_res = {}

        for key in vars(new_results[0]).keys():
            if type(vars(new_results[0])[key]) is not int and type(vars(new_results[0])[key]) is not float:
                continue
            values = []
            for res in new_results:
                values.append(vars(res)[key])
            if len(values) > 1:
                if "DISTANCE" in key:
                    agg_res[key] = (round(statistics.mean(values), 2), statistics.stdev(values))
                else:
                    agg_res[key] = (round(statistics.mean(values), 1), statistics.stdev(values))
            else:
                agg_res[key] = (values[0], 0)
        return agg_res


class MonitorTester:

    # Common test config
    def __init__(self):
        self.NUM_NODES = 150
        self.NUM_TO_HIDE = 10
        self.NUM_SOURCES = 10
        self.NUM_TARGETS = 10
        self.CNODES = 100
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

    def get_real_dataset_triple(self, path, theta=None):
        full = networkx.read_edgelist(path, create_using=networkx.DiGraph if self.WEIGHT_KWARGS["directed"] else networkx.Graph, nodetype=int)
        self.NUM_NODES = full.number_of_nodes()
        nth = self.DISTRIBUTION.value["function"](full, self.NUM_TO_HIDE)
        part = self.CLOSURE.value["function"](full.copy(), nth)
        if theta is None:
            recv, _ = InfluentialNodeRecovery(part, self.NUM_TO_HIDE, 2)
        else:
            recv, _ = ExpandGraph(part, self.NUM_TO_HIDE, theta)
        print("done")
        return full, part, recv

    def test_setup(self, full, part, recv):
        # Generate the weights for the full graph
        self.WEIGHT.value["function"](full, attribute="weight", force=True, **self.WEIGHT_KWARGS)
        # Copy the weights to the other two graphs
        SetSameWeightsToOtherGraphs(full, [part, recv])

        # Assign random edges to the newly reconstructed edges
        self.WEIGHT.value["function"](recv, attribute="weight", force=False, **self.WEIGHT_KWARGS)

        # Choose sources and targets (they have to be in all 3 graphs)
        valid_nodes = set(part.nodes())

        if len(valid_nodes) < self.NUM_SOURCES + self.NUM_TARGETS:
            print(f"Cannot continue with the algorithm, as there are not enough nodes in partial graph to "
                  f"select {self.NUM_SOURCES} sources and {self.NUM_TARGETS} targets")
            return

        sources = list(random.sample(list(valid_nodes), self.NUM_SOURCES))
        for src in sources:
            valid_nodes.remove(src)
        targets = list(random.sample(list(valid_nodes), self.NUM_TARGETS))

        # Compute the set of virtual nodes
        # virtual_set = GetVirtualNodesByDifference(part, recv)
        virtual_set = GetVirtualNodesByNodeLabel(recv, "RECV")

        return sources, targets, virtual_set

    def perform_source_identification_test(self, full, part, recv):
        print("started task")
        real_sources, _, virtual_set = self.test_setup(full, part, recv)

        source_boost = full.number_of_nodes() / (full.number_of_nodes() - len(virtual_set))
        cprint(bcolors.OKCYAN, "Boosting source identification by a factor of: ", source_boost)

        # infected_nodes_full = IndependentCascadeWithMonitors(full, real_sources, [], 8)
        # infected_nodes_part = IndependentCascadeWithMonitors(part, real_sources, [], 8)
        # infected_nodes_recv = IndependentCascadeWithMonitors(recv, real_sources, [], 8)

        infected_full = IndependentCascadeWithMonitors(full, seeds=real_sources, monitors=[], steps=20)
        infected_part = IndependentCascadeWithMonitors(part, seeds=real_sources, monitors=[], steps=20)
        infected_recv = IndependentCascadeWithMonitors(recv, seeds=real_sources, monitors=[], steps=20)

        flat_full = set()
        for lis in infected_full:
            flat_full = flat_full.union(lis)

        flat_part = set()
        for lis in infected_part:
            flat_part = flat_part.union(lis)

        flat_recv = set()
        for lis in infected_recv:
            flat_recv = flat_recv.union(lis)

        # flat_full = AdaptiveIntervalCascade(full, real_sources, steps=1000, full_intervals=[75, 150], max_iterations=100, return_interval=False)
        # flat_part = AdaptiveIntervalCascade(part, real_sources, steps=1000, full_intervals=[75, 150], max_iterations=100, return_interval=False)
        # flat_recv = AdaptiveIntervalCascade(recv, real_sources, steps=1000, full_intervals=[75, 150], max_iterations=100, return_interval=False)

        inf_subgraph_full = GetInfectedSubgraphs(full, flat_full)
        inf_subgraph_part = GetInfectedSubgraphs(part, flat_part)
        inf_subgraph_recv = GetInfectedSubgraphs(recv, flat_recv)

        est_sources_full, _ = IdentifySources(full, len(real_sources), inf_subgraph_full)
        est_sources_part, _ = IdentifySources(part, len(real_sources), inf_subgraph_part)
        est_sources_recv, discarded = IdentifySources(recv, int(source_boost * len(real_sources)), inf_subgraph_recv, virtual_set=virtual_set)

        if len(est_sources_full) == 0 or len(est_sources_part) == 0 or len(est_sources_recv) == 0:
            cprint(bcolors.FAIL, "DISCARDING RESULT")
            return None

        # Result gathering
        results = SourceIdentificationTestReport()
        results.NUM_EST_SOURCES_FULL = len(est_sources_full)
        results.NUM_EST_SOURCES_PART = len(est_sources_part)
        results.NUM_EST_SOURCES_RECV = len(est_sources_recv)
        results.VIRTUAL_SOURCES = discarded

        for node in est_sources_full:
            if node in real_sources:
                results.TP_FULL += 1
            else:
                results.FP_FULL += 1

        for node in est_sources_part:
            if node in real_sources:
                results.TP_PART += 1
            else:
                results.FP_PART += 1

        for node in est_sources_recv:
            if node in real_sources:
                results.TP_RECV += 1
            else:
                results.FP_RECV += 1

        for node in real_sources:
            if node not in est_sources_full:
                results.FN_FULL += 1
            if node not in est_sources_part:
                results.FN_PART += 1
            if node not in est_sources_recv:
                results.FN_RECV += 1

        # Distance measurements
        print("Measuring distances...")
        full_distance_dict = {}
        for est_source in est_sources_full:
            full_distance_dict[est_source] = GetDistanceToClosestRealSource(full, est_source, real_sources)
        for num in range(7):
            results.__setattr__("FULL_DISTANCE_" + str(num), operator.countOf(full_distance_dict.values(), num))

        part_distance_dict = {}
        for est_source in est_sources_part:
            part_distance_dict[est_source] = GetDistanceToClosestRealSource(part, est_source, real_sources)
        for num in range(7):
            results.__setattr__("PART_DISTANCE_" + str(num), operator.countOf(part_distance_dict.values(), num))

        recv_distance_dict = {}
        for est_source in est_sources_recv:
            recv_distance_dict[est_source] = GetDistanceToClosestRealSource(recv, est_source, real_sources)
        for num in range(7):
            results.__setattr__("RECV_DISTANCE_" + str(num), operator.countOf(recv_distance_dict.values(), num))
        print("Done measuring!")

        """
        print("Measuring Distances...")
        dst_full = all_pairs_dijkstra_path_length(full.to_undirected())

        avg = 0
        for est_source in est_sources_full:
            min_dst = 1000
            for real_source in real_sources:
                if est_source in dst_full and real_source in dst_full[est_source]:
                    dst = dst_full[est_source][real_source]
                    if dst < min_dst:
                        min_dst = dst
            avg += dst
        results.FULL_DISTANCE = round(avg / len(real_sources), 1)

        print("Full distances measured...")

        dst_part = all_pairs_dijkstra_path_length(part.to_undirected())
        avg = 0
        for est_source in est_sources_part:
            min_dst = 1000
            for real_source in real_sources:
                if est_source in dst_part and real_source in dst_part[est_source]:
                    dst = dst_part[est_source][real_source]
                    if dst < min_dst:
                        min_dst = dst
            avg += dst
        results.PART_DISTANCE = round(avg / len(real_sources), 1)

        print("Part distances measured...")

        dst_recv = all_pairs_dijkstra_path_length(recv.to_undirected())
        avg = 0
        for est_source in est_sources_recv:
            min_dst = 1000
            for real_source in real_sources:
                if est_source in dst_recv and real_source in dst_recv[est_source]:
                    dst = dst_recv[est_source][real_source]
                    if dst < min_dst:
                        min_dst = dst
            avg += dst
        results.RECV_DISTANCE = round(avg / len(real_sources), 1)

        print("Recv distances measured...")
        """

        return results


    def perform_test(self, full, part, recv):

        sources, targets, virtual_set = self.test_setup(full, part, recv)

        # Place the monitors on the 3 graphs
        cprint(bcolors.OKGREEN, "Running monitor placement...")

        # Run the monitor placement algorithm on all the 3 graphs
        monitors_full, _ = PlaceMonitors(full, sources, targets, c_nodes=20, verbose=True)
        monitors_part, _ = PlaceMonitors(part, sources, targets, c_nodes=20, verbose=True)
        monitors_recv, _ = PlaceMonitors(recv, sources, targets, c_nodes=self.CNODES, virtual_set=virtual_set, verbose=True)

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

    def perform_th_test(self, full, part, recv, threshold):

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

        # Now the different part: increase the number of sources as to increase the number of infected nodes above the threshold
        sources = []
        full_done = False
        part_done = False
        recv_done = False
        sources_full = []
        sources_part = []
        sources_recv = []

        # Targets are fixed
        targets = list(random.sample(list(valid_nodes), self.NUM_TARGETS))
        for trg in targets:
            valid_nodes.remove(trg)

        while not (full_done and part_done and recv_done):

            if len(valid_nodes) < self.NUM_SOURCES + self.NUM_TARGETS:
                print(f"Cannot continue with the algorithm, as there are not enough nodes in partial graph to "
                      f"select {self.NUM_SOURCES} sources and {self.NUM_TARGETS} targets")
                return

            # Add more sources to the graph
            new_sources = list(random.sample(list(valid_nodes), self.NUM_SOURCES))
            sources.extend(new_sources)
            for src in new_sources:
                valid_nodes.remove(src)

            # Compute the set of virtual nodes
            virtual_set = GetVirtualNodesByNodeLabel(recv, "RECV")

            # Perform IC to see the results and gather them
            if not full_done:
                ic_full_full = IndependentCascadeWithMonitors(full, sources, [], 100)
                cascade_results = GatherCascadeResults(ic_full_full, full, sources, targets, [])
                if cascade_results["num_of_infected"] > threshold:
                    sources_full = sources
                    full_done = True


            if not part_done:
                ic_full_part = IndependentCascadeWithMonitors(part, sources, [], 100)
                cascade_results = GatherCascadeResults(ic_full_part, full, sources, targets, [])
                if cascade_results["num_of_infected"] > threshold:
                    sources_part = sources
                    part_done = True

            if not recv_done:
                ic_full_recv = IndependentCascadeWithMonitors(recv, sources, [], 100)
                cascade_results = GatherCascadeResults(ic_full_recv, full, sources, targets, [])
                if cascade_results["num_of_infected"] > threshold:
                    sources_recv = sources
                    recv_done = True

        monitors_full, _ = PlaceMonitors(full, sources_full, targets, verbose=True, c_nodes=100)
        monitors_part, _ = PlaceMonitors(part, sources_part, targets, verbose=True, c_nodes=100)
        monitors_recv, _ = PlaceMonitors(recv, sources_recv, targets, virtual_set=virtual_set, c_nodes=100, verbose=True)

        ic_res = IndependentCascadeWithMonitors(full, sources, monitors_full, 100)
        cascade_results = GatherCascadeResults(ic_res, full, sources, targets, [])
        finf_full = cascade_results["num_of_infected"]

        ic_res = IndependentCascadeWithMonitors(full, sources, monitors_part, 100)
        cascade_results = GatherCascadeResults(ic_res, full, sources, targets, [])
        finf_part = cascade_results["num_of_infected"]

        ic_res = IndependentCascadeWithMonitors(full, sources, monitors_recv, 100)
        cascade_results = GatherCascadeResults(ic_res, full, sources, targets, [])
        finf_recv = cascade_results["num_of_infected"]

        print("FULL (", len(sources_full), sources_full)
        print("PART (", len(sources_part), sources_part)
        print("RECV (", len(sources_recv), sources_recv)

        print("FULL (", len(monitors_full), monitors_full)
        print("PART (", len(monitors_part), monitors_part)
        print("RECV (", len(monitors_recv), monitors_recv)

        print("FULL", finf_full)
        print("PART", finf_part)
        print("RECV", finf_recv)


    def perform_fixed_monitors_test(self, full, part, recv, num_monitors):

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

        # Now the different part: increase the number of sources as to increase the number of infected nodes above the threshold
        sources = []
        full_done = False
        part_done = False
        recv_done = False
        sources_full = []
        sources_part = []
        sources_recv = []
        monitors_full = []
        monitors_part = []
        monitors_recv = []

        # Targets are fixed
        targets = list(random.sample(list(valid_nodes), self.NUM_TARGETS))
        for trg in targets:
            valid_nodes.remove(trg)

        while not (full_done and part_done and recv_done):

            if len(valid_nodes) < self.NUM_SOURCES:
                cprint(bcolors.FAIL, f"Cannot continue with the algorithm, as there are not enough nodes in partial graph to "
                      f"select {self.NUM_SOURCES} sources and {self.NUM_TARGETS} targets")
                num_monitors = -1  # disable hard limit and let the algorithm continue
            else:
                # Add more sources to the test
                new_sources = list(random.sample(list(valid_nodes), self.NUM_SOURCES))
                sources.extend(new_sources)
                print("extended sources: new len", len(sources))
                for src in new_sources:
                    valid_nodes.remove(src)

            # Compute the set of virtual nodes
            virtual_set = GetVirtualNodesByNodeLabel(recv, "RECV")

            if not full_done:
                monitors_full, _ = PlaceMonitors(full, sources, targets, c_nodes=100, verbose=True)
                if len(monitors_full) >= num_monitors:
                    sources_full = sources.copy()
                    full_done = True
                    print("full done! with", len(sources_full))


            if not part_done:
                monitors_part, _ = PlaceMonitors(part, sources, targets, c_nodes=100, verbose=True)
                if len(monitors_part) >= num_monitors:
                    sources_part = sources.copy()
                    part_done = True
                    print("part done! with", len(sources_part))

            if not recv_done:
                monitors_recv, _ = PlaceMonitors(recv, sources, targets, c_nodes=100, virtual_set=virtual_set, verbose=True)
                if len(monitors_recv) >= num_monitors:
                    sources_recv = sources.copy()
                    recv_done = True
                    print("recv done! with", len(sources_recv))


        ic_res = IndependentCascadeWithMonitors(full, sources, monitors_full, 100)
        cascade_results = GatherCascadeResults(ic_res, full, sources, targets, [])
        finf_full = cascade_results["num_of_infected"]

        ic_res = IndependentCascadeWithMonitors(full, sources, monitors_part, 100)
        cascade_results = GatherCascadeResults(ic_res, full, sources, targets, [])
        finf_part = cascade_results["num_of_infected"]

        ic_res = IndependentCascadeWithMonitors(full, sources, monitors_recv, 100)
        cascade_results = GatherCascadeResults(ic_res, full, sources, targets, [])
        finf_recv = cascade_results["num_of_infected"]

        print("FULL (", len(sources_full), sources_full)
        print("PART (", len(sources_part), sources_part)
        print("RECV (", len(sources_recv), sources_recv)

        print("FULL (", len(monitors_full), monitors_full)
        print("PART (", len(monitors_part), monitors_part)
        print("RECV (", len(monitors_recv), monitors_recv)

        print("FULL", finf_full)
        print("PART", finf_part)
        print("RECV", finf_recv)

        DrawGraph(full, graph_name="FULL")
        DrawGraph(part, graph_name="PART")
        DrawGraph(recv, graph_name="RECV")


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
        mtr.PERC_RECOVERED_HIDDEN = mtr.NUM_RECOVERED / mtr.NUM_HIDDEN if mtr.PERC_HIDDEN != 0 else "<no hidden nodes>"
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
        # # Nodes saved per monitor
        mtr.FULL_NODES_SAVED_PER_MONITOR = round((mtr.NUM_NODES_FULL - mtr.NUM_INFECTED_FF) / mtr.NUM_MONITORS_FULL, 3)
        mtr.PART_NODES_SAVED_PER_MONITOR = round((mtr.NUM_NODES_PART - mtr.NUM_INFECTED_FP) / mtr.NUM_MONITORS_PART, 3)
        mtr.RECV_NODES_SAVED_PER_MONITOR = round((mtr.NUM_NODES_RECV - mtr.NUM_INFECTED_FR) / mtr.NUM_MONITORS_RECV, 3)
        # Targets saved per monitor
        mtr.FULL_NODES_SAVED_PER_MONITOR = round((mtr.NUM_TARGETS - mtr.INFECTED_TARGETS_FF) / mtr.NUM_MONITORS_FULL, 6)
        mtr.PART_NODES_SAVED_PER_MONITOR = round((mtr.NUM_TARGETS - mtr.INFECTED_TARGETS_FP) / mtr.NUM_MONITORS_PART, 6)
        mtr.RECV_NODES_SAVED_PER_MONITOR = round((mtr.NUM_TARGETS - mtr.INFECTED_TARGETS_FR) / mtr.NUM_MONITORS_RECV, 6)

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
    results = Parallel(n_jobs=10)(delayed(single_test_repeat)(**{
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
    }) for i in range(40))

    agg_results = MonitorTestReport.aggregate_results(results)
    for key in agg_results:
        print(key, agg_results[key])


########################################################################################################################
# --- REAL DATASET TESTING ---
########################################################################################################################


def real_dataset_test():
    mt = MonitorTester()
    mt.TRIPLE_INDEX = -1
    mt.NUM_TO_HIDE = 1000
    mt.NUM_SOURCES = 100
    mt.NUM_TARGETS = 20
    mt.GENERATION = EGraphGenerationFunction.ERealGraph
    mt.GENERATION_KWARGS = {}
    mt.CLOSURE = EClosureFunction.ETotalClosure
    mt.CLOSURE_KWARGS = {}
    mt.DISTRIBUTION = ENodeHidingSelectionFunction.EUniformDistribution
    mt.DISTRIBUTION_KWARGS = {}
    mt.WEIGHT = EWeightSetterFunction.EInDegreeWeights
    mt.WEIGHT_KWARGS = {"min_val": 0, "max_val": 0.1, "directed": True}
    mt.DATASET_PATH = ROOT_DIR + "/Datasets"
    mt.FOLDER = "Real"
    mt.PRINT_TO_FILE = None
    mt.TEST_PARAMS = ""
    f, p, r = mt.get_real_dataset_triple(os.path.join(mt.DATASET_PATH, "Real", "facebook_combined.txt"))
    return mt, f, p, r


def parallel_real_test_repeat():
    mt, full, part, recv = real_dataset_test()
    # Main call for parallelization.
    results = Parallel(n_jobs=10)(delayed(mt.perform_test)(full, part, recv) for _ in range(10))
    # Result aggregation and print
    agg_results = MonitorTestReport.aggregate_results(results)
    for key in agg_results:
        print(key, agg_results[key])


def parallel_si_test_repeat():
    mt = MonitorTester()
    mt.TRIPLE_INDEX = -1
    mt.NUM_TO_HIDE = 750
    mt.NUM_SOURCES = 20
    mt.NUM_TARGETS = 20
    mt.GENERATION = EGraphGenerationFunction.ERandomSparseDirectedGraph
    mt.GENERATION_KWARGS = {}
    mt.CLOSURE = EClosureFunction.ETotalClosure
    mt.CLOSURE_KWARGS = {}
    mt.DISTRIBUTION = ENodeHidingSelectionFunction.EUniformDistribution
    mt.DISTRIBUTION_KWARGS = {}
    mt.WEIGHT = EWeightSetterFunction.EInDegreeWeights
    mt.WEIGHT_KWARGS = {"min_val": 0, "max_val": 1, "directed": True}
    mt.DATASET_PATH = ROOT_DIR + "/Datasets"
    mt.FOLDER = "Synthetic"
    mt.PRINT_TO_FILE = None
    mt.TEST_PARAMS = ""

    # full, part, recv = mt.read_synthetic_dataset(path=mt.DATASET_PATH, folder=mt.FOLDER, index=496)  #525, 43.2, 45.6
    # full, part, recv = mt.generate_synthetic_dataset()
    import os
    # full, part, recv = mt.get_real_dataset_triple(os.path.join(mt.DATASET_PATH, "Real", "as_route_views.txt"))
    full, part, recv = mt.get_real_dataset_triple(os.path.join(mt.DATASET_PATH, "Real", "email-Eu-core.txt"))
    # theta = np.array([[0.58716, 0.375428], [0.34779, 0.790919]])

    print("Ended recovery")

    # Main call for parallelization.
    results = Parallel(n_jobs=10)(delayed(mt.perform_source_identification_test)(full, part, recv) for _ in range(10))

    # Result aggregation and print
    agg_results = SourceIdentificationTestReport.aggregate_results(results)

    array = []
    for key in agg_results:
        array.append(agg_results[key])
        print(key, agg_results[key])

    import pandas as pd
    import os

    df = pd.DataFrame(array).T
    file = "F:\Backup\Projects\PyCharm\Thesis\ThesisProject\Test\SourceIdentification\Result.xlsx"
    os.remove(file)
    df.to_excel(excel_writer=file)
    os.startfile(file)


def parallel_main_test_repeat():
    mt = MonitorTester()
    mt.TRIPLE_INDEX = -1
    mt.NUM_TO_HIDE = 10
    mt.NUM_SOURCES = 20
    mt.NUM_TARGETS = 20
    mt.GENERATION = EGraphGenerationFunction.ERandomSparseDirectedGraph
    mt.GENERATION_KWARGS = {}
    mt.CLOSURE = EClosureFunction.ETotalClosure
    mt.CLOSURE_KWARGS = {}
    mt.DISTRIBUTION = ENodeHidingSelectionFunction.EUniformDistribution
    mt.DISTRIBUTION_KWARGS = {}
    mt.WEIGHT = EWeightSetterFunction.EInDegreeWeights
    mt.WEIGHT_KWARGS = {"min_val": 0, "max_val": 0.1, "directed": True}
    mt.DATASET_PATH = ROOT_DIR + "/Datasets"
    mt.FOLDER = "Synthetic"
    mt.PRINT_TO_FILE = None
    mt.TEST_PARAMS = ""

    mt.CNODES = 10

    full, part, recv = mt.generate_synthetic_dataset()

    # Main call for parallelization.
    results = Parallel(n_jobs=10)(delayed(mt.perform_test)(full, part, recv) for _ in range(30))

    # Result aggregation and print
    agg_results = MonitorTestReport.aggregate_results(results)

    array = []
    for key in agg_results:
        array.append(agg_results[key])
        print(key, agg_results[key])

    # import pandas as pd
    # import os
    #
    # df = pd.DataFrame(array).T
    # file = f"F:\Backup\Projects\PyCharm\Thesis\ThesisProject\Test\SourceIdentification\Result{mt.CNODES}.xlsx"
    # os.remove(file)
    # df.to_excel(excel_writer=file)
    # os.startfile(file)



if __name__ == "__main__":

    import atexit

    def exit_handler():
        import winsound
        winsound.Beep(1000, 1000)

    atexit.register(exit_handler)

    # parallel_real_test_repeat()
    parallel_si_test_repeat()
    # parallel_main_test_repeat()

    # mt = MonitorTester()
    # mt.TRIPLE_INDEX = -1
    # mt.NUM_NODES = 150  # 250
    # mt.NUM_TO_HIDE = 100  # 20
    # mt.NUM_SOURCES = 10
    # mt.NUM_TARGETS = 10
    # mt.GENERATION = EGraphGenerationFunction.ECorePeripheryDirectedGraph
    # mt.GENERATION_KWARGS = {}
    # mt.CLOSURE = EClosureFunction.ECrawlerClosure
    # mt.CLOSURE_KWARGS = {}
    # mt.DISTRIBUTION = ENodeHidingSelectionFunction.EUniformDistribution
    # mt.DISTRIBUTION_KWARGS = {}
    # mt.WEIGHT = EWeightSetterFunction.EUniformWeights
    # mt.WEIGHT_KWARGS = {"min_val": 0, "max_val": 0.1}
    # # mt.WEIGHT_KWARGS = {}
    # mt.DATASET_PATH = ROOT_DIR + "/Datasets"
    # mt.FOLDER = "Synthetic"
    # mt.PRINT_TO_FILE = None
    # mt.TEST_PARAMS = ""
    #
    # full, part, recv = GenerateRandomGraphTriple(
    #     mt.NUM_NODES, mt.NUM_TO_HIDE, mt.GENERATION.value["function"], mt.GENERATION_KWARGS, mt.DISTRIBUTION.value["function"],
    #     mt.DISTRIBUTION_KWARGS, mt.CLOSURE.value["function"], mt.CLOSURE_KWARGS, None, "deg", False)
    #
    # index = WriteGraphTriple(mt.DATASET_PATH, mt.FOLDER, GenerateGraphFilename(
    #         mt.NUM_NODES, mt.NUM_TO_HIDE, mt.GENERATION.value["short_name"],
    #         mt.DISTRIBUTION.value["short_name"], mt.CLOSURE.value["short_name"],
    #         mt.WEIGHT.value["short_name"]), full, part, recv)
    #
    # mt.TRIPLE_INDEX = index

    # parallel_test_repeat(mt)


    # mt.perform_fixed_monitors_test(full, part, recv, 30)



