from Test.Common.DatasetGenerator import GenerateRandomGraphTriple, ParallelDatasetGenerationSeed, \
    SetSameWeightsToOtherGraphs
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Utilities.DrawGraph import DrawGraph
from Utilities.PrintResultsSeeding import Avg, CreateTable, AddRow
from Seeding.IC_model import *
from prettytable import PrettyTable
from definitions import ROOT_DIR
import time

###################### TEST PARAMETERS  ######################
# You can change the following parameters for various purposes
NUM_ITER = 5  # recommended not to set higher
SEED_RANGE = 32  # recommended not to set higher
NUM_RUN = 40

NUM_NODES = 300
# MIN_EDGES = 100
NODES_TO_DELETE = 50
DISTRIBUTION = UniformDistribution
HIDING = TotalNodeClosure
##############################################################

# ParallelDatasetGenerationSeed(NUM_NODES, MIN_EDGES, NODES_TO_DELETE, DISTRIBUTION, HIDING, num_of_graphs=30,
#                              file_path=ROOT_DIR + "/Datasets/Seeding/Synthetic_3/")


test_file = open("../Results/IC_model/Test_2", "a")
test_file.write("--- GRAPH PARAMETERS ---" +
                "\nModel: IC" +
                "\nNum. of nodes: " + str(NUM_NODES) +
                "\nNum. of hidden nodes: " + str(NODES_TO_DELETE) +
                "\nDistribution function: Uniform" +
                "\nHiding function: Total Node Closure" +
                "\n\n")
iter = 0

for i in range(5, 10):

    test_file.write(str(iter + 1) + ")\n")
    t = CreateTable()

    print("iter = " + str(iter))

    full = nx.read_weighted_edgelist(ROOT_DIR + "/Datasets/Seeding/Synthetic_1/" + str(i) + "_full_"
                                     + str(NUM_NODES) + "_hid_" + str(NODES_TO_DELETE) + ".txt",
                                     create_using=nx.DiGraph)
    part = nx.read_weighted_edgelist(ROOT_DIR + "/Datasets/Seeding/Synthetic_1/" + str(i) + "_part_"
                                     + str(NUM_NODES) + "_hid_" + str(NODES_TO_DELETE) + ".txt",
                                     create_using=nx.DiGraph)
    recv = nx.read_weighted_edgelist(ROOT_DIR + "/Datasets/Seeding/Synthetic_1/" + str(i) + "_recv_"
                                     + str(NUM_NODES) + "_hid_" + str(NODES_TO_DELETE) + ".txt",
                                     create_using=nx.DiGraph)

    InitGraphParametersIC(full)
    InitGraphParametersIC(part)
    InitGraphParametersIC(recv)
    SetSameWeightsToOtherGraphs(full, [part, recv])

    for k in range(1, SEED_RANGE, 5):
        print("\nk = " + str(k))

        start = time.time()

        mean_full_vote = ParallelSIMVoterank(full, k, num_run=NUM_RUN)
        mean_part_vote = ParallelSIMVoterank(part, k, num_run=NUM_RUN)
        mean_recv_vote = ParallelSIMVoterank(recv, k, num_run=NUM_RUN)

        print("vote rank done")

        mean_full_basic = ParallelSIMBasicGreedy(full, k, num_iter=NUM_RUN)
        mean_part_basic = ParallelSIMBasicGreedy(part, k, num_iter=NUM_RUN)
        mean_recv_basic = ParallelSIMBasicGreedy(recv, k, num_iter=NUM_RUN)

        print("basic greedy done")

        end = time.time()

        print("\nExecution time for iteration " + str(iter + 1) + ": " + str(end - start))

        t = AddRow(t, k,
                   round(mean_full_vote, 2),
                   round(mean_part_vote, 2),
                   round(mean_recv_vote, 2),
                   round(mean_full_basic, 2),
                   round(mean_part_basic, 2),
                   round(mean_recv_basic, 2))

    iter += 1
    test_file.write(str(t) + "\n\n")
