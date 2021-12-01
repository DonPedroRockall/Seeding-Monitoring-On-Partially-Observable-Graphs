from Test.Common.DatasetGenerator import GenerateRandomGraphTriple, ParallelDatasetGeneration, \
    ParallelVarHiddenGeneration
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Test.Common.GraphGenerator import *
from Test.Common.Utility import GetVirtualNodesByLabel, IsVirtualNode, SetSameWeightsToOtherGraphs
from Utilities.DrawGraph import DrawGraph
from Utilities.PrintResultsSeeding import *
from Seeding.IC_model import *
from prettytable import PrettyTable
from definitions import ROOT_DIR
import time

####################### TEST PARAMETERS  #######################
# You can change the following parameters for various test cases
GRAPH_START_IDX = 4
GRAPH_END_IDX = 9
SEED_RANGE = 10
NUM_RUN = 40
NUM_CORES = 4


NUM_NODES = 1500
NODES_TO_DELETE = 1000
GENERATION = GNCConnectedDirectedGraph
DISTRIBUTION = DegreeDistribution
HIDING = TotalNodeClosure
INF_THRESH = 5
INF_CENTR = "deg"
GRAPH_FILE_PATH = "/Datasets/Seeding/Variable_Hidden_Degree/"
TEST_FILE_PATH = "../Results/IC_model/Variable_Hidden_Degree.txt"
################################################################

test_file = open(TEST_FILE_PATH, "a")
test_file.write("--- GRAPH PARAMETERS ---" +
                "\nModel: IC" +
                "\nNum. of nodes: " + str(NUM_NODES) +
                "\nDistribution function: Degree" +
                "\nHiding function: Total Node Closure" +
                "\nNum. runs per type of graph: " + str(NUM_RUN) +
                "\n")

t = CreateTableComparison()

full = nx.read_weighted_edgelist(ROOT_DIR + GRAPH_FILE_PATH + 'full_graph.txt',
                                 create_using=nx.DiGraph)

InitGraphParametersIC(full)

for i in range(100, NUM_NODES-100, 200):

    part = nx.read_weighted_edgelist(ROOT_DIR + GRAPH_FILE_PATH + "part_" + str(i) + "_hidden.txt",
                                     create_using=nx.DiGraph)
    recv = nx.read_weighted_edgelist(ROOT_DIR + GRAPH_FILE_PATH + "recv_" + str(i) + "_hidden.txt",
                                     create_using=nx.DiGraph)

    virtuals = GetVirtualNodesByLabel(part, recv)

    InitGraphParametersIC(part)
    InitGraphParametersIC(recv)
    SetSameWeightsToOtherGraphs(full, [part, recv])

    mean_real_vote = list()

    for k in range(1, SEED_RANGE, 2):

        # VOTERANK EVALUATION
        vr_recv = SIMVoterank(recv, k, restricted_nodes=virtuals)

        mean_real_vote.append(ParallelRunIC(full, vr_recv, num_cores=NUM_CORES, num_run=NUM_RUN))

        end = time.time()

        print(str(i) + "-" + str(k) + " finished voterank")

    # Adding results to the table
    t = AddRowComparison(t, i, mean_real_vote[0], mean_real_vote[1], mean_real_vote[2], mean_real_vote[3], mean_real_vote[4])


test_file.write("\n" + str(t) + "\n")
