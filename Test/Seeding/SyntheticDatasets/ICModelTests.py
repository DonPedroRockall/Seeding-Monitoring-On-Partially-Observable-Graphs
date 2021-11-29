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
SEED_RANGE = 9
NUM_RUN = 40
NUM_CORES = 4
VOTE_RANK_ONLY = True  # select 'False' to run both Basic Greedy and Voterank, 'True' to run Voterank only

NUM_NODES = 300
NODES_TO_DELETE = [100, 150, 200]
GENERATION = GNCConnectedDirectedGraph
DISTRIBUTION = DegreeDistribution
HIDING = TotalNodeClosure
INF_THRESH = None
INF_CENTR = "deg"
GRAPH_FILE_PATH = "/Datasets/Seeding/Small_Degree/"
TEST_FILE_PATH = "../Results/IC_model/Tests_Revisited_Degree.txt"
################################################################

'''
ParallelVarHiddenGeneration(NUM_NODES, GENERATION, DISTRIBUTION, HIDING, INF_THRESH, INF_CENTR,
                            file_path=ROOT_DIR + GRAPH_FILE_PATH)
'''

ParallelDatasetGeneration(NUM_NODES, NODES_TO_DELETE[0], GENERATION, DISTRIBUTION, HIDING, INF_THRESH, INF_CENTR,
                          num_of_graphs=8, file_path=ROOT_DIR+GRAPH_FILE_PATH)
ParallelDatasetGeneration(NUM_NODES, NODES_TO_DELETE[1], GENERATION, DISTRIBUTION, HIDING, INF_THRESH, INF_CENTR,
                          num_of_graphs=8, file_path=ROOT_DIR+GRAPH_FILE_PATH)
ParallelDatasetGeneration(NUM_NODES, NODES_TO_DELETE[2], GENERATION, DISTRIBUTION, HIDING, INF_THRESH, INF_CENTR,
                          num_of_graphs=8, file_path=ROOT_DIR+GRAPH_FILE_PATH)
'''


test_file = open(TEST_FILE_PATH, "a")
test_file.write("--- GRAPH PARAMETERS ---" +
                "\nModel: IC" +
                "\nNum. of nodes: " + str(NUM_NODES) +
                "\nNum. of hidden nodes: " + str(NODES_TO_DELETE) +
                "\nDistribution function: Uniform" +
                "\nHiding function: Total Node Closure" +
                "\nNum. runs per type of graph: " + str(NUM_RUN) +
                "\n")

exec_t_basic = list()
exec_t_vote = list()

iter = 0

for i in range(GRAPH_START_IDX, GRAPH_END_IDX):

    if VOTE_RANK_ONLY:
        t = CreateTableVoteOnly()
    else:
        t = CreateTableBoth()

    test_file.write("\n\n" + str(iter + 1) + ")")

    full = nx.read_weighted_edgelist(ROOT_DIR + GRAPH_FILE_PATH + str(i) + "_full"
                                     + "_hid" + str(NODES_TO_DELETE) + "_tresh" + str(INF_THRESH) + ".txt",
                                     create_using=nx.DiGraph)
    part = nx.read_weighted_edgelist(ROOT_DIR + GRAPH_FILE_PATH + str(i) + "_part"
                                     + "_hid" + str(NODES_TO_DELETE) + "_tresh" + str(INF_THRESH) + ".txt",
                                     create_using=nx.DiGraph)
    recv = nx.read_weighted_edgelist(ROOT_DIR + GRAPH_FILE_PATH + str(i) + "_recv"
                                     + "_hid" + str(NODES_TO_DELETE) + "_tresh" + str(INF_THRESH) + ".txt",
                                     create_using=nx.DiGraph)

    virtuals = GetVirtualNodesByLabel(part, recv)

    InitGraphParametersIC(full)
    InitGraphParametersIC(part)
    InitGraphParametersIC(recv)
    SetSameWeightsToOtherGraphs(full, [part, recv])

    for k in range(1, SEED_RANGE):
        start = time.time()

        # VOTERANK EVALUATION
        vr_full = SIMVoterank(full, k)
        vr_part = SIMVoterank(part, k)
        vr_recv = SIMVoterank(recv, k, restricted_nodes=virtuals)

        mean_full_vote = ParallelRunIC(full, vr_full, num_cores=NUM_CORES, num_run=NUM_RUN)
        mean_part_vote = ParallelRunIC(part, vr_part, num_cores=NUM_CORES, num_run=NUM_RUN)
        mean_recv_vote = ParallelRunIC(recv, vr_recv, num_cores=NUM_CORES, num_run=NUM_RUN)
        mean_real_vote = ParallelRunIC(full, vr_recv, num_cores=NUM_CORES, num_run=NUM_RUN)

        end = time.time()

        print(str(iter+1) + "-" + str(k) + " finished voterank")

        exec_t_vote.append(end - start)

        if not VOTE_RANK_ONLY:
            start = time.time()

            # BASIC GREEDY EVALUATION
            mean_full_basic = ParallelSIMBasicGreedy(full, k, num_cores=NUM_CORES, num_run=NUM_RUN)
            mean_part_basic = ParallelSIMBasicGreedy(part, k, num_cores=NUM_CORES, num_run=NUM_RUN)
            mean_recv_basic = ParallelSIMBasicGreedy(recv, k, num_cores=NUM_CORES, num_run=NUM_RUN,
                                                     restricted_set=virtuals)
            mean_real_basic = ParallelSIMBasicGreedy(full, k, num_cores=NUM_CORES, num_run=NUM_RUN,
                                                     restricted_set=virtuals)

            end = time.time()

            print(str(iter + 1) + "-" + str(k) + " finished basic greedy")

            exec_t_basic.append(end - start)

        # Adding results to the table
        if VOTE_RANK_ONLY:
            t = AddRowVoteOnly(t, k,
                               round(mean_full_vote, 2),
                               round(mean_part_vote, 2),
                               round(mean_recv_vote, 2),
                               round(mean_real_vote, 2),
                               )
        else:
            t = AddRowBoth(t, k,
                           round(mean_full_basic, 2),
                           round(mean_part_basic, 2),
                           round(mean_recv_basic, 2),
                           round(mean_real_basic, 2),
                           round(mean_full_vote, 2),
                           round(mean_part_vote, 2),
                           round(mean_recv_vote, 2),
                           round(mean_real_vote, 2),
                           )

    test_file.write("\n" + str(t) + "\n")

    if not VOTE_RANK_ONLY:
        test_file.write("\nAverage exec time for Basic Greedy: " + str(round(Avg(exec_t_basic), 2)) + " secs")
        exec_t_basic.clear()

    test_file.write("\nAverage exec time for Voterank: " + str(round(Avg(exec_t_vote), 2)) + " secs")
    exec_t_vote.clear()

    iter += 1'''
