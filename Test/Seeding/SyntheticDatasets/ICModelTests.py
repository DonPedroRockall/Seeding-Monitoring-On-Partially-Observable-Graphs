from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Utilities.DrawGraph import DrawGraph
from Utilities.PrintResultsSeeding import Avg, CreateTable, AddRow
from Seeding.IC_model import *
from prettytable import PrettyTable

###################### TEST PARAMETERS  ######################
# You can change the following parameters for various purposes
NUM_RUN = 50
NUM_ITER = 5  # recommended not to set higher
SEED_RANGE = 22  # recommended not to set higher
NUM_NODES = 300
MIN_EDGES = 100
NODES_TO_DELETE = 30
DISTRIBUTION = UniformDistribution
HIDING = TotalNodeClosure
##############################################################


active_per_run_full = list()
active_per_run_part = list()
active_per_run_rec = list()

mean_full_vote = list()
mean_part_vote = list()
mean_rec_vote = list()
mean_full_basic = list()
mean_part_basic = list()
mean_rec_basic = list()

file_obj = open("../Results/IC_model/Test_1", "a")
t = CreateTable()
file_obj.write("--- GRAPH PARAMETERS ---" + "\nNum. of nodes: " + str(NUM_NODES) + "\nMin. num. of edges: " +
               str(MIN_EDGES) + "\nNum. of nodes to delete: " + str(NODES_TO_DELETE) + "\n")

for k in range(1, SEED_RANGE, 5):

    for _ in range(NUM_ITER):
        full, part, rec = GenerateRandomGraphTriple(NUM_NODES, MIN_EDGES, NODES_TO_DELETE,
                                                    distribution_function=DISTRIBUTION, hiding_function=HIDING)

        InitGraphParametersIC(full)
        InitGraphParametersIC(part)
        InitGraphParametersIC(rec)

        for _ in range(NUM_RUN):
            active_per_run_full.append(len(SIMVoterank(full, k)))
            active_per_run_part.append(len(SIMVoterank(part, k)))
            active_per_run_rec.append(len(SIMVoterank(rec, k)))

        # PrintResultsSeeding("VOTE RANK", Avg(active_per_run_full), Avg(active_per_run_part), Avg(active_per_run_rec))

        mean_full_vote.append(Avg(active_per_run_full))
        mean_part_vote.append(Avg(active_per_run_part))
        mean_rec_vote.append(Avg(active_per_run_rec))

        active_per_run_full.clear()
        active_per_run_part.clear()
        active_per_run_rec.clear()

        for _ in range(NUM_RUN):
            active_per_run_full.append(len(SIMBasicGreedy(full, k)))
            active_per_run_part.append(len(SIMBasicGreedy(part, k)))
            active_per_run_rec.append(len(SIMBasicGreedy(rec, k)))

        # PrintResultsSeeding("BASIC GREEDY", Avg(active_per_run_full), Avg(active_per_run_part), Avg(active_per_run_rec))

        mean_full_basic.append(Avg(active_per_run_full))
        mean_part_basic.append(Avg(active_per_run_part))
        mean_rec_basic.append(Avg(active_per_run_rec))

        active_per_run_full.clear()
        active_per_run_part.clear()
        active_per_run_rec.clear()

    t = AddRow(t, k, round(Avg(mean_full_vote), 2), round(Avg(mean_part_vote), 2), round(Avg(mean_rec_vote), 2),
               round(Avg(mean_full_basic), 2), round(Avg(mean_part_basic), 2), round(Avg(mean_rec_basic), 2))

file_obj.write(str(t) + "\n\n")
