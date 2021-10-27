from Test.Common.DatasetGenerator import GenerateRandomGraphTriple
from Test.Common.DistributionFunctions import *
from Test.Common.HidingFunctions import *
from Utilities.DrawGraph import DrawGraph
from Utilities.PrintResultsSeeding import PrintResultsSeeding, Avg
from Seeding.IC_model import *

active_per_run_full = list()
active_per_run_part = list()
active_per_run_rec = list()
k = 1

mean_full_vote = list()
mean_part_vote = list()
mean_rec_vote = list()
mean_full_basic = list()
mean_part_basic = list()
mean_rec_basic = list()

print("Max seeds: " + str(k) + "\n\n")

for _ in range(20):
    print("Iteration: " + str(_))
    full, part, rec = GenerateRandomGraphTriple(100, 30, 10, distribution_function=UniformDistribution)

    InitGraphParametersIC(full)
    InitGraphParametersIC(part)
    # InitGraphParametersIC(rec)

    for _ in range(50):
        active_per_run_full.append(len(SIMVoterank(full, k)))
        active_per_run_part.append(len(SIMVoterank(part, k)))
        # active_per_run_rec.append(len(SIMVoterank(rec, k)))

    PrintResultsSeeding("VOTE RANK", k, Avg(active_per_run_full), Avg(active_per_run_part), 0)

    mean_full_vote.append(Avg(active_per_run_full))
    mean_part_vote.append(Avg(active_per_run_part))

    active_per_run_full.clear()
    active_per_run_part.clear()
    active_per_run_rec.clear()

    for _ in range(50):
        active_per_run_full.append(len(SIMBasicGreedy(full, k)))
        active_per_run_part.append(len(SIMBasicGreedy(part, k)))
        # active_per_run_rec.append(len(SIMBasicGreedy(rec, k)))

    PrintResultsSeeding("BASIC GREEDY", k, Avg(active_per_run_full), Avg(active_per_run_part), 0)

    mean_full_basic.append(Avg(active_per_run_full))
    mean_part_basic.append(Avg(active_per_run_part))

    active_per_run_full.clear()
    active_per_run_part.clear()
    active_per_run_rec.clear()

print("\n\n############### FINAL RESULTS ###############")
print("\nNum. of seeds: " + str(k))
print("\nTOTAL MEAN - VOTE RANK")
print("FULL: " + str(Avg(mean_full_vote)))
print("PART: " + str(Avg(mean_part_vote)))
print("\nTOTAL MEAN - BASIC GREEDY")
print("FULL: " + str(Avg(mean_full_basic)))
print("PART: " + str(Avg(mean_part_basic)))
