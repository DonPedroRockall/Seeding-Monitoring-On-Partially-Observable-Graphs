"""
from multiprocessing import Pool, Manager

import networkx as nx
import Monitoring.Monitor
from Monitor import *
from Utilities.DrawGraph import DrawGraph


def read_direct_weighted_graph(pathname):
    infile = open(pathname, 'r')
    graph = nx.DiGraph()
    for line in infile:
        if "#" not in line:
            u, v, p = line.split()
            graph.add_edge(u, v, act_prob=float(p))
    return graph


def run(*args):
    steps, graph, k, queue, interval = list(args)[0]
    i = queue.get()
    pathname = 'wiki-Vote_results_multi_sources/test_' + str(i) + '_k_' + str(k) + '.csv'

    # out_file = open(pathname, 'a')
    # writer = csv.writer(out_file, delimiter=',')
    # if os.stat(pathname).st_size == 0:
    #     heading = ['Sources'] + [''] * (k - 1) + ['Num_Infected_Nodes', 'Num_Edges', 'Algo'] + [''] * (k - 1) + [
    #         'Accuracy_Algo', 'Time_Algo', 'IMeterSort'] + [''] * (k - 1) + ['Accuracy_IMeterSort', 'Time_IMeterSort']
    #     writer.writerow(heading)
    process(steps, graph, k, interval)
    queue.put(i)



graph = read_direct_weighted_graph("../Datasets/Wiki-Vote-Rand.txt")
Monitoring.Monitor.PlaceMonitors(graph, 4, 2)

k = 4
steps = 2
n_jobs = 30
processes = 16

intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
queue = Manager().Queue()

for i in range(n_jobs):
    queue.put(i)

for k in range(2, 5):
    for interval in range(0, len(intervals), 2):
        pool = Pool(processes)
        pool.map(run, [(steps, graph, k, queue, intervals[interval:interval + 2]) for i in range(n_jobs)])
"""
import random

import networkx as nx

from Monitoring.Monitor import MonitorPlacement


graph = nx.DiGraph()
k = 5
num_sources = 3
num_targets = 3
monitor_budget = 10
sources = list()
targets = list()

while graph.size() == 0 or not nx.is_weakly_connected:
    graph = nx.generators.gn_graph(250, create_using=nx.DiGraph)
    k += 1
nx.set_edge_attributes(graph, 0, "weight")

print("Weak connectivity achieved for k =", k)
print("Randomizing weights...")

for u, v, data in graph.edges(data=True):
    data["weight"] = random.random()
print("Weights Randomized")


while len(sources) < num_sources:
    sources.append(random.choice(list(graph.nodes())))

while len(targets) < num_targets:
    node = random.choice(list(graph.nodes()))
    if node not in sources:
        targets.append(node)

print("Chosed sources:", sources, "Chosed Targets:", targets)

monitors = MonitorPlacement(graph, sources, targets, monitor_budget)

print("Monitors:", monitors)

