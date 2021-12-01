from __future__ import division

import random

from definitions import ROOT_DIR

"Debug mode on/off"

import networkx as nx
from random import choice
from camerini import Camerini
from independent_cascade_opt import independent_cascade, get_infected_subgraphs
import DiffusionModels.IndependentCascade
from DiffusionModels.IndependentCascade import IndependentCascadeWithMonitors
import os
import csv
from timeit import default_timer as timer
from multiprocessing import Pool, Manager


def read_direct_weighted_graph(pathname):
    infile = open(pathname, 'r')
    graph = nx.DiGraph()
    for line in infile:
        if "#" not in line:
            u, v, p = line.split()
            graph.add_edge(u, v, act_prob=float(p))
    return graph


def find_roots(branching):
    roots = []
    for node in branching.nodes():
        if branching.in_edges([node]) == []:
            roots.append(node)
    return roots


def adaptive_cascade(graph, random_sources, steps, interval):
    i_low = 0
    i_high = 0
    no_inf = 0
    i = 0
    while True:
        if i > 35:
            return None
        if steps == 0:
            return None

        # infected_nodes = DiffusionModels.IndependentCascade.IndependentCascadeWithMonitors(graph, random_sources, steps)
        # infected_nodes = DiffusionModels.IndependentCascade.IndependentCascade\
        #     .independent_cascade(G=graph, seeds=random_sources, steps=steps)
        ic = IndependentCascadeWithMonitors(G=graph, seeds=random_sources, monitors=[], steps=steps)
        infected_nodes = set()
        for sublist in ic:
            infected_nodes = infected_nodes.union(sublist)

        if no_inf > 2:
            return None

        if infected_nodes is None:
            no_inf += 1
            continue
            # print ('Len infected nodes',len(infected_nodes),i_low, i_high, steps,interval)
        if len(infected_nodes) < interval[0]:
            i_low += 1
        else:
            if len(infected_nodes) > interval[1]:
                i_high += 1
            else:
                print('Success for', random_sources, 'k', k, 'interval', interval)
                return infected_nodes
        i += 1

        if infected_nodes is not None:
            if i_low > 4:
                steps += 1
                i_low = 0
                i_high = 0
            if i_high > 4:
                steps -= 1
                i_low = 0
                i_high = 0


def process(graph, steps, k, interval):
    while True:
        random_sources = list(random.sample(list(graph.nodes()), k))

        print('Random sources: ', random_sources)

        infected_nodes = adaptive_cascade(graph, random_sources, steps, interval)
        if infected_nodes is not None:
            break
        else:
            print('fail to ', k, interval)
    print('# infected', len(infected_nodes))
    infected_subgraphs = get_infected_subgraphs(graph, infected_nodes)
    print('# subgraphs', len(infected_subgraphs))

    camerini = Camerini(graph, attr='weight')

    edges = 0
    solutions = camerini.find_roots_branching(k, scores=True, subgraphs=infected_subgraphs)

    for subgraph in infected_subgraphs:
        edges += len(subgraph.edges())

    sources = []
    for element in solutions:
        sources.append(element[0])

    return sources


def run(*args):
    steps, graph, k, queue, interval = list(args)[0]
    i = queue.get()
    pathname = 'wiki-Vote_results_multi_sources/test_' + str(i) + '_k_' + str(k) + '.csv'

    out_file = open(pathname, 'a')
    writer = csv.writer(out_file, delimiter=',')
    if os.stat(pathname).st_size == 0:
        heading = ['Sources'] + [''] * (k - 1) + ['Num_Infected_Nodes', 'Num_Edges', 'Algo'] + [''] * (k - 1) + [
            'Accuracy_Algo', 'Time_Algo', 'IMeterSort'] + [''] * (k - 1) + ['Accuracy_IMeterSort', 'Time_IMeterSort']
        writer.writerow(heading)
    process(graph, steps, k, interval)
    queue.put(i)


if __name__ == "__main__":
    k = 4
    steps = 2

    pathname_graph = ROOT_DIR + "/Datasets/Real/Wiki-Vote/Wiki-Vote.txt"
    graph: nx.DiGraph = nx.read_edgelist(pathname_graph, create_using=nx.DiGraph(), nodetype=int, comments='#')

    for u, v, data in graph.edges(data=True):
        data["weight"] = random.random() * 0.1

    intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
    process(graph, steps, k, interval=intervals)




# n_jobs = 30
# processes = 16
#
# intervals = [100, 250, 500, 650, 1000, 1200, 1500, 1700, 2100, 2700]
# queue = Manager().Queue()
# for i in range(n_jobs):
#     queue.put(i)
# for k in range(2, 5):
#     for interval in range(0, len(intervals), 2):
#         pool = Pool(processes)
#         pool.map(run, [(steps, graph, k, queue, intervals[interval:interval + 2]) for i in range(n_jobs)])
# -------------------------------------------------------
    # for process in processes:
    #   process.start()

    # for process in processes:
    #   process.join()
    # # Parallel(n_jobs=-1)(delayed(run)(steps, graph, k, i) for i in range(n_jobs) )

    # process(steps, graph, random_sources, k, writer=writer)
