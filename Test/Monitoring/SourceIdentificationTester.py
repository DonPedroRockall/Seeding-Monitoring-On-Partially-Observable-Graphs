import random
import networkx as nx
from Monitoring.SourceIdentification.Camerini import CameriniAlgorithm
from DiffusionModels.IndependentCascade import GetInfectedSubgraphs, AdaptiveCascade
from definitions import ROOT_DIR


def process(graph, steps, k, interval):
    while True:
        random_sources = list(random.sample(list(graph.nodes()), k))
        infected_nodes = AdaptiveCascade(graph, random_sources, steps, interval)
        if infected_nodes is not None:
            break
    infected_subgraphs = GetInfectedSubgraphs(graph, infected_nodes)

    camerini = CameriniAlgorithm(graph, attr='weight')

    # edges = 0
    solutions = camerini.find_roots_branching(k, scores=True, subgraphs=infected_subgraphs)
    sources = []
    for solution in solutions:
        sources.append(solution[0])

    print("SOURCES:", sources)
    print("RANDOM_SOURCES:", random_sources)
    return solutions

    # for subgraph in infected_subgraphs:
    #     edges += len(subgraph.edges())
    #
    # sources = []
    # for element in solutions:
    #     sources.append(element[0])
    #
    # return sources





if __name__ == "__main__":
    k = 20
    steps = 2

    pathname_graph = ROOT_DIR + "/Datasets/Real/Wiki-Vote.txt"
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
