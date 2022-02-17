from Common.ColorPrints import bcolors, cprint
from Monitoring.SourceIdentification.Camerini import CameriniAlgorithm


def IdentifySources(graph, k, infected_subgraphs=None, virtual_set={}):
    """
    Performs a Source Identification algorithm on a graph. Activation probability attribute for graph edges is "weight"
    :param graph:               The graph on which to perform the algorithm
    :param k:                   The number of estimated sources. The algorithm will try to identify all the k sources,
                                but in all cases no more than k sources will be returned
    :param infected_subgraphs:  The infected subgraphs. This parameter is external and is required to perform the source
                                identification. It is obtained by choosing some number of random sources and performing
                                an adaptive cascade on a graph using those random sources. For example of usage, please
                                refer to the function named "process" in "SourceIdentificationTester.py"
    :param virtual_set:         The set of virtual (recovered) nodes. Sources cannot be identified in virtual nodes,
                                therefore any node identified as source belonging to this set will be discarded
    :return:                    The sources found
    """
    camerini = CameriniAlgorithm(graph, attr="weight")
    solutions = camerini.find_roots_branching(k, scores=False, subgraphs=infected_subgraphs)
    sources = []
    discarded = 0
    for solution in solutions:
        if type(solution) is int or type(solution) is str:
            if solution not in virtual_set:
                sources.append(solution)
            else:
                discarded += 1

        else:
            if solution[0] not in virtual_set:
                sources.append(solution[0])
            else:
                discarded += 1

    return sources, discarded

