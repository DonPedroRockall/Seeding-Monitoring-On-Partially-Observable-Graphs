from Monitoring.SourceIdentification.Camerini import CameriniAlgorithm


def IdentifySources(graph, k, infected_subgraphs=None):
    """
    Performs a Source Identification algorithm on a graph. Activation probability attribute for graph edges is "weight"
    :param graph:               The graph on which to perform the algorithm
    :param k:                   The number of estimated sources. The algorithm will try to identify all the k sources,
                                but in all cases no more than k sources will be returned
    :param infected_subgraphs:  The infected subgraphs. This parameter is external and is required to perform the source
                                identification. It is obtained by choosing some number of random sources and performing
                                an adaptive cascade on a graph using those random sources. For example of usage, please
                                refer to the function named "process" in "SourceIdentificationTester.py"
    :return:                    The sources found
    """
    camerini = CameriniAlgorithm(graph, attr="weight")
    solutions = camerini.find_roots_branching(k, scores=False, subgraphs=infected_subgraphs)
    sources = []
    for solution in solutions:
        if type(solution) is int:
            sources.append(solution)
        else:
            sources.append(solution[0])

    return solutions

