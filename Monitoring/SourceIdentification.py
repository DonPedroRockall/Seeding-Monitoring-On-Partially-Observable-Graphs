def find_roots(branching):
    roots = []
    for node in branching.nodes():
        if not branching.in_edges([node]):
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

        infected_nodes = independent_cascade(graph, random_sources, steps)

        if no_inf > 2:
            return None

        if infected_nodes is None:
            no_inf += 1
            continue
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


def process(steps, graph, k, interval):
    while True:
        random_sources = set()
        list_graph_nodes = list(graph.nodes())
        while len(random_sources) < k:
            random_sources.add(random.choice(list_graph_nodes))

        print('Random sources: ', random_sources)

        infected_nodes = adaptive_cascade(graph, random_sources, steps, interval)
        if infected_nodes is not None:
            break
        else:
            print('fail to ', k, interval)
    print('# infected', len(infected_nodes))
    infected_subgraphs = get_infected_subgraphs(graph, infected_nodes)
    print('# subgraphs', len(infected_subgraphs))

    camerini = Camerini(graph, attr='act_prob')
    # print ('# nodes',len(infected_subgraphs[0].nodes()))

    # print ('Camerini ranking:')
    # branchings = camerini.ranking(k, infected_subgraphs[0])
    # print ('obvious roots:',camerini.find_roots(infected_subgraphs[0]))
    # branchings = camerini.ranking(k, graph, root='root')

    # for branching in branchings:
    #   print (camerini.find_roots(branching))

    edges = 0
    print("Starting to find root branches")
    solutions = camerini.find_roots_branching(k, scores=True, subgraphs=infected_subgraphs)
    print("Root branching search ended")

    for subgraph in infected_subgraphs:
        edges += len(subgraph.edges())

    sources = []
    for element in solutions:
        sources.append(element[0])

    return sources

    # accuracy_algo = sum([1 for node in sources if node in random_sources])
    #
    # imeter_solutions = set()
    # start = timer()
    # if len(infected_subgraphs) >= k:
    #     for subgraph in infected_subgraphs:
    #         if subgraph.size() == 0:
    #             imeter_solutions.update(subgraph.nodes())
    #             continue
    #         imeter_solutions.update(IMeterSort(subgraph))
    # else:
    #     subgraphs_and_shares = [[subgraph, camerini.get_graph_score(subgraph),
    #                              int(k / len(infected_subgraphs)) if len(subgraph.edges()) != 0 else 1] for subgraph in
    #                             infected_subgraphs]
    #     subgraphs_and_shares = sorted(subgraphs_and_shares, key=lambda x: x[1], reverse=True)
    #     j = 0
    #
    #     remains = k - sum([elem[2] for elem in subgraphs_and_shares])
    #     for i in range(remains, 0, -1):
    #         if len(subgraphs_and_shares[j][0].edges()) != 0:
    #             subgraphs_and_shares[j][2] += 1
    #             j += 1
    #     for elem in subgraphs_and_shares:
    #         if elem[0].size() == 0:
    #             imeter_solutions.update(elem[0].nodes())
    #             continue
    #         imeter_solutions.update(IMeterSort(elem[0], k=elem[2]))
    # time_IMeterSort = timer() - start
    # print('IMeter solution', imeter_solutions)
    # accuracy_IMeterSort = sum([1 for node in imeter_solutions if node in random_sources])
    #
    # return
##########################################################################################