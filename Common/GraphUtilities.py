import networkx as nx


def GetDistanceToClosestRealSource(graph: nx.DiGraph, est_source, real_sources, max_distance=4):
    queue = [est_source]
    cur_distance = 0
    visited = {est_source}
    next_set = set()

    while len(queue) > 0:
        curnode = queue.pop()
        if curnode in real_sources:
            return cur_distance
        for neigh in graph.neighbors(curnode):
            if neigh not in visited:
                next_set.add(neigh)
        if len(queue) == 0 and cur_distance <= max_distance:
            queue.extend(next_set)
            cur_distance += 1

    return max_distance + 1



def GetVirtualNodesByDifference(part: nx.DiGraph, recv: nx.DiGraph):
    """
    Returns the set of virtual nodes. Virtual nodes are defined as the nodes that are present in recv graph but not in
    part graph. A node is present if it has the same label
    :param part:        The Partial graph
    :param recv:        The Recovered graph
    :return:
    """
    virtuals = set()
    for node in recv.nodes():
        if node not in part.nodes():
            virtuals.add(node)
    return virtuals


def GetVirtualNodesByNodeLabel(recv: nx.DiGraph, label: str):
    virtuals = set()
    for node in recv.nodes():
        if str(node).startswith(label):
            virtuals.add(node)
    return virtuals


def SetSameWeightsToOtherGraphs(original_graph: nx.Graph, other_graphs: list):
    """
    Copies all the attributes of original_graph to all the other graphs in other_graphs list, without altering the
    structure of the graph(s) itself. (I.E.: it will not create new nodes or new edges, every graph will stay the same)
    :param original_graph:
    :param other_graphs:
    :return:
    """
    for u, v, data in original_graph.edges(data=True):
        for graph in other_graphs:
            if graph.has_edge(u, v):
                for key in data:
                    graph[u][v][key] = data[key]


def GenerateReportFilename(num_of_nodes: int, num_hidden: int, num_src: int, num_trg: int, gen_func, hiding_func, closure_func, weight_func):
    """Generates a short filename in which are indicated all the main properties of a graph for a report"""
    return f"num{num_of_nodes}_hid{num_hidden}_src{num_src}_trg{num_trg}_gen{gen_func}_dst{hiding_func}_cls{closure_func}_wgt{weight_func}.txt"


def GenerateGraphFilename(num_of_nodes: int, num_hidden: int, gen_func, hiding_func, closure_func, weight_func):
    """Generates a short filename in which are indicated all the main properties of a graph"""
    return f"num{num_of_nodes}_hid{num_hidden}_gen{gen_func}_dst{hiding_func}_cls{closure_func}_wgt{weight_func}.txt"
