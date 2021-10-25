import networkx as nx


def MultiDiGraphToDiGraph(graph: nx.MultiDiGraph, attr="weight"):
    """
    Build a DiGraph from a MultiDiGraph with the following properties:
    - It has all the exact nodes of the MultiDiGraph
    - Replaces all weighted parallel edges with a single edge which weights as much as the sum of the weights of all the
        parallel edges it replaces
    - Has the same attributes, except weight that is modified as above
    - The graph is directed, and thus edges (u, v) are considered distinct from (v, u)
    :param graph:       The MultiDiGraph to transform
    :param attr:        The attribute to modify (default: "weight")
    :return:            The corresponding DiGraph as described above
    """
    # For every node in graph
    for node in graph.nodes():
        # Look for adjacent nodes
        for adj_node in graph[node]:
            # If adjacent node has an edge to the first node
            # Or our graph have several edges from the first to the adjacent node
            if len(graph[node][adj_node]) > 1:
                # Iterate over all parallel edges
                all_parallel_edges = list(graph[node][adj_node])
                total_weight = 0
                # Compute total weight
                for edge_key in all_parallel_edges:
                    total_weight += graph[node][adj_node][edge_key][attr]
                    # If the key is not 0, then remove the edge, so that for each node only the edge with key 0 remains
                    if edge_key > 0:
                        graph.remove_edge(node, adj_node, edge_key)
                # Set the weight for the remaining edge with key 0 to total_weight
                graph[node][adj_node][0][attr] = total_weight


    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.nodes())
    digraph.add_edges_from(graph.edges(data=True))
    return digraph


# Test case for the function
if __name__ == "__main__":
    g = nx.MultiDiGraph()
    g.add_nodes_from([1, 2, 3, 4, 5])
    g.add_edge(1, 2, None, weight=1)
    g.add_edge(1, 2, None, weight=1.5)
    g.add_edge(2, 1, None, weight=0.1)
    g.add_edge(3, 4, None, weight=4.0)
    g.add_edge(3, 4, None, weight=4.2)
    g.add_edge(3, 4, None, weight=4.3)
    g.add_edge(4, 3, None, weight=8.4)
    g.add_edge(4, 3, None, weight=8.7)
    dig = MultiDiGraphToDiGraph(g)
    for u, v, data in dig.edges(data=True):
        print(u, v, data)

