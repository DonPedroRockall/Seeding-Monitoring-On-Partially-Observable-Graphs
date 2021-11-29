import os
import networkx

from Test.Common.Utility import GenerateGraphFilename


def ReadGraphTriple(path, folder, index, graph_type=networkx.DiGraph):
    """
    Reads a previously saved graph triple
    :param path:            Dataset path
    :param folder:          Specific folder where the triple is written to
    :param index:           Index of the graph
    :return:
    """

    full_path = None
    part_path = None
    recv_path = None

    for filename in os.listdir(str(os.path.join(path, folder))):
        if filename.startswith("ind" + str(index) + "_FULL_"):
            full_path = filename
        elif filename.startswith("ind" + str(index) + "_PART_"):
            part_path = filename
        elif filename.startswith("ind" + str(index) + "_RECV_"):
            recv_path = filename

    if full_path is None or part_path is None or recv_path is None:
        raise FileNotFoundError("Cannot find the graph files for the specified index")

    full = networkx.read_edgelist(full_path, create_using=graph_type, nodetype=int)
    part = networkx.read_edgelist(part_path, create_using=graph_type, nodetype=int)
    recv = networkx.read_edgelist(recv_path, create_using=graph_type, nodetype=str)

    # Convert the nodes of recv to int, except for the virtual nodes (that are names RECVX) that will remain strings
    mapping = {}
    for node in recv.nodes():
        if node.startswith("RECV"):
            mapping[node] = node
        else:
            mapping[node] = int(node)
    networkx.relabel_nodes(recv, mapping, copy=False)

    return full, part, recv


def WriteGraphTriple(path, folder, filename_template, full, part, recv):

    i = 0
    while True:

        # Search for a free index
        freeIndex = True
        for filename in os.listdir(str(os.path.join(path, folder))):
            if filename.startswith("ind" + str(i) + "_"):
                i += 1
                freeIndex = False
                break

        # Once found a free index, write the graphs
        if freeIndex:
            full_path = os.path.join(path, folder, "ind" + str(i) + "_FULL_" + filename_template)
            part_path = os.path.join(path, folder, "ind" + str(i) + "_PART_" + filename_template)
            recv_path = os.path.join(path, folder, "ind" + str(i) + "_RECV_" + filename_template)

            networkx.write_edgelist(full, full_path)
            networkx.write_edgelist(part, part_path)
            networkx.write_edgelist(recv, recv_path)

            return



