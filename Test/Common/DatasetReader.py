import os

import networkx


def ReadGraphTriple(path, index=0, hid=150, dist="deg", type=networkx.DiGraph):
    """
    Reads a previously saved graph triple
    :param path:            Path where to read the triple from (should include a final /)
    :param index:
    :param hid:
    :param dist:
    :return:
    """
    path_full = path + str(index) + "_full_hid" + str(hid) + "_" + dist + ".txt"
    path_part = path + str(index) + "_part_hid" + str(hid) + "_" + dist + ".txt"
    path_recv = path + str(index) + "_recv_hid" + str(hid) + "_" + dist + ".txt"

    full = networkx.read_edgelist(path_full, create_using=type, nodetype=int)
    part = networkx.read_edgelist(path_part, create_using=type, nodetype=int)
    recv = networkx.read_weighted_edgelist(path_recv, create_using=type, nodetype=int)

    return full, part, recv


def WriteGraphTriple(path, full, part, recv, hidden, distribution="deg"):
    i = 0
    while True:
        full_path = path + str(i) + "_full_hid" + str(hidden) + "_" + distribution + ".txt"
        part_path = path + str(i) + "_part_hid" + str(hidden) + "_" + distribution + ".txt"
        recv_path = path + str(i) + "_recv_hid" + str(hidden) + "_" + distribution + ".txt"

        if os.path.isfile(full_path) or os.path.isfile(part_path) or os.path.isfile(recv_path):
            i += 1
        else:
            networkx.write_weighted_edgelist(full, full_path)
            networkx.write_weighted_edgelist(part, part_path)
            networkx.write_weighted_edgelist(recv, recv_path)
            return
