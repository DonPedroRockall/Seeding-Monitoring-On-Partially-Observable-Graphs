from prettytable import PrettyTable


def AddRow(t: PrettyTable, k, vr_full, vr_part, vr_recv, bg_full, bg_part, bg_recv):
    t.add_row([k, vr_full, vr_part, vr_recv, bg_full, bg_part, bg_recv])
    return t


def Avg(lst: list):
    return sum(lst) / len(lst)


def CreateTable():
    t = PrettyTable(['k', 'VR_Full', 'VR_Part', 'VR_Recv', 'BG_Full', 'BG_Part', 'BG_Recv'])
    return t
