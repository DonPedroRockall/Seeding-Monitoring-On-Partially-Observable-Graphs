from prettytable import PrettyTable


def AddRow(t: PrettyTable, k, vr_full, vr_part, vr_recv):
    t.add_row([k, vr_full, ,vr_part vr_recv])
    return t


def Avg(lst: list):
    return sum(lst) / len(lst)


def CreateTable():
    t = PrettyTable(['k', 'VR_Full', 'VR_Part', 'VR_Recv'])
    return t