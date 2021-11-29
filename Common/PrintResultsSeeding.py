from prettytable import PrettyTable


def AddRowVoteOnly(t: PrettyTable, k, vr_full, vr_part, vr_recv, vr_real):
    t.add_row([k, vr_full, vr_part, vr_recv, vr_real])
    return t


def AddRowBoth(t: PrettyTable, k, bg_full, bg_part, bg_recv, bg_real, vr_full, vr_part, vr_recv, vr_real):
    t.add_row([k, bg_full, bg_part, bg_recv, bg_real, vr_full, vr_part, vr_recv, vr_real])
    return t


def AddRowComparison(t: PrettyTable, hidden, one, three, five, seven, nine):
    t.add_row([hidden, one, three, five, seven, nine])
    return t


def Avg(lst: list):
    return sum(lst) / len(lst)


def CreateTableVoteOnly():
    t = PrettyTable(['k', 'VR_Full', 'VR_Part', 'VR_Recv', 'VR_Real'])
    return t


def CreateTableBoth():
    t = PrettyTable(['k', 'BG_Full', 'BG_Part', 'BG_Recv', 'BG_Real', 'VR_Full', 'VR_Part', 'VR_Recv', 'VR_Real'])
    return t


def CreateTableComparison():
    t = PrettyTable(['Hidden nodes', '1 seed', '3 seeds', '5 seeds', '7 seeds', '9 seeds'])
    return t
