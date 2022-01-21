import os

import numpy

from definitions import ROOT_DIR
from pandas import DataFrame


def RetrieveResults():
    g_path = ROOT_DIR + "/Test/Monitoring/Results/SyntheticDatasets/TotalClosure/Weights0-01/GNC"
    table = numpy.ndarray(shape=(10, 12))

    for folder in os.listdir(g_path):

        vertical_index = int(int(folder[0:-1]) / 10) - 1
        print(g_path + "/" + folder)

        for file in os.listdir(g_path + "/" + folder):
            if file.startswith("test_0"):
                f = open(g_path + "/" + folder + "/" + file)
                content = f.read()
                results = content.split("==============================================================")[1]
                reslines = results.split("\n")[1:-1]
                print(reslines)

                for resline in reslines:
                    temp = resline.split(" (")
                    if len(temp) < 2:
                        continue
                    t1 = temp[1]
                    # res_dict[temp[0]] = temp[1].split(", ")[0]
                    if temp[0] == "NUM_INFECTED_FF":
                        table[vertical_index, 0] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "NUM_INFECTED_FP":
                        table[vertical_index, 1] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "NUM_INFECTED_FR":
                        table[vertical_index, 2] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "NON_SOURCE_INF_FF":
                        table[vertical_index, 3] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "NON_SOURCE_INF_FP":
                        table[vertical_index, 4] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "NON_SOURCE_INF_FR":
                        table[vertical_index, 5] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "INFECTED_TARGETS_FF":
                        table[vertical_index, 6] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "INFECTED_TARGETS_FP":
                        table[vertical_index, 7] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "INFECTED_TARGETS_FR":
                        table[vertical_index, 8] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "NUM_MONITORS_FULL":
                        table[vertical_index, 9] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "NUM_MONITORS_PART":
                        table[vertical_index, 10] = round(float(t1.split(", ")[0]), 1)
                    elif temp[0] == "NUM_MONITORS_RECV":
                        table[vertical_index, 11] = round(float(t1.split(", ")[0]), 1)

    df = DataFrame(table)
    df.to_excel("tab.xlsx")


if __name__ == "__main__":
    RetrieveResults()
