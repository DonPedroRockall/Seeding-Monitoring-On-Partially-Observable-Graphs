def PrintResultsSeeding(algorithm: str, k: int, mean_full, mean_part,
                        mean_rec):
    print("############### " + algorithm + " ###############")
    print("Full graph - Mean of activated nodes: " + str(mean_full))
    print("Partial graph - Mean of activated nodes: " + str(mean_part))
    # print("--- RECOVERED GRAPH ---")
    # print("Mean of activated nodes: " + str(mean_rec)
    # print("############################################")


def Avg(lst: list):
    return sum(lst) / len(lst)
