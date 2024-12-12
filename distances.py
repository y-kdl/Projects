from scipy.spatial import distance
import numpy as np

def distance_selection(selected_distance, list1, list2):
    list1 = np.array(list1, dtype=float)
    list2 = np.array(list2, dtype=float)
    if selected_distance == "Euclidean":
        dist = distance.euclidean(list1, list2)
    elif selected_distance == "Canberra":
        dist = distance.canberra(list1, list2)
    elif selected_distance == "Manhattan":
        dist = distance.cityblock(list1, list2)
    elif selected_distance == "Chebyshev":
        dist = distance.chebyshev(list1, list2)
    elif selected_distance == "Minkowsky":
        dist = distance.minkowski(list1, list2)
    return dist