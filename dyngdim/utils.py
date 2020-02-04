"""additional functions"""
import numpy as np


def delta_measure(graph, node):
    """create a delta measure with the correct mass"""
    total_degree = sum([graph.degree(u, weight="weight") for u in graph])
    measure = np.zeros(len(graph))
    measure[node] = total_degree / (
        len(graph) * graph.degree(node, weight="weight")
    )
    return measure
