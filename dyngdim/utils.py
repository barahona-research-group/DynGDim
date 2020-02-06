"""additional functions"""
import numpy as np
import warnings

def delta_measure(graph, node):
    """create a delta measure with the correct mass"""
    total_degree = sum([graph.degree(u, weight="weight") for u in graph])
    measure = np.zeros(len(graph))
    measure[node] = total_degree / (
        len(graph) * graph.degree(node, weight="weight")
    )
    return measure

def averaging(a, axis=None, weights=None):
        a = np.asanyarray(a)
        wgt = np.asanyarray(weights)
        result_dtype = np.result_type(a.dtype, wgt.dtype)
        scl = wgt.sum(axis=axis, dtype=result_dtype)       
        warnings.filterwarnings("ignore")
        avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis)/scl
        return avg