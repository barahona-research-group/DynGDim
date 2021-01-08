"""compute relative dimension from single source"""
import numpy as np

from dyngdim import run_single_source, initial_measure
from dyngdim.io import save_single_source_results
from generate_grid import generate_grid

graph = generate_grid()


t_min = -5
t_max = 0.0
n_t = 200

id_0 = int(len(graph) / 2)
id_0 = 10

times = np.logspace(t_min, t_max, n_t)

# set the source
measure = initial_measure(graph, [id_0])
results = run_single_source(graph, times, measure)
save_single_source_results(results)
