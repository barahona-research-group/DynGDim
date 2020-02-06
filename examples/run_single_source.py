"""compute relative dimension from single source"""
import numpy as np

from dyngdim import run_single_source
from dyngdim.utils import delta_measure
from dyngdim.io import save_single_source_results

from generate_grid import generate_grid

graph = generate_grid()


t_min = -2
t_max = 1.0
n_t = 500

id_0 = int(len(graph)/2)

times = np.logspace(t_min, t_max, n_t)

# set the source
results = run_single_source(graph, times, delta_measure(graph, id_0))
save_single_source_results(results)
