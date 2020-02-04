"""compute relative dimension from single source"""
import numpy as np

from dyngdim import run_single_source
from dyngdim.utils import delta_measure
from dyngdim.io import save_single_source_results

from generate_grid import generate_grid

graph = generate_grid()


t_min = -5
t_max = 0.5
n_t = 200

id_0 = 4  # int(n_node/2)

times = np.logspace(t_min, t_max, n_t)

# set the source
results = run_single_source(graph, times, delta_measure(graph, id_0))
save_single_source_results(results)
