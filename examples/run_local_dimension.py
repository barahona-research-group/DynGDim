"""compute relative dimensions from all sources"""
import numpy as np

from dyngdim import run_local_dimension
from dyngdim.io import save_local_dimensions
from generate_grid import generate_grid

graph = generate_grid()

t_min = -2
t_max = 1.0
n_t = 20
n_workers = 4

times = np.logspace(t_min, t_max, n_t)

local_dimensions = run_local_dimension(graph, times, n_workers=n_workers)
save_local_dimensions(times, local_dimensions)
