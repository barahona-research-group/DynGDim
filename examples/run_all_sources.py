"""compute relative dimensions from all sources"""
import numpy as np

from dyngdim import run_all_sources
from dyngdim.io import save_all_sources_relative_dimensions
from generate_grid import generate_grid

graph = generate_grid()

t_min = -2
t_max = 1.0
n_t = 10
n_workers = 4

times = np.logspace(t_min, t_max, n_t)

relative_dimensions = run_all_sources(graph, times, n_workers=n_workers)[0]
save_all_sources_relative_dimensions(relative_dimensions)
