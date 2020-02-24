"""plot relative dimensions from all sources"""
import matplotlib.pyplot as plt
import numpy as np

from dyngdim import compute_global_dimension
from dyngdim.io import load_local_dimensions
from dyngdim.plotting import plot_local_dimensions
from generate_grid import generate_grid

graph = generate_grid()

times, local_dimensions = load_local_dimensions()
times = times[:-1]  # last point has numerical errors
local_dimensions = local_dimensions[:-1]

pos = [[u, 0] for u in graph]
plot_local_dimensions(graph, local_dimensions, times[:-1], pos=pos)

global_dimension = compute_global_dimension(local_dimensions)

plt.figure()
plt.imshow(
    local_dimensions,
    origin="auto",
    aspect="auto",
    extent=(0, len(graph), np.log10(times[0]), np.log10(times[-1])),
)
plt.colorbar(label="local dimension")
plt.xlabel("node id")
plt.ylabel("log10(time)")

plt.figure()
plt.semilogx(times, global_dimension, "-")
plt.xlabel("time")
plt.ylabel("Global dimension")
plt.show()
