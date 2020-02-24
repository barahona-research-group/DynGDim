"""plot relative dimensions from all sources"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dyngdim.io import load_all_sources_relative_dimensions
from dyngdim.plotting import plot_all_sources
from generate_grid import generate_grid

relative_dimensions = np.array(load_all_sources_relative_dimensions())
plot_all_sources(relative_dimensions)

graph = generate_grid()
pos = [[u, 0] for u in graph]

plt.figure()
nx.draw(
    graph, pos=pos, node_color=relative_dimensions.sum(1), cmap=plt.get_cmap("coolwarm")
)

plt.show()
