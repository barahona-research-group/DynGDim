"""plot relatice dimension from single source"""
import numpy as np
import matplotlib.pyplot as plt

from dyngdim.io import load_single_source_results
from dyngdim.plotting import plot_single_source
from dyngdim import run_single_source, initial_measure
import networkx as nx


def generate_grid():
    """generate the grid"""
    n_node = 500
    dim = 1
    periodic = False

    graph = nx.grid_graph(dim * [n_node], periodic=periodic)

    for u, v in graph.edges():
        graph[u][v]["weight"] = 1.0

    return nx.convert_node_labels_to_integers(graph, label_attribute="pos")


graph = generate_grid()


t_min = -4.5
t_max = 1.0
n_t = 2000

id_0 = int(len(graph) / 3)

times = np.logspace(t_min, t_max, n_t)

# set the source
measure = initial_measure(graph, [id_0])
results = run_single_source(graph, times, measure)

plot_single_source(results, with_trajectories=True)

plt.figure(figsize=(4, 3))
x = np.linspace(0, 1, len(results["relative_dimensions"]))
plt.plot(x, results["relative_dimensions"])
plt.axvline(x[id_0], c="k")
plt.xlabel("Position")
plt.ylabel("Relative dimension")
plt.gca().set_xlim(0, 1)
plt.savefig("rel_dim_interval.pdf", bbox_inches="tight")
plt.show()
