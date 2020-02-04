import networkx as nx
import numpy as np

from dyngdim import run_single_source 
from dyngdim.plotting import plot_single_source 
from dyngdim.io import save_single_source_results

n_node = 101
dim = 1
periodic = False

t_min = -5
t_max = .5
n_t = 200

id_0 = int(n_node/2)

graph = nx.grid_graph(dim * [n_node], periodic=periodic)
graph = nx.convert_node_labels_to_integers(graph, label_attribute='pos')

times = np.logspace(t_min, t_max, n_t)

#set the source
source_id = [id_0,]
initial_measure = np.zeros(len(graph))
deg = np.array([len(graph[i]) for i in source_id])
initial_measure[source_id] = 1./len(source_id)

results = run_single_source(graph, times, initial_measure)
save_single_source_results(results)
