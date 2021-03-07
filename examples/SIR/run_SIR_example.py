import networkx as nx
from pathlib import Path

import numpy as np

from dyngdim.sir import analyse_graph


if __name__ == "__main__":

    times = np.logspace(-3.5, 1.2, 500)
    betas = np.logspace(-0.7, 1.5, 10)
    n_runs = 20
    n_workers = 80

    n_nodes = 500
    ps = np.logspace(-2, -0.5, 10)

    for i, p in enumerate(ps):
        print(f"computing p = {p}")

        folder = f"output_sm_{i}"
        G = nx.newman_watts_strogatz_graph(n_nodes, 2, p, seed=0)
        analyse_graph(
            G, times=times, betas=betas, n_runs=n_runs, n_workers=n_workers, folder=folder
        )
        nx.write_gpickle(G, f"{folder}/graph.gpickle")
