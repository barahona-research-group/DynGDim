"""function to generate the same grid in the examples"""
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
