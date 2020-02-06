"""main functions"""
import multiprocessing
import time

import networkx as nx
import numpy as np
import scipy as sc
from tqdm import tqdm

from .utils import delta_measure


class Worker:
    """worker for computing relative dimensions"""

    def __init__(self, graph, laplacian, times, spectral_gap):
        self.laplacian = laplacian
        self.times = times
        self.spectral_gap = spectral_gap
        self.graph = graph

    def __call__(self, node):
        print("node %s..." % node)
        time_0 = time.time()
        initial_measure = delta_measure(self.graph, node)
        node_trajectories = compute_node_trajectories(
            self.laplacian, initial_measure, self.times, disable_tqdm=True
        )
        print(
            "node %s... done in %.2f seconds"
            % (node, np.round(time.time() - time_0, 2))
        )
        return extract_relative_dimensions(
            self.times, node_trajectories, self.spectral_gap
        )[:2]


def run_all_sources(graph, times, use_spectral_gap=True, n_workers=1):
    """main function to compute all relative dimensions of a graph"""
    laplacian, spectral_gap = construct_laplacian(
        graph, use_spectral_gap=use_spectral_gap
    )

    worker = Worker(graph, laplacian, times, spectral_gap)
    pool = multiprocessing.Pool(n_workers)

    out = np.array(pool.map(worker, graph.nodes))

    relative_dimensions = out[:, 0]
    peak_times = out[:, 1]

    np.fill_diagonal(relative_dimensions, 0)
    np.fill_diagonal(peak_times, 0)

    return relative_dimensions, peak_times


def run_single_source(graph, times, initial_measure, use_spectral_gap=True):
    """main function to compute relative dimensions"""
    laplacian, spectral_gap = construct_laplacian(
        graph, use_spectral_gap=use_spectral_gap
    )

    node_trajectories = compute_node_trajectories(laplacian, initial_measure, times)
    (
        relative_dimensions,
        peak_times,
        peak_amplitudes,
        diffusion_coefficient,
    ) = extract_relative_dimensions(times, node_trajectories, spectral_gap)

    results = {
        "relative_dimensions": relative_dimensions,
        "peak_amplitudes": peak_amplitudes,
        "peak_times": peak_times,
        "diffusion_coefficient": diffusion_coefficient,
        "times": times,
    }

    return results


def run_local_dimension(graph, times, use_spectral_gap=True, n_workers=1):
    """  computing the local dimensionality of each node """

    relative_dimensions, peak_times = run_all_sources(
        graph, times, use_spectral_gap, n_workers
    )

    local_dimensions = []
    for time_horizon in times:
        relative_dimensions_reduced = relative_dimensions.copy()
        relative_dimensions_reduced[peak_times > time_horizon] = 0
        local_dimensions.append(relative_dimensions_reduced.mean(1))

    return np.array(local_dimensions)


def compute_global_dimension(local_dimensions):
    """ Computing the global dimensiona of the graph """
    return local_dimensions.mean(1)


def construct_laplacian(graph, laplacian_tpe="normalized", use_spectral_gap=True):
    """construct the Laplacian matrix"""

    if laplacian_tpe == "normalized":
        degrees = np.array([graph.degree(i, weight="weight") for i in graph.nodes])
        laplacian = sc.sparse.diags(1.0 / degrees).dot(nx.laplacian_matrix(graph))
        # laplacian = nx.laplacian_matrix(graph).dot(sc.sparse.diags(1.0 / degrees))
    else:
        raise Exception(
            "Any other laplacian type than normalized are not implemented as they will not work"
        )

    if use_spectral_gap:
        spectral_gap = abs(sc.sparse.linalg.eigs(laplacian, which="SM", k=2)[0][1])
        laplacian /= spectral_gap
    else:
        spectral_gap = 1.0

    return laplacian, spectral_gap


def heat_kernel(laplacian, timestep, measure):
    """compute matrix exponential on a measure"""
    return sc.sparse.linalg.expm_multiply(-timestep * laplacian, measure)


def compute_node_trajectories(laplacian, initial_measure, times, disable_tqdm=False):
    """compute node trajectories from diffusion dynamics"""
    node_trajectories = [
        heat_kernel(laplacian, times[0], initial_measure),
    ]
    for i in tqdm(range(len(times) - 1), disable=disable_tqdm):
        node_trajectories.append(
            heat_kernel(times[i + 1] - times[i], laplacian, node_trajectories[-1])
        )
    return np.array(node_trajectories)


def extract_relative_dimensions(times, node_trajectories, spectral_gap):
    """compute the relative dimensions, given a trajectory p_t
    dim_th is a threshold to select unreachable nodes,
    dim smaller than min(eff_dim)+dim_th are dim=0"""

    # set the diffusion coefficient
    diffusion_coefficient = 0.5 / spectral_gap

    # find the peaks
    peak_amplitudes = np.max(node_trajectories, axis=0)
    peak_times = times[np.argmax(node_trajectories, axis=0)]

    # compute the effective dimension
    relative_dimensions = (
        -2.0
        * np.log(peak_amplitudes)
        / (1.0 + np.log(peak_times) + np.log(4.0 * diffusion_coefficient * np.pi))
    )

    # set un-defined dimensions to 0
    relative_dimensions[np.isnan(relative_dimensions)] = 0
    relative_dimensions[relative_dimensions < 0] = 0

    return relative_dimensions, peak_times, peak_amplitudes, diffusion_coefficient
