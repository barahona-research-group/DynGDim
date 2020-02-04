"""main functions"""
import numpy as np
import scipy as sc
from tqdm import tqdm
import networkx as nx


def run_single_source(graph, times, initial_measure, use_spectral_gap=True):
    """main function to compute relative dimensions"""
    laplacian, spectral_gap = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)

    node_trajectories = compute_node_trajectories(laplacian, initial_measure, times)
    (
        relative_dimensions,
        peak_amplitudes,
        peak_times,
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


def construct_laplacian(graph, laplacian_tpe="normalized", use_spectral_gap=True):
    """construct the Laplacian matrix"""

    if laplacian_tpe == "normalized":
        degrees = np.array([graph.degree[i] for i in graph.nodes])
        laplacian = sc.sparse.diags(1.0 / degrees).dot(nx.laplacian_matrix(graph))
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


# compute all neighbourhood densities
def heat_kernel(laplacian, timestep, measure):
    """compute matrix exponential on a measure"""
    return sc.sparse.linalg.expm_multiply(-timestep * laplacian, measure)


def compute_node_trajectories(laplacian, initial_measure, times):
    """compute node trajectories from diffusion dynamics"""
    node_trajectories = [
        heat_kernel(laplacian, times[0], initial_measure),
    ]
    for i in tqdm(range(len(times) - 1)):
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

    return relative_dimensions, peak_amplitudes, peak_times, diffusion_coefficient
