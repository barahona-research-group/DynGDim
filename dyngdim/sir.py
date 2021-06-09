"""Module to compare SIR simulations with local dimensions."""
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import EoN
import networkx as nx
import numpy as np
import pylab as plt
from scipy.stats import pearsonr
from tqdm import tqdm

from dyngdim.dyngdim import run_local_dimension


def single_run(G, beta, node, mu=1):
    """Run SIR using EoN"""
    R = EoN.fast_SIR(G, beta, mu, initial_infecteds=node)[-1]
    return (R[-1] - 1) / (len(G) - 1)


def several_runs(node, n_runs, G, beta, mu=1):
    """Run several SIR simulations."""
    return [single_run(G, beta, node, mu=mu) for _ in range(n_runs)]


def run(n_runs, G, beta, n_workers=1, mu=1, nodes=None):
    """Run several SIR simulations, and returns mean, percentiles and chi values."""
    if nodes is None:
        nodes = G
    res_all, mean, perc_l, perc_u = [], [], [], []
    with Pool(processes=n_workers) as p:
        for res in list(
            tqdm(
                p.imap(partial(several_runs, n_runs=n_runs, G=G, beta=beta, mu=mu), nodes),
                total=len(nodes),
            )
        ):
            res_all += res
            m = np.mean(res)
            mean.append(m)
            perc_l.append(abs(m - np.percentile(res, 25)))
            perc_u.append(abs(m - np.percentile(res, 75)))

    percs = [perc_l, perc_u]
    res_all = np.array(res_all)
    chi = np.mean((res_all - np.mean(res_all)) ** 2) / np.mean(res_all ** 2)
    infect = np.mean(res_all)
    return mean, percs, chi, infect


def compute_corr(n_runs, G, beta, local_dimensions, n_workers=1, mu=1, nodes=None):
    """Compute the correlation between local dimension and SIR dynamnics.

    Returns mean sir values, percentiles, correlations witth local dimension and chi values.
    """
    corrs = []
    mean_sir, perc_sir, chi, infect = run(n_runs, G, beta, n_workers=n_workers, mu=mu, nodes=nodes)
    for dim in local_dimensions:
        dim[np.isnan(dim)] = 0
        corrs.append(pearsonr(mean_sir, dim)[0])
    corrs = np.array(corrs)
    corrs[np.isnan(corrs)] = 0
    return mean_sir, perc_sir, corrs, chi, infect


def get_beta_critical(G, mu=1):
    """Compute beta criticals.

    Returns beta critical based on eigenvalues and degrees

    TODO: based on message passing.
    """

    A = nx.adjacency_matrix(G)
    w, _ = np.linalg.eigh(A.toarray())
    beta_crit_eig = mu / np.max(w)
    print("Beta critical (1/eig) = ", beta_crit_eig)

    degree = np.array([len(G[u]) for u in G])
    beta_crit_deg = np.mean(degree) / (np.mean(degree ** 2) - np.mean(degree))
    print("Beta critical (degree) = ", beta_crit_deg)

    return beta_crit_eig, beta_crit_deg


def scan_beta(
    G,
    betas,
    times,
    n_runs,
    local_dimensions,
    n_workers=1,
    plot_folder=None,
    data_folder=None,
    mu=1,
    nodes=None,
):
    """Compute correlations with a beta scan.

    Returns the scan of correlationa and the chis values.
    """
    Path(data_folder).mkdir(exist_ok=True, parents=True)
    Path(plot_folder).mkdir(exist_ok=True, parents=True)
    corr_scan = []
    chis = []
    infects = []
    for beta in betas:
        print("computing beta = ", beta)
        mean_sir, std_sir, corrs, chi, infect = compute_corr(
            n_runs,
            G,
            beta,
            local_dimensions,
            n_workers=n_workers,
            mu=mu,
            nodes=nodes,
        )
        chis.append(chi)
        infects.append(infect)
        corr_scan.append(corrs)

        if data_folder:
            pickle.dump(
                [times, beta, mean_sir, std_sir, corrs, chi, infect],
                open(f"{data_folder}/sir_{np.round(beta, 4)}.pkl", "wb"),
            )
        if plot_folder:
            plt.figure()
            plt.errorbar(
                local_dimensions[np.argmax(corrs)], mean_sir, yerr=std_sir, fmt="+", elinewidth=0.1
            )
            plt.xlabel("local dim")
            plt.ylabel("sir")
            plt.savefig(f"{plot_folder}/sir_{np.round(beta, 4)}.pdf")
            plt.close()

            plt.figure()
            plt.semilogx(times, corrs)
            plt.axvline(times[np.argmax(corrs)])
            plt.gca().set_ylim(0, 1)
            plt.savefig(f"{plot_folder}/corr_{np.round(beta, 4)}.pdf")
            plt.close()
    return corr_scan, chis, infects


def analyse_graph(
    G, times=None, betas=None, n_runs=100, n_workers=4, folder="output", mu=1, nodes=None
):
    """Run SIR comparision with local dimension."""
    if times is None:
        times = np.logspace(-3.5, 1.2, 1000)
    if betas is None:
        betas = np.logspace(-0.7, 1.5, 50)

    if not Path(folder).exists():
        Path(folder).mkdir()

    if not (Path(folder) / "figures").exists():
        (Path(folder) / "figures").mkdir()

    local_dimensions = run_local_dimension(G, times, n_workers=n_workers, nodes=nodes)
    corr_scan, chis, infects = scan_beta(
        G,
        betas,
        times,
        n_runs,
        local_dimensions,
        n_workers=n_workers,
        plot_folder=f"{folder}/figures",
        data_folder=f"{folder}/data",
        mu=mu,
        nodes=nodes,
    )
    beta_crit_eig, beta_crit_deg = get_beta_critical(G)
    pickle.dump(
        [times, betas, beta_crit_eig, beta_crit_deg, corr_scan, chis, infects],
        open(f"{folder}/corr_scan.pkl", "wb"),
    )
    return times, betas, beta_crit_eig, beta_crit_deg, corr_scan, chis


def plot_analysis(folder, vmin=0.7, with_beta_crit=False, with_chi=False):
    """Plot SIR analysis."""
    times, betas, beta_crit_eig, beta_crit_deg, corr_scan, chis, infects = pickle.load(
        open(f"{folder}/corr_scan.pkl", "rb")
    )
    best = []
    for row in corr_scan:
        best.append(times[np.argmax(row)])

    plt.figure(figsize=(5, 2))
    plt.pcolormesh(
        times[::5], betas, np.array(corr_scan)[:, ::5], cmap="YlOrBr", shading="nearest", vmin=vmin
    )
    plt.plot(best, betas, "--C0")

    if with_beta_crit:
        plt.axhline(beta_crit_eig, c="r")
        plt.axhline(beta_crit_deg, c="b")

    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel("beta")

    if with_chi:
        plt.twiny()
        plt.plot(chis, betas)

    plt.twiny()
    plt.plot(infects, betas, c="k")
    plt.xlabel("infectability")
    # plt.twiny()
    plt.savefig(f"{folder}/corr_scan.pdf", bbox_inches="tight")
