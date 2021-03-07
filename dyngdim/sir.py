"""Module to compare SIR simulations with local dimensions."""
import networkx as nx
from pathlib import Path
import pickle as pickle
from scipy.stats import pearsonr

from tqdm import tqdm
from epydemic import SIR, StochasticDynamics
import pylab as plt

import numpy as np

from functools import partial
from multiprocessing import Pool


from dyngdim.dyngdim import run_local_dimension


def single_run(G, beta, node, mu=1):
    """Run SIR model with single infected node and return fraction of removed nodes."""
    Model = SIR()
    E = StochasticDynamics(Model, G)

    param = dict()
    param[SIR.P_REMOVE] = mu
    param[SIR.P_INFECT] = beta
    param[SIR.P_INFECTED] = 0.0
    E.setUp(param)

    Model.changeCompartment(node, Model.INFECTED)
    return 1.0 * (E.do({})["epydemic.SIR.R"] - 1) / (len(G) - 1)


def several_runs(node, n_runs, G, beta, mu=1):
    """Run several SIR simulations."""
    return [single_run(G, beta, node, mu=mu) for _ in range(n_runs)]


def run(n_runs, G, beta, n_workers=1, mu=1):
    """Run several SIR simulations, and returns mean, percentiles and chi values."""
    res_all, mean, perc_l, perc_u = [], [], [], []
    with Pool(processes=n_workers) as p:
        for res in list(
            tqdm(
                p.imap(partial(several_runs, n_runs=n_runs, G=G, beta=beta, mu=mu), G.nodes),
                total=len(G.nodes),
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
    return mean, percs, chi


def compute_corr(n_runs, G, beta, times, local_dimensions, n_workers=1, mu=1):
    """Compute the correlation between local dimension and SIR dynamnics.

    Returns mean sir values, percentiles, correlations witth local dimension and chi values.
    """
    corrs = []
    mean_sir, perc_sir, chi = run(n_runs, G, beta, n_workers=n_workers, mu=mu)
    for dim in local_dimensions:
        dim[np.isnan(dim)] = 0
        corrs.append(pearsonr(mean_sir, dim)[0])
    corrs = np.array(corrs)
    corrs[np.isnan(corrs)] = 0
    return mean_sir, perc_sir, corrs, chi


def get_beta_critical(G, mu=1):
    """Compute beta criticals.

    Returns beta critical based on eigenvalues and degrees

    TODO: based on message passing.
    """

    A = nx.adjacency_matrix(G)
    w, v = np.linalg.eigh(A.toarray())
    beta_crit_eig = mu / np.max(w)
    print("Beta critical (1/eig) = ", beta_crit_eig)

    degree = np.array([len(G[u]) for u in G])
    beta_crit_deg = np.mean(degree) / (np.mean(degree ** 2) - np.mean(degree))
    print("Beta critical (degree) = ", beta_crit_deg)

    return beta_crit_eig, beta_crit_deg


def scan_beta(G, betas, times, n_runs, local_dimensions, n_workers=1, plot_folder=None, mu=1):
    """Compute correlations with a beta scan.

    Returns the scan of correlationa and the chis values.
    """
    corr_scan = []
    chis = []
    for beta in betas:
        print("computing beta = ", beta)
        mean_sir, std_sir, corrs, chi = compute_corr(
            n_runs, G, beta, times, local_dimensions, n_workers=n_workers, mu=mu
        )
        chis.append(chi)
        corr_scan.append(corrs)
        if plot_folder:
            plt.figure()
            plt.errorbar(local_dimensions[np.argmax(corrs)], mean_sir, yerr=std_sir, fmt="+")
            plt.xlabel("local dim")
            plt.ylabel("sir")
            plt.savefig(f"{plot_folder}/sir_{np.round(beta, 2)}.pdf")
            plt.close()

            plt.figure()
            plt.semilogx(times, corrs)
            plt.axvline(times[np.argmax(corrs)])
            plt.gca().set_ylim(0, 1)
            plt.savefig(f"{plot_folder}/corr_{np.round(beta, 2)}.pdf")
            plt.close()
    return corr_scan, chis


def analyse_graph(G, times=None, betas=None, n_runs=100, n_workers=4, folder="output", mu=1):
    """Run SIR comparision with local dimension."""
    if times is None:
        times = np.logspace(-3.5, 1.2, 1000)
    if betas is None:
        betas = np.logspace(-0.7, 1.5, 50)

    if not Path(folder).exists():
        Path(folder).mkdir()

    if not (Path(folder) / "figures").exists():
        (Path(folder) / "figures").mkdir()

    local_dimensions = run_local_dimension(G, times, n_workers=n_workers)
    corr_scan, chis = scan_beta(
        G,
        betas,
        times,
        n_runs,
        local_dimensions,
        n_workers=n_workers,
        plot_folder=f"{folder}/figures",
        mu=mu,
    )
    beta_crit_eig, beta_crit_deg = get_beta_critical(G)
    pickle.dump(
        [times, betas, beta_crit_eig, beta_crit_deg, corr_scan, chis],
        open(f"{folder}/corr_scan.pkl", "wb"),
    )
    return times, betas, beta_crit_eig, beta_crit_deg, corr_scan, chis


def plot_analysis(folder, vmin=0.7):
    """Plot SIR analysis."""
    times, betas, beta_crit_eig, beta_crit_deg, corr_scan, chis = pickle.load(
        open(f"{folder}/corr_scan.pkl", "rb")
    )
    plt.figure()
    plt.pcolormesh(times, betas, corr_scan, cmap="YlOrBr", shading="nearest", vmin=vmin)

    plt.axhline(beta_crit_eig, c="r")
    plt.axhline(beta_crit_deg, c="b")
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.twiny()
    plt.plot(chis, betas)
    plt.savefig(f"{folder}/corr_scan.pdf")