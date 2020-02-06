"""plotting functions"""
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx


def plot_all_sources(relative_dimensions):
    """plot relative dimensionf computed from all sources"""
    relative_dimensions = relative_dimensions + np.diag(
        len(relative_dimensions) * [np.nan]
    )

    plt.figure()
    plt.imshow(relative_dimensions, cmap=plt.get_cmap("coolwarm"))
    plt.colorbar(label="Rwlative dimension")


def plot_single_source(
    results, ds=[1, 2, 3], folder="./"
):  # pylint: disable=dangerous-default-value
    """plot the relative dimensions"""

    plt.figure()
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.2, 1], width_ratios=[1, 0.05])

    gs.update(wspace=0.05)
    gs.update(hspace=0.00)

    ax1 = plt.subplot(gs[0, 0])
    plt.hist(
        np.log10(results["peak_times"]),
        bins=int(len(results["times"]) / 10),
        density=False,
        log=True,
        range=(
            np.log10(results["times"][0] * 0.9),
            np.log10(results["times"][-1] * 1.1),
        ),
        color="0.5",
    )
    ax1.set_xlim(
        np.log10(results["times"][0] * 0.9), np.log10(results["times"][-1] * 1.1)
    )
    ax1.set_xticks([])

    ax2 = plt.subplot(gs[1, 0])

    def f(d):
        f = (
            results["times"] ** (-d / 2.0)
            * np.exp(-d / 2.0)
            / (4.0 * results["diffusion_coefficient"] * np.pi) ** (0.5 * d)
        )
        return f

    for d in ds:
        ax2.plot(results["times"], f(d), "--", lw=2, label=r"$d_{rel} =$" + str(d))

    cmap = plt.get_cmap("coolwarm")

    nan_id = np.argwhere(results["relative_dimensions"] == 0)
    relative_dimension_nan = results["relative_dimensions"].copy()
    relative_dimension_nan[nan_id] = np.nan

    plt.scatter(
        results["peak_times"][nan_id],
        results["peak_amplitudes"][nan_id],
        c="k",
        s=50,
        cmap=cmap,
    )
    plt.scatter(
        results["peak_times"],
        results["peak_amplitudes"],
        c=relative_dimension_nan,
        s=50,
        cmap=cmap,
    )

    plt.xscale("log")
    plt.yscale("log")

    ax2.set_xlim(results["times"][0] * 0.9, results["times"][-1] * 1.1)
    ax2.set_ylim(
        np.min(results["peak_amplitudes"]) * 0.9,
        np.max(results["peak_amplitudes"]) * 1.1,
    )
    ax1.set_xticks([])

    ax_cb = plt.subplot(gs[1, 1])
    cbar = plt.colorbar(ax=ax2, cax=ax_cb)
    cbar.set_label(r"${\rm Relative\,\, dimension}\,\, d_{rel}$")

    ax2.set_xlabel(r"$\rm Peak\,\,  time\, \,  (units\, \,  of\, \,  \lambda_2)$")
    ax2.set_ylabel(r"$\rm Peak\,\, amplitude$")
    ax2.legend()

    plt.savefig(folder + "/relative_dimension.svg")


def plot_local_dimensions(
    graph, local_dimension, times, pos=None, folder="./local_dimension_figs"
):
    """plot local dimensions"""

    if not os.path.isdir(folder):
        os.mkdir(folder)

    if pos is None:
        pos = nx.spring_layout(graph)

    plt.figure()

    vmin = np.nanmin(local_dimension)
    vmax = np.nanmax(local_dimension)

    for time_index, time_horizon in enumerate(times):
        plt.figure()

        node_size = (
            local_dimension[time_index, :] / np.max(local_dimension[time_index, :]) * 20
        )

        cmap = plt.cm.coolwarm

        node_order = np.argsort(node_size)

        for n in node_order:
            nodes = nx.draw_networkx_nodes(
                graph,
                pos=pos,
                nodelist=[n,],
                node_size=node_size[n],
                cmap=cmap,
                node_color=[
                    local_dimension[time_index, n]
                    / np.max(local_dimension[time_index, :]),
                ],
                vmin=vmin,
                vmax=vmax,
            )

        plt.colorbar(nodes, label="Local Dimension")

        weights = np.array([graph[i][j]["weight"] for i, j in graph.edges])
        nx.draw_networkx_edges(graph, pos=pos, alpha=0.5, width=2 * weights)

        plt.suptitle("Time Horizon {:.2e}".format(time_horizon), fontsize=14)
        plt.savefig(folder + "/local_dimension_{}.svg".format(time_index))
        plt.close()
