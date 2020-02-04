"""plotting functions"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_single_source(results, ds=[1, 2, 3], folder="./"):
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
