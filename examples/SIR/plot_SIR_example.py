from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from dyngdim.sir import plot_analysis


if __name__ == "__main__":
    bests = []
    best_corrs = []
    infects_all = []
    ps_id = []
    folders = sorted(
        [str(folder) for folder in Path(".").iterdir() if (folder / "corr_scan.pkl").exists()]
    )
    print(folders)

    for folder in folders:
        plot_analysis(folder, vmin=0.5)
        ps_id.append(int(str(folder).split("_")[-1]))
        times, betas, beta_crit_eig, beta_crit_deg, corr_scan, chis, infects = pickle.load(
            open(f"{folder}/corr_scan.pkl", "rb")
        )
        best = []
        best_corr = []
        for row in corr_scan:
            best.append(times[np.argmax(row)])
            best_corr.append(np.max(row))

        bests.append(best)
        best_corrs.append(best_corr)
        infects_all.append(infects)

    ps = np.logspace(-2, -0.5, 10)
    plt.figure(figsize=(3, 4))
    colors = plt.cm.jet(np.linspace(0, 1, len(ps_id)))
    cmappable = ScalarMappable(norm=LogNorm(ps[0], ps[-1]), cmap="jet")

    for i in np.argsort(ps_id):
        infects = np.array(infects_all[i])
        shift = betas[np.argmin(abs(infects - 0.5))]
        plt.loglog(bests[i], betas / shift, c=colors[i], label=np.around(ps[i], 3))
    plt.legend()
    plt.xlabel("best Markov time")
    plt.ylabel(r"$\frac{\beta}{\beta_\mathrm{crit}}$")
    plt.colorbar(cmappable, label="p")
    plt.savefig("best.pdf", bbox_inches="tight")

    plt.figure(figsize=(3, 4))
    # for i, (best, infects, p_id) in enumerate(zip(best_corrs, infects_all, ps_id)):
    print(ps_id)
    for i in np.argsort(ps_id):
        best = np.array(best_corrs[i])
        infects = np.array(infects_all[i])
        shift = betas[np.argmin(abs(infects - 0.5))]
        plt.semilogy(best, betas / shift, c=colors[i], label=np.around(ps[i], 3))
        # plt.axvline(shift)
        # plt.plot(betas / shift, infects)

    plt.colorbar(cmappable, label="p")
    plt.legend()
    plt.xlabel("best pearson correlation")
    plt.ylabel(r"$\frac{\beta}{\beta_\mathrm{crit}}$")
    plt.savefig("best_corrs.pdf", bbox_inches="tight")
