import pylab as plt
import pickle


def plot_scan(folder):

    times, betas, beta_crit_eig, beta_crit_deg, corr_scan, chis = pickle.load(
        open(f"{folder}/corr_scan.pkl", "rb")
    )
    plt.figure()
    plt.pcolormesh(times, betas, corr_scan, cmap="YlOrBr", shading="nearest", vmin=0.7)

    plt.axhline(beta_crit_eig, c="r")
    plt.axhline(beta_crit_deg, c="b")
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.twiny()
    plt.plot(chis, betas)
    plt.savefig(f"{folder}/corr_scan.pdf")


if __name__ == "__main__":
    plot_scan("output")
