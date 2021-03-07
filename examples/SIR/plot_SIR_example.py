from pathlib import Path

from dyngdim.sir import plot_analysis

if __name__ == "__main__":
    for folder in Path(".").iterdir():
        if (folder / "corr_scan.pkl").exists():
            plot_analysis(folder)
