from dyngdim.plotting import plot_single_source 
from dyngdim.io import load_single_source_results
import matplotlib.pyplot as plt

results = load_single_source_results()
plot_single_source(results)

plt.show()
