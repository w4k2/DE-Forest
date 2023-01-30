import glob
import json
import os
import matplotlib.pyplot as plt


# Paths contains the list of paths, where runhistory.json files exist
paths = glob.glob("/home/joannagrzyb/work/DE-Forest/**/runhistory.json", recursive=True)

# Load data to as json object and retrieve parameters values
index = []
bac_score = []
for path in paths:
    try:
        f = open(path)
        data = json.load(f)

        for row in data["data"]:
            index.append(row[0])
            bac_score.append(row[9]["score_bac"])

        bootstraps = []
        metric_names = []
        n_clfs = []
        p_sizes = []
        for row in data["configs"].values():
            bootstraps.append(row["bootstrap"])
            metric_names.append(row["metric_name"])
            n_clfs.append(row["n_classifiers"])
            p_sizes.append(row["p_size"])            
    except:
        print("Error loading data!")

# Closing file
f.close()

fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

axs.flat[0].stem(index, n_clfs)
axs.flat[0].set_title("The number of classifiers")
axs.flat[0].set_ylabel("n_classifiers")
axs.flat[0].set_xlabel("Iterations")
axs.flat[0].grid(color='lightgray')
axs.flat[0].set(ylim=(0, max(n_clfs)+1))

ax_bac = axs.flat[0].twinx()
ax_bac.plot(index, bac_score, color='rebeccapurple')
ax_bac.set_ylabel("BAC score", color='rebeccapurple')
plt.fill_between(index, bac_score, alpha=0.2, color='rebeccapurple')
ax_bac.tick_params(axis='y', colors='rebeccapurple')
ax_bac.set(ylim=(0.5, 1))

axs.flat[1].stem(index, p_sizes)
axs.flat[1].set_title("Population size")
axs.flat[1].set_ylabel("p_size")
axs.flat[1].set_xlabel("Iterations")
axs.flat[1].grid(color='lightgray')
axs.flat[1].set(ylim=(0, max(p_sizes)+1))

ax_bac = axs.flat[1].twinx()
ax_bac.plot(index, bac_score, color='rebeccapurple')
ax_bac.set_ylabel("BAC score", color='rebeccapurple')
plt.fill_between(index, bac_score, alpha=0.2, color='rebeccapurple')
ax_bac.tick_params(axis='y', colors='rebeccapurple')
ax_bac.set(ylim=(0.5, 1))

axs.flat[2].stem(index, bootstraps)
axs.flat[2].set_title("Bootstrapsping")
axs.flat[2].set_ylabel("Bootstrap")
axs.flat[2].set_xlabel("Iterations")
axs.flat[2].grid(color='lightgray')

ax_bac = axs.flat[2].twinx()
ax_bac.plot(index, bac_score, color='rebeccapurple')
ax_bac.set_ylabel("BAC score", color='rebeccapurple')
plt.fill_between(index, bac_score, alpha=0.2, color='rebeccapurple')
ax_bac.tick_params(axis='y', colors='rebeccapurple')
ax_bac.set(ylim=(0.5, 1))

axs.flat[3].stem(index, metric_names)
axs.flat[3].set_title("Metric name")
axs.flat[3].set_ylabel("metric_name")
axs.flat[3].set_xlabel("Iterations")
axs.flat[3].grid(color='lightgray')

ax_bac = axs.flat[3].twinx()
ax_bac.plot(index, bac_score, color='rebeccapurple')
ax_bac.set_ylabel("BAC score", color='rebeccapurple')
plt.fill_between(index, bac_score, alpha=0.2, color='rebeccapurple')
ax_bac.tick_params(axis='y', colors='rebeccapurple')
ax_bac.set(ylim=(0.5, 1))

fig.suptitle('SMAC optimization')

if not os.path.exists("smac_outputs/plots/"):
    os.makedirs("smac_outputs/plots/")
filepath = "smac_outputs/plots/smac_plot"
plt.savefig(filepath + ".png", bbox_inches='tight')
plt.savefig(filepath + ".eps", format='eps', bbox_inches='tight')
plt.clf()
