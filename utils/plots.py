import os
import numpy as np
import matplotlib.pyplot as plt


def optimization_plot(res_history, y_label, dataset_name, filename):
    val = [e.opt.get("F")[0] for e in res_history]
    values = [-value for value in val]
    plt.plot(np.arange(int(len(val))), values, "ro")
    plt.ylabel(y_label)
    plt.xlabel("N_gen")
    plt.title("Optimization")
    plt.grid(True, color="silver", linestyle=":", axis='both', which='both')
    plt.ylim(0, 1)
    # Save plot
    file_path = "results/experiment0/optimization/%s/%s" % (dataset_name, filename)
    if not os.path.exists("results/experiment0/optimization/%s/" % (dataset_name)):
        os.makedirs("results/experiment0/optimization/%s/" % (dataset_name))
    plt.savefig(file_path+".png", bbox_inches='tight')
    plt.savefig(file_path+".eps", format='eps', bbox_inches='tight')
    plt.close()


# Save plot to the file png and eps of quality
def process_plot(dataset_name, metric_name, clf_names, n_folds, clf_name):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.legend()
    plt.legend(reversed(plt.legend().legendHandles), clf_names, framealpha=1)
    plt.grid(True, color="silver", linestyle=":")
    plt.ylabel(metric_name)
    plt.xlabel("Data")
    plt.axis([0, n_folds, 0, 1])
    plt.gcf().set_size_inches(10, 5)  # Get the current figure
    # Save plot
    file_path = "results/experiment0/process_plots/%s/process_plot_%s_%s" % (dataset_name, clf_name, metric_name)
    if not os.path.exists("results/experiment0/process_plots/%s/" % (dataset_name)):
        os.makedirs("results/experiment0/process_plots/%s/" % (dataset_name))
    plt.savefig(file_path+".png", bbox_inches='tight')
    plt.savefig(file_path+".eps", format='eps', bbox_inches='tight')
    plt.clf()  # Clear the current figure
    plt.close()


def diversity_bar_plot(diversity, diversity_measures, methods_ens_alias, experiment_name):
    for metric_id, metric in enumerate(diversity_measures):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(methods_ens_alias, diversity[:, metric_id], width=0.4, color=["#2F3441", "#877669", "#877669", "#A6B1A2", "#E2B3A9"])
        # , "#877669", "#A6B1A2", "#E2B3A9"])

        plt.grid(True, color="silver", linestyle=":", axis='y', which='major')
        plt.ylabel(f"{metric}", fontsize=14)
        plt.xlabel("Methods", fontsize=14)
        plt.xticks(rotation="vertical")
        plt.gcf().set_size_inches(6, 6)
        # Save plot
        filename = "results/%s/diversity_plot/diversity_bar_plot_%s_%s" % (experiment_name, metric, experiment_name)
        if not os.path.exists("results/%s/diversity_plot/" % (experiment_name)):
            os.makedirs("results/%s/diversity_plot/" % (experiment_name))
        plt.savefig(filename+".png", bbox_inches='tight')
        plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
        plt.clf()
        plt.close()
