import os
import numpy as np
import matplotlib.pyplot as plt
from .load_datasets import load_dataset
from .datasets_table_description import calc_imbalance_ratio


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


# def result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
#     imbalance_ratios = []
#     for dataset_path in dataset_paths:
#         X, y = load_dataset(dataset_path)
#         IR = calc_imbalance_ratio(X, y)
#         imbalance_ratios.append(IR)
#     IR_argsorted = np.argsort(imbalance_ratios)
#     for metric_id, metric in enumerate(metrics_alias):
#         if not os.path.exists("results/%s/tables/" % experiment_name):
#             os.makedirs("results/%s/tables/" % experiment_name)
#         with open("results/%s/tables/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
#             for id, arg in enumerate(IR_argsorted):
#                 id += 1
#                 line = "%d" % (id)
#                 line_values = []
#                 line_values = mean_scores[arg, metric_id, :]
#                 max_value = np.amax(line_values)
#                 for clf_id, clf_name in enumerate(methods):
#                     if mean_scores[arg, metric_id, clf_id] == max_value:
#                         line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
#                     else:
#                         line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
#                 line += " \\\\"
#                 print(line, file=file)


def result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables/" % experiment_name):
            os.makedirs("results/%s/tables/" % experiment_name)
        with open("results/%s/tables/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s}" % (metric), file=file)
            columns = "r"
            for i in methods:
                columns += " c"

            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{Dataset name} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for dataset_id, dataset_path in enumerate(dataset_paths):
                line = "$%s$" % (dataset_path)
                line_values = []
                line_values = mean_scores[dataset_id, metric_id, :]
                max_value = np.amax(line_values)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[dataset_id, metric_id, clf_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[dataset_id, metric_id, clf_id], stds[dataset_id, metric_id, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[dataset_id, metric_id, clf_id], stds[dataset_id, metric_id, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)


def result_tables_for_time(dataset_paths, mean_times_folds, methods, experiment_name):
    if not os.path.exists("results/%s/tables/" % experiment_name):
        os.makedirs("results/%s/tables/" % experiment_name)
    with open("results/%s/tables/time_%s.tex" % (experiment_name, experiment_name), "w+") as file:
        print("\\begin{table}[!ht]", file=file)
        print("\\centering", file=file)
        print("\\caption{Time [s]}", file=file)
        columns = "r"
        for i in methods:
            columns += " c"

        print("\\scalebox{0.4}{", file=file)
        print("\\begin{tabular}{%s}" % columns, file=file)
        print("\\hline", file=file)
        columns_names = "\\textbf{Dataset name} &"
        for name in methods:
            name = name.replace("_", "-")
            columns_names += f'\\textbf{{{name}}} & '
        columns_names = columns_names[:-3]
        columns_names += "\\\\"
        print(columns_names, file=file)
        print("\\hline", file=file)
        for dataset_id, dataset_path in enumerate(dataset_paths):
            line = "$%s$" % (dataset_path)
            line_values = []
            line_values = mean_times_folds[dataset_id, :]
            max_value = np.amax(line_values)
            for clf_id, clf_name in enumerate(methods):
                if mean_times_folds[dataset_id, clf_id] == max_value:
                    line += " & \\textbf{%0.3f}" % (mean_times_folds[dataset_id, clf_id])
                else:
                    line += " & %0.3f" % (mean_times_folds[dataset_id, clf_id])
            line += " \\\\"
            print(line, file=file)
        print("\\end{tabular}}", file=file)
        print("\\end{table}", file=file)
