import os
import numpy as np
from .load_datasets import load_dataset
from .datasets_table_description import calc_imbalance_ratio


def result_tables_IR(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    imbalance_ratios = []
    for dataset_path in dataset_paths:
        X, y = load_dataset(dataset_path)
        IR = calc_imbalance_ratio(X, y)
        imbalance_ratios.append(IR)
    IR_argsorted = np.argsort(imbalance_ratios)
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables_IR/" % experiment_name):
            os.makedirs("results/%s/tables_IR/" % experiment_name)
        with open("results/%s/tables_IR/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s}" % (metric), file=file)
            columns = "r"
            for i in methods:
                columns += " c"
            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{ID} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for id, arg in enumerate(IR_argsorted):
                id += 1
                line = "%d" % (id)
                # lineir = "$%s$" % (dataset_paths[arg])
                # print(line, lineir)
                line_values = []
                line_values = mean_scores[arg, metric_id, :]
                max_value = np.amax(line_values)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[arg, metric_id, clf_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)


def result_tables_features(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    X_features = []
    for dataset_path in dataset_paths:
        X, y = load_dataset(dataset_path)
        X_features.append(X.shape[1])
    X_features_argsorted = np.argsort(X_features)
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables_features/" % experiment_name):
            os.makedirs("results/%s/tables_features/" % experiment_name)
        with open("results/%s/tables_features/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s}" % (metric), file=file)
            columns = "r"
            for i in methods:
                columns += " c"
            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{ID} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for id, arg in enumerate(X_features_argsorted):
                id += 1
                line = "%d" % (id)
                # lineir = "$%s$" % (dataset_paths[arg])
                # print(line, lineir)
                line_values = []
                line_values = mean_scores[arg, metric_id, :]
                max_value = np.amax(line_values)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[arg, metric_id, clf_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)


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


def result_tables_for_time(dataset_names, imbalance_ratios, sum_times, methods, experiment_name):
    IR_argsorted = np.argsort(imbalance_ratios)
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
        columns_names = "\\textbf{ID} &"
        for name in methods:
            name = name.replace("_", "-")
            columns_names += f'\\textbf{{{name}}} & '
        columns_names = columns_names[:-3]
        columns_names += "\\\\"
        print(columns_names, file=file)
        print("\\hline", file=file)
    
        for id, arg in enumerate(IR_argsorted):
            id += 1
            # ds_name = dataset_names[arg].replace("_", "\\_")
            # line = "%d & \\emph{%s}" % (id, ds_name)
            line = "%d" % (id)
            for clf_id, clf_name in enumerate(methods):
                line += " & %0.3f" % (sum_times[arg, clf_id])
            line += " \\\\"
            print(line, file=file)
        print("\\end{tabular}}", file=file)
        print("\\end{table}", file=file)
