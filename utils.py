import io
import re
import os
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from pathlib import Path
from math import sqrt, ceil
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def parse_keel_dat(dat_file):
    with open(dat_file, "r") as fp:
        data = fp.read()
        header, payload = data.split("@data\n")

    attributes = re.findall(
        r"@[Aa]ttribute (.*?)[ {](integer|real|.*)", header)
    output = re.findall(r"@[Oo]utput[s]? (.*)", header)

    dtype_map = {"integer": np.int, "real": np.float}

    columns, types = zip(*attributes)
    types = [*map(lambda _: dtype_map.get(_, np.object), types)]
    dtype = dict(zip(columns, types))

    # Replace missing values with NaN in datasets
    data = pd.read_csv(io.StringIO(payload), names=columns, dtype=dtype, na_values=[" <null>"])

    # Replace NaN values with most frequent values
    if data.isnull().values.any():
        imputer = SimpleImputer(strategy='most_frequent')
        imputer = imputer.fit(data)
        data = imputer.transform(data)
        data = pd.DataFrame(data, columns=columns)

    if not output:  # if it was not found
        output = columns[-1]
    target = data[output]
    data.drop(labels=output, axis=1, inplace=True)

    return data, target


def prepare_X_y(data, target):
    class_encoder = LabelEncoder()
    target = class_encoder.fit_transform(target.values.ravel())
    return data.values, target


def load_dataset(dataset_path, return_X_y=True):
    data, target = parse_keel_dat(dataset_path)
    if return_X_y:
        return prepare_X_y(data, target)
    return Bunch(data=data, target=target, filename=dataset_path)


def dataset_description(dataset_paths):
    datasets_names = []
    X_all = []
    y_all = []
    class_ratios = []
    number_of_features = []
    number_of_classes = []
    for dataset_path in dataset_paths:
        ds = Path(dataset_path).stem
        datasets_names.append(ds)
        X, y = load_dataset(dataset_path)
        unique, counts = np.unique(y, return_counts=True)
        print(ds, unique, counts)
        for cl, counter in zip(unique, counts):
            if cl == 0:
                class_ratio = str(counter)
            else:
                class_ratio += ":" + str(counter)
        class_ratios.append(class_ratio)
        n_features = X.shape[1]
        number_of_features.append(n_features)
        X_all.append(X)
        y_all.append(y)
        number_of_classes.append(len(unique))
    number_of_features_sorted = np.argsort(number_of_features)
    with open("results/datasets_description.tex", "w+") as file:
        for id, arg in enumerate(number_of_features_sorted):
            id += 1
            number_of_objects = len(y_all[arg])
            dataset_name = datasets_names[arg].replace("_", "\\_")
            print("%d & \\emph{%s} & %s & %d & %d\\\\" % (id, dataset_name, number_of_features[arg], number_of_objects, number_of_classes[arg]), file=file)


def result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    number_of_features = []
    for dataset_path in dataset_paths:
        X, y = load_dataset(dataset_path)
        n_features = X.shape[1]
        number_of_features.append(n_features)
    number_of_features_sorted = np.argsort(number_of_features)
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables/" % experiment_name):
            os.makedirs("results/%s/tables/" % experiment_name)
        with open("results/%s/tables/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
            for id, arg in enumerate(number_of_features_sorted):
                id += 1
                line = "%d" % (id)
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


def pairs_metrics_multi_grid_all(method_names, data_np, experiment_name, dataset_paths, metrics, filename, ref_methods, offset, treshold=0.5):
    # Load data
    data = {}
    for dataset_id, dataset_path in enumerate(dataset_paths):
        dataset_name = Path(dataset_path).stem
        for method_id, method_name in enumerate(method_names):
            for metric_id, metric in enumerate(metrics):
                try:
                    if metric == "Gmean2" or metric == "F1score":
                        continue
                    else:
                        data[(method_name, dataset_name, metric)] = data_np[dataset_id, metric_id, method_id]
                except:
                    print("None is ", method_name, dataset_name, metric)
                    data[(method_name, dataset_name, metric)] = None
                    print(data[(method_name, dataset_name, metric)])

    # Remove unnecessary metrics
    if "Gmean2" in metrics:
        metrics.remove("Gmean2")
    if "F1score" in metrics:
        metrics.remove("F1score")

    fig, axes = plt.subplots(len(metrics), len(ref_methods))
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    # Init/clear ranks
    for index_i, ref_method in enumerate(ref_methods):
        for index_j, metric in enumerate(metrics):
            ranking = {}
            for method_name in method_names:
                ranking[method_name] = {
                    "win": 0, "lose": 0, "tie": 0, "error": 0}

            # Pair tests
            for dataset in tqdm(dataset_paths, "Rank %s" % (metric)):
                dataset_name = Path(dataset).stem
                method_1 = ref_method
                for j, method_2 in enumerate(method_names):
                    if method_1 == method_2:
                        continue
                    try:
                        statistic, p_value = stats.ranksums(data[(method_1, dataset_name, metric)], data[(
                            method_2, dataset_name, metric)])
                        if p_value < treshold:
                            if statistic > 0:
                                ranking[method_2]["win"] += 1
                            else:
                                ranking[method_2]["lose"] += 1
                        else:
                            ranking[method_2]["tie"] += 1
                    except:
                        ranking[method_2]["error"] += 1
                        print("Exception", method_1, method_2,
                              dataset_name, metric)

            # Count ranks
            rank_win = []
            rank_tie = []
            rank_lose = []
            rank_error = []

            method_names_c = [x for x in method_names if x != ref_method]
            for method_name in method_names_c:
                rank_win.append(ranking[method_name]['win'])
                rank_tie.append(ranking[method_name]['tie'])
                rank_lose.append(ranking[method_name]['lose'])
                try:
                    rank_error.append(ranking[method_name]['error'])
                except Exception:
                    pass

            rank_win.reverse()
            rank_tie.reverse()
            rank_lose.reverse()
            rank_error.reverse()

            rank_win = np.array(rank_win)
            rank_tie = np.array(rank_tie)
            rank_lose = np.array(rank_lose)
            rank_error = np.array(rank_error)
            ma = method_names_c.copy()
            ma.reverse()

            # Plotting
            try:
                axes[index_j, index_i].barh(
                    ma, rank_error, color="green", height=0.9)
                axes[index_j, index_i].barh(
                    ma, rank_win, left=rank_error, color="green", height=0.9)
                axes[index_j, index_i].barh(
                    ma, rank_tie, left=rank_error + rank_win, color="gold", height=0.9)
                axes[index_j, index_i].barh(
                    ma, rank_lose, left=rank_error + rank_win + rank_tie, color="crimson", height=0.9)
                axes[index_j, index_i].set_xlim([0, len(dataset_paths)])
            except Exception:
                axes[index_j, index_i].barh(
                    ma, rank_win, color="green", height=0.9)
                axes[index_j, index_i].barh(
                    ma, rank_tie, left=rank_win, color="gold", height=0.9)
                axes[index_j, index_i].barh(
                    ma, rank_lose, left=rank_win + rank_tie, color="crimson", height=0.9)
                axes[index_j, index_i].set_xlim([0, len(dataset_paths)])

            # Name of the metric only on the left side of the figure
            axes[index_j, 0].text(offset, (index_j*0.15), metric.upper(), fontsize=12, weight="bold")
            # Name of the reference method only on the top of the figure
            axes[0, index_i].text(index_i, 3, ref_method, fontsize=12, weight="bold")

            # Calculate and plot critical difference
            N_of_streams = len(dataset_paths)
            critical_difference = ceil(
                N_of_streams / 2 + 1.96 * sqrt(N_of_streams) / 2)
            if len(dataset_paths) < 25:
                axes[index_j, index_i].axvline(
                    critical_difference, 0, 1, linestyle="--", linewidth=3, color="black")
            else:
                axes[index_j, index_i].axvline(
                    critical_difference, 0, 1, linestyle="--", linewidth=3, color="black")

    if not os.path.exists("results/%s/ranking/" % (experiment_name)):
        os.makedirs("results/%s/ranking/" % (experiment_name))
    plt.gcf().set_size_inches(9, 9)
    filepath = "results/%s/ranking/%s" % (experiment_name, filename)
    plt.savefig(filepath + ".png", bbox_inches='tight')
    plt.savefig(filepath + ".eps", format='eps', bbox_inches='tight')
    plt.clf()


def pairs_metrics_multi_grid(method_names, data_np, experiment_name, dataset_paths, metrics, filename, ref_method, offset, treshold=0.5):

    # Load data
    data = {}
    for dataset_id, dataset_path in enumerate(dataset_paths):
        dataset_name = Path(dataset_path).stem
        for method_id, method_name in enumerate(method_names):
            for metric_id, metric in enumerate(metrics):
                try:
                    if metric == "Gmean2" or metric == "F1score":
                        continue
                    else:
                        data[(method_name, dataset_name, metric)] = data_np[dataset_id, metric_id, method_id]
                except:
                    print("None is ", method_name, dataset_name, metric)
                    data[(method_name, dataset_name, metric)] = None
                    print(data[(method_name, dataset_name, metric)])

    # Remove unnecessary metrics
    if "Gmean2" in metrics:
        metrics.remove("Gmean2")
    if "F1score" in metrics:
        metrics.remove("F1score")
    fig, axes = plt.subplots(len(metrics), len(ref_method))
    # print(fig, axes, len(ref_methods))
    fig.subplots_adjust(hspace=0.6, wspace=0.6)

    # Init/clear ranks
    for index_j, metric in enumerate(metrics):
        ranking = {}
        for method_name in method_names:
            ranking[method_name] = {"win": 0, "lose": 0, "tie": 0, "error": 0}

        # Pair tests
        for dataset in tqdm(dataset_paths, "Rank %s" % (metric)):
            dataset_name = Path(dataset).stem
            method_1 = ref_method[0]
            for j, method_2 in enumerate(method_names):
                if method_1 == method_2:
                    continue
                try:
                    statistic, p_value = stats.ranksums(data[(method_1, dataset_name, metric)], data[(method_2, dataset_name, metric)])
                    if p_value < treshold:
                        if statistic > 0:
                            ranking[method_2]["win"] += 1
                        else:
                            ranking[method_2]["lose"] += 1
                    else:
                        ranking[method_2]["tie"] += 1
                except:
                    ranking[method_2]["error"] += 1
                    print("Exception", method_1, method_2, dataset_name, metric)

        # Count ranks
        rank_win = []
        rank_tie = []
        rank_lose = []
        rank_error = []

        method_names_c = [x for x in method_names if x != ref_method[0]]
        # print(method_names_c)
        for method_name in method_names_c:
            rank_win.append(ranking[method_name]['win'])
            rank_tie.append(ranking[method_name]['tie'])
            rank_lose.append(ranking[method_name]['lose'])
            try:
                rank_error.append(ranking[method_name]['error'])
            except Exception:
                pass

        rank_win.reverse()
        rank_tie.reverse()
        rank_lose.reverse()
        rank_error.reverse()

        rank_win = np.array(rank_win)
        rank_tie = np.array(rank_tie)
        rank_lose = np.array(rank_lose)
        rank_error = np.array(rank_error)
        ma = method_names_c.copy()
        ma.reverse()

        # Plotting
        try:
            axes[index_j].barh(
                ma, rank_error, color="blue", height=0.9)
            axes[index_j].barh(
                ma, rank_win, left=rank_error, color="green", height=0.9)
            axes[index_j].barh(
                ma, rank_tie, left=rank_error + rank_win, color="gold", height=0.9)
            axes[index_j].barh(
                ma, rank_lose, left=rank_error + rank_win + rank_tie, color="crimson", height=0.9)
            axes[index_j].set_xlim([0, len(dataset_paths)])
        except Exception:
            axes[index_j].barh(
                ma, rank_win, color="blue", height=0.9)
            axes[index_j].barh(
                ma, rank_tie, left=rank_win, color="gold", height=0.9)
            axes[index_j].barh(
                ma, rank_lose, left=rank_win + rank_tie, color="crimson", height=0.9)
            axes[index_j].set_xlim([0, len(dataset_paths)])

        # Name of the metric only on the left side of the figure
        axes[index_j].text(offset, index_j*0.1-0.6, metric.upper(), fontsize=12, weight="bold")
        # Name of the reference method only on the top of the figure
        axes[0].text(0, 2, ref_method[0], fontsize=12, weight="bold")

        # Calculate and plot critical difference
        N_of_streams = len(dataset_paths)
        critical_difference = ceil(
            N_of_streams / 2 + 1.96 * sqrt(N_of_streams) / 2)
        if len(dataset_paths) < 25:
            axes[index_j].axvline(
                critical_difference, 0, 1, linestyle="--", linewidth=3, color="black")
        else:
            axes[index_j].axvline(
                critical_difference, 0, 1, linestyle="--", linewidth=3, color="black")

    if not os.path.exists("results/%s/ranking/" % (experiment_name)):
        os.makedirs("results/%s/ranking/" % (experiment_name))
    plt.gcf().set_size_inches(3, 6)
    filepath = "results/%s/ranking/%s" % (experiment_name, filename)
    plt.savefig(filepath + ".png", bbox_inches='tight')
    plt.savefig(filepath + ".eps", format='eps', bbox_inches='tight')
    plt.clf()


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
