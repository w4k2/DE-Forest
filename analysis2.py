import os
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

from methods.DE_Forest import DifferentialEvolutionForest
from methods.Random_FS import RandomFS
from utils.wilcoxon_ranking import pairs_metrics_multi_grid_all, pairs_metrics_multi_line
from utils.plots import result_tables, result_tables_IR, result_tables_features, result_tables_for_time


base_estimator = DecisionTreeClassifier(random_state=1234)

# Parallelization
n_proccess = 16
methods = {
    "DE_Forest_constr":
        DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=15, metric_name="BAC", bootstrap=True, random_state_cv=222, p_size=107, constraints=True),
    "RandomFS":
        RandomFS(base_classifier=base_estimator, n_classifiers=15, bootstrap=False, max_features_selected=True),
    "RandomFS_b":
        RandomFS(base_classifier=base_estimator, n_classifiers=15, bootstrap=True, max_features_selected=True),
    "DT":
        DecisionTreeClassifier(random_state=1234),
}
method_names = methods.keys()

metrics_alias = [
    "BAC",
    "Gmean_pr",
    "F1score",
    "Recall",
    "Specificity",
    "Precision"]

DATASETS_DIR = "datasets/"
dataset_paths = []
for root, _, files in os.walk(DATASETS_DIR):
    print(root, files)
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append(os.path.join(root, filename))

n_splits = 2
n_repeats = 5
n_folds = n_splits * n_repeats
n_methods = len(methods)
n_metrics = len(metrics_alias)
n_datasets = len(dataset_paths)
# Load data from file
data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))
methods_names = list(methods.keys())
time_for_all = np.zeros((n_datasets, len(methods), n_folds))
mean_times_folds = np.zeros((n_datasets, len(methods)))

experiment_name = "experiment2_constr"

for dataset_id, dataset_path in enumerate(dataset_paths):
    dataset_name = Path(dataset_path).stem
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                filename = "results/%s/raw_results/%s/%s/%s.csv" % (experiment_name, metric, dataset_name, clf_name)
                if not os.path.isfile(filename):
                    print("File not exist - %s" % filename)
                    # continue
                scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                data_np[dataset_id, metric_id, clf_id] = scores
                mean_score = np.mean(scores)
                mean_scores[dataset_id, metric_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, metric_id, clf_id] = std
            except:
                print("Error loading data!", dataset_name, clf_name, metric)
        try:
            filename = "results/%s/time_results/%s/%s_time.csv" % (experiment_name, dataset_name, clf_name)
            if not os.path.isfile(filename):
                # print("File not exist - %s" % filename)
                continue
            times = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
            mean_time_score = np.mean(times)
            mean_times_folds[dataset_id, clf_id] = mean_time_score
        except:
            print("Error loading time data!", dataset_name, clf_name)

# Results in form of one .tex table of each metric
# result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Results in form of one .tex table of each metric sorted by IR
# result_tables_IR(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Results in form of one .tex table of each metric sorted by number of features
# result_tables_features(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Wilcoxon ranking grid - statistic test for all methods
# pairs_metrics_multi_grid_all(method_names=method_names, data_np=data_np, experiment_name=experiment_name, dataset_paths=dataset_paths, metrics=metrics_alias, filename="ex2_wilcoxon_all", ref_methods=list(method_names)[0:2], offset=-10)

# Wilcoxon ranking line - statistic test for my method vs the remaining methods
pairs_metrics_multi_line(method_names=list(method_names), data_np=data_np, experiment_name=experiment_name, dataset_paths=dataset_paths, metrics=metrics_alias, filename="ex2_wilcoxon", ref_methods=list(method_names))

# Time results in form of .tex table
# result_tables_for_time(dataset_paths, mean_times_folds, methods, experiment_name)