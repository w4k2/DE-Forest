import os
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from methods.SOORF import SingleObjectiveOptimizationRandomForest
from methods.Random_FS import RandomFS
from utils import result_tables, pairs_metrics_multi_grid_all, dataset_description, process_plot


base_estimator = DecisionTreeClassifier(random_state=0)
# Parallelization
n_proccess = 5
methods = {
    "SOORF_a0":
        SingleObjectiveOptimizationRandomForest(base_classifier=base_estimator, n_classifiers=10, metric_name="Aggregate", alpha=0, bootstrap=False, n_proccess=n_proccess),
    "SOORF_a02":
        SingleObjectiveOptimizationRandomForest(base_classifier=base_estimator, n_classifiers=10, metric_name="Aggregate", alpha=0.2, bootstrap=False, n_proccess=n_proccess),
    "SOORF_a04":
        SingleObjectiveOptimizationRandomForest(base_classifier=base_estimator, n_classifiers=10, metric_name="Aggregate", alpha=0.4, bootstrap=False, n_proccess=n_proccess),
    "SOORF_a06":
        SingleObjectiveOptimizationRandomForest(base_classifier=base_estimator, n_classifiers=10, metric_name="Aggregate", alpha=0.6, bootstrap=False, n_proccess=n_proccess),
    "SOORF_a08":
        SingleObjectiveOptimizationRandomForest(base_classifier=base_estimator, n_classifiers=10, metric_name="Aggregate", alpha=0.8, bootstrap=False, n_proccess=n_proccess),
    "SOORF_a1":
        SingleObjectiveOptimizationRandomForest(base_classifier=base_estimator, n_classifiers=10, metric_name="Aggregate", alpha=1, bootstrap=False, n_proccess=n_proccess),
}

metrics_alias = [
    "ACC",
    "BAC",
    "Gmean",
    "F1score",
    "Recall",
    "Specificity",
    "Precision"]

DATASETS_DIR = "datasets/"
# DATASETS_DIR = "d/"
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

for dataset_id, dataset_path in enumerate(dataset_paths):
    dataset_name = Path(dataset_path).stem
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                filename = "results/pre_experiment/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
                if not os.path.isfile(filename):
                    # print("File not exist - %s" % filename)
                    continue
                scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                data_np[dataset_id, metric_id, clf_id] = scores
                mean_score = np.mean(scores)
                mean_scores[dataset_id, metric_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, metric_id, clf_id] = std
            except:
                print("Error loading data!", dataset_name, clf_name, metric)

            # Save process plots of metrics of each dataset
            # process_plot(dataset_name, metric, methods_names, n_folds, clf_name)

# print(mean_scores)

# All datasets with description in the table
# dataset_description(dataset_paths)

experiment_name = "pre_experiment"
# Results in form of one .tex table of each metric
result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)


# Wilcoxon ranking grid - statistic test for all methods
# pairs_metrics_multi_grid_all(method_names=list(methods.keys()), data_np=data_np, experiment_name="experiment0", dataset_paths=dataset_paths, metrics=metrics_alias, filename="ex0_ranking_plot_grid_all", ref_methods=list(methods.keys())[0:2], offset=-1.25)
