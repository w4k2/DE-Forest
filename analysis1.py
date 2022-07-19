import os
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from methods.DE_Forest import DifferentialEvolutionForest
from methods.Random_FS import RandomFS
# from utils.datasets_table_description import make_description_table
from utils.wilcoxon_ranking import pairs_metrics_multi_grid_all, pairs_metrics_multi_grid
from utils.plots import process_plot, diversity_bar_plot, result_tables


base_estimator = DecisionTreeClassifier(random_state=1234)

# Parallelization
n_proccess = 16
methods = {
    "DE_Forest_gm":
        DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=10, metric_name="gm", alpha=1, bootstrap=False, n_proccess=n_proccess, random_state_cv=222),
    "DE_Forest_AUC":
        DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=10, metric_name="AUC", alpha=1, bootstrap=False, n_proccess=n_proccess, random_state_cv=222),
    "DE_Forest_bac":
        DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=10, metric_name="BAC", alpha=1, bootstrap=False, n_proccess=n_proccess, random_state_cv=222),
    "DE_Forest_gm_b":
        DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=10, metric_name="gm", alpha=1, bootstrap=True, n_proccess=n_proccess, random_state_cv=222),
    "DE_Forest_AUC_b":
        DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=10, metric_name="AUC", alpha=1, bootstrap=True, n_proccess=n_proccess, random_state_cv=222),
    "DE_Forest_bac_b":
        DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=10, metric_name="Accuracy", alpha=1, bootstrap=True, n_proccess=n_proccess),
    # "DE_Forest_a1_bac_p":
    #     DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=10, metric_name="BAC", alpha=1, bootstrap=False, n_proccess=n_proccess, random_state_cv=222, pruning=True),
    # "DE_Forest_a1_bac_bp":
    #     DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=10, metric_name="BAC", alpha=1, bootstrap=True, n_proccess=n_proccess, random_state_cv=222, pruning=True),
    "RandomFS":
        RandomFS(base_classifier=base_estimator, n_classifiers=10, bootstrap=False, max_features_selected=True),
    # "RandomFS_all_feat":
    #     RandomFS(base_classifier=base_estimator, n_classifiers=10, bootstrap=False, max_features_selected=False),
    "RandomFS_b":
        RandomFS(base_classifier=base_estimator, n_classifiers=10, bootstrap=True, max_features_selected=True),
    # "RandomFS_b_all_feat":
        # RandomFS(base_classifier=base_estimator, n_classifiers=10, bootstrap=True, max_features_selected=False),
    "DT":
        DecisionTreeClassifier(random_state=1234),
    "RF":
        RandomForestClassifier(random_state=0, n_estimators=10, bootstrap=False),
    "RF_b":
        RandomForestClassifier(random_state=0, n_estimators=10, bootstrap=True),
}

method_names = methods.keys()
print(method_names)

metrics_alias = [
    # "ACC",
    "BAC",
    "Gmean",
    "F1score",
    "Recall",
    "Specificity",
    "Precision"]

# DATASETS_DIR = "dtest/"
# DATASETS_DIR = "datasets/"
DATASETS_DIR = "ds10/"
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

diversity_measures = ["Entropy", "KW", "Disagreement", "Q statistic"]
diversity = np.zeros((n_datasets, len(method_names), n_folds, len(diversity_measures)))

for dataset_id, dataset_path in enumerate(dataset_paths):
    dataset_name = Path(dataset_path).stem
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                filename = "results/experiment1/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
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

            # Save process plots of metrics of each dataset - nie wiem co to, nie za bardzo działa, pusty wykres
            # process_plot(dataset_name, metric, methods_names, n_folds, clf_name)

            for div_measure_id, div_measure in enumerate(diversity_measures):
                try:
                    filename = "results/experiment1/diversity_results/%s/%s_diversity.csv" % (dataset_name, clf_name)
                    if not os.path.isfile(filename):
                        # print("File not exist - %s" % filename)
                        continue
                    diversity_raw = np.genfromtxt(filename, delimiter=' ', dtype=np.float32)
                    if np.isnan(diversity_raw).all():
                        pass
                    else:
                        diversity_raw = np.nan_to_num(diversity_raw)
                        diversity[dataset_id, clf_id] = diversity_raw
                except:
                    print("Error loading diversity data!", dataset_name, clf_name, div_measure)

diversity_m = np.mean(diversity, axis=2)
diversity_mean = np.mean(diversity_m, axis=0)
# print(mean_scores)

# All datasets with description in the table
# make_description_table()

experiment_name = "experiment1"
# Results in form of one .tex table of each metric
result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Wilcoxon ranking grid - statistic test for all methods
# pairs_metrics_multi_grid_all(method_names=method_names, data_np=data_np, experiment_name=experiment_name, dataset_paths=dataset_paths, metrics=metrics_alias, filename="ex1_wilcoxon_all", ref_methods=list(method_names)[0:6], offset=-10)

# Diversity bar Plotting
# diversity_bar_plot(diversity_mean, diversity_measures, method_names, experiment_name=experiment_name)
