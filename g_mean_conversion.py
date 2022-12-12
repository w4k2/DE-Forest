import os
import numpy as np
import math
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

from methods.DE_Forest import DifferentialEvolutionForest
from methods.Random_FS import RandomFS


# This program convert G-mean score calculate from imblearn library as the squared root of the product of the sensitivity and specificity into the squared root of the product of the recall and precision

# Change experiment_name and corresponding methods
# experiment_name = "experiment1" # DE_Forest
experiment_name = "experiment2_constr" # DE_Forest_constr

base_estimator = DecisionTreeClassifier(random_state=1234)
methods = {
    # "DE_Forest":
    #     DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=15, metric_name="BAC", bootstrap=True, random_state_cv=222, p_size=107),
    "DE_Forest_constr":
        DifferentialEvolutionForest(base_classifier=base_estimator, n_classifiers=15, metric_name="BAC", bootstrap=True, random_state_cv=222, p_size=107, constraints=True),
    "RandomFS":
        RandomFS(base_classifier=base_estimator, n_classifiers=15, bootstrap=False, max_features_selected=True),
    "RandomFS_b":
        RandomFS(base_classifier=base_estimator, n_classifiers=15, bootstrap=True, max_features_selected=True),
    "DT":
        DecisionTreeClassifier(random_state=1234),
}

metrics_base = [
    "Recall",
    "Precision"
    ]

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
n_metrics = len(metrics_base)
n_datasets = len(dataset_paths)
data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
methods_names = list(methods.keys())
Gmean_pr_np = np.zeros((n_folds))

# Read data of Precision and Recall from file
for dataset_id, dataset_path in enumerate(dataset_paths):
    dataset_name = Path(dataset_path).stem
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_base):
            try:
                filename = "results/%s/raw_results/%s/%s/%s.csv" % (experiment_name, metric, dataset_name, clf_name)
                if not os.path.isfile(filename):
                    print("File not exist - %s" % filename)
                    # continue
                scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                data_np[dataset_id, metric_id, clf_id] = scores
            except:
                print("Error loading data!", dataset_name, clf_name, metric)

for dataset_id, dataset_path in enumerate(dataset_paths):
    dataset_name = Path(dataset_path).stem
    for clf_id, clf_name in enumerate(methods):
        Gmean_pr_list = []
        for fold_id in range(n_folds):
            # Convert G-mean metric to sqrt(Recall*Precision)
            Gmean_pr = math.sqrt(data_np[dataset_id, 0, clf_id, fold_id] * data_np[dataset_id, 1, clf_id, fold_id])
            Gmean_pr_np[fold_id] = Gmean_pr
        filepath = "results/%s/raw_results/Gmean_pr/%s" % (experiment_name, dataset_name)
        filename = "%s/%s.csv" % (filepath, clf_name)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        np.savetxt(fname=filename, fmt="%f", X=Gmean_pr_np)
