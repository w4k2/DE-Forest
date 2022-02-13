import numpy as np
import os
import time
from joblib import Parallel, delayed
import logging
import traceback
from pathlib import Path
import warnings
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from methods.SOORF import SingleObjectiveOptimizationRandomForest
from methods.Random_FS import RandomFS
from utils import load_dataset, optimization_plot

"""
Datasets are from KEEL repository.
"""


base_estimator = DecisionTreeClassifier(random_state=1234)
n_proccess = 4
methods = {
    "RandomFS":
        RandomFS(base_classifier=base_estimator, n_classifiers=10, bootstrap=False),
    "DT":
        DecisionTreeClassifier(random_state=1234),
    "RF":
        RandomForestClassifier(random_state=0, n_estimators=10, bootstrap=False),
    # "RandomFS_b":
    #     RandomFS(base_classifier=base_estimator, n_classifiers=10, bootstrap=True),
    # "RF_b":
    #     RandomForestClassifier(random_state=0, n_estimators=10, bootstrap=True),
}

# Repeated Stratified K-Fold cross validator
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
n_folds = n_splits * n_repeats

DATASETS_DIR = "datasets/"
dataset_paths = []
for root, _, files in os.walk(DATASETS_DIR):
    print(root, files)
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append(os.path.join(root, filename))

metrics = [
    accuracy_score,
    balanced_accuracy_score,
    geometric_mean_score,
    f1_score,
    recall_score,
    specificity_score,
    precision_score
    ]
metrics_alias = [
    "ACC",
    "BAC",
    "Gmean",
    "F1score",
    "Recall",
    "Specificity",
    "Precision"]

if not os.path.exists("textinfo"):
    os.makedirs("textinfo")
logging.basicConfig(filename='textinfo/experiment0.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("--------------------------------------------------------------------------------")
logging.info("-------                        NEW EXPERIMENT                            -------")
logging.info("--------------------------------------------------------------------------------")


def compute(dataset_id, dataset_path):
    logging.basicConfig(filename='textinfo/experiment0.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
    try:
        warnings.filterwarnings("ignore")
        print("START: %s" % (dataset_path))
        logging.info("START - %s" % (dataset_path))
        start = time.time()

        X, y = load_dataset(dataset_path)
        # Normalization - transform data to [0, 1]
        X = MinMaxScaler().fit_transform(X, y)
        scores = np.zeros((len(metrics), len(methods), n_folds))
        diversity = np.zeros((len(methods), n_folds, 4))
        dataset_name = Path(dataset_path).stem

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            for clf_id, clf_name in enumerate(methods):
                start_method = time.time()
                clf = clone(methods[clf_name])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Scores for each metric
                for metric_id, metric in enumerate(metrics):
                    if metric_id >= 2:
                        # average="weighted" : this alters ‘macro’ to account for label imbalance
                        scores[metric_id, clf_id, fold_id] = metric(y_test, y_pred, average="weighted")
                    else:
                        scores[metric_id, clf_id, fold_id] = metric(y_test, y_pred)

                # Diversity
                calculate_diversity = getattr(clf, "calculate_diversity", None)
                if callable(calculate_diversity):
                    diversity[clf_id, fold_id] = clf.calculate_diversity()
                else:
                    diversity[clf_id, fold_id] = None
                print(diversity[clf_id, fold_id])

                end_method = time.time() - start_method
                logging.info("DONE METHOD %s - %s (Time: %d [s])" % (clf_name, dataset_path, end_method))
                print("DONE METHOD %s - %s (Time: %d [s])" % (clf_name, dataset_path, end_method))

        # Save results to csv
        for clf_id, clf_name in enumerate(methods):
            # Save metric results
            for metric_id, metric in enumerate(metrics_alias):
                filename = "results/experiment0/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
                if not os.path.exists("results/experiment0/raw_results/%s/%s/" % (metric, dataset_name)):
                    os.makedirs("results/experiment0/raw_results/%s/%s/" % (metric, dataset_name))
                np.savetxt(fname=filename, fmt="%f", X=scores[metric_id, clf_id, :])
            # Save diversity results
            filename = "results/experiment0/diversity_results/%s/%s_diversity.csv" % (dataset_name, clf_name)
            if not os.path.exists("results/experiment0/diversity_results/%s/" % (dataset_name)):
                os.makedirs("results/experiment0/diversity_results/%s/" % (dataset_name))
            np.savetxt(fname=filename, fmt="%f", X=diversity[clf_id, :, :])

        end = time.time() - start
        logging.info("DONE - %s (Time: %d [s])" % (dataset_path, end))
        print("DONE - %s (Time: %d [s])" % (dataset_path, end))

    except Exception as ex:
        logging.exception("Exception in %s" % (dataset_path))
        print("ERROR: %s" % (dataset_path))
        traceback.print_exc()
        print(str(ex))


for dataset_id, dataset_path in enumerate(dataset_paths):
    compute(dataset_id, dataset_path)
