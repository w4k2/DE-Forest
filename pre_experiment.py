import numpy as np
import os
import time
import logging
import traceback
from pathlib import Path

from joblib import Parallel, delayed
import warnings
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

from methods.SOORF import SingleObjectiveOptimizationRandomForest
from utils import load_dataset

"""
Datasets are from KEEL repository.
"""

base_estimator = DecisionTreeClassifier(random_state=1234)
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
logging.basicConfig(filename='textinfo/pre_experiment.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("--------------------------------------------------------------------------------")
logging.info("-------                        NEW EXPERIMENT                            -------")
logging.info("--------------------------------------------------------------------------------")


def compute(dataset_id, dataset_path):
    logging.basicConfig(filename='textinfo/pre_experiment.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
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
        diversities_cor_q_k = np.zeros((len(methods), n_folds, 3))
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

                end_method = time.time() - start_method
                logging.info("DONE METHOD %s - %s (Time: %d [s])" % (clf_name, dataset_path, end_method))
                print("DONE METHOD %s - %s (Time: %d [s])" % (clf_name, dataset_path, end_method))

                # Take diversity results from DiversityTests: correlation, Q-statistic, Cohen's k
                diversities_cor_q_k[clf_id, fold_id] = clf.diversities

                # Save values from optimization
                val = [e.opt.get("F")[0] for e in clf.res_history]
                optimization_values = [-value for value in val]
                filename = "results/pre_experiment/optimization/%s/%s_agg_%d.csv" % (dataset_name, clf_name, fold_id)
                if not os.path.exists("results/pre_experiment/optimization/%s/" % (dataset_name)):
                    os.makedirs("results/pre_experiment/optimization/%s/" % (dataset_name))
                np.savetxt(fname=filename, fmt="%f", X=optimization_values)

        # Save results to csv
        for clf_id, clf_name in enumerate(methods):
            # Save metric results
            for metric_id, metric in enumerate(metrics_alias):
                filename = "results/pre_experiment/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
                if not os.path.exists("results/pre_experiment/raw_results/%s/%s/" % (metric, dataset_name)):
                    os.makedirs("results/pre_experiment/raw_results/%s/%s/" % (metric, dataset_name))
                np.savetxt(fname=filename, fmt="%f", X=scores[metric_id, clf_id, :])
            # Save diversity results
            filename = "results/pre_experiment/diversity_results/%s/%s_diversity.csv" % (dataset_name, clf_name)
            if not os.path.exists("results/pre_experiment/diversity_results/%s/" % (dataset_name)):
                os.makedirs("results/pre_experiment/diversity_results/%s/" % (dataset_name))
            np.savetxt(fname=filename, fmt="%f", X=diversity[clf_id, :, :])
            # Save diversity results from DiversityTests: correlation, Q-statistic, Cohen's k
            filename = "results/pre_experiment/diversities_cor_q_k/%s/%s_cqk.csv" % (dataset_name, clf_name)
            if not os.path.exists("results/pre_experiment/diversities_cor_q_k/%s/" % (dataset_name)):
                os.makedirs("results/pre_experiment/diversities_cor_q_k/%s/" % (dataset_name))
            np.savetxt(fname=filename, fmt="%f", X=diversities_cor_q_k[clf_id, :, :])

        end = time.time() - start
        logging.info("DONE - %s (Time: %d [s])" % (dataset_path, end))
        print("DONE - %s (Time: %d [s])" % (dataset_path, end))

    except Exception as ex:
        logging.exception("Exception in %s" % (dataset_path))
        print("ERROR: %s" % (dataset_path))
        traceback.print_exc()
        print(str(ex))


# Multithread; n_jobs - number of threads, where -1 all threads, safe for my computer 2
Parallel(n_jobs=1)(
                delayed(compute)
                (dataset_id, dataset_path)
                for dataset_id, dataset_path in enumerate(dataset_paths)
                )
