from skopt import BayesSearchCV
from skopt.plots import plot_objective
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import os
import warnings
from pathlib import Path
from joblib import Parallel, delayed

from methods.DE_Forest import DifferentialEvolutionForest
from utils.load_datasets import load_dataset



warnings.filterwarnings("ignore")
# DATASETS_DIR = "datasets/"
DATASETS_DIR = "dtest/"
dataset_paths = []
for root, _, files in os.walk(DATASETS_DIR):
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append(os.path.join(root, filename))

def compute(dataset_id, dataset_path):
    X, y = load_dataset(dataset_paths[0])
    # Normalization - transform data to [0, 1]
    X = MinMaxScaler().fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

    base_estimator = DecisionTreeClassifier(random_state=1234)
    opt = BayesSearchCV(
        DifferentialEvolutionForest(base_estimator),
        {
            # 'n_classifiers': [5],
            # 'p_size': [100],
            # 'metric_name': ['BAC'],
            # 'bootstrap': ['False'],
            'n_classifiers': [5, 10, 25],
            'p_size': [100, 200, 500],
            # 'n_classifiers': (5, 25),
            # 'p_size': (100, 500),
            'metric_name': ['BAC', 'AUC', 'GM'],
            'bootstrap': ['True', 'False'],  # categorical parameter
        },
        n_iter=32,
        scoring="balanced_accuracy",
        cv=3
    )

    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))

    _ = plot_objective(opt.optimizer_results_[0],
                    dimensions=["n_classifiers", "p_size", "metric_name", "bootstrap"],
                    n_minimum_search=int(1e8)
                    )
    
    dataset_name = Path(dataset_path).stem
    if not os.path.exists("results/experiment0/tune_plots/"):
                os.makedirs("results/experiment0/tune_plots/")
    plt.savefig("experiment0/tune_plots/tune_%s.png" % dataset_name, bbox_inches='tight')


# Multithread; n_jobs - number of threads, where -1 all threads, safe for my computer 2
Parallel(n_jobs=-1)(
                delayed(compute)
                (dataset_id, dataset_path)
                for dataset_id, dataset_path in enumerate(dataset_paths)
                )


# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_objective, plot_histogram

# from sklearn.datasets import load_digits
# from sklearn.svm import LinearSVC, SVC
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt

# X, y = load_digits(n_class=10, return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # pipeline class is used as estimator to enable
# # search over different model types
# pipe = Pipeline([
#     ('model', SVC())
# ])

# # single categorical value of 'model' parameter is
# # sets the model class
# # We will get ConvergenceWarnings because the problem is not well-conditioned.
# # But that's fine, this is just an example.
# linsvc_search = {
#     'model': [LinearSVC(max_iter=1000)],
#     'model__C': (1e-6, 1e+6, 'log-uniform'),
# }

# # explicit dimension classes can be specified like this
# svc_search = {
#     'model': Categorical([SVC()]),
#     'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
#     'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
#     'model__degree': Integer(1,8),
#     'model__kernel': Categorical(['linear', 'poly', 'rbf']),
# }

# opt = BayesSearchCV(
#     pipe,
#     # (parameter space, # of evaluations)
#     [(svc_search, 40), (linsvc_search, 16)],
#     cv=3
# )

# opt.fit(X_train, y_train)

# print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(X_test, y_test))
# print("best params: %s" % str(opt.best_params_))

# _ = plot_objective(opt.optimizer_results_[0],
#                    dimensions=["C", "degree", "gamma", "kernel"],
#                    n_minimum_search=int(1e8))
# # plt.show()
# plt.savefig("plotSVC.png", bbox_inches='tight')