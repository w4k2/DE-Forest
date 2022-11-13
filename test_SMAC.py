# import logging

# logging.basicConfig(level=logging.INFO)

# import itertools
# import warnings

# import numpy as np
# from ConfigSpace.hyperparameters import (
#     CategoricalHyperparameter,
#     UniformFloatHyperparameter,
#     UniformIntegerHyperparameter,
# )
# from sklearn import datasets
# from sklearn.exceptions import ConvergenceWarning
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.metrics import balanced_accuracy_score, make_scorer

# # Import ConfigSpace and different types of parameters
# from smac.configspace import ConfigurationSpace
# from smac.facade.smac_mf_facade import SMAC4MF

# # Import SMAC-utilities
# from smac.scenario.scenario import Scenario

# from methods.DE_Forest import DifferentialEvolutionForest
# from utils.load_datasets import load_dataset

# from pathlib import Path
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import MinMaxScaler
from __future__ import annotations
import os
import time
from datetime import datetime


import itertools
import warnings

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from smac import MultiFidelityFacade, Scenario

# now = datetime.now() # current date and time
# date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
# print("date and time:",date_time)
# # We load the MNIST-dataset (a widely used benchmark) and split it into a list of binary datasets
# digits = datasets.load_digits()
# instances = [[str(a) + str(b)] for a, b in itertools.combinations(digits.target_names, 2)]
# # print(instances)

# # DATASETS_DIR = "dtest/"
# # dataset_paths = []
# # for root, _, files in os.walk(DATASETS_DIR):
# #     for filename in filter(lambda _: _.endswith('.dat'), files):
# #         dataset_paths.append(os.path.join(root, filename))
# # instances = [dataset_paths] # sprawdź czy to jest list[list[str]]
# # print(instances, type(instances))

# # base_estimator = DecisionTreeClassifier(random_state=1234)


# def generate_instances(a: int, b: int):
#     """
#     Function to select data for binary classification from the digits dataset
#     a & b are the two classes
#     """
#     # get indices of both classes
#     indices = np.where(np.logical_or(a == digits.target, b == digits.target))
#     # print(indices)

#     # get data
#     data = digits.data[indices]
#     target = digits.target[indices]

#     return data, target


# # Target Algorithm
# def sgd_from_cfg(cfg, seed, instance):
#     """Creates a SGD classifier based on a configuration and evaluates it on the
#     digits dataset using cross-validation.

#     Parameters:
#     -----------
#     cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
#         Configuration containing the parameters.
#         Configurations are indexable!
#     seed: int or RandomState
#         used to initialize the svm's random generator
#     instance: str
#         used to represent the instance to use (the 2 classes to consider in this case)

#     Returns:
#     --------
#     float
#         A crossvalidated mean score for the SGD classifier on the loaded data-set.
#     """

#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=ConvergenceWarning)
#         start_time = time.time()

#         # SGD classifier using given configuration
#         clf = SGDClassifier(
#             loss="log_loss",
#             penalty="elasticnet",
#             alpha=cfg["alpha"],
#             l1_ratio=cfg["l1_ratio"],
#             learning_rate=cfg["learning_rate"],
#             eta0=cfg["eta0"],
#             max_iter=30,
#             early_stopping=True,
#             random_state=seed,
#         )

#         # clf = DifferentialEvolutionForest(
#         #     base_classifier=base_estimator, 
#         #     n_classifiers=cfg["n_classifiers"], 
#         #     metric_name=cfg["metric_name"],
#         #     bootstrap=cfg["bootstrap"],
#         #     p_size=cfg["p_size"],
#         #     random_state_cv=222,
#         # )

#         # get instance
#         data, target = generate_instances(int(instance[0]), int(instance[1]))
#         # print(int(instance[0]), int(instance[1]))
#         # print(data, target)

#         # print(instances[0])
#         # X, y = load_dataset(instances[0])
#         # X = MinMaxScaler().fit_transform(X, y)

#         cv = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)  # to make CV splits consistent
#         scores = cross_val_score(clf, data, target, scoring=make_scorer(balanced_accuracy_score), cv=cv)

#         # scores = cross_val_score(clf, X, y, scoring=make_scorer(balanced_accuracy_score), cv=cv)

#         end_time = time.time() - start_time
#         additional_info = {
#             "score_bac": np.mean(scores),
#             "time": end_time,
#         }
#     return (1 - np.mean(scores), additional_info)


# if __name__ == "__main__":
#     start_time = time.time()
#     # Build Configuration Space which defines all parameters and their ranges
#     cs = ConfigurationSpace()

#     # We define a few possible parameters for the SGD classifier
#     alpha = UniformFloatHyperparameter("alpha", 0, 1, default_value=1.0)
#     l1_ratio = UniformFloatHyperparameter("l1_ratio", 0, 1, default_value=0.5)
#     learning_rate = CategoricalHyperparameter(
#         "learning_rate", choices=["constant", "invscaling", "adaptive"], default_value="constant"
#     )
#     eta0 = UniformFloatHyperparameter("eta0", 0.00001, 1, default_value=0.1, log=True)
#     # Add the parameters to configuration space
#     cs.add_hyperparameters([alpha, l1_ratio, learning_rate, eta0])


#     # # We define a few possible parameters for the SGD classifier
#     # bootstrap = CategoricalHyperparameter("bootstrap", choices=["True", "False"], default_value="False")
#     # metric_name = CategoricalHyperparameter("metric_name", choices=["BAC", "AUC", "GM"], default_value="BAC")
#     # n_classifiers = UniformIntegerHyperparameter("n_classifiers", 5, 25, default_value=10)
#     # p_size = UniformIntegerHyperparameter("p_size", 100, 500, default_value=200)
#     # # Add the parameters to configuration space
#     # cs.add_hyperparameters([bootstrap, metric_name, n_classifiers, p_size])

#     # SMAC scenario object
#     scenario = Scenario(
#         {
#             "run_obj": "quality",  # we optimize quality (alternative to runtime)
#             "wallclock-limit": 100,  # max duration to run the optimization (in seconds)
#             "cs": cs,  # configuration space
#             "deterministic": True,
#             "limit_resources": True,  # Uses pynisher to limit memory and runtime
#             "memory_limit": 3072,  # adapt this to reasonable value for your hardware
#             "cutoff": 3,  # runtime limit for the target algorithm
#             "instances": instances,  # Optimize across all given instances
#             "output_dir": "smac_output_ex/%s/" % (date_time), 
#             # "output_dir": "smac_output2/aaa/",  # You can save to output directory, but we lost the original name of the folder (date and time)
#             # "instance_file": ?
#             "shared_model": True, 
#             "input_psmac_dirs": "smac_output_ex/%s*/" % (date_time),
#         }
#     )

#     # intensifier parameters
#     # if no argument provided for budgets, hyperband decides them based on the number of instances available
#     intensifier_kwargs = {
#         "initial_budget": 1,
#         "max_budget": 45,
#         "eta": 3,
#         # You can also shuffle the order of using instances by this parameter.
#         # 'shuffle' will shuffle instances before each SH run and 'shuffle_once'
#         # will shuffle instances once before the 1st SH iteration begins
#         "instance_order": None,
#     }

#     # To optimize, we pass the function to the SMAC-object
#     smac = SMAC4MF(
#         scenario=scenario,
#         rng=np.random.RandomState(42),
#         tae_runner=sgd_from_cfg,
#         intensifier_kwargs=intensifier_kwargs,
#     )

#     # Example call of the function
#     # It returns: Status, Cost, Runtime, Additional Infos
#     def_costs = []
#     for i in instances:
#         cost = smac.get_tae_runner().run(cs.get_default_configuration(), i[0])[1]
#         def_costs.append(cost)
#     print("Value for default configuration: %.4f" % (np.mean(def_costs)))

#     # Start optimization
#     try:
#         incumbent = smac.optimize()
#     finally:
#         incumbent = smac.solver.incumbent

#     inc_costs = []
#     for i in instances:
#         cost = smac.get_tae_runner().run(incumbent, i[0])[1]
#         inc_costs.append(cost)
#     print("Optimized Value: %.4f" % (np.mean(inc_costs)))

#     end_time = time.time() - start_time
#     print(end_time)

#     if not os.path.exists("smac_outputs/%s" % (date_time)):
#                 os.makedirs("smac_outputs/%s" % (date_time))
#     with open("smac_outputs/%s/time.txt" % (date_time), 'w') as f:
#         f.write(end_time)


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class DigitsDataset:
    def __init__(self) -> None:
        self._data = datasets.load_digits()

    def get_instances(self) -> list[str]:
        """Create instances from the dataset which include two classes only."""
        return [f"{classA}-{classB}" for classA, classB in itertools.combinations(self._data.target_names, 2)]

    def get_instance_features(self) -> dict[str, list[int | float]]:
        """Returns the mean and variance of all instances as features."""
        features = {}
        for instance in self.get_instances():
            data, _ = self.get_instance_data(instance)
            features[instance] = [np.mean(data), np.var(data)]

        return features

    def get_instance_data(self, instance: str) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve data from the passed instance."""
        # We split the dataset into two classes
        classA, classB = instance.split("-")
        indices = np.where(np.logical_or(int(classA) == self._data.target, int(classB) == self._data.target))

        data = self._data.data[indices]
        target = self._data.target[indices]

        return data, target


class SGD:
    def __init__(self, dataset: DigitsDataset) -> None:
        self.dataset = dataset

    @property
    def configspace(self) -> ConfigurationSpace:
        """Build the configuration space which defines all parameters and their ranges for the SGD classifier."""
        cs = ConfigurationSpace()

        # We define a few possible parameters for the SGD classifier
        alpha = Float("alpha", (0, 1), default=1.0)
        l1_ratio = Float("l1_ratio", (0, 1), default=0.5)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        eta0 = Float("eta0", (0.00001, 1), default=0.1, log=True)
        # Add the parameters to configuration space
        cs.add_hyperparameters([alpha, l1_ratio, learning_rate, eta0])

        return cs

    def train(self, config: Configuration, instance: str, seed: int = 0) -> float:
        """Creates a SGD classifier based on a configuration and evaluates it on the
        digits dataset using cross-validation."""

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # SGD classifier using given configuration
            clf = SGDClassifier(
                loss="log",
                penalty="elasticnet",
                alpha=config["alpha"],
                l1_ratio=config["l1_ratio"],
                learning_rate=config["learning_rate"],
                eta0=config["eta0"],
                max_iter=30,
                early_stopping=True,
                random_state=seed,
            )

            # get instance
            data, target = self.dataset.get_instance_data(instance)

            cv = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)  # to make CV splits consistent
            scores = cross_val_score(clf, data, target, cv=cv)

        return 1 - np.mean(scores)


if __name__ == "__main__":

    start_time = time.time()

    # current date and time
    now = datetime.now() 
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    dataset = DigitsDataset()
    model = SGD(dataset)

    scenario = Scenario(
        model.configspace,
        walltime_limit=30,  # We want to optimize for 30 seconds
        n_trials=5000,  # We want to try max 5000 different trials
        min_budget=1,  # Use min one instance
        max_budget=45,  # Use max 45 instances (if we have a lot of instances we could constraint it here)
        instances=dataset.get_instances(),
        instance_features=dataset.get_instance_features(),
    )

    # Create our SMAC object and pass the scenario and the train method
    smac = MultiFidelityFacade(
        scenario,
        model.train,
        overwrite=True,
    )

    # Now we start the optimization process
    incumbent = smac.optimize()

    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    end_time = time.time() - start_time
    print(end_time)

    if not os.path.exists("smac_outputs/%s" % (date_time)):
                os.makedirs("smac_outputs/%s" % (date_time))
    with open("smac_outputs/%s/time.txt" % (date_time), 'w') as f:
        f.write(str(end_time))