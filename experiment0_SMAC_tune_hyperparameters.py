from __future__ import annotations
import warnings
import numpy as np
import time
from datetime import datetime
import os
from pathlib import Path

from smac import MultiFidelityFacade, Scenario
from ConfigSpace import Categorical, ConfigurationSpace, Integer

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from methods.DE_Forest import DifferentialEvolutionForest
from utils.load_datasets import load_dataset

# logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# SMAC Version: V2.0.0a1


DATASETS_DIR = "datasets_pre_experiment/"
# DATASETS_DIR = "dtest/"
dataset_paths = []
for root, _, files in os.walk(DATASETS_DIR):
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append(os.path.join(root, filename))
instances = dataset_paths

base_estimator = DecisionTreeClassifier(random_state=1234)

# Target Algorithm
def DE_forest_from_cfg(cfg, seed, instance):
    """Creates a DE_forest classifier based on a configuration and evaluates it using cross-validation.

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!
    seed: int or RandomState
        used to initialize the random generator of cross-validation
    instance: str
        used to represent the instance to use (the dataset to consider in this case)

    Returns:
    --------
    float
        A crossvalidated mean score (bac) for the DE_forest classifier on the loaded data-set.
    additional_info
        A dictionary of additional informations.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        start_time = time.time()

        clf = DifferentialEvolutionForest(
            base_classifier=base_estimator, 
            n_classifiers=cfg["n_classifiers"], 
            metric_name=cfg["metric_name"],
            bootstrap=cfg["bootstrap"],
            p_size=cfg["p_size"],
            random_state_cv=222,
        )

        X, y = load_dataset(str(instance))
        X = MinMaxScaler().fit_transform(X, y)

        cv = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)  # to make CV splits consistent

        scores = cross_val_score(clf, X, y, scoring=make_scorer(balanced_accuracy_score), cv=cv)

        end_time = time.time() - start_time
        additional_info = {
            "score_bac": np.mean(scores),
            "time": end_time,
        }
        print(additional_info)

    return (1 - np.mean(scores), additional_info)

if __name__ == "__main__":
    start_time = time.time()

    # Build Configuration Space which defines all parameters and their ranges
    configspace = ConfigurationSpace()

    # Define a few possible parameters for the classifier
    bootstrap = Categorical("bootstrap", ["True", "False"], default="False")
    metric_name = Categorical("metric_name", ["BAC", "AUC", "GM"], default="BAC")
    n_classifiers = Integer("n_classifiers", (5, 25), default=10)
    p_size = Integer("p_size", (100, 500), default=200)
    # Add the parameters to configuration space
    configspace.add_hyperparameters([bootstrap, metric_name, n_classifiers, p_size])
    
    # current date and time
    now = datetime.now() 
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    scenario = Scenario(
        configspace,
        walltime_limit=3000,  # We want to optimize for 3000 seconds
        n_trials=5000,  # We want to try max 5000 different trials
        min_budget=1,  # Use min one instance
        max_budget=len(instances),  # Use max as many instances as we have datasets
        instances=instances,
        output_directory=Path("smac_outputs/%s" % (date_time)),
        n_workers = 16, # 16 rdzeni na serwerze, uruchamiaj na max. 2 na własnym komputerze
        instance_features={item:[idx, idx+1] for idx, item in enumerate(instances)}
    )

    # Create our SMAC object and pass the scenario and the train method
    smac = MultiFidelityFacade(
        scenario,
        DE_forest_from_cfg,
        overwrite=True,
    )

    # Now we start the optimization process
    incumbent = smac.optimize()

    default_cost = smac.validate(configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    end_time = time.time() - start_time
    print(end_time)

    if not os.path.exists("smac_outputs/%s" % (date_time)):
                os.makedirs("smac_outputs/%s" % (date_time))
    with open("smac_outputs/%s/time.txt" % (date_time), 'w') as f:
        f.write(str(end_time))

# Wynik z optymalizacji jest zapisany w pliku stats.json
# Wynik z badań na serwerze z 15.11.2022 dla 5 datasetów. to:
    # "bootstrap": "True",
    # "metric_name": "BAC",
    # "n_classifiers": 15,
    # "p_size": 107

# Wynik z badań na serwerze z 06.12.2022 dla 5 datasetów., (G-mean liczony na precision i recall) to:
    # "bootstrap": "True",
    # "metric_name": "BAC",
    # "n_classifiers": 15,
    # "p_size": 107