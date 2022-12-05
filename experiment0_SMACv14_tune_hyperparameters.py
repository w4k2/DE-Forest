import logging
import warnings
import numpy as np
import time
from datetime import datetime
import os

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

# Import ConfigSpace and different types of parameters
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from methods.DE_Forest import DifferentialEvolutionForest
from utils.load_datasets import load_dataset

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# SMAC Version: V1.4
# Old version of SMAC, NOT USE IT


DATASETS_DIR = "dtest/"
dataset_paths = []
for root, _, files in os.walk(DATASETS_DIR):
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append([os.path.join(root, filename)])
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
    cs = ConfigurationSpace()

    # Define a few possible parameters for the classifier
    bootstrap = CategoricalHyperparameter("bootstrap", choices=["True", "False"], default_value="False")
    metric_name = CategoricalHyperparameter("metric_name", choices=["BAC", "AUC", "GM"], default_value="BAC")
    n_classifiers = UniformIntegerHyperparameter("n_classifiers", 5, 25, default_value=10)
    p_size = UniformIntegerHyperparameter("p_size", 100, 500, default_value=200)
    # Add the parameters to configuration space
    cs.add_hyperparameters([bootstrap, metric_name, n_classifiers, p_size])
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "wallclock-limit": 3000,  # 100 max duration to run the optimization (in seconds).
            "cs": cs,  # configuration space
            "deterministic": True,
            "limit_resources": True,  # Uses pynisher to limit memory and runtime
            "memory_limit": 3072,  # adapt this to reasonable value for your hardware
            # "cutoff": 3,  # runtime limit for the target algorithm
            "instances": instances,  # Optimize across all given instances
            "output_dir": "smac_output/%s/" % (date_time),
            # 2 parameters to use Parallelism
            "shared_model": True, 
            "input_psmac_dirs": "smac_output/%s/" % (date_time),
        }
    )

    # intensifier parameters
    # if no argument provided for budgets, hyperband decides them based on the number of instances available
    intensifier_kwargs = {
        "initial_budget": 1,
        "max_budget": 45,
        "eta": 3,
        "instance_order": None,
    }

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=DE_forest_from_cfg,
        intensifier_kwargs=intensifier_kwargs,
    )

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos (mean score, time)
    def_costs = []
    for i in instances:
        cost = smac.get_tae_runner().run(cs.get_default_configuration(), i[0])[1]
        def_costs.append(cost)
    print("Value for default configuration: %.4f" % (np.mean(def_costs)))

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_costs = []
    for i in instances:
        cost = smac.get_tae_runner().run(incumbent, i[0])[1]
        inc_costs.append(cost)
    print("Optimized Value: %.4f" % (np.mean(inc_costs)))

    end_time = time.time() - start_time
    print(end_time)

    # Wyniki z optymalizacji są w pliku traj.json - ostatni wiersz to najlepsze wartości

    # sprawdzić czy gdy jest wiele instancji, to ten najlepszy wynik jest uśrednieniem dla wszystkich datasetów czy jak to się robi
    # ODP: SMAC then returns the algorithm that had the best performance across all the instances.

    # TODO:
    # - zapisać wyniki jakości bac do pliku - zapisuje się w runhistory
    # - zapisać czasy do pliku - zapisuje się w runhistory
    # - sprawdzić uruchomienie dla wielu-wątków ( czy to parallel działa? dostaję odpowiedz w terminalu (INFO:smac.runhistory.runhistory:Entry was not added to the runhistory because existing runs will not overwritten.), może źle podaję tą nazwę folderu dla input), za 3 uruchomieniem już nie ma takiego komunikatu

    # uruchom jeszcze raz, zeby zobaczyc jak dziala z tym czasem 3000