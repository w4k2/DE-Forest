import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import mode
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from .optimization import Optimization
from .bootstrap_optimization import BootstrapOptimization
from utils.utils_diversity import calc_diversity_measures

# from pymoo.core.problem import starmap_parallelized_eval
from multiprocessing.pool import ThreadPool
import multiprocessing


class DifferentialEvolutionForest(BaseEstimator):
    def __init__(self, base_classifier, metric_name="GM", alpha=0.5, n_classifiers=10, test_size=0.5, objectives=1, p_size=100, predict_decision="MV", bootstrap=False, n_proccess=8, random_state_cv=222, pruning=False):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.classes = None
        self.test_size = test_size
        self.objectives = objectives
        self.p_size = p_size
        self.selected_features = []
        self.predict_decision = predict_decision
        self.metric_name = metric_name
        self.alpha = alpha
        self.bootstrap = bootstrap
        self.n_proccess = n_proccess
        self.random_state_cv = random_state_cv
        self.pruning = pruning

    def partial_fit(self, X, y, classes=None):
        self.X, self.y = X, y
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(self.y, return_inverse=True)
        n_features = X.shape[1]

        cross_validation = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=self.random_state_cv)
        # the number of processes to be used
        # pool = multiprocessing.Pool(self.n_proccess)
        # Bootstrap
        X_b = []
        y_b = []
        if self.bootstrap is True:
            for random_state in range(self.n_classifiers):
                # Prepare bootstrap sample
                Xy_bootstrap = resample(X, y, replace=True, random_state=random_state)
                X_b.append(Xy_bootstrap[0])
                y_b.append(Xy_bootstrap[1])
            # Create optimization problem
            # problem = BootstrapOptimization(X, y, X_b, y_b, test_size=self.test_size, estimator=self.base_classifier, n_features=n_features, n_classifiers=self.n_classifiers, metric_name=self.metric_name, alpha=self.alpha, runner=pool.starmap, func_eval=starmap_parallelized_eval)
            problem = BootstrapOptimization(X, y, X_b, y_b, test_size=self.test_size, estimator=self.base_classifier, n_features=n_features, n_classifiers=self.n_classifiers, metric_name=self.metric_name, alpha=self.alpha)
            algorithm = DE(
                pop_size=self.p_size,
                sampling=LHS(),
                variant="DE/rand/1/bin",
                CR=0.9,
                dither="vector",
                jitter=False
                )
        else:
            # Create optimization problem
            # problem = Optimization(X, y, test_size=self.test_size, estimator=self.base_classifier, n_features=n_features, n_classifiers=self.n_classifiers, metric_name=self.metric_name, alpha=self.alpha, cross_validation=cross_validation, runner=pool.starmap, func_eval=starmap_parallelized_eval)
            problem = Optimization(X, y, test_size=self.test_size, estimator=self.base_classifier, n_features=n_features, n_classifiers=self.n_classifiers, metric_name=self.metric_name, alpha=self.alpha, cross_validation=cross_validation)
            algorithm = DE(
                pop_size=self.p_size,
                sampling=LHS(),
                variant="DE/rand/1/bin",
                CR=0.9,
                dither="vector",
                jitter=False
                )
        # It has also been found that setting CR to a low value, e.g., CR=0.2 helps to optimize separable functions since it fosters the search along the coordinate axes. On the contrary, this choice is not effective if parameter dependence is encountered, which frequently occurs in real-world optimization problems rather than artificial test functions. So for parameter dependence, the choice of CR=0.9 is more appropriate.
        # One strategy to introduce adaptive weights (F) during one run. The option allows the same dither to be used in one iteration (‘scalar’) or a different one for each individual (‘vector).
        # Another strategy for adaptive weights (F). Here, only a very small value is added or subtracted to the F used for the crossover for each individual.
        # termination = get_termination("n_gen", 10)
        res = minimize(problem,
                       algorithm,
                       # termination,
                       seed=1,
                       save_history=True,
                       verbose=False)
        # verbose=True)
        self.res_history = res.history

        # F returns all Pareto front solutions (quality) in form [-accuracy]
        self.quality = res.F
        # X returns values of selected features
        # print(res.X)
        # print("Wynik", res.F)
        for result_opt in res.X:
            if result_opt > 0.5:
                feature = True
                self.selected_features.append(feature)
            else:
                feature = False
                self.selected_features.append(feature)

        self.selected_features = np.array_split(self.selected_features, self.n_classifiers)
        # self.selected_features is the vector of selected of features for each model in the ensemble, so bootstrap in this loop ensure different bootstrap data for each model
        for id, sf in enumerate(self.selected_features):
            if self.bootstrap is True:
                X_train = X_b[id]
                y_train = y_b[id]
                candidate = clone(self.base_classifier).fit(X_train[:, sf], y_train)
                # Add candidate to the ensemble
                self.ensemble.append(candidate)
            else:
                candidate = clone(self.base_classifier).fit(X[:, sf], y)
                # Add candidate to the ensemble
                self.ensemble.append(candidate)

        # Pruning based on balanced_accuracy_score
        if self.pruning:
            bac_array = []
            for sf, clf in zip(self.selected_features, self.ensemble):
                y_pred = clf.predict(X[:, sf])
                bac = balanced_accuracy_score(y, y_pred)
                bac_array.append(bac)
            bac_arg_sorted = np.argsort(bac_array)
            self.ensemble_arr = np.array(self.ensemble)
            # The percent of deleted models, ex. 0.3 from 10 models = 30 % models will be deleted
            pruned = 0.3
            pruned_indx = int(pruned * len(self.ensemble))
            selected_models = bac_arg_sorted[pruned_indx:]
            self.ensemble_arr = self.ensemble_arr[selected_models]
            self.ensemble = self.ensemble_arr.tolist()

            selected_features_list = [sf.tolist() for sf in self.selected_features]
            selected_features_arr = np.array(selected_features_list)
            self.selected_features = selected_features_arr[selected_models, :]

    def fit(self, X, y, classes=None):
        self.ensemble = []
        self.partial_fit(X, y, classes)

    def ensemble_support_matrix(self, X):
        # Ensemble support matrix
        return np.array([member_clf.predict_proba(X[:, sf]) for member_clf, sf in zip(self.ensemble, self.selected_features)])

    def predict(self, X):
        # Prediction based on the Average Support Vectors
        if self.predict_decision == "ASV":
            ens_sup_matrix = self.ensemble_support_matrix(X)
            average_support = np.mean(ens_sup_matrix, axis=0)
            prediction = np.argmax(average_support, axis=1)
        # Prediction based on the Majority Voting
        elif self.predict_decision == "MV":
            predictions = np.array([member_clf.predict(X[:, sf]) for member_clf, sf in zip(self.ensemble, self.selected_features)])
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)

    def calculate_diversity(self):
        '''
        entropy_measure_e: E varies between 0 and 1, where 0 indicates no difference and 1 indicates the highest possible diversity.
        kw - Kohavi-Wolpert variance
        Q-statistic: <-1, 1>
        Q = 0 statistically independent classifiers
        Q < 0 classifiers commit errors on different objects
        Q > 0 classifiers recognize the same objects correctly
        '''
        if len(self.ensemble) > 1:
            # All measures for whole ensemble
            self.entropy_measure_e, self.k0, self.kw, self.disagreement_measure, self.q_statistic_mean = calc_diversity_measures(self.X, self.y, self.ensemble, self.selected_features, p=0.01)

            return(self.entropy_measure_e, self.kw, self.disagreement_measure, self.q_statistic_mean)
