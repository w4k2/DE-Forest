import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils import resample
from scipy.stats import mode
from random import choice
import math
from utils.utils_diversity import calc_diversity_measures

# Setup to print all dataset array
# np.set_printoptions(threshold=sys.maxsize)


"""
Metoda wybiera cechy losowo - nowo zaimplementowany Random Forest, oryginalnie z artykułu (w sklearn jest inna implementacja)
"""


class RandomFS(BaseEstimator):
    def __init__(self, base_classifier, n_classifiers=10, test_size=0.5, objectives=1, predict_decision="MV", bootstrap=False, max_features_selected=True):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.classes = None
        self.test_size = test_size
        self.objectives = objectives
        self.selected_features = []
        self.predict_decision = predict_decision
        self.bootstrap = bootstrap
        self.max_features_selected = max_features_selected

    def partial_fit(self, X, y, classes=None):
        self.X, self.y = X, y
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(self.y, return_inverse=True)
        n_features = X.shape[1]
        # Random feature selection
        # It selects features randomly, but max. number of features is <= sqrt(n_features)
        features = []
        if self.max_features_selected:
            for clf in range(self.n_classifiers):
                max_features = int(math.sqrt(n_features))
                count_1 = max_features
                count_0 = n_features - max_features
                total_count = n_features
                for f in range(n_features):
                    random_number = np.random.randint(0, total_count)
                    if random_number < count_1:
                        features.append(True)
                        count_1 = count_1 - 1
                    else:
                        features.append(False)
                        count_0 = count_0 - 1
                    total_count = total_count - 1
        else:
            for clf in range(self.n_classifiers):
                for f in range(n_features):
                    features.append(choice([True, False]))
        self.selected_features = features
        self.selected_features = np.array_split(self.selected_features, self.n_classifiers)

        # self.selected_features is the vector of selected of features for each model in the ensemble, so bootstrap in this loop ensure different bootstrap data for each model
        random_state = 1
        for sf in self.selected_features:
            if self.bootstrap is True:
                # Prepare bootstrap sample
                Xy_bootstrap = resample(X, y, replace=True, random_state=random_state)
                random_state += 1
                X_b = Xy_bootstrap[0]
                y_b = Xy_bootstrap[1]
                candidate = clone(self.base_classifier).fit(X_b[:, sf], y_b)
                # Add candidate to the ensemble
                self.ensemble.append(candidate)
            else:
                candidate = clone(self.base_classifier).fit(X[:, sf], y)
                # Add candidate to the ensemble
                self.ensemble.append(candidate)

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
