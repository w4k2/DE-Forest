import numpy as np
import math
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from scipy.stats import mode
from pymoo.core.problem import ElementwiseProblem


class BootstrapOptimizationConstr(ElementwiseProblem):
    def __init__(self, X, y, X_b, y_b, test_size, estimator, n_features, metric_name, alpha, objectives=1, n_classifiers=10, **kwargs):
        # runner, func_eval,
        self.estimator = estimator
        self.test_size = test_size
        self.objectives = objectives
        self.n_features = n_features
        self.n_classifiers = n_classifiers
        self.X = X
        self.y = y
        self.classes_, _ = np.unique(self.y, return_inverse=True)
        self.X_b = X_b
        self.y_b = y_b
        self.metric_name = metric_name
        self.alpha = alpha

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        n_variable = self.n_classifiers * self.n_features
        xl_binary = [0] * n_variable
        xu_binary = [1] * n_variable

        # !!! Set n_ieq_constr=0, if constraint is not used

        super().__init__(n_var=n_variable, n_obj=objectives, n_ieq_constr=1, n_eq_constr=0, xl=xl_binary, xu=xu_binary, **kwargs)
        # runner=runner, func_eval=func_eval,

    def predict(self, X, selected_features, ensemble):
        predictions = np.array([member_clf.predict(X[:, sf]) for member_clf, sf in zip(ensemble, selected_features)])
        prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]

    # x: a two dimensional matrix where each row is a point to evaluate and each column a variable
    def validation(self, x, classes=None):
        ensemble = []
        selected_features = []
        for result_opt in x:
            if result_opt > 0.5:
                feature = True
                selected_features.append(feature)
            else:
                feature = False
                selected_features.append(feature)

        selected_features = np.array_split(selected_features, self.n_classifiers)

        for id, sf in enumerate(selected_features):
            # If at least one element in sf is True
            if True in sf:
                X_train = self.X_b[id]
                y_train = self.y_b[id]
                candidate = clone(self.estimator).fit(X_train[:, sf], y_train)
                ensemble.append(candidate)
        # If at least one element in self.selected_features is True
        for index in range(self.n_classifiers):
            if True in selected_features[index]:
                pass
            else:
                self.metric = 0
                return self.metric

        y_pred = self.predict(self.X, selected_features, ensemble)
        if self.metric_name == "BAC":
            self.metric = balanced_accuracy_score(self.y, y_pred)
        elif self.metric_name == "GM":
            self.metric = geometric_mean_score(self.y, y_pred)
        elif self.metric_name == "AUC":
            self.metric = roc_auc_score(self.y, y_pred)
        return self.metric

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)
        # Function F is always minimize, but the minus sign (-) before F means maximize
        f1 = -1 * scores
        out["F"] = f1

        # Function constraint to select specific numbers of features:
        # number = int((1 - self.scale_features) * self.n_features)
        # out["G"] = (self.n_features - np.sum(x[2:]) - number) ** 2
        # print(out["G"])
        # print((x[2:]))

        # print(x)
        selected_features = []
        for result_opt in x:
            if result_opt > 0.5:
                feature = True
                selected_features.append(feature)
            else:
                feature = False
                selected_features.append(feature)
        selected_features = np.array_split(selected_features, self.n_classifiers)
        # print(selected_features, np.sum(selected_features, axis=1), max(np.sum(selected_features, axis=1)), self.n_features, 0.5 * self.n_features)
        # to co jest po prawej stronie ma być <= 0
        out["G"] = max(np.sum(selected_features, axis=1)) - int(0.5 * self.n_features)
        # print(out["G"])
