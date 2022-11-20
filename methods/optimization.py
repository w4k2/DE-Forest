import numpy as np
import math
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from scipy.stats import mode
from pymoo.core.problem import ElementwiseProblem


class Optimization(ElementwiseProblem):
    def __init__(self, X, y, test_size, estimator, n_features, metric_name, alpha, cross_validation, objectives=1, n_classifiers=10, **kwargs):
        # runner, func_eval,
        self.estimator = estimator
        self.test_size = test_size
        self.objectives = objectives
        self.n_features = n_features
        self.n_classifiers = n_classifiers
        self.X = X
        self.y = y
        self.classes_, _ = np.unique(self.y, return_inverse=True)
        self.metric_name = metric_name
        self.alpha = alpha
        self.cross_validation = cross_validation

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        n_variable = self.n_classifiers * self.n_features
        xl_binary = [0] * n_variable
        xu_binary = [1] * n_variable

        super().__init__(n_var=n_variable, n_obj=objectives,
                         n_constr=0, xl=xl_binary, xu=xu_binary, **kwargs)
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
        scores = np.zeros((self.cross_validation.get_n_splits()))
        selected_features = np.array_split(selected_features, self.n_classifiers)
        for fold_id, (train, test) in enumerate(self.cross_validation.split(self.X, self.y)):
            for sf in selected_features:
                # If at least one element in sf is True
                if True in sf:
                    X_train = self.X[train]
                    y_train = self.y[train]
                    X_test = self.X[test]
                    y_test = self.y[test]
                    candidate = clone(self.estimator)
                    candidate.fit(X_train[:, sf], y_train)
                    ensemble.append(candidate)

            for index in range(self.n_classifiers):
                # If at least one element in selected_features is True
                if True in selected_features[index]:
                    pass
                else:
                    scores = [0, 0]
                    return np.mean(scores, axis=0)
            y_pred = self.predict(X_test, selected_features, ensemble)
            if self.metric_name == "BAC":
                scores[fold_id] = balanced_accuracy_score(y_test, y_pred)
            elif self.metric_name == "GM":
                scores[fold_id] = geometric_mean_score(y_test, y_pred)
            elif self.metric_name == "AUC":
                scores[fold_id] = roc_auc_score(y_test, y_pred, multi_class='ovo')
        return np.mean(scores, axis=0)

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)

        # Function F is always minimize, but the minus sign (-) before F means maximize
        f1 = -1 * scores
        out["F"] = f1

        # Function constraint to select specific numbers of features:
        # number = int((1 - self.scale_features) * self.n_features)
        # out["G"] = (self.n_features - np.sum(x[2:]) - number) ** 2

        # print(x)
        # out["G"] = int(math.sqrt(self.n_features)) - np.sum(x[0:self.n_classifiers])
        # print(out["G"])
