import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from pymoo.core.problem import ElementwiseProblem
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.stats import mode
from EnsembleDiversityTests.EnsembleDiversityTests import DiversityTests


class Optimization(ElementwiseProblem):
    def __init__(self, X, y, test_size, estimator, n_features, metric_name, alpha, objectives=1, n_classifiers=10, **kwargs):
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
        # self.models = {}

        self.test_size = 0
        if self.test_size != 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, stratify=self.y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = np.copy(self.X), np.copy(self.X), np.copy(self.y), np.copy(self.y)

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        n_variable = self.n_classifiers * self.n_features
        xl_binary = [0] * n_variable
        xu_binary = [1] * n_variable

        super().__init__(n_var=n_variable, n_obj=objectives,
                         n_constr=0, xl=xl_binary, xu=xu_binary, **kwargs)

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
        for sf in selected_features:
            # If at least one element in sf is True
            if True in sf:
                # key = np.array2string(sf.astype(int), separator="")[1:-1]
                # # print(key)
                # if key in self.models.keys():
                #     print("MEM")
                #     ensemble.append(self.models[key])
                # else:
                #     candidate = clone(self.estimator).fit(self.X_train[:, sf], self.y_train)
                #     ensemble.append(candidate)
                #     self.models[key] = candidate

                candidate = clone(self.estimator).fit(self.X_train[:, sf], self.y_train)
                ensemble.append(candidate)

        # If at least one element in selected_features is True
        for index in range(self.n_classifiers):
            if True in selected_features[index]:
                pass
            else:
                self.metric = [0, 0]
                return self.metric
        y_pred = self.predict(self.X_test, selected_features, ensemble)
        if self.metric_name == "Accuracy":
            self.metric = [accuracy_score(self.y_test, y_pred)]
        elif self.metric_name == "BAC":
            self.metric = [balanced_accuracy_score(self.y_test, y_pred)]
        elif self.metric_name == "Aggregate":
            accuracy = accuracy_score(self.y_test, y_pred)
            predictions = []
            names = []
            for clf_ind, member_clf in enumerate(ensemble):
                predictions.append(member_clf.predict(self.X[:, selected_features[clf_ind]]).tolist())
                names.append(str(clf_ind))
            test_class = DiversityTests(predictions, names, self.y_test)
            diversities = test_class.get_avg_pairwise(print_flag=False)
            correlation = (diversities[0] + 1) / 2
            self.metric = [self.alpha * accuracy + (1 - self.alpha) * correlation]
            # print("METRIC")
            # print(self.metric)
        return self.metric

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)
        # print(x, scores)

        # Function F is always minimize, but the minus sign (-) before F means maximize
        f1 = -1 * scores[0]
        # f2 = -1 * scores[1]
        # out["F"] = anp.column_stack(np.array([f1, f2]))
        out["F"] = f1

        # Function constraint to select specific numbers of features:
        # number = int((1 - self.scale_features) * self.n_features)
        # out["G"] = (self.n_features - np.sum(x[2:]) - number) ** 2
        # print(out["G"])
        # print((x[2:]))
