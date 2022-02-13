import numpy as np
# import autograd.numpy as anp
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from pymoo.core.problem import ElementwiseProblem
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.stats import mode
from utils_diversity import calc_diversity_measures
from EnsembleDiversityTests.EnsembleDiversityTests import DiversityTests


class BootstrapOptimization(ElementwiseProblem):
    def __init__(self, X, y, X_b, y_b, test_size, estimator, n_features, metric_name, alpha, objectives=1, n_classifiers=10, **kwargs):
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

        # self.test_size = 0
        # if self.test_size != 0:
        #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, stratify=self.y)
        # else:
        #     self.X_train, self.X_test, self.y_train, self.y_test = np.copy(self.X), np.copy(self.X), np.copy(self.y), np.copy(self.y)

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        n_variable = self.n_classifiers * self.n_features
        xl_binary = [0] * n_variable
        xu_binary = [1] * n_variable

        super().__init__(n_var=n_variable, n_obj=objectives,
                         n_constr=0, xl=xl_binary, xu=xu_binary, **kwargs)
    #
    # def calculate_diversity(self):
    #     '''
    #     entropy_measure_e: E varies between 0 and 1, where 0 indicates no difference and 1 indicates the highest possible diversity.
    #     kw - Kohavi-Wolpert variance
    #     Q-statistic: <-1, 1>
    #     Q = 0 statistically independent classifiers
    #     Q < 0 classifiers commit errors on different objects
    #     Q > 0 classifiers recognize the same objects correctly
    #     '''
    #     if len(self.ensemble) > 1:
    #         # All measures for whole ensemble
    #         self.entropy_measure_e, self.k0, self.kw, self.disagreement_measure, self.q_statistic_mean = calc_diversity_measures(self.X, self.y, self.ensemble, self.selected_features, p=0.01)
    #
    #         return(self.entropy_measure_e, self.kw, self.disagreement_measure, self.q_statistic_mean)

    def predict(self, X, selected_features, ensemble):
        # Prediction based on the Majority Voting
        # print("SHAPE TEST X:")
        # for sf in self.selected_features:
        #     print(np.shape(X[:, sf]))
        predictions = np.array([member_clf.predict(X[:, sf]) for member_clf, sf in zip(ensemble, selected_features)])
        prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]

    # x: a two dimensional matrix where each row is a point to evaluate and each column a variable
    def validation(self, x, classes=None):
        ensemble = []
        selected_features = []
        # self.classes_ = classes
        # if self.classes_ is None:
        #     self.classes_, _ = np.unique(self.y, return_inverse=True)

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
                self.metric = [0, 0]
                return self.metric

        y_pred = self.predict(self.X, selected_features, ensemble)
        if self.metric_name == "Accuracy":
            self.metric = [accuracy_score(self.y, y_pred)]
        elif self.metric_name == "Aggregate":
            accuracy = accuracy_score(self.y, y_pred)
            # print(accuracy)
            # L = len(self.ensemble)
            # entropy = np.mean(
            #     (
            #         L // 2
            #         - np.abs(
            #             np.sum(
            #                 self.y[np.newaxis, :] == np.array([member_clf.predict(self.X[:, self.selected_features[clf_ind]]) for clf_ind, member_clf in enumerate(self.ensemble)]),
            #                 axis=0,
            #             )
            #             - L // 2
            #         )
            #     )
            #     / (L / 2)
            # )
            # entropy = calculate_diversity()
            # print(entropy)
            # self.metric = [self.alpha * accuracy + (1 - self.alpha) * entropy]
            predictions = []
            names = []
            for clf_ind, member_clf in enumerate(ensemble):
                predictions.append(member_clf.predict(self.X[:, selected_features[clf_ind]]).tolist())
                names.append(str(clf_ind))
            test_class = DiversityTests(predictions, names, self.y)
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
        # f1 = -1 * scores
        # print(f1)
        # f2 = -1 * scores[1]
        # out["F"] = anp.column_stack(np.array([f1, f2]))
        out["F"] = f1

        # print(f1.shape)

        # Function constraint to select specific numbers of features:
        # number = int((1 - self.scale_features) * self.n_features)
        # out["G"] = (self.n_features - np.sum(x[2:]) - number) ** 2
        # print(out["G"])
        # print((x[2:]))
