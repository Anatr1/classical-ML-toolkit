import numpy as np
import scipy.optimize

DIMENSIONS = 4

def J_binary(w, b, dataset, labels, lambda_, pi, SVM=False):
    norm_term = lambda_ / 2 * (np.linalg.norm(w) ** 2)
    sum_term_positive = 0
    sum_term_negative = 0

    for i in range(dataset.shape[1]):
        if labels[i] == 1:
            sum_term_positive += np.log1p(np.exp(-np.dot(w.T, dataset[:, i]) - b))
        else:
            sum_term_negative += np.log1p(np.exp(np.dot(w.T, dataset[:, i]) + b))

    if not SVM:
        j = (
            norm_term
            + (pi / dataset[:, labels == 1].shape[1]) * sum_term_positive
            + ((1 - pi) / dataset[:, labels == 0].shape[1]) * sum_term_negative
        )
    else:
        j = (
            norm_term
            + (pi / dataset[:, labels == 1].shape[1]) * sum_term_positive
            + ((1 - pi) / dataset[:, labels == -1].shape[1]) * sum_term_negative
        )
    return j

def binary_logreg_obj(v, dataset, labels, l, prior=0.5, SVM=False):
    w, b = v[0:-1], v[-1]

    j = J_binary(w, b, dataset, labels, l, prior, SVM)
    return j

def J_multiclass(w, b, dataset, labels, lambda_, pi, SVM=False):
    norm_term = lambda_ / 2 * (np.linalg.norm(w) ** 2)
    sum_term = 0

    for i in range(dataset.shape[1]):
        # Ensure that w and dataset[:, i] have compatible shapes for the dot product operation
        w_reshaped = w[:dataset.shape[0]].reshape(-1, 1)
        dataset_reshaped = dataset[:, i].reshape(-1, 1)
        sum_term += np.log1p(np.exp(-np.dot(w_reshaped.T, dataset_reshaped) - b[labels[i]]))

    if not SVM:
        j = norm_term + (1 / dataset.shape[1]) * sum_term
    else:
        j = norm_term + (1 / dataset.shape[1]) * sum_term
    return j

def multiclass_logreg_obj(v, dataset, labels, l, prior=0.5, SVM=False):
    w, b = v[0:-len(set(labels))], v[-len(set(labels)):]

    j = J_multiclass(w, b, dataset, labels, l, prior, SVM)
    return j

class LogisticRegression:
    def __init__(self) -> None:
        self.x_estimated_minimum_position = 0
        self.f_objective_value_at_minimum = 0
        self.d_info = ""

    def train(self, dataset, labels, lambda_, pi=0.5, SVM=False):
        (
            self.x_estimated_minimum_position,
            self.f_objective_value_at_minimum,
            self.d_info,
        ) = scipy.optimize.fmin_l_bfgs_b(
            binary_logreg_obj,
            np.zeros(dataset.shape[0] + 1),
            args=(dataset, labels, lambda_, pi, SVM),
            approx_grad=True,
        )

    def getScores(self, X):
        scores = np.dot(self.x_estimated_minimum_position[0:-1], X) + self.x_estimated_minimum_position[-1]
        return scores

    def predict(self, X):
        scores = self.getScores(X)
        prediction = (scores > 0).astype(int)
        return prediction
    
class MulticlassLogisticRegression:
    def __init__(self) -> None:
        self.x_estimated_minimum_position = 0
        self.f_objective_value_at_minimum = 0
        self.d_info = ""

    def train(self, dataset, labels, lambda_, pi=0.5, SVM=False):
        (
            self.x_estimated_minimum_position,
            self.f_objective_value_at_minimum,
            self.d_info,
        ) = scipy.optimize.fmin_l_bfgs_b(
            multiclass_logreg_obj,
            np.zeros((dataset.shape[0], DIMENSIONS + 1)),
            args=(dataset, labels, lambda_, pi, SVM),
            approx_grad=True,
        )

    def getScores(self, X):
        # Ensure that x_estimated_minimum_position and X have compatible shapes for the dot product operation
        x_estimated_minimum_position_reshaped = self.x_estimated_minimum_position[:X.shape[0]].reshape(-1, 1)
        scores = np.dot(x_estimated_minimum_position_reshaped.T, X) + self.x_estimated_minimum_position[-1]
        return scores

    def predict(self, X):
        scores = self.getScores(X)
        prediction = np.argmax(scores, axis=0)
        return prediction
