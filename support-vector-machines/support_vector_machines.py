import scipy.optimize as scopt
import numpy as np
from itertools import repeat

def LD(α, H):
    grad = np.dot(H, α) - np.ones(H.shape[1])
    return (
        (1 / 2) * np.dot(np.dot(α.T, H), α) - np.dot(α.T, np.ones(H.shape[1])),
        grad,
    )


class LinearSVM:
    def __init__(self, dataset_training, labels_training, dataset_evaluation, labels_evaluation, C, K=10, threshold=0, print_to_console=-1):
        self.dataset_training = dataset_training
        self.labels_training = labels_training
        self.dataset_evaluation = dataset_evaluation
        self.labels_evaluation = labels_evaluation
        self.C = C
        self.K = K
        self.threshold = threshold
        self.print_to_console = print_to_console

    def getDatasetExtended(self, dataset):
        row = np.zeros(dataset.shape[1]) + self.K 
        dataset_extended = np.vstack([dataset, row])

        return dataset_extended

    def getHmodified(self):
        dataset_training_extended = self.getDatasetExtended(self.dataset_training)  
        X = np.dot(dataset_training_extended.T, dataset_training_extended)

        labels_training_as_column = self.labels_training.reshape(self.labels_training.size, 1) 
        labels_training_as_row = self.labels_training.reshape(1, self.labels_training.size) 
        Z = np.dot(labels_training_as_column, labels_training_as_row) 

        H = Z * X

        return H

    def primalObjective(self, w, dataset_training_extended, f_value):
        norm_w = (1 / 2) * np.linalg.norm(2) ** 2

        m = np.zeros(self.labels_training.size)
        for i in range(self.labels_training.size):
            vett = [0, 1 - self.labels_training[i] * (np.dot(w.T, dataset_training_extended[:, i]))]
            m[i] = vett[np.argmax(vett)]

        primal_loss = norm_w + self.C * np.sum(m)
        dual_loss = -f_value
        duality_gap = primal_loss - dual_loss

        return primal_loss, dual_loss, duality_gap

    def linearSVM(self):
        # Alphas must be 0 <= αi <= C
        bounds = list(repeat((0, self.C), self.dataset_training.shape[1]))
        alphas = np.zeros(self.dataset_training.shape[1])
        H = self.getHmodified()

        minimum_pos, f_value, _ = scopt.fmin_l_bfgs_b(LD, alphas, args=(H,), bounds=bounds, iprint=self.print_to_console, factr=1.0)

        # We can now recover the primal solution
        AZ = minimum_pos * self.labels_training
        dataset_training_extended = self.getDatasetExtended(self.dataset_training)
        w = np.sum(AZ.reshape(1, len(AZ)) * dataset_training_extended, axis=1)

        # Compute the scores
        dataset_evaluation_extended = self.getDatasetExtended(self.dataset_evaluation)
        scores = np.dot(w.T, dataset_evaluation_extended)

        # Predict labels
        predicted_labels = (scores > self.threshold) * 1
        predicted_labels[predicted_labels == 0] = -1 

        # Let's check how bad we did
        n_correct = np.array(predicted_labels == self.labels_evaluation).sum()
        accuracy = 100 * n_correct / self.labels_evaluation.size
        error_rate = 100 - accuracy

        primal_loss, dual_loss, duality_gap = self.primalObjective(w, dataset_training_extended, f_value)

        return primal_loss, dual_loss, duality_gap, error_rate

    def train(self): 
        bounds = list(repeat((0, self.C), self.dataset_training.shape[1]))
        alphas = np.zeros(self.dataset_training.shape[1])
        H = self.getHmodified()

        self.minimum_pos, f_value, _ = scopt.fmin_l_bfgs_b(LD, alphas, args=(H,), bounds=bounds, iprint=self.print_to_console, factr=1.0)

    def getScores(self):
        # We can now recover the primal solution
        AZ = self.minimum_pos * self.labels_training
        dataset_training_extended = self.getDatasetExtended(self.dataset_training)
        w = np.sum(AZ.reshape(1, len(AZ)) * dataset_training_extended, axis=1)

        # Compute the scores
        dataset_evaluation_extended = self.getDatasetExtended(self.dataset_evaluation)
        scores = np.dot(w.T, dataset_evaluation_extended)

        return scores

    def predict(self):
        scores = self.getScores()
        predicted_labels = (scores > self.threshold) * 1
        predicted_labels[predicted_labels == 0] = -1

        return predicted_labels


class PolynomialSVM:
    def __init__(self, dataset_training, labels_training, dataset_evaluation, labels_evaluation, C, c, d, K=10, threshold=0, print_to_console=-1):
        self.dataset_training = dataset_training
        self.labels_training = labels_training
        self.dataset_evaluation = dataset_evaluation
        self.labels_evaluation = labels_evaluation
        self.C = C
        self.K = K
        self.d = d
        self.c = c
        self.threshold = threshold
        self.print_to_console = print_to_console

    def getKernelPoly(self):
        # Compute the kernel function k(xi, xj) + epsilon = (x1Tx2 + c)^d + K^2
        kernel = (np.dot(self.dataset_training.T, self.dataset_training) + self.c) ** self.d + self.K ** 2

        return kernel

    def getHPoly(self):
        labels_training_as_column = self.labels_training.reshape(self.labels_training.size, 1) 
        labels_training_as_row    = self.labels_training.reshape(1, self.labels_training.size)  
        Z = np.dot(labels_training_as_column, labels_training_as_row) 

        H = Z * self.getKernelPoly()

        return H

    def polynomialSVM(self):
        # Alphas must be 0 <= αi <= C
        bounds = list(repeat((0, self.C), self.dataset_training.shape[1])) 
        alphas = np.zeros(self.dataset_training.shape[1])
        H = self.getHPoly()

        minimum_pos, f_value, _ = scopt.fmin_l_bfgs_b(LD, alphas, args=(H,), bounds=bounds, iprint=self.print_to_console, factr=1.0)

        # Compute the scores
        AZ = (minimum_pos*self.labels_training).reshape(1, self.dataset_training.shape[1])
        # Kernel computed with test sample
        kernel_part = (np.dot(self.dataset_training.T, self.dataset_evaluation) + self.c) ** self.d + self.K ** 2
        scores = np.sum(np.dot(AZ, kernel_part), axis=0)

        # Predict labels
        predicted_labels = (scores > self.threshold) * 1
        predicted_labels[predicted_labels == 0] = -1

        # Let's check how bad we did
        n_correct = np.array(predicted_labels == self.labels_evaluation).sum()
        accuracy = 100 * n_correct / self.labels_evaluation.size
        error_rate = 100 - accuracy

        dual_loss = -f_value

        return dual_loss, error_rate

    def train(self):
        bounds = list(repeat((0, self.C), self.dataset_training.shape[1]))
        alphas = np.zeros(self.dataset_training.shape[1])
        H = self.getHPoly()

        self.minimum_pos, f_value, _ = scopt.fmin_l_bfgs_b(LD, alphas, args=(H,), bounds=bounds, iprint=self.print_to_console, factr=1.0)

    def getScores(self):
        # Compute the scores
        AZ = (self.minimum_pos*self.labels_training).reshape(1, self.dataset_training.shape[1])
        # Kernel computed with test sample
        kernel_part = (np.dot(self.dataset_training.T, self.dataset_evaluation) + self.c) ** self.d + self.K ** 2
        scores = np.sum(np.dot(AZ, kernel_part), axis=0)

        return scores

    def predict(self):
        scores = self.getScores()
        predicted_labels = (scores > self.threshold) * 1
        predicted_labels[predicted_labels == 0] = -1

        return predicted_labels


class RadialBasisFunctionSVM:
    def __init__(self, dataset_training, labels_training, dataset_evaluation, labels_evaluation, C, gamma, K=10, threshold=0, print_to_console=-1):
        self.dataset_training = dataset_training
        self.labels_training = labels_training
        self.dataset_evaluation = dataset_evaluation
        self.labels_evaluation = labels_evaluation
        self.C = C
        self.K = K
        self.gamma = gamma
        self.threshold = threshold
        self.print_to_console = print_to_console    

    def getKernelRBF(self):
        # Compute the kernel function k(xi, xj) + epsilon = exp(-gamma * ||x1 - x2||^2) + K^2
        kernel = np.zeros((self.dataset_training.shape[1], self.dataset_training.shape[1]))
        for i in range(self.dataset_training.shape[1]):
            for j in range(self.dataset_training.shape[1]):
                x1 = self.dataset_training[:, i]
                x2 = self.dataset_training[:, j]
                kernel[i, j] = np.exp(-self.gamma * (np.linalg.norm(x1 - x2)**2)) + self.K**2

        return kernel  

    def getHRBF(self):
        labels_training_as_column = self.labels_training.reshape(self.labels_training.size, 1) 
        labels_training_as_row    = self.labels_training.reshape(1, self.labels_training.size)  
        Z = np.dot(labels_training_as_column, labels_training_as_row)

        H = Z * self.getKernelRBF()

        return H

    def RBFSVM(self):
        # Alphas must be 0 <= αi <= C
        bounds = list(repeat((0, self.C), self.dataset_training.shape[1])) 
        alphas = np.zeros(self.dataset_training.shape[1])
        H = self.getHRBF()

        minimum_pos, f_value, _ = scopt.fmin_l_bfgs_b(LD, alphas, args=(H,), bounds=bounds, iprint=self.print_to_console, factr=1.0)

        # Compute the scores
        AZ = (minimum_pos*self.labels_training).reshape(1, self.dataset_training.shape[1])
        # Kernel computed with test sample
        kernel_part = np.zeros((self.dataset_training.shape[1], self.dataset_evaluation.shape[1]))
        for i in range(self.dataset_training.shape[1]):
            for j in range(self.dataset_evaluation.shape[1]):
                x1 = self.dataset_training[:, i]
                x2 = self.dataset_evaluation[:, j]
                kernel_part[i, j] = np.exp(-self.gamma * (np.linalg.norm(x1 - x2)**2)) + self.K**2
        scores = np.sum(np.dot(AZ, kernel_part), axis=0)

        # Predict labels
        predicted_labels = (scores > self.threshold) * 1
        predicted_labels[predicted_labels == 0] = -1

        # Let's check how bad we did
        n_correct = np.array(predicted_labels == self.labels_evaluation).sum()
        accuracy = 100 * n_correct / self.labels_evaluation.size
        error_rate = 100 - accuracy

        dual_loss = -f_value

        return dual_loss, error_rate

    def train(self):
        bounds = list(repeat((0, self.C), self.dataset_training.shape[1]))
        alphas = np.zeros(self.dataset_training.shape[1])
        H = self.getHRBF()

        self.minimum_pos, f_value, _ = scopt.fmin_l_bfgs_b(LD, alphas, args=(H,), bounds=bounds, iprint=self.print_to_console, factr=1.0)

    def getScores(self):
        # Compute the scores
        AZ = (self.minimum_pos*self.labels_training).reshape(1, self.dataset_training.shape[1])
        # Kernel computed with test sample
        kernel_part = np.zeros((self.dataset_training.shape[1], self.dataset_evaluation.shape[1]))
        for i in range(self.dataset_training.shape[1]):
            for j in range(self.dataset_evaluation.shape[1]):
                x1 = self.dataset_training[:, i]
                x2 = self.dataset_evaluation[:, j]
                kernel_part[i, j] = np.exp(-self.gamma * (np.linalg.norm(x1 - x2)**2)) + self.K**2
        scores = np.sum(np.dot(AZ, kernel_part), axis=0)

        return scores

    def predict(self):
        scores = self.getScores()
        predicted_labels = (scores > self.threshold) * 1
        predicted_labels[predicted_labels == 0] = -1

        return predicted_labels


