import scipy
import numpy as np


def mcol(v):
    # 1-dim vectors -> column vectors.
    return v.reshape((v.size, 1))


def mrow(v):
    # 1-dim vectors -> row vectors.
    return v.reshape(1, v.size)


def getLogLikelihood(x, mean, sigma):
    return (
        -(x.shape[0] / 2) * np.log(2 * np.pi)
        - (0.5) * (np.linalg.slogdet(sigma)[1])
        - (0.5)
        * np.multiply((np.dot((x - mean).T, np.linalg.inv(sigma))).T, (x - mean)).sum(
            axis=0
        )
    )


class GaussianClassifierFullCovariance:
    def __init__(self) -> None:
        self.mean_class_0 = 0
        self.mean_class_1 = 0
        self.mean_class_2 = 0
        self.sigma_class_0 = 0
        self.sigma_class_1 = 0
        self.sigma_class_2 = 0
        self.pi_class_0 = 0
        self.pi_class_1 = 0
        self.pi_class_2 = 0

    def train(self, dataset, labels):
        self.mean_class_0 = mcol(dataset[:, labels == 0].mean(axis=1))
        self.mean_class_1 = mcol(dataset[:, labels == 1].mean(axis=1))
        self.mean_class_2 = mcol(dataset[:, labels == 2].mean(axis=1))

        self.sigma_class_0 = np.cov(dataset[:, labels == 0])
        self.sigma_class_1 = np.cov(dataset[:, labels == 1])
        self.sigma_class_2 = np.cov(dataset[:, labels == 2])

        self.pi_class_0 = dataset[:, labels == 0].shape[1] / dataset.shape[1]
        self.pi_class_1 = dataset[:, labels == 1].shape[1] / dataset.shape[1]
        self.pi_class_2 = dataset[:, labels == 2].shape[1] / dataset.shape[1]

    def predict(self, X):
        ll_class_0 = getLogLikelihood(X, self.mean_class_0, self.sigma_class_0)
        ll_class_1 = getLogLikelihood(X, self.mean_class_1, self.sigma_class_1)
        ll_class_2 = getLogLikelihood(X, self.mean_class_2, self.sigma_class_2)

        S = np.vstack((ll_class_0, ll_class_1, ll_class_2))

        S_joint = S + mcol(
            np.array(
                [
                    np.log(self.pi_class_0),
                    np.log(self.pi_class_1),
                    np.log(self.pi_class_2),
                ]
            )
        )

        # Compute marginal log densities
        marginal_log_densities = mrow(scipy.special.logsumexp(S_joint, axis=0))

        # Get predictions
        log_posteriors = S_joint - marginal_log_densities
        predictions = np.argmax(log_posteriors, axis=0)

        return predictions

    def logInfo(self):
        print(f"\nGaussianClassifierFullCovariance:")
        print(f"\nMean class 0:\n{self.mean_class_0}")
        print(f"\nMean class 1:\n{self.mean_class_1}")
        print(f"\nMean class 2:\n{self.mean_class_2}")
        print(f"\nSigma class 0:\n{self.sigma_class_0}")
        print(f"\nSigma class 1:\n{self.sigma_class_1}")
        print(f"\nSigma class 2:\n{self.sigma_class_2}")
        print(f"\nPi class 0:\n{self.pi_class_0}")
        print(f"\nPi class 1:\n{self.pi_class_1}")
        print(f"\nPi class 2:\n{self.pi_class_2}")

class GaussianClassifierNaiveBayes:
    def __init__(self) -> None:
        self.mean_class_0 = 0
        self.mean_class_1 = 0
        self.mean_class_2 = 0
        self.sigma_class_0 = 0
        self.sigma_class_1 = 0
        self.sigma_class_2 = 0
        self.pi_class_0 = 0
        self.pi_class_1 = 0
        self.pi_class_2 = 0

    def train(self, dataset, labels):
        self.mean_class_0 = mcol(dataset[:, labels == 0].mean(axis=1))
        self.mean_class_1 = mcol(dataset[:, labels == 1].mean(axis=1))
        self.mean_class_2 = mcol(dataset[:, labels == 2].mean(axis=1))

        identity_matrix = np.identity(dataset.shape[0])
        self.sigma_class_0 = np.multiply(
            np.cov(dataset[:, labels == 0]), identity_matrix
        )
        self.sigma_class_1 = np.multiply(
            np.cov(dataset[:, labels == 1]), identity_matrix
        )
        self.sigma_class_2 = np.multiply(
            np.cov(dataset[:, labels == 2]), identity_matrix
        )

        self.pi_class_0 = dataset[:, labels == 0].shape[1] / dataset.shape[1]
        self.pi_class_1 = dataset[:, labels == 1].shape[1] / dataset.shape[1]
        self.pi_class_2 = dataset[:, labels == 2].shape[1] / dataset.shape[1]

    def predict(self, X):
        ll_class_0 = getLogLikelihood(X, self.mean_class_0, self.sigma_class_0)
        ll_class_1 = getLogLikelihood(X, self.mean_class_1, self.sigma_class_1)
        ll_class_2 = getLogLikelihood(X, self.mean_class_2, self.sigma_class_2)

        S = np.vstack((ll_class_0, ll_class_1, ll_class_2))

        S_joint = S + mcol(
            np.array(
                [
                    np.log(self.pi_class_0),
                    np.log(self.pi_class_1),
                    np.log(self.pi_class_2),
                ]
            )
        )

        # Compute marginal log densities
        marginal_log_densities = mrow(scipy.special.logsumexp(S_joint, axis=0))

        # Get predictions
        log_posteriors = S_joint - marginal_log_densities
        predictions = np.argmax(log_posteriors, axis=0)

        return predictions

    def logInfo(self):
        print(f"\nGaussianClassifierNaiveBayes:")
        print(f"\nMean class 0:\n{self.mean_class_0}")
        print(f"\nMean class 1:\n{self.mean_class_1}")
        print(f"\nMean class 2:\n{self.mean_class_2}")
        print(f"\nSigma class 0:\n{self.sigma_class_0}")
        print(f"\nSigma class 1:\n{self.sigma_class_1}")
        print(f"\nSigma class 2:\n{self.sigma_class_2}")
        print(f"\nPi class 0:\n{self.pi_class_0}")
        print(f"\nPi class 1:\n{self.pi_class_1}")
        print(f"\nPi class 2:\n{self.pi_class_2}")

class GaussianClassifierFullTiedCovariance:
    def __init__(self) -> None:
        self.mean_class_0 = 0
        self.mean_class_1 = 0
        self.mean_class_2 = 0
        self.sigma_class_0 = 0
        self.sigma_class_1 = 0
        self.sigma_class_2 = 0
        self.sigma_tied = 0
        self.pi_class_0 = 0
        self.pi_class_1 = 0
        self.pi_class_2 = 0

    def train(self, dataset, labels):
        self.mean_class_0 = mcol(dataset[:, labels == 0].mean(axis=1))
        self.mean_class_1 = mcol(dataset[:, labels == 1].mean(axis=1))
        self.mean_class_2 = mcol(dataset[:, labels == 2].mean(axis=1))

        self.sigma_class_0 = np.cov(dataset[:, labels == 0])
        self.sigma_class_1 = np.cov(dataset[:, labels == 1])
        self.sigma_class_2 = np.cov(dataset[:, labels == 2])
        
        self.sigma_tied = (
            1
            / (dataset.shape[1])
            * (
                dataset[:, labels == 0].shape[1] * self.sigma_class_0
                + dataset[:, labels == 1].shape[1] * self.sigma_class_1
                + dataset[:, labels == 2].shape[1] * self.sigma_class_2
            )
        )

        self.pi_class_0 = dataset[:, labels == 0].shape[1] / dataset.shape[1]
        self.pi_class_1 = dataset[:, labels == 1].shape[1] / dataset.shape[1]
        self.pi_class_2 = dataset[:, labels == 2].shape[1] / dataset.shape[1]

    def predict(self, X):
        ll_class_0 = getLogLikelihood(X, self.mean_class_0, self.sigma_tied)
        ll_class_1 = getLogLikelihood(X, self.mean_class_1, self.sigma_tied)
        ll_class_2 = getLogLikelihood(X, self.mean_class_2, self.sigma_tied)
        
        S = np.vstack((ll_class_0, ll_class_1, ll_class_2))

        S_joint = S + mcol(
            np.array(
                [
                    np.log(self.pi_class_0),
                    np.log(self.pi_class_1),
                    np.log(self.pi_class_2),
                ]
            )
        )
        
        # Compute marginal log densities
        marginal_log_densities = mrow(scipy.special.logsumexp(S_joint, axis=0))

        # Get predictions
        log_posteriors = S_joint - marginal_log_densities
        predictions = np.argmax(log_posteriors, axis=0)

        return predictions

    def logInfo(self):
        print(f"\nGaussianClassifierFull:")
        print(f"\nMean class 0:\n{self.mean_class_0}")
        print(f"\nMean class 1:\n{self.mean_class_1}")
        print(f"\nMean class 2:\n{self.mean_class_2}")
        print(f"\nSigma class 0:\n{self.sigma_class_0}")
        print(f"\nSigma class 1:\n{self.sigma_class_1}")
        print(f"\nSigma class 2:\n{self.sigma_class_2}")
        print(f"\nSigma tied:\n{self.sigma_tied}")
        print(f"\nPi class 0:\n{self.pi_class_0}")
        print(f"\nPi class 1:\n{self.pi_class_1}")
        print(f"\nPi class 2:\n{self.pi_class_2}")

class GaussianClassifierNaiveBayesTiedCovariance:
    def __init__(self) -> None:
        self.mean_class_0 = 0
        self.mean_class_1 = 0
        self.mean_class_2 = 0
        self.sigma_class_0 = 0
        self.sigma_class_1 = 0
        self.sigma_class_2 = 0
        self.sigma_tied = 0
        self.pi_class_0 = 0
        self.pi_class_1 = 0
        self.pi_class_2 = 0

    def train(self, dataset, labels):
        self.mean_class_0 = mcol(dataset[:, labels == 0].mean(axis=1))
        self.mean_class_1 = mcol(dataset[:, labels == 1].mean(axis=1))
        self.mean_class_2 = mcol(dataset[:, labels == 2].mean(axis=1))

        identity_matrix = np.identity(dataset.shape[0])
        self.sigma_class_0 = np.multiply(
            np.cov(dataset[:, labels == 0]), identity_matrix
        )
        self.sigma_class_1 = np.multiply(
            np.cov(dataset[:, labels == 1]), identity_matrix
        )
        self.sigma_class_2 = np.multiply(
            np.cov(dataset[:, labels == 2]), identity_matrix
        )

        self.sigma_tied = (
            1 / (dataset.shape[1]) * (
                dataset[:, labels == 0].shape[1] * self.sigma_class_0
                + dataset[:, labels == 1].shape[1] * self.sigma_class_1
                + dataset[:, labels == 2].shape[1] * self.sigma_class_2
            )
        )

        self.pi_class_0 = dataset[:, labels == 0].shape[1] / dataset.shape[1]
        self.pi_class_1 = dataset[:, labels == 1].shape[1] / dataset.shape[1]
        self.pi_class_2 = dataset[:, labels == 2].shape[1] / dataset.shape[1]

    def predict(self, X):
        ll_class_0 = getLogLikelihood(X, self.mean_class_0, self.sigma_tied)
        ll_class_1 = getLogLikelihood(X, self.mean_class_1, self.sigma_tied)
        ll_class_2 = getLogLikelihood(X, self.mean_class_2, self.sigma_tied)

        S = np.vstack((ll_class_0, ll_class_1, ll_class_2))

        S_joint = S + mcol(
            np.array(
                [
                    np.log(self.pi_class_0),
                    np.log(self.pi_class_1),
                    np.log(self.pi_class_2),
                ]
            )
        )

        # Compute marginal log densities
        marginal_log_densities = mrow(scipy.special.logsumexp(S_joint, axis=0))

        # Get predictions
        log_posteriors = S_joint - marginal_log_densities
        predictions = np.argmax(log_posteriors, axis=0)

        return predictions
    
    def logInfo(self):
        print(f"\nGaussianClassifierNaiveBayesTiedCovariance:")
        print(f"\nMean class 0:\n{self.mean_class_0}")
        print(f"\nMean class 1:\n{self.mean_class_1}")
        print(f"\nMean class 2:\n{self.mean_class_2}")
        print(f"\nSigma class 0:\n{self.sigma_class_0}")
        print(f"\nSigma class 1:\n{self.sigma_class_1}")
        print(f"\nSigma class 2:\n{self.sigma_class_2}")
        print(f"\nSigma tied:\n{self.sigma_tied}")
        print(f"\nPi class 0:\n{self.pi_class_0}")
        print(f"\nPi class 1:\n{self.pi_class_1}")
        print(f"\nPi class 2:\n{self.pi_class_2}")
