import numpy as np


def mcol(v):
    # 1-dim vectors -> column vectors.
    return v.reshape((v.size, 1))


def computePCA(covariance_matrix, dataset, m):
    # Use linalg.eigh to get eigenvalues and eigenvectors of C,
    # then computes its principal components and projection matrix
    _, eigenvectors = np.linalg.eigh(covariance_matrix)
    principal_components = eigenvectors[:, ::-1][:, 0:m]

    projection_matrix = np.dot(principal_components.T, dataset)

    return projection_matrix


def PCA(dataset, m):
    # First we need to center the data
    centered_dataset = dataset - mcol(dataset.mean(axis=1))

    # Compute covariance matrix
    covariance_matrix = (1 / centered_dataset.shape[1]) * (
        np.dot(centered_dataset, centered_dataset.T)
    )

    return computePCA(covariance_matrix, dataset, m)
