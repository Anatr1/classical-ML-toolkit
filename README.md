# Classical ML Toolkit

<img src="./logo.jpeg" alt="drawing" width="100"/> 

This repo contains a collection of classical machine learning algorithms and tools implemented in Python. The goal is to provide a simple and easy to understand implementation of the algorithms, so that they can be used as a reference implementation for students and practitioners alike.

## Contents
- Dimensionality Reduction:
    - [Principal Component Analysis](./dimensionality-reduction/PCA.py)
    - [Linear Discriminant Analysis](./dimensionality-reduction/LDA.py)
- Cross Validation:
    - [K-Fold Cross Validation](./cross-validation/cross_validation.py)
    - [Leave-One-Out Cross Validation](./cross-validation/cross_validation.py)
- Gaussian Classifiers:
    - [Multivariate Gaussian Classifier (Full Covariance)](./gaussian-classifiers/multivariate_gaussian_classifiers.py)
    - [Multivariate Gaussian Classifier (Naive Bayes Diagonal Covariance)](./gaussian-classifiers/multivariate_gaussian_classifiers.py)
    - [Multivariate Gaussian Classifier (Tied Full Covariance)](./gaussian-classifiers/multivariate_gaussian_classifiers.py)
    - [Multivariate Gaussian Classifier (Tied Naive Bayes Diagonal Covariance)](./gaussian-classifiers/multivariate_gaussian_classifiers.py)
- Logistic Regression:
    - [Logistic Regression](./logistic-regression/logistic_regression.py)
- Support Vector Machines:
    - [Linear SVM](./support-vector-machines/support_vector_machines.py)
    - [Polynomial Kernel SVM](./support-vector-machines/support_vector_machines.py)
    - [Radial Basis Function Kernel SVM](./support-vector-machines/support_vector_machines.py)
- Coming (somewhat) Soon:
    - Gaussian Mixture Models
    - K-Means Clustering
    - Perceptron and basic neural networks
    - Backpropagation
    - Examples
    - Documentation
    - And more!

## Usage

The algorithms are implemented as classes, with the following methods:
- `train(X, y)`: Train the model on the given data.
- `predict(X)`: Predict the labels for the given data.

Implementation may vary slightly between algorithms, but the general idea is the same.
