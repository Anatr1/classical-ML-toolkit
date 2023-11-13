import numpy as np

DIMENSIONS = 4

def computeAccuracy(predictions, labels):
    accurate_predictions = np.array(predictions == labels).sum()
    accuracy = accurate_predictions / labels.size * 100
    return accuracy


def computeErrorRate(predictions, labels):
    accuracy = computeAccuracy(predictions, labels)
    errorRate = 100 - accuracy
    return errorRate


def LOOCrossValidation(classifier, dataset, labels, lambda_=0):
    # Performs Leave One Out Cross Validation on a given dataset and labels.
    # Returns the average error rate and the standard deviation.
    k = len(dataset[0])

    error_rates = []

    print(f"Performing Leave One Out Cross Validation with {k} samples...")
    for i in range(k):
        training_dataset = []
        for dim in range(DIMENSIONS):
            training_dataset.append([])
        training_labels = []

        validation_dataset = []
        for dim in range(DIMENSIONS):
            validation_dataset.append([dataset[dim][i]])
        validation_labels = labels[i]
        
        for dim in range(DIMENSIONS):
            for j in range(k):
                if j != i:
                    training_dataset[dim].append(dataset[dim][j])
                    if len(training_labels) < k - 1:
                        training_labels.append(labels[j])

        training_dataset = np.array(training_dataset)
        training_labels = np.array(training_labels)
        validation_dataset = np.array(validation_dataset)
        validation_labels = np.array(validation_labels)

        classifier.train(training_dataset, training_labels, lambda_)
        error_rates.append(computeErrorRate(classifier.predict(validation_dataset), validation_labels))

    error_rates = np.array(error_rates)
    print(f"Average error rate: {round(np.mean(error_rates), 2)}%")
    print(f"Standard deviation: {round(np.std(error_rates), 2)}%")

def KFoldCrossValidation(classifier, dataset, labels, k=3, lambda_=0):
    # Performs K-Fold Cross Validation on a given dataset and labels.
    # Returns the average error rate and the standard deviation.

    error_rates = []

    print(f"Performing K-Fold Cross Validation with {k} folds...")
    for i in range(k):
        training_dataset = []
        for dim in range(DIMENSIONS):
            training_dataset.append([])
        training_labels = []

        validation_dataset = []
        for dim in range(DIMENSIONS):
            validation_dataset.append([])
        validation_labels = []

        for j in range(len(dataset[0])):
            if j % k == i:
                for dim in range(DIMENSIONS):
                    validation_dataset[dim].append(dataset[dim][j])
                validation_labels.append(labels[j])
            else:
                for dim in range(DIMENSIONS):
                    training_dataset[dim].append(dataset[dim][j])
                training_labels.append(labels[j])

        training_dataset = np.array(training_dataset)
        training_labels = np.array(training_labels)
        validation_dataset = np.array(validation_dataset)
        validation_labels = np.array(validation_labels)

        classifier.train(training_dataset, training_labels, lambda_)
        error_rates.append(computeErrorRate(classifier.predict(validation_dataset), validation_labels))

    error_rates = np.array(error_rates)
    print(f"Average error rate: {round(np.mean(error_rates), 2)}%")
    print(f"Standard deviation: {round(np.std(error_rates), 2)}%")
