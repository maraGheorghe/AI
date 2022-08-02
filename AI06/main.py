import math

import pandas as pd
import os
import matplotlib.pyplot as plt

crtDir = os.getcwd()
file_sports = os.path.join(crtDir, 'data', 'sport.csv')
file_flowers = os.path.join(crtDir, 'data', 'flowers.csv')
file_binary_flowers = os.path.join(crtDir, 'data', 'binary_flowers.csv')
sports = pd.read_csv(file_sports)
flowers = pd.read_csv(file_flowers)


def plot_predictions(feature):
    indexes = [i for i in range(len(sports[feature]))]
    real, = plt.plot(indexes, sports[feature], 'ro', label='real')
    computed, = plt.plot(indexes, sports['Predicted' + feature], 'bo', label='computed')
    plt.legend([real, (real, computed)], ["Real", "Computed"])
    plt.show()


def calculate_error_regression_sum(real, computed):
    error = 0
    for r, c in zip(real, computed):
        for index in range(len(r)):
            error += abs(r[index] - c[index])
    return error / len(real[0])


def calculate_error_regression_sqrt(real, computed):
    error = 0
    for r, c in zip(real, computed):
        for index in range(len(r)):
            error += (r[index] - c[index]) ** 2
    return math.sqrt(error / len(real[0]))


print("I")
print('Sum:', calculate_error_regression_sum([sports['Weight'], sports['Waist'], sports['Pulse']],
                                             [sports['PredictedWeight'], sports['PredictedWaist'],
                                              sports['PredictedPulse']]))

print('Sqrt:', calculate_error_regression_sqrt([sports['Weight'], sports['Waist'], sports['Pulse']],
                                               [sports['PredictedWeight'], sports['PredictedWaist'],
                                                sports['PredictedPulse']]))


def evaluate_classification(real, computed, label_names):
    acc = sum([1 if real[i] == computed[i] else 0 for i in range(0, len(real))]) / len(real)
    TP = {}
    FP = {}
    TN = {}
    FN = {}
    for label in label_names:
        TP[label] = sum(
            [1 if (real[i] == label and computed[i] == label) else 0 for i in range(len(real))])
        FP[label] = sum(
            [1 if (real[i] != label and computed[i] == label) else 0 for i in range(len(real))])
        TN[label] = sum(
            [1 if (real[i] != label and computed[i] != label) else 0 for i in range(len(real))])
        FN[label] = sum(
            [1 if (real[i] == label and computed[i] != label) else 0 for i in range(len(real))])
        print(label, TP[label], FP[label], TN[label], FN[label])
    precision = {}
    recall = {}
    for label in label_names:
        precision[label] = TP[label] / (TP[label] + FP[label])
        recall[label] = TP[label] / (TP[label] + FN[label])
    print(precision, recall)
    return acc, precision, recall


print('\nII')
flowers_types = list(set(flowers['Type']))
accuracy, precisions, recalls = evaluate_classification(flowers['Type'], flowers['PredictedType'],
                                                        flowers_types)
print('Accuracy:', accuracy)
print('Precision for', flowers_types[0], 'is', precisions[flowers_types[0]])
print('Precision for', flowers_types[1], 'is', precisions[flowers_types[1]])
print('Precision for', flowers_types[2], 'is', precisions[flowers_types[2]])
print('Recall for', flowers_types[0], 'is', recalls[flowers_types[0]])
print('Recall for', flowers_types[1], 'is', recalls[flowers_types[1]])
print('Recall for', flowers_types[2], 'is', recalls[flowers_types[2]])


def calculate_loss_regression_sum(real, computed):
    error = 0
    for r, c in zip(real, computed):
        for index in range(len(r)):
            error += abs(r[index] - c[index])
    return error


print('\nI extra')
print('CE regression:', calculate_loss_regression_sum([sports['Weight'], sports['Waist'], sports['Pulse']],
                                                      [sports['PredictedWeight'], sports['PredictedWaist'],
                                                       sports['PredictedPulse']]))


def evaluate_loss_binary_classification(real, computed, positive):
    real_outputs = [[1, 0] if label == positive else [0, 1] for label in real]
    no_of_classes = len(set(real))
    dataset_CE = 0.0
    for i in range(len(real)):
        sample_CE = - sum([real_outputs[i][j] * math.log(computed[i][j]) for j in range(no_of_classes)])
        dataset_CE += sample_CE
    mean_CE = dataset_CE / len(real)
    return mean_CE


print('\nII extra')
real_values = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
computed_outputs = [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6]]
print('CE binary:', evaluate_loss_binary_classification(real_values, computed_outputs, 'spam'))


def evaluate_multi_class_loss(targetValues, rawOutputs):
    expected_values = [math.exp(value) for value in rawOutputs]
    sum_for_expected_values = sum(expected_values)
    map_outputs = [value / sum_for_expected_values for value in expected_values]
    sample_CE = - sum([targetValues[j] * math.log(map_outputs[j]) for j in range(len(targetValues))])
    return sample_CE


print('\nIII extra')
print('CE multi-class', evaluate_multi_class_loss([0, 1, 0, 0, 0], [-0.5, 1.2, 0.1, 2.4, 0.3]))


def evaluate_multi_label_loss(targetValues, rawOutputs):
    mapOutputs = [1 / (1 + math.exp(-val)) for val in rawOutputs]
    sample_CE = - sum([targetValues[j] * math.log(mapOutputs[j]) for j in range(len(targetValues))])
    return sample_CE


print('\nIV extra')
print('CE multi-label:', evaluate_multi_label_loss([0, 1, 0, 0, 1], [-0.5, 1.2, 0.1, 2.4, 0.3]))
