from math import exp
from random import random


def sigmoid(x):
    return 1 / (1 + exp(-x))


class MyLogisticRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coefficient_ = []

    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        self.coefficient_ = [random() for _ in range(len(x[0]) + 1)]
        for epoch in range(noEpochs):
            for i in range(len(x)):
                y_computed = sigmoid(self.evaluate(x[i], self.coefficient_))
                crt_error = y_computed - y[i]
                for j in range(0, len(x[0])):
                    self.coefficient_[j + 1] = self.coefficient_[j + 1] - learningRate * crt_error * x[i][j]
                self.coefficient_[0] = self.coefficient_[0] - learningRate * crt_error * 1
        self.intercept_ = self.coefficient_[0]
        self.coefficient_ = self.coefficient_[1:]

    def evaluate(self, xi, coefficient):
        yi = coefficient[0]
        for j in range(len(xi)):
            yi += coefficient[j + 1] * xi[j]
        return yi

    def predict_one_sample(self, sample_features):
        threshold = 0.5
        coefficients = [self.intercept_] + [c for c in self.coefficient_]
        computed_float_value = self.evaluate(sample_features, coefficients)
        computed01_value = sigmoid(computed_float_value)
        computed_label = 0 if computed01_value < threshold else 1
        return computed_label

    def predict(self, in_test):
        computed_labels = [self.predict_one_sample(sample) for sample in in_test]
        return computed_labels
