import random
from statistics import mean


class BGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coefficient_ = []

    # def fit(self, x, y, learningRate=0.001, noEpochs=1000):
    #     self.coefficient_ = [random.random() for _ in range(len(x[0]) + 1)]
    #     for epoch in range(noEpochs):
    #         coefficients = [0] * (len(x[0]) + 1)
    #         for i in range(len(x)):  # for each sample from the training data
    #             y_computed = self.eval(x[i])  # estimate the output
    #             crtError = y_computed - y[i]  # compute the error for the current sample
    #             for j in range(0, len(x[0])):  # update the coefficients
    #                 coefficients[j] = coefficients[j] - learningRate * crtError * x[i][j]
    #             coefficients[len(x[0])] = coefficients[len(x[0])] - learningRate * crtError * 1
    #         for index in range(len(self.coefficient_)):
    #             self.coefficient_[index] += coefficients[index]
    #     self.intercept_ = self.coefficient_[-1]
    #     self.coefficient_ = self.coefficient_[:-1]

    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        self.coefficient_ = [random.random() for _ in range(len(x[0]) + 1)]
        for epoch in range(noEpochs):
            errors = []
            for i in range(len(x)):
                y_computed = self.eval(x[i])
                errors.append(y_computed - y[i])
            error = mean(errors)
            for i in range(len(x)):
                for j in range(0, len(x[0])):
                    self.coefficient_[j] = self.coefficient_[j] - learningRate * error * x[i][j]
                self.coefficient_[len(x[0])] = self.coefficient_[len(x[0])] - learningRate * error * 1
        self.intercept_ = self.coefficient_[-1]
        self.coefficient_ = self.coefficient_[:-1]

    def eval(self, xi):
        yi = self.coefficient_[-1]
        for j in range(len(xi)):
            yi += self.coefficient_[j] * xi[j]
        return yi

    def predict(self, x):
        y_computed = [self.eval(xi) for xi in x]
        return y_computed
