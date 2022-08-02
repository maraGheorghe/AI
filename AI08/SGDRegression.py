import random


class SGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coefficient_ = []

    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        self.coefficient_ = [random.random() for _ in range(len(x[0]) + 1)]
        for epoch in range(noEpochs):
            for i in range(len(x)):  # for each sample from the training data
                y_computed = self.eval(x[i])  # estimate the output
                crt_error = y_computed - y[i]  # compute the error for the current sample
                for j in range(0, len(x[0])):  # update the coefficients
                    self.coefficient_[j] = self.coefficient_[j] - learningRate * crt_error * x[i][j]
                self.coefficient_[len(x[0])] = self.coefficient_[len(x[0])] - learningRate * crt_error * 1
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
