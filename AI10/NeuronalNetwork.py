import numpy as np


class NeuralNetwork:

    def __init__(self, hidden_layer_size=12, max_iter=7000, learning_rate=.001):
        self.__weights = []
        self.__hidden_layer_size = hidden_layer_size
        self.__max_iter = max_iter
        self.__learning_rate = learning_rate

    def __softmax(self, x):
        exp_vector = np.exp(x)
        return exp_vector / exp_vector.sum(axis=1, keepdims=True)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def fit(self, x, y):
        no_features = len(x[0])
        no_outputs = len(set(y))
        new_y = np.zeros((len(y), no_outputs))
        for i in range(len(y)):
            new_y[i, y[i]] = 1
        y = new_y
        weight_ih = np.random.rand(no_features, self.__hidden_layer_size)  # input X hidden
        coefficient_ih = np.random.randn(self.__hidden_layer_size)
        weight_ho = np.random.rand(self.__hidden_layer_size, no_outputs)  # hidden  X output
        coefficient_ho = np.random.randn(no_outputs)
        for epoch in range(self.__max_iter):
            y_ih = np.dot(x, weight_ih) + coefficient_ih  # forward propagation
            y_ih_sigmoid = self.__sigmoid(y_ih)
            y_output = np.dot(y_ih_sigmoid, weight_ho) + coefficient_ho
            y_output_softmax = self.__softmax(y_output)
            error = y_output_softmax - y  # back propagation
            error_weight_ho = np.dot(y_ih_sigmoid.T, error)
            error_coefficient_ho = error
            error_dah = np.dot(error, weight_ho.T)
            dah_dzh = self.__sigmoid_derivative(y_ih)
            dzh_dwh = x
            error_weight_ih = np.dot(dzh_dwh.T, dah_dzh * error_dah)
            error_coefficient_ih = error_dah * dah_dzh
            weight_ih -= self.__learning_rate * error_weight_ih
            coefficient_ih -= self.__learning_rate * error_coefficient_ih.sum(axis=0)
            weight_ho -= self.__learning_rate * error_weight_ho
            coefficient_ho -= self.__learning_rate * error_coefficient_ho.sum(axis=0)
        self.__weights = [weight_ih, coefficient_ih, weight_ho, coefficient_ho]

    def predict(self, x):
        weight_ih, coefficient_ih, weight_ho, coefficient_ho = self.__weights
        y_ih = np.dot(x, weight_ih) + coefficient_ih
        y_ih_sigmoid = self.__sigmoid(y_ih)
        y_output = np.dot(y_ih_sigmoid, weight_ho) + coefficient_ho
        y_output_softmax = self.__softmax(y_output)
        computed_output = [list(output).index(max(output)) for output in y_output_softmax]
        return computed_output
