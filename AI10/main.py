import itertools
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn import neural_network
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import NeuronalNetwork


def load_data_flowers():
    data = load_iris()
    input_data = data['data']
    output_data = data['target']
    outputs_name = data['target_names']
    feature_names = list(data['feature_names'])
    feature_1 = [feat[feature_names.index('sepal length (cm)')] for feat in input_data]
    feature_2 = [feat[feature_names.index('sepal width (cm)')] for feat in input_data]
    feature_3 = [feat[feature_names.index('petal length (cm)')] for feat in input_data]
    feature_4 = [feat[feature_names.index('petal width (cm)')] for feat in input_data]
    input_data = [[feat[feature_names.index('sepal length (cm)')],
                   feat[feature_names.index('sepal width (cm)')],
                   feat[feature_names.index('petal length (cm)')],
                   feat[feature_names.index('petal width (cm)')]] for feat in input_data]
    return input_data, output_data, outputs_name, feature_1, feature_2, feature_3, feature_4, feature_names


def load_data_digit():
    data = load_digits()
    input_data = data.images
    output_data = data['target']
    outputs_name = data['target_names']
    return input_data, output_data, outputs_name


def plot_data_four_features(input_data, output_data, feature_names, title=None):
    x = [i[0] for i in input_data]
    y = [i[1] for i in input_data]
    z = [i[2] for i in input_data]
    v = [i[3] for i in input_data]
    figure = px.scatter_3d(x=x, y=y, z=z, symbol=v, color=output_data, title=title,
                           labels=dict(x=feature_names[0], y=feature_names[1], z=feature_names[2],
                                       symbol=feature_names[3], color="Type"))
    figure.update_layout(legend=dict(orientation="v", yanchor='top', xanchor="right"))
    figure.show()


def plot_histogram_feature(feature, variableName):
    plt.hist(feature, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plot_histogram_data(output_data, outputs_name, title):
    plt.hist(output_data, 10)
    plt.title('Histogram of ' + title)
    plt.xticks(np.arange(len(outputs_name)), outputs_name)
    plt.show()


def train_and_test(input_data, output_data):
    indexes = [i for i in range(len(input_data))]
    train_sample = np.random.choice(indexes, int(0.8 * len(input_data)), replace=False)
    test_sample = [i for i in indexes if i not in train_sample]
    train_inputs = [input_data[i] for i in train_sample]
    train_outputs = [output_data[i] for i in train_sample]
    test_inputs = [input_data[i] for i in test_sample]
    test_outputs = [output_data[i] for i in test_sample]
    return train_inputs, train_outputs, test_inputs, test_outputs


def normalisation(train_data, test_data):
    scaler = StandardScaler()
    if not isinstance(train_data[0], list):
        trainData = [[d] for d in train_data]
        testData = [[d] for d in test_data]
        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(train_data)
        normalisedTrainData = scaler.transform(train_data)
        normalisedTestData = scaler.transform(test_data)
    return normalisedTrainData, normalisedTestData


def classifier_by_tool(train_inputs, train_outputs, test_inputs, test_outputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=100, solver='sgd',
                                              verbose=0, random_state=1, learning_rate_init=.1)
    classifier.fit(train_inputs, train_outputs)
    computed_outputs = classifier.predict(test_inputs)
    print('Accuracy by tool:', classifier.score(test_inputs, test_outputs))
    return computed_outputs


def classifier_by_me(train_inputs, train_outputs, test_inputs):
    classifier = NeuronalNetwork.NeuralNetwork(hidden_layer_size=10)
    classifier.fit(np.array(train_inputs), np.array(train_outputs))
    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs


def evaluate(test_outputs, computed_labels, output_names):
    confusion_matrix_calculated = confusion_matrix(test_outputs, computed_labels)
    acc = sum([confusion_matrix_calculated[i][i] for i in range(len(output_names))]) / len(test_outputs)
    prec = {}
    rec = {}
    for i in range(len(output_names)):
        prec[output_names[i]] = confusion_matrix_calculated[i][i] / sum([confusion_matrix_calculated[j][i]
                                                                         for j in range(len(output_names))])
        rec[output_names[i]] = confusion_matrix_calculated[i][i] / sum([confusion_matrix_calculated[i][j]
                                                                        for j in range(len(output_names))])
    print('Accuracy: ', acc)
    print('Precision: ', prec)  # TP/TP+FP - cate din cele gasite sunt relevante
    print('Recall: ', rec)  # TP/TP+FN - cate  relevante au fost gasite
    return confusion_matrix_calculated


def plotConfusionMatrix(cm, class_names, title):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                 horizontalalignment='center',
                 color='white' if cm[row, column] > thresh else 'black')
    plt.ylabel('Real label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def flatten_data(train_data, test_data):
    def flatten(mat):
        x = []
        for line in mat:
            for el in line:
                x.append(el)
        return x
    train_data = [flatten(data) for data in train_data]
    test_data = [flatten(data) for data in test_data]
    return train_data, test_data


if __name__ == '__main__':
    print("IRIS")
    inputs, outputs, outputNames, feature1, feature2, feature3, feature4, featureNames = load_data_flowers()
    plot_data_four_features(inputs, outputs, featureNames, "Initial data for flowers")
    plot_histogram_feature(feature1, featureNames[0])
    plot_histogram_feature(feature2, featureNames[1])
    plot_histogram_feature(feature3, featureNames[2])
    plot_histogram_feature(feature4, featureNames[3])
    plot_histogram_data(outputs, outputNames, 'Flowers class')
    trainInputs, trainOutputs, testInputs, testOutputs = train_and_test(inputs, outputs)
    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    plot_data_four_features(trainInputs, trainOutputs, featureNames, "Normalised flowers' data")
    computedOutputs = classifier_by_tool(trainInputs, trainOutputs, testInputs, testOutputs)
    print('Computed:', list(computedOutputs))
    print('Real:    ', testOutputs)
    print()
    computedOutputsByMe = classifier_by_me(trainInputs, trainOutputs, testInputs)
    print('Computed by me:', computedOutputsByMe)
    print('Real:          ', testOutputs)
    confusion_matrix_by_me = evaluate(np.array(testOutputs), np.array(computedOutputsByMe), outputNames)
    plotConfusionMatrix(confusion_matrix_by_me, outputNames, "Iris classification by me")

    print('\n\nDIGITS')
    inputs, outputs, outputNames = load_data_digit()
    plot_histogram_data(outputs, outputNames, 'Digits Class')
    trainInputs, trainOutputs, testInputs, testOutputs = train_and_test(inputs, outputs)
    trainInputs, testInputs = flatten_data(trainInputs, testInputs)
    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    computedOutputs = classifier_by_tool(trainInputs, trainOutputs, testInputs, testOutputs)
    print('Computed:', list(computedOutputs))
    print('Real:    ', testOutputs)
    print()
    computedOutputsByMe = classifier_by_me(trainInputs, trainOutputs, testInputs)
    print('Computed by me:', computedOutputsByMe)
    print('Real:          ', testOutputs)
    confusion_matrix_by_me = evaluate(np.array(testOutputs), np.array(computedOutputsByMe), outputNames)
    plotConfusionMatrix(confusion_matrix_by_me, outputNames, "Digits classification by me")
