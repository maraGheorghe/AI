import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neural_network
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import completeness_score

from KMeans import MyKMeans


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


def plot_data_flowers(input_data, output_data, feature_names):
    sns.scatterplot(x=[X[2] for X in input_data],
                    y=[X[3] for X in input_data],
                    hue=output_data,
                    palette="deep",
                    legend=None,
                    s=100)
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    plt.title("All flowers initial data")
    plt.show()


def load_data_spam(filename):
    file = pd.read_csv(filename)
    input_data = [value for value in file["emailText"]]
    output_data = [value for value in file["emailType"]]
    label_names = list(set(output_data))
    return input_data, output_data, label_names


def load_data_emotions(filename):
    file = pd.read_csv(filename)
    input_data = [value for value in file["Text"]]
    output_data = [value for value in file["Sentiment"]]
    label_names = list(set(output_data))
    return input_data, output_data, label_names


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


def extract_features_bag_of_words(train_inputs, test_inputs):  # BAG OF WORDS
    vec = CountVectorizer()
    train_features = vec.fit_transform(train_inputs)
    test_features = vec.transform(test_inputs)
    return train_features.toarray(), test_features.toarray()


def extract_features_tf_idf(train_inputs, test_inputs, max_features):  # TF-IDF
    vec = TfidfVectorizer(max_features=max_features)
    train_features = vec.fit_transform(train_inputs)
    test_features = vec.fit_transform(test_inputs)
    return train_features.toarray(), test_features.toarray()


def extract_features_hashing(train_inputs, test_inputs, n_features):  # HASHING - bag of words that uses hash codes
    vec = HashingVectorizer(n_features=n_features)
    train_features = vec.fit_transform(train_inputs)
    test_features = vec.fit_transform(test_inputs)
    return train_features.toarray(), test_features.toarray()


def predict_by_tool(train_features, test_features, label_names, classes):
    unsupervisedClassifier = KMeans(n_clusters=classes, random_state=0)
    unsupervisedClassifier.fit(train_features)
    computed_indexes = unsupervisedClassifier.predict(test_features)
    computed_outputs = [label_names[value] for value in computed_indexes]
    return computed_outputs


def predict_by_me(train_features, test_features, label_names, classes):
    my_unsupervised_classifier = MyKMeans(n_clusters=classes)
    my_unsupervised_classifier.fit(train_features)
    my_centroids, computed_indexes = my_unsupervised_classifier.evaluate(test_features)
    computed_outputs = [label_names[value] for value in computed_indexes]
    return computed_outputs, my_centroids, computed_indexes


def predict_supervised(train_inputs, train_outputs, test_inputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(25, 40, 20), activation='relu', max_iter=1000,
                                              solver='sgd',
                                              verbose=0, random_state=1, learning_rate_init=.01)
    classifier.fit(train_inputs, train_outputs)
    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs


def predict_hybrid(train_inputs, train_outputs, test_inputs, test_outputs):  # semi-supervised
    n = 100  # 100 inputs will be labeled
    classifier = neural_network.MLPClassifier()
    classifier.fit(train_inputs[:n], train_outputs[:n])
    computed_outputs = classifier.predict(test_inputs)
    prev_acc = accuracy_score(test_outputs, computed_outputs)

    unsupervised_classifier = KMeans(n_clusters=n, random_state=0)
    x = unsupervised_classifier.fit_transform(train_inputs)  # distance matrix points - centroids
    representative_indexes = np.argmin(x, axis=0)
    representative_inputs = [train_inputs[i] for i in representative_indexes]
    representative_outputs = [list(train_outputs)[x] for x in representative_indexes]
    classifier = neural_network.MLPClassifier()
    classifier.fit(representative_inputs, representative_outputs)  # fit with the most representative data
    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs, prev_acc


def plot_result_flowers(test_inputs, test_outputs, centroids, classification):
    sns.scatterplot(x=[X[2] for X in test_inputs],
                    y=[X[3] for X in test_inputs],
                    hue=test_outputs,
                    style=classification,
                    palette="deep",
                    legend=None,
                    s=100)
    plt.plot([x[2] for x in centroids],
             [y[3] for y in centroids],
             'k+',
             markersize=10)
    plt.title("Test data flowers classification")
    plt.show()


if __name__ == '__main__':
    print("IRIS")
    inputs, outputs, labelsNames, feature1, feature2, feature3, feature4, featureNames = load_data_flowers()
    plot_data_flowers(inputs, outputs, featureNames)
    trainInputs, trainOutputs, testInputs, testOutputs = train_and_test(inputs, outputs)
    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    computedOutput = predict_by_tool(trainInputs, testInputs, labelsNames, len(set(labelsNames)))
    print('Completeness score by tool:', completeness_score(testOutputs, computedOutput))
    computedOutput, centroids, computedIndexes = \
        predict_by_me(trainInputs, testInputs, labelsNames, len(set(labelsNames)))
    print('Completeness score by me:', completeness_score(testOutputs, computedOutput))
    plot_result_flowers(testInputs, testOutputs, centroids, computedIndexes)

    print("\nSPAM")
    inputs, outputs, labelsNames = load_data_spam('data/spam.csv')
    trainInputs, trainOutputs, testInputs, testOutputs = train_and_test(inputs, outputs)
    # trainFeatures, testFeatures = extract_features_bag_of_words(trainInputs, testInputs)
    # trainFeatures, testFeatures = extract_features_tf_idf(trainInputs, testInputs, 500)
    trainFeatures, testFeatures = extract_features_hashing(trainInputs, testInputs, 2 ** 10)
    computedOutputs = predict_by_tool(trainFeatures, testFeatures, labelsNames, len(set(labelsNames)))
    myComputedOutputs, centroids, computedIndexes = \
        predict_by_me(trainFeatures, testFeatures, labelsNames, len(set(labelsNames)))
    inverseTestOutputs = ['spam' if elem == 'ham' else 'ham' for elem in testOutputs]
    accuracyByTool = accuracy_score(testOutputs, computedOutputs)
    accuracyByToolInverse = accuracy_score(inverseTestOutputs, computedOutputs)
    print('Accuracy score by tool:', max(accuracyByTool, accuracyByToolInverse))
    accuracyByMe = accuracy_score(testOutputs, myComputedOutputs)
    accuracyByMeInverse = accuracy_score(inverseTestOutputs, myComputedOutputs)
    print('Accuracy score by me:', max(accuracyByMe, accuracyByMeInverse))
    print('Output computed by tool:  ', computedOutputs)
    print('Output computed by me:    ', myComputedOutputs)
    print('Real output:              ', testOutputs)

    print("\nEMOTIONS")
    inputs, outputs, labelsNames = load_data_emotions('data/reviews_mixed.csv')
    trainInputs, trainOutputs, testInputs, testOutputs = train_and_test(inputs, outputs)
    # trainFeatures, testFeatures = extract_features_bag_of_words(trainInputs, testInputs)
    trainFeatures, testFeatures = extract_features_tf_idf(trainInputs, testInputs, 150)
    # trainFeatures, testFeatures = extract_features_hashing(trainInputs, testInputs, 2 ** 10)
    computedOutputs = predict_by_tool(trainFeatures, testFeatures, labelsNames, len(set(labelsNames)))
    myComputedOutputs, centroids, computedIndexes = \
        predict_by_me(trainFeatures, testFeatures, labelsNames, len(set(labelsNames)))
    supervisedOutput = predict_supervised(trainFeatures, trainOutputs, testFeatures)
    hybridOutput, prevAcc = predict_hybrid(trainFeatures, trainOutputs, testFeatures, testOutputs)
    inverseTestOutputs = ['negative' if elem == 'positive' else 'positive' for elem in testOutputs]
    accuracyByTool = accuracy_score(testOutputs, computedOutputs)
    accuracyByToolInverse = accuracy_score(inverseTestOutputs, computedOutputs)
    print('Accuracy score by tool:', max(accuracyByTool, accuracyByToolInverse))
    accuracyByMe = accuracy_score(testOutputs, myComputedOutputs)
    accuracyByMeInverse = accuracy_score(inverseTestOutputs, myComputedOutputs)
    print('Accuracy score by me:', max(accuracyByMe, accuracyByMeInverse))
    print('Accuracy score supervised:', accuracy_score(testOutputs, supervisedOutput))
    print('Accuracy score hybrid before KMeans:', prevAcc)
    print('Accuracy score hybrid after KMeans:', accuracy_score(testOutputs, hybridOutput))
    print('Output computed by tool:  ', computedOutputs)
    print('Output computed by me:    ', myComputedOutputs)
    print('Output for supervised:    ', list(supervisedOutput))
    print('Output for hybrid:        ', list(hybridOutput))
    print('Real output:              ', testOutputs)
