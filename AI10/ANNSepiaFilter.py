from PIL import Image
import numpy as np
from sklearn import neural_network
from main import train_and_test, evaluate, plotConfusionMatrix, plot_histogram_data


def process_image(path):
    img = Image.open(path)
    img = np.asarray(img)
    processed = []
    for i in img:
        processed += list(i)
    return np.ravel(processed) / 255.0


def process_folder():
    inputs_data = []
    outputs_data = []
    for i in range(1, 100):
        inputs_data.append(process_image('data/original/' + str(i) + '.jpg'))
        outputs_data.append(0)
        inputs_data.append(process_image('data/sepia/' + str(i) + '-sepia.jpg'))
        outputs_data.append(1)
    return inputs_data, outputs_data


def train_by_tool(train_inputs, train_outputs, test_inputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(12, 25, 12), max_iter=10000)
    classifier.fit(train_inputs, train_outputs)
    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs


if __name__ == '__main__':
    outputNames = ['original', 'sepia']
    imgSize = 64
    inputData, outputData = process_folder()
    trainInputs, trainOutputs, testInputs, testOutputs = train_and_test(inputData, outputData)
    trainInputs = np.array(trainInputs)
    trainOutputs = np.array(trainOutputs)
    testInputs = np.array(testInputs)
    testOutputs = np.array(testOutputs)
    plot_histogram_data(trainOutputs, outputNames, 'original and sepia images')
    computedOutputs = train_by_tool(trainInputs, trainOutputs, testInputs)
    print(computedOutputs)
    print(testOutputs)
    confusion_matrix = evaluate(testOutputs, computedOutputs, outputNames)
    plotConfusionMatrix(confusion_matrix, outputNames, "Sepia ANN classification")
