import cv2
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import tensorflow as tf
import os
import numpy as np
from main import evaluate, plotConfusionMatrix, plot_histogram_data


def process_folder(directory, img_size):
    data = []
    for label in outputNames:
        path = os.path.join(directory, label)
        class_num = outputNames.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


def train_and_test(data):
    indexes = [i for i in range(len(data))]
    train_sample = np.random.choice(indexes, int(0.8 * len(data)), replace=False)
    test_sample = [i for i in indexes if i not in train_sample]
    train = [data[i] for i in train_sample]
    test = [data[i] for i in test_sample]
    return train, test


def inputs_outputs_normalisation(train, test, img_size):
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    for feature, label in train:
        train_input.append(feature)
        train_output.append(label)
    for feature, label in test:
        test_input.append(feature)
        test_output.append(label)
    train_input = np.array(train_input) / 255.0
    test_input = np.array(test_input) / 255.0
    train_input.reshape(-1, img_size, img_size, 1)
    train_output = np.array(train_output)
    test_input.reshape(-1, img_size, img_size, 1)
    test_output = np.array(test_output)
    return train_input, train_output, test_input, test_output


def train_by_tool(train_input, train_output, test_input, test_output, img_size):
    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(img_size, img_size, 3)))  # first layer
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, padding="same", activation="relu"))  # second layer
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    model.add(Flatten())  # output layer
    model.add(Dense(2, activation="softmax"))
    opt = adam_v2.Adam(learning_rate=.001)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_input, train_output, validation_data=(test_input, test_output), epochs=35)
    return model.predict(x=test_input)


if __name__ == '__main__':
    outputNames = ['original', 'sepia']
    imgSize = 64
    totalData = process_folder('data', imgSize)
    trainData, testData, = train_and_test(totalData)
    trainInput, trainOutput, testInput, testOutput = inputs_outputs_normalisation(trainData, testData, imgSize)
    plot_histogram_data(trainOutput, outputNames, "original and sepia images")
    computedOutputs = train_by_tool(trainInput, trainOutput, testInput, testOutput, imgSize)
    computedOutputs = [list(elem).index(max(list(elem))) for elem in computedOutputs]
    print(computedOutputs)
    confusion_matrix = evaluate(testOutput, computedOutputs, outputNames)
    plotConfusionMatrix(confusion_matrix, outputNames, "Sepia CNN classification")
