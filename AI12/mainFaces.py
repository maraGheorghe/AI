import cv2
from glob import glob
import dlib
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
import tensorflow as tf
from keras.optimizers import adam_v2
import numpy as np
from mainEmoticons import plot_histogram_data, plotConfusionMatrix, evaluate


def normalisation(train_input, test_input, img_size):
    train_input = np.array(train_input) / 255.0
    test_input = np.array(test_input) / 255.0
    train_input.reshape(-1, img_size, img_size, 1)
    test_input.reshape(-1, img_size, img_size, 1)
    return train_input, test_input


def crop_image(image, cropped):
    x, y, w, h = cropped
    image_cropped = image[y:h, x:w]
    return image_cropped


def extract_faces(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return gray
    face = faces[0]
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cropped_face = crop_image(gray, (x1, y1, x2, y2))
    return cropped_face


def process_folder(directory, labels):
    input = []
    output = []
    for label in labels:
        path = directory + '/' + label + '/*'
        for file in glob(path)[:3000]:
            input.append(extract_faces(file))
            output.append(labels.index(label))
    return np.array([[[[i, i, i] for i in row] for row in image] for image in input]), np.array(output)


def train_by_tool_faces(train_input, train_output, test_input, test_output):
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    opt = adam_v2.Adam(learning_rate=.001)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_input, train_output, validation_data=(test_input, test_output), epochs=35)
    return model.predict(x=test_input)


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()  # OpenCV - haar
    outputNames = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    trainInput, trainOutput = process_folder('data/faces/train', outputNames)
    testInput, testOutput = process_folder('data/faces/test', outputNames)
    trainInput, testInput = normalisation(trainInput, testInput, 48)
    plot_histogram_data(trainOutput, outputNames, "face emotions")
    computedOutputs = train_by_tool_faces(trainInput, trainOutput, testInput, testOutput)
    computedOutputs = [list(elem).index(max(list(elem))) for elem in computedOutputs]
    confusion_matrix = evaluate(testOutput, computedOutputs, outputNames)
    plotConfusionMatrix(confusion_matrix, outputNames, "Face emotions classification")
    for i in range(len(testOutput)):
        if computedOutputs[i] != testOutput[i]:
            plt.imshow(testInput[i])
            plt.title('Real: ' + outputNames[testOutput[i]] + ', computed: ' + outputNames[computedOutputs[i]])
            plt.show()
