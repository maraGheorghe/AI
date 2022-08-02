import pandas as pd
import numpy as np
from sklearn import linear_model


def load_data(filename, input_features, output_feature):
    file = pd.read_csv(filename)
    features = []
    for feature in input_features:
        features.append([float(value) for value in file[feature]])
    output_feature = [float(value) for value in file[output_feature]]
    return features, output_feature


def prepare_data(features, result):
    matrix = []
    for index, elems in enumerate(zip(features[0], features[1])):
        first, second = elems
        if pd.isna(first) or pd.isna(second) or first == second == 0 or [first, second] in matrix:
            features[0].pop(index)
            features[1].pop(index)
            result.pop(index)
        else:
            matrix.append([first, second])
    return features, result


def train_and_test(features, result):
    np.random.seed(5)
    indexes = [i for i in range(len(result))]
    train_sample_indexes = np.random.choice(indexes, int(0.8 * len(result)), replace=False)
    validation_sample_indexes = [i for i in range(len(result)) if i not in train_sample_indexes]
    train_features = []
    validation_features = []
    for feature in features:
        train_features.append([feature[i] for i in train_sample_indexes])
        validation_features.append([feature[i] for i in validation_sample_indexes])
    train_result = [result[i] for i in train_sample_indexes]
    validation_result = [result[i] for i in validation_sample_indexes]
    return train_features, train_result, validation_features, validation_result


def linear_regression_by_tool(train_features, train_result):
    xx = [[x, y] for x, y in zip(train_features[0], train_features[1])]
    regressor_result = linear_model.LinearRegression()
    regressor_result.fit(xx, train_result)
    return regressor_result


def linear_regression(features, result):  # (XT * X) ** (-1) * (XT) * Y
    XT = [[1] * len(features[0])] + features
    XTX = []
    for row1 in XT:
        line = []
        for row2 in XT:
            line.append(sum([x * y for x, y in zip(row1, row2)]))
        XTX.append(line)
    XTX_inverse = get_matrix_inverse(XTX)
    XTX_inverse_XT = []
    for row in XTX_inverse:
        line = []
        for row1 in transpose_matrix(XT):
            line.append(sum([x * y for x, y in zip(row, row1)]))
        XTX_inverse_XT.append(line)
    XTX_inverse_XTY = []
    for row in XTX_inverse_XT:
        XTX_inverse_XTY.append(sum([x * y for x, y in zip(row, result)]))
    return XTX_inverse_XTY


def get_matrix_minor(matrix, i, j):
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]


def get_matrix_determinant(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    determinant = 0
    for column in range(len(matrix)):
        determinant += ((-1) ** column) * matrix[0][column] * get_matrix_determinant(
            get_matrix_minor(matrix, 0, column))
    return determinant


def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))


def get_matrix_inverse(matrix):  # X* / determinant; X* = XT * (-1) ** (r + c)
    determinant = get_matrix_determinant(matrix)
    if len(matrix) == 2:
        return [[matrix[1][1] / determinant, -1 * matrix[0][1] / determinant],
                [-1 * matrix[1][0] / determinant, matrix[0][0] / determinant]]
    matrixX = []
    for row in range(len(matrix)):
        matrixX_row = []
        for column in range(len(matrix)):
            minor = get_matrix_minor(matrix, row, column)
            matrixX_row.append(((-1) ** (row + column)) * get_matrix_determinant(minor))
        matrixX.append(matrixX_row)
    matrixX = transpose_matrix(matrixX)
    for row in range(len(matrixX)):
        for column in range(len(matrixX)):
            matrixX[row][column] = matrixX[row][column] / determinant
    return matrixX


def calculate_y(coefficients, features):
    y = []
    for f1, f2 in zip(*features):
        y.append(coefficients[0] + f1 * coefficients[1] + f2 * coefficients[2])
    return y


def test(computed_output, validation_output):
    error = 0.0
    for t1, t2 in zip(computed_output, validation_output):
        error += (t1 - t2) ** 2
    error = error / len(validation_output)
    print('Prediction error: ', error)


if __name__ == '__main__':
    file_v1 = 'data/v1_world-happiness-report-2017.csv'
    file_v2 = 'data/v2_world-happiness-report-2017.csv'
    file_v3 = 'data/v3_world-happiness-report-2017.csv'
    inputs, output = load_data(file_v3, ['Economy..GDP.per.Capita.', 'Freedom'],
                               'Happiness.Score')
    inputs, output = prepare_data(inputs, output)
    train_inputs, train_outputs, validation_inputs, validation_outputs = train_and_test(inputs, output)
    regressor = linear_regression_by_tool(train_inputs, train_outputs)
    print('Regressor calculated by tool:', [] + [regressor.intercept_] + list(regressor.coef_))
    train_regressor = linear_regression(train_inputs, train_outputs)
    print('Regressor calculated by me:', train_regressor)
    test(calculate_y(train_regressor, validation_inputs), validation_outputs)
