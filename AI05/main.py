import os
from math import sqrt

import matplotlib.pyplot as plt

from ACO import ACO
from Graph import Graph


def readGraphFromTSP(fileName):
    f = open(fileName, "r")
    line = f.readline()
    while "DIMENSION" not in line:
        line = f.readline()
    noNodes = int(line.split(': ')[1])
    while "NODE" not in line:
        line = f.readline()
    coord = []
    dictionary = {}
    for _ in range(noNodes):
        line = f.readline()
        elems = line.split(' ')
        dictionary[float(elems[0])] = [float(elems[1]), float(elems[2])]
        coord.append([float(elems[1]), float(elems[2])])
    f.close()
    matrix = []
    for i in dictionary.values():
        x1, y1 = i
        matrix.append([])
        for j in dictionary.values():
            x2, y2 = j
            matrix[-1].append(round(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
    return {'noNodes': noNodes,
            'coord': coord,
            'matrix': matrix}


def plotCities(coords, path=[]):
    x = []
    y = []
    for coord in coords:
        x.append(coord[0])
        y.append(coord[1])
    plt.plot(x, y, 'b*', markersize=10)
    for i, j in zip(path + path[-1:], path[1:] + path[:1]):
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='g')
    plt.show()


def tests():
    print("EIL51")
    file = os.path.join(crtDir, 'data', 'eil51.tsp')
    result = readGraphFromTSP(file)
    aco_test = ACO(51, 30, 1.0, 10.0, 0.7, 3)
    graph_test = Graph(result['matrix'], result['coord'], result['noNodes'])
    assert (aco_test.solve(graph_test)[1] - 426 <= 50)
    print("DJ38")
    file = os.path.join(crtDir, 'data', 'dj38.tsp')
    result = readGraphFromTSP(file)
    aco_test = ACO(38, 50, 1.0, 10.0, 0.7, 3)
    graph_test = Graph(result['matrix'], result['coord'], result['noNodes'])
    assert (aco_test.solve(graph_test)[1] - 6656 <= 100)
    print("\nWI29")
    file = os.path.join(crtDir, 'data', 'wi29.tsp')
    result = readGraphFromTSP(file)
    aco_test = ACO(29, 50, 1.0, 10.0, 0.5, 3)
    graph_test = Graph(result['matrix'], result['coord'], result['noNodes'])
    assert (aco_test.solve(graph_test)[1] - 27603 <= 2000)
    # print("\nQA194")
    # file = os.path.join(crtDir, 'data', 'qa194.tsp')
    # result = readGraphFromTSP(file)
    # aco_test = ACO(194, 10, 3.0, 10.0, 0.5, 10)
    # graph_test = Graph(result['matrix'], result['coord'], result['noNodes'])
    # assert (aco_test.solve(graph_test)[1] - 9352 <= 1000)


if __name__ == '__main__':
    crtDir = os.getcwd()
    tests()
    file_name = input('Give the name of the TSP file: ')
    file_name += ".tsp"
    file_path = os.path.join(crtDir, 'data', file_name)
    cities = readGraphFromTSP(file_path)
    print('Nodes:', cities['noNodes'], '\nCoordinates:', cities['coord'], '\nMatrix:', cities['matrix'])
    plotCities(cities['coord'])
    ants = int(input('Give the number of ants: '))
    iterations = int(input('Give the number of iterations: '))
    alpha = float(input('Give the importance of the pheromone: '))
    beta = float(input('Give the importance of the visibility: '))
    rho = float(input('Give the pheromone residual coefficient (value between 0 and 1): '))
    q = int(input('Give the pheromone intensity: '))
    dynamic = bool(int(input('Dynamic? True = 1/False = 0: ')))
    aco = ACO(ants, iterations, alpha, beta, rho, q, dynamic)
    graph = Graph(cities['matrix'], cities['coord'], cities['noNodes'])
    path, cost = aco.solve(graph)
    print('\nThe cost is:', cost, 'and the path is:', path, '.')
    plotCities(cities['coord'], path)
