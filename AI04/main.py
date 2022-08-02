import os
import sys
import warnings
from math import sqrt
from random import seed, randint

from GA import GA
from MyChromosome import MyChromosome
from MyExtraChromosome import MyExtraChromosome

warnings.simplefilter('ignore')


def readGraphFromTSP(fileName):
    f = open(fileName, "r")
    line = f.readline()
    while "DIMENSION" not in line:
        line = f.readline()
    noNodes = int(line.split(': ')[1])
    noEdges = noNodes * noNodes
    while "NODE" not in line:
        line = f.readline()
    dictionary = {}
    for _ in range(noNodes):
        line = f.readline()
        elems = line.split(' ')
        dictionary[int(elems[0])] = [int(elems[1]), int(elems[2])]
    matrix = []
    for i in dictionary.values():
        x1, y1 = i
        matrix.append([])
        for j in dictionary.values():
            x2, y2 = j
            matrix[-1].append(round(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
    graph = {'noNodes': noNodes,
             'matrix': matrix,
             'noEdges': noEdges,
             'source': randint(1, noNodes),
             'destination': randint(1, noNodes)}
    f.close()
    return graph


def readGraphFromFile(fileName):
    f = open(fileName, "r")
    n = int(f.readline())
    matrix = []
    for i in range(n):
        matrix.append([])
        line = f.readline()
        elems = line.split(",")
        for j in range(n):
            matrix[-1].append(int(elems[j]))
    noEdges = 0
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                if j > i:
                    noEdges += 1
            else:
                matrix[i][j] = INF
    graph = {'noNodes': n,
             'matrix': matrix,
             'noEdges': noEdges,
             'source': int(f.readline()),
             'destination': int(f.readline())}
    f.close()
    return graph


def shortest_path_all_nodes(path, param):
    matrix = param['matrix']
    length = 0
    for index in range(len(path) - 1):
        length += matrix[path[index] - 1][path[index + 1] - 1]
    length += matrix[path[0] - 1][path[-1] - 1]
    return length


def shortest_path(path, param):
    matrix = param['matrix']
    length = 0
    for index in range(len(path) - 1):
        length += matrix[path[index] - 1][path[index + 1] - 1]
    return length


def call_ga(graph, population_size=500, no_of_generations=100,
            chromosome_type=MyChromosome, function=shortest_path_all_nodes):
    # IN:
    # graph - dictionarul ce retine informatii despre orase si distantele dintre acestea, sub fora de graf,
    # OUT:
    # the_bests - solutiile cu cel mai bun fitness gasite in ultima generatie
    gaParam = {'popSize': population_size, 'noGen': no_of_generations, 'chromosome': chromosome_type}
    problems_parameters = graph
    problems_parameters['function'] = function
    best_chromosomes = []
    ga = GA(gaParam, problems_parameters)
    ga.initialisation()
    ga.evaluation()
    gens = []
    for generation in range(gaParam['noGen']):
        # ga.oneGeneration()
        ga.oneGenerationElitism()
        # ga.oneGenerationSteadyState()
        best_chromosome = ga.bestChromosome()
        gens = ga.population
        # print('Best solution in generation ' + str(generation) + ' is: x = ' + str(best_chromosome.representation)
        #     + ' with f(x) = ' + str(best_chromosome.fitness))
        best_chromosomes.append(best_chromosome)
    the_best = best_chromosomes[0]
    for good in best_chromosomes:
        if the_best.fitness > good.fitness:
            the_best = good
    the_bests = [the_best]
    for possible in gens:
        if the_best.fitness == possible.fitness and possible not in the_bests:
            the_bests.append(possible)
    return the_bests


def tests():
    print("Easy01")
    path = os.path.join(crtDir, 'data', 'easy01.txt')
    graph = readGraphFromFile(path)
    try:
        assert (call_ga(graph, 50, 50)[0].fitness == 14)
        assert (call_ga(graph, 50, 50, MyExtraChromosome, shortest_path)[0].fitness == 5)
        print("Passed")
    except AssertionError:
        print("Failed")
    print("Easy02")
    path = os.path.join(crtDir, 'data', 'easy02.txt')
    graph = readGraphFromFile(path)
    try:
        assert (call_ga(graph, 75, 50)[0].fitness == 19)
        assert (call_ga(graph, 75, 50, MyExtraChromosome, shortest_path)[0].fitness == 5)
        print("Passed")
    except AssertionError:
        print("Failed")
    print("Easy03")
    path = os.path.join(crtDir, 'data', 'easy03.txt')
    graph = readGraphFromFile(path)
    try:
        assert (call_ga(graph, 100, 100)[0].fitness == 28)
        assert (call_ga(graph, 100, 100, MyExtraChromosome, shortest_path)[0].fitness == 4)
        print("Passed")
    except AssertionError:
        print("Failed")
    print("Medium01")
    path = os.path.join(crtDir, 'data', 'medium01.txt')
    graph = readGraphFromFile(path)
    try:
        assert (call_ga(graph, 200, 200)[0].fitness == 31)
        assert (call_ga(graph, 200, 200, MyExtraChromosome, shortest_path)[0].fitness == 6)
        print("Passed")
    except AssertionError:
        print("Failed")
    print("Medium02")
    path = os.path.join(crtDir, 'data', 'medium02.txt')
    graph = readGraphFromFile(path)
    try:
        assert (call_ga(graph, 200, 150)[0].fitness == 23)
        print("Passed")
    except AssertionError:
        print("Failed")
    print("Hard01")
    path = os.path.join(crtDir, 'data', 'hard01.txt')
    graph = readGraphFromFile(path)
    try:
        assert (call_ga(graph, 500, 500)[0].fitness == 291)
        print("Passed")
    except AssertionError:
        print("Failed")
    print("Hard02")
    path = os.path.join(crtDir, 'data', 'hard02.txt')
    graph = readGraphFromFile(path)
    try:
        assert (call_ga(graph, 1000, 500)[0].fitness == 292)
        print("Passed")
    except AssertionError:
        print("Failed")


if __name__ == '__main__':
    INF = sys.maxsize
    crtDir = os.getcwd()
    tests()
    file_name = input("Give the name of the GML file: ")
    file_name += ".txt"
    file_path = os.path.join(crtDir, 'data', file_name)
    cities = readGraphFromFile(file_path)
    # cities = readGraphFromTSP(file_path)
    print('Nodes:', cities['noNodes'], '\nMatrix:', cities['matrix'], '\nEdges:', cities['noEdges'],
          '\nSource:', cities['source'], '\nDestination:', cities['destination'])
    population = int(input("Give the size of the population:"))
    generations = int(input("Give the number of generations:"))
    bests = call_ga(cities, population, generations)
    for chromosome in bests:
        print(chromosome)
    print('There are', len(bests), 'solutions with fit', bests[0].fitness, '.')
    with open('data/result', 'a') as f:
        sys.stdout = f
        print('\nFor the file: ', file_name, 'the result is: ')
        for chromosome in bests:
            print(chromosome)
        print('There are', len(bests), 'solutions with fit', bests[0].fitness, '.')
        print()
    sys.stdout = sys.__stdout__
    shortest = call_ga(cities, population, generations, MyExtraChromosome, shortest_path)
    print('\nAnd the shortest paths from source', cities['source'], 'to', cities['destination'], 'are:')
    for chromosome in shortest:
        print(chromosome)
    print('There are', len(shortest), 'solutions with fit', shortest[0].fitness, '.')
    with open('data/result', 'a') as f:
        sys.stdout = f
        print('\nAnd the shortest paths from source', cities['source'], 'to', cities['destination'], 'are:')
        for chromosome in shortest:
            print(chromosome)
        print('There are', len(shortest), 'solutions with fit', shortest[0].fitness, '.')
        print()
