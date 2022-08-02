import os
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from GA import GA

warnings.simplefilter('ignore')


def greedyCommunitiesDetectionByTool(network):
    # Input: a graph
    # Output: list of community index (for every node)

    from networkx.algorithms import community

    A = np.matrix(network["mat"])
    G = nx.from_numpy_matrix(A)
    communities_generator = community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    sorted(map(sorted, top_level_communities))
    communities = [0 for node in range(network['noNodes'])]
    index = 1
    for community in sorted(map(sorted, top_level_communities)):
        for node in community:
            communities[node] = index
        index += 1
    return communities


def readNetworkFromGML(file):
    network = nx.read_gml(file, label='id')
    mat = nx.adj_matrix(network)
    matrix = [[0] * len(network.nodes) for i in range(len(network.nodes))]
    for i in range(len(mat.nonzero()[0])):
        matrix[mat.nonzero()[0][i]][mat.nonzero()[1][i]] = 1
    net = {'noNodes': len(network.nodes),
           'mat': matrix,
           'noEdges': len(network.edges),
           'degrees': [degree[1] for degree in network.degree()]
           }
    return net


def plotNetwork(network, communities=None):
    if communities is None:
        communities = [1] * network['noNodes']
    np.random.seed(123)
    A = np.matrix(network["mat"])
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(5, 5))
    nx.draw_networkx_nodes(G, pos, node_size=50, cmap=plt.cm.cool, node_color=communities)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.show()


def modularity(communities, param):
    noNodes = param['noNodes']
    mat = param['mat']
    degrees = param['degrees']
    noEdges = param['noEdges']
    M = 2 * noEdges
    Q = 0.0
    for i in range(0, noNodes):
        for j in range(0, noNodes):
            if communities[i] == communities[j]:
                Q += (mat[i][j] - degrees[i] * degrees[j] / M)
    return Q * 1 / M


# Diferenta dintre modularitatea simpla si aceasta este faptul ca aceasta incearca sa gaseasca si comunitatile mici
# din comunitatile mari. Parametrul lmbd, care este default 0.5, ne spune cat de dens este graful:  0 - nu exista
# muchii, 1 - e graf complet.
def modularity_density(communities, param, lmbd=0.5):
    G = nx.from_numpy_matrix(np.matrix(param['mat']))
    my_communities = [[] for _ in range(param['noOfCommunities'])]
    for i in range(param['noNodes']):
        my_communities[communities[i] - 1].append(i)
    Q = 0.0
    for community in my_communities:
        sub = nx.subgraph(G, community)
        sub_n = sub.number_of_nodes()
        interior_degrees = []
        exterior_degrees = []
        for node in sub:
            interior_degrees.append(sub.degree(node))
            exterior_degrees.append(G.degree(node) - sub.degree(node))
        try:
            Q += (1 / sub_n) * ((2 * lmbd * np.sum(interior_degrees)) - (2 * (1 - lmbd) * np.sum(exterior_degrees)))
        except ZeroDivisionError:
            pass
    return Q


# Incearca, de asemenea, sa evite limita de rezolutie, gasind si comunitatile mici din comunitatile mari.
# Se bazeaza pe distributia Student - se calculeaza raritatea statistica a unei comunitati.
def z_modularity(communities, param):
    G = nx.from_numpy_matrix(np.matrix(param['mat']))
    my_communities = [[] for _ in range(param['noOfCommunities'])]
    for i in range(param['noNodes']):
        my_communities[communities[i] - 1].append(i)
    edges = G.number_of_edges()
    Q = 0.0
    mmc = 0
    dc2m = 0
    for community in my_communities:
        sub = nx.subgraph(G, community)
        sub_n = sub.number_of_nodes()
        dc = 0
        for node in sub:
            dc += G.degree(node)
        mmc = sub_n / edges
        dc2m += (dc / (2 * edges)) ** 2
    try:
        Q = (mmc - dc2m) / np.sqrt(dc2m * (1 - dc2m))
    except ZeroDivisionError:
        pass
    return Q


# IN: network - dictionarul ce retine informatii despre retea
#    no - integer, numarul de comunitati dat ca paramentru
# OUT: the_best - MyChromosome, cromozomul cel mai bun din toate generatiile
def call_ga(network, no):
    gaParam = {'popSize': 100, 'noGen': 100}
    problems_parameters = network
    problems_parameters['noOfCommunities'] = no
    # problems_parameters['function'] = modularity
    # problems_parameters['function'] = modularity_density
    problems_parameters['function'] = z_modularity
    best_chromosomes = []
    ga = GA(gaParam, problems_parameters)
    ga.initialisation()
    ga.evaluation()
    for generation in range(gaParam['noGen']):
        ga.oneGeneration()
        # ga.oneGenerationElitism()
        # ga.oneGenerationSteadyState()
        best_chromosome = ga.bestChromosome()
        print('Best solution in generation ' + str(generation) + ' is: x = ' + str(best_chromosome.representation)
              + ' with repartition : ' + str(best_chromosome.repartition) + ' and f(x) = ' + str(
            best_chromosome.fitness))
        best_chromosomes.append(best_chromosome)
    the_best = best_chromosomes[1]
    for chromosome in best_chromosomes:
        if the_best.fitness < chromosome.fitness:
            the_best = chromosome
    return the_best


def tests_for_one(name, path, number):
    # O modularitate buna are valoarea cuprinsa intre 0.3 si 0.7.
    net = readNetworkFromGML(path)
    communities_tool = greedyCommunitiesDetectionByTool(net)
    the_best = call_ga(net, number)
    count = 0
    for x, y in zip(communities_tool, the_best.representation):
        if x != y:
            count += 1
    plotNetwork(net, the_best.representation)
    print('For network:', name, 'there are', count, 'differences.')
    print('Best chromosome has representation:', the_best.representation, 'repartition:', the_best.repartition,
          'and fitness:', the_best.fitness)


def tests():
    tests_for_one("Dolphins", "data/dolphins.gml", 2)
    tests_for_one("Football", "data/football.gml", 2)
    tests_for_one("Karate", "data/karate.gml", 2)
    tests_for_one("Krebs", "data/karate.gml", 2)
    tests_for_one("Lesmis", "data/lesmis.gml", 2)
    tests_for_one("MAP", "data/map.gml", 2)
    tests_for_one("Polbooks", "data/polbooks.gml", 2)
    tests_for_one("Nouns", "data/adjnoun.gml", 2)
    # tests_for_one("Power", "data/power.gml", 25)


if __name__ == '__main__':
    crtDir = os.getcwd()
    tests()
    file_name = input("Give the name of the GML file: ")
    communities_number = int(input("Give the number of communities:"))
    file_name += ".gml"
    file_path = os.path.join(crtDir, 'data', file_name)
    graph = readNetworkFromGML(file_path)
    best = call_ga(graph, communities_number)
    print(best)
    plotNetwork(graph, best.representation)
