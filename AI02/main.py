import os
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')


def readNet(fileName):
    f = open(fileName, "r")
    net = {}
    n = int(f.readline())
    net['noNodes'] = n
    mat = []
    for i in range(n):
        mat.append([])
        line = f.readline()
        elems = line.split(" ")
        for j in range(n):
            mat[-1].append(int(elems[j]))
    net["mat"] = mat
    degrees = []
    noEdges = 0
    for i in range(n):
        d = 0
        for j in range(n):
            if mat[i][j] == 1:
                d += 1
            if j > i:
                noEdges += mat[i][j]
        degrees.append(d)
    net["noEdges"] = noEdges
    net["degrees"] = degrees
    f.close()
    return net


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


def greedyCommunitiesDetection(network, no_of_communities):
    # Input: a graph
    # Output: list of community index (for every node)

    def most_crossed_edge():
        edge_tuple = list(nx.edge_betweenness_centrality(G).items())  # [(muchia, coeficientul)]
        most_crossed = max(edge_tuple, key=lambda item: item[1])[0]
        # cea care are coeficientul cel mai mare (= este cea mai traversata in cel mai scurt drum), trebuie eliminata
        return most_crossed

    A = np.matrix(network["mat"])
    G = nx.from_numpy_matrix(A)
    while len(list(nx.connected_components(G))) < no_of_communities:
        source, destination = most_crossed_edge()
        G.remove_edge(source, destination)
        # se sterge muchia cea mai traversata pana cand avem atatea componente conexe cate comunitati ne dorim

    communities = [1] * network['noNodes']
    color = 0
    for community in nx.connected_components(G):
        color += 1
        for node in community:
            communities[node] = color
    # se da o culoare specifica comunitaii din care face parte fiecarui nod din graf
    return communities


def make_dictionary_of_result(communities):
    dictionary = {}
    for color in communities:
        if color in dictionary.keys():
            dictionary[color] += 1
        else:
            dictionary[color] = 1
    print(dictionary)
    return dictionary


def tests():
    print("Doplhins")
    path = os.path.join(crtDir, 'data', 'dolphins.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 2)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("Karate")
    path = os.path.join(crtDir, 'data', 'karate.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 2)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("Krebs")
    path = os.path.join(crtDir, 'data', 'krebs.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 2)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("Football")
    path = os.path.join(crtDir, 'data', 'football.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 2)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("Les Miserables")
    path = os.path.join(crtDir, 'data', 'lesmis.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 2)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("Nouns")
    path = os.path.join(crtDir, 'data', 'adjnoun.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 2)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("NetScience")
    path = os.path.join(crtDir, 'data', 'netscience.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 397)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("PolBooks")
    path = os.path.join(crtDir, 'data', 'polbooks.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 2)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("Power")
    path = os.path.join(crtDir, 'data', 'power.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 25)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))
    print("MAP")
    path = os.path.join(crtDir, 'data', 'map.gml')
    net = readNetworkFromGML(path)
    assert (make_dictionary_of_result(greedyCommunitiesDetection(net, 2)) == make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(net)))


if __name__ == '__main__':
    crtDir = os.getcwd()
    tests()
    file_name = input("Give the name of the GML file: ")
    communities_number = int(input("Give the number of communities:"))
    file_name += ".gml"
    file_path = os.path.join(crtDir, 'data', file_name)
    graph = readNetworkFromGML(file_path)
    make_dictionary_of_result(
        greedyCommunitiesDetectionByTool(graph))
    obtained_communities = greedyCommunitiesDetection(graph, communities_number)
    print(obtained_communities)
    plotNetwork(graph, obtained_communities)
    with open('data/result', 'a') as f:
        sys.stdout = f
        print("For the file: " + file_name + " and the number of communities "
              + str(communities_number) + ", the result is: ")
        print("Communities are: ")
        print(obtained_communities)
        make_dictionary_of_result(obtained_communities)
        print()
