import random
from random import randint


class MyExtraChromosome:
    def __init__(self, problemsParameters=None):
        self.__problemsParameters = problemsParameters
        self.__fitness = 0.0
        self.__representation = []
        self.__init_representation()

    @property
    def representation(self):
        return self.__representation

    @property
    def fitness(self):
        return self.__fitness

    @representation.setter
    def representation(self, chromosome=[]):
        self.__representation = chromosome

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    def crossover(self, c):
        # se face o taietura acolo unde se gaseste un nod comun pe aceeasi pozitie in amandoi cromozomii si se ia
        # prima parte de la unul, iar restul de la celalat cromozom pana nu mai raman elemente neadaugate in lista din
        # acesta din urma
        if len(self.__representation) == 2 or len(c.__representation) == 2:
            offspring = MyExtraChromosome(c.__problemsParameters)
            offspring.representation = c.__representation
            return offspring
        same_positions = []
        for index in range(min(len(self.__representation), len(c.__representation))):
            if self.__representation[index] == c.__representation[index]:
                same_positions.append(index)
        cut_index = randint(0, len(same_positions) - 1)
        cut = same_positions[cut_index]
        new_representation = self.__representation[:cut + 1]
        for i in range(cut + 1, len(c.__representation)):
            if c.__representation[i] in new_representation:
                continue
            new_representation.append(c.__representation[i])
        offspring = MyExtraChromosome(c.__problemsParameters)
        offspring.representation = new_representation
        return offspring

    def mutation(self):
        # se alege un mod random care sa fie diferit de sursa si de destinatie, iar apoi se cauta un alt drum care sa
        # inlocuiasca nodul respectiv
        if len(self.__representation) == 2:
            return
        node = randint(1, len(self.__representation) - 2)
        before = self.__representation[node - 1]
        after = self.__representation[node + 1]
        used = self.__representation[:node] + self.__representation[node + 2:]
        new_representation = [before]
        while new_representation[-1] != after:
            new_node = randint(1, self.__problemsParameters['noNodes'])
            while new_node in used:
                new_node = randint(1, self.__problemsParameters['noNodes'])
            new_representation.append(new_node)
            used.append(new_node)
        self.__representation = self.__representation[:node - 1] + new_representation + self.__representation[node + 2:]

    def __str__(self):
        return 'Chromosome: ' + str(self.__representation) + ' has fit: ' + str(self.__fitness) + '.'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__representation == c.__representation and self.__fitness == c.__fitness

    def __init_representation(self):
        # se pune sursa in reprezentare, apoi se populeaza aleator lista pana cand se ajunge la destinatie
        self.__representation.append(self.__problemsParameters['source'])
        while self.__representation[-1] != self.__problemsParameters['destination']:
            node = randint(1, self.__problemsParameters['noNodes'])
            while node in self.__representation:
                node = randint(1, self.__problemsParameters['noNodes'])
            self.__representation.append(node)
