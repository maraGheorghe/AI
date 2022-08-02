import random
from random import randint


class MyChromosome:
    def __init__(self, problemsParameters=None):
        self.__problemsParameters = problemsParameters
        self.__fitness = 0.0
        self.__representation = []
        self.__repartition = {}
        self.__init_representation()

    @property
    def representation(self):
        return self.__representation

    @property
    def repartition(self):
        return self.__repartition

    @property
    def fitness(self):
        return self.__fitness

    @representation.setter
    def representation(self, chromosome=[]):
        self.__representation = chromosome

    @repartition.setter
    def repartition(self, repartition={}):
        self.__repartition = repartition

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    def crossover(self, c):
        cut = randint(0, len(self.__representation) - 1)
        new_representation = []
        for i in range(cut):
            new_representation.append(self.__representation[i])
        for i in range(cut, len(self.__representation)):
            new_representation.append(c.__representation[i])
        new_repartition = {}
        for community in new_representation:
            if community in new_repartition:
                new_repartition[community] += 1
            else:
                new_repartition[community] = 1
        for community in range(1, self.__problemsParameters['noOfCommunities'] + 1):
            if community not in new_repartition:
                position = randint(0, len(self.__representation) - 1)
                while new_repartition[new_representation[position]] == 1:
                    position = randint(0, len(self.__representation) - 1)
                new_repartition[new_representation[position]] -= 1
                new_repartition[community] = 1
                self.__representation[position] = community
        offspring = MyChromosome(c.__problemsParameters)
        offspring.representation = new_representation
        offspring.repartition = new_repartition
        return offspring

    def mutation(self):
        no_of_communities = self.__problemsParameters['noOfCommunities']
        position = randint(0, len(self.__representation) - 1)
        value = randint(1, no_of_communities)
        while self.__repartition[self.__representation[position]] == 1 or self.__representation[position] == value:
            position = randint(0, len(self.__representation) - 1)
        self.__repartition[self.__representation[position]] -= 1
        self.__repartition[value] += 1
        self.__representation[position] = value

    def __str__(self):
        return '\nChromosome: ' + str(self.__representation) + ' has fit: ' + str(self.__fitness) \
               + ' and repartition: ' + str(self.__repartition) + '.'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__representation == c.__repres and self.__fitness == c.__fitness

    def __init_representation(self):
        no_of_communities = self.__problemsParameters['noOfCommunities']
        self.__representation = [i for i in range(1, no_of_communities + 1)]
        for community in range(1, no_of_communities + 1):
            self.__repartition[community] = 1
        for _ in range(self.__problemsParameters['noNodes'] - no_of_communities):
            value = randint(1, no_of_communities)
            self.__representation.append(value)
            self.__repartition[value] += 1
        random.shuffle(self.__representation)
