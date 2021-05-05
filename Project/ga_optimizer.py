import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set()
from sklearn.model_selection import train_test_split

np.random.seed(10)
from random import sample

# paper: https://arxiv.org/pdf/2006.12703.pdf
# code example: https://heartbeat.fritz.ai/hyperparameter-optimization-with-genetic-algorithms-in-kotlin-75e9c5a1e5ab

# rf toy example
from sklearn.datasets import load_wine
from random import sample
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import metrics


class GA_OPT():
    def __init__(self, params, data, pop_size=2, mutation_prob=.1, random_select=.5, retain=.1):
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.params = params
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.random_select = random_select
        self.retain = retain

    def generate_random_classifier(self):
        '''
        generates a random forest instance with randomized parameters
        :return: random forest instance
        '''
        rand_params = {}
        for key, values in self.params.items():
            rand_params[key] = np.random.choice(values)
        ind = rfc()
        ind.set_params(**rand_params)
        return ind

    def generate_pop(self):
        '''
        (2) create the inital population
        :return: a list of <pop_size> randomized random forest instances
        '''
        population = []
        for i in range(self.pop_size):
            ind = self.generate_random_classifier()
            population.append((ind, 0))
        return np.array(population)

    def eval(self, ind):
        '''
        (3) Evaluate the fitness of each individual in the current generation
        :param ind: invidual (random forest instance) of the popululation
        :return: accuracy of the given individual
        '''
        ind.fit(self.X_train, self.y_train)
        y_pred = ind.predict(self.X_test)
        acc = metrics.accuracy_score(self.y_test, y_pred)
        return acc

    def evaluate_pop(self, population):
        '''
        (4) Sort each individual according to their fitness value, from high to low
        :param population:
        :return:
        '''
        individuals = [x[0] for x in population]
        fitness = [self.eval(ind) for ind in individuals]
        new_population = [x for x in zip(individuals, fitness)]
        new_population.sort(key=lambda x: x[1], reverse=True)
        return new_population

    def run(self):
        # generate initial population
        population = self.generate_pop()

        # evaluate initial population
        population = self.evaluate_pop(population)

        print(population)


if __name__ == '__main__':
    #toy dataset
    iris = load_wine()
    X, y = iris.data, iris.target
    data = train_test_split(X, y, test_size=0.2)

    #chromosome (param) representation
    n_estimators = [1, 2, 3]
    criterion = ['gini', 'entropy']
    max_depth = [1, 2, 3]
    params = {'n_estimators': n_estimators, 'criterion': criterion, 'max_depth': max_depth}

    #optimizer
    optimizer = GA_OPT(params, data, pop_size=10)
    optimizer.run()
