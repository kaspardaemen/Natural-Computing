import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd

sns.set()
from sklearn.model_selection import train_test_split

np.random.seed(10)
from random import sample

# paper: https://arxiv.org/pdf/2006.12703.pdf
# code example: https://heartbeat.fritz.ai/hyperparameter-optimization-with-genetic-algorithms-in-kotlin-75e9c5a1e5ab

# rf toy example
from sklearn import  datasets
from random import sample
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import metrics


class GA_OPT():
    def __init__(self, params, data, pop_size=2, mutation_rate=.2, random_select=.3, survival_rate=.3, n_generations=10):
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.params = params
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.random_select = random_select
        self.survival_rate = survival_rate
        self.n_generations = n_generations
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
        return ind, rand_params
    def generate_pop(self):
        '''
        (2) create the inital population
        :return: a list of <pop_size> randomized random forest instances
        '''

        population = []
        for i in range(self.pop_size):
            ind, rand_params = self.generate_random_classifier()
            population.append((ind, rand_params, 0))
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
        params = [x[1] for x in population]
        fitness = [self.eval(ind) for ind in individuals]
        new_population = [x for x in zip(individuals, params,  fitness)]
        new_population.sort(key=lambda x: x[2], reverse=True)
        return new_population
    def mutate(self, params_dict):
        param_keys = [x for x in params_dict.keys()]
        rand_key = np.random.choice(param_keys)
        rand_value = np.random.choice(self.params[rand_key])

        mutated_params = params_dict.copy()
        mutated_params[rand_key] = rand_value
        return mutated_params
    def crossover(self, p1, p2):
        '''
        Crossover function to produce two children
        :param p1: first parent
        :param p2: second parent
        :return: c (child produced by the crossover operation of the two parents)
        '''
        p1_params = p1[1]
        p2_params = p2[1]
        child_params = {}
        for key, values in self.params.items():
            if(stats.bernoulli.rvs(p=0.5)):
                child_params[key] = p1_params[key]
            else:
                child_params[key] = p2_params[key]

        if (stats.bernoulli.rvs(p=self.mutation_rate)):
            child_params = self.mutate(child_params)
        classifier = rfc(**child_params)
        child = (classifier, child_params, 0)
        return child
    def evolve(self, current_pop):
        '''
        (8) Produce new individuals.
        :return: The next population
        '''
        #TODO: Nog opdelen in meerdere functies
        fittest_len = int(self.survival_rate*len(current_pop))
        #(5) select the fittest individuals:
        fittest_parents = current_pop[:fittest_len]
        #(6) Allow some less fit individuals to survive
        lucky_parents = []
        for unfit_parent in current_pop[fittest_len:]:
            if(stats.bernoulli.rvs(p=self.random_select)):
                lucky_parents.append(unfit_parent)
        parents = fittest_parents + lucky_parents
        children = []
        for i in range(self.pop_size):
            parent_indices = range(len(parents))
            p1_index, p2_index = np.random.choice(parent_indices, size=2, replace=False)
            p1, p2 = parents[p1_index], parents[p2_index]
            child = self.crossover(p1,p2)
            children.append(child)
        return children
    def get_gen_results(self, current_pop):
        acc_list = np.mean([x[2] for x in current_pop])
        return acc_list

    def run(self):
        # generate initial population
        #shape of population:= tuple(rf instance, param object, fitness)
        population = self.generate_pop()
        hist = []
        # evaluate initial population
        population = self.evaluate_pop(population)
        hist.append(self.get_gen_results(population))
        for g in range(1,self.n_generations+1):
            print(f'generation {g}')
            new_gen = self.evolve(population)
            new_gen = self.evaluate_pop(new_gen)
            hist.append(self.get_gen_results(new_gen))
            population = new_gen
        plt.plot(hist)
        plt.xlabel('generation')
        plt.ylabel('best accuracy')
        plt.show()

        print(f'best solution: {population[0][1]}')

if __name__ == '__main__':
    #toy dataset
    ds = datasets.load_breast_cancer()
    X, y = ds.data, ds.target
    data = train_test_split(X, y, test_size=0.2)

    #chromosome (param) representation
    n_estimators = [int(x) for x in np.linspace(1,10,1)]
    criterion = ['gini', 'entropy']
    max_depth = [int(x) for x in np.linspace(1,10,1)]
    max_features = ['auto', 'sqrt', 'log2']
    max_samples = [int(x) for x in np.linspace(1,10,1)]
    params = {'n_estimators': n_estimators, 'criterion': criterion, 'max_depth': max_depth, 'max_features': max_features, 'max_samples': max_samples}

    #optimizer
    optimizer = GA_OPT(params, data, pop_size=20, n_generations=50)
    optimizer.run()
