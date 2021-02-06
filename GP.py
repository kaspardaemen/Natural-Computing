# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:20:42 2021

@author: remco
"""

import deap.gp as gp
from deap import creator 
from deap import base
from deap import creator
from deap import tools
import numpy as np
import operator
import math
import random
import deap.algorithms as algorithms

data =[[-1.0, 0.0000],[-0.9, -0.1629],[-0.8, -0.2624],[-0.7, -0.3129],[-0.6, -0.3264],[-0.5, -0.3125],[-0.4, -0.2784],
        [-0.3, -0.2289], [-0.2, -0.1664],[-0.1, -0.0909],[0, 0.0],[0.1, 0.1111],[0.2, 0.2496],[0.3, 0.4251],[0.4, 0.6496],
        [0.5, 0.9375], [0.6, 1.3056],[0.7, 1.7731],[0.8, 2.3616],[0.9, 3.0951],[1.0, 4.0000]]

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
    
def protectedLog(base):
    try: 
        return math.log(base)
    except ValueError:
        return 1
    

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
#pset.addPrimitive(operator.neg, 1) #deze nodig?
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(protectedLog, 1)
pset.addEphemeralConstant("A", lambda: random.randint(-1,1))


pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    # sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    abserrors = (np.abs(func(x[0]) - x[1]) for x in points)
    #print(points)
    return math.fsum(abserrors) / len(points),

toolbox.register("evaluate", evalSymbReg, points=data)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)
    
pop = toolbox.population(n=1000)
hof = tools.HallOfFame(1) 
pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0, 50, stats=mstats,
                                       halloffame=hof, verbose=True)

#print(pop)
print(len(hof))
print(hof[0])

#plot min fitness (a) 
#plot min size (b)
