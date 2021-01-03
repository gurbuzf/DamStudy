###########################################
#                                         #
# Author: Faruk Gurbuz DATE:6-26-2020     #
# email:faruk-gurbuz@uiowa.edu            #
###########################################

import numpy as np # version > 1.8
import pandas as pd
import matplotlib.pyplot as plt

def InitialPopulation(n_chromosomes, n_genes):
    '''
    Returns a 2D numpy array consisting of binary string 
    with a size of (n_chromosomes, n_genes)
    '''
    init_pop = np.random.randint(low=0, high=2, size = (n_chromosomes, n_genes))
    np.random.shuffle(init_pop)
    init_pop[0] = np.array([1 for _ in range(n_genes)])
    return init_pop

def InitialPopulation2(n_chromosomes, n_genes):
    '''
    Returns a 2D numpy array consisting of numbers in [0, 0.25, 0.50, 0.75, 1.0] 
    with a size of (n_chromosomes, n_genes)
    '''
    init_pop = np.random.choice([0, 0.25, 0.50, 0.75, 1], size=(n_chromosomes, n_genes))
    np.random.shuffle(init_pop)
    init_pop[0] = np.array([1 for _ in range(n_genes)])
    return init_pop

def MatingPoolSelection(population, fitnesses, n_parents=None, selection='best', k=3):
    ''' Returns parents for cross-over
    
    WARNING: This code is based on higher fitness values. If minimization problem is the case, arrange 'fitnesses' array accordingly.
             Be carefull when using tournament selection. It can get stuck in an infinite loop in some cases. 
    INPUT:

        population: 2D np array, size:(n_chromosomes, n_genes)

        fitnesses : a np vector, size:(1, n_chromosomes), includes fitnesses of individuals(chromosomes)

        n_parents: int, default:population.shape[0]/2 (floor if even), number of parents to be selected

        selection: str, default:'best', parent selection algorithms, 
                    options: best, selects best individuals from population for mating
                             roulette: selects individuals using Roulette Wheel Selection algorithm
                             tournament : selects individuals using K-way tournament selection algorithm 
                             rank : simple linear selection algoritm   
                             
        k: int, default:3, number of individuals in tournament
    '''
    N = population.shape[0]

    if n_parents==None: n_parents = np.floor(N/2).astype(int)

    if selection == 'best':
        idx = np.argpartition(fitnesses, - n_parents)[-n_parents:]
        indices = idx[np.argsort((-fitnesses)[idx])]
        # return indices

    elif selection == 'roulette':
        fitness_offset = fitnesses.copy()
        if fitnesses.min() < 0: # make it work when negative values exist in fittnesses
            offset = fitnesses.min()
            fitness_offset = fitnesses + abs(offset)
        total_fitness = np.sum(fitness_offset)
        #TODO: probaility calculation may results in Error. Sometime P becomes None when total_fitness is zero #DONE!
        relative_fitness = fitness_offset/total_fitness
        relative_fitness[np.isnan(relative_fitness)] = 0 #set 0 if None value comes out from division
        try:
            indices = np.random.choice(N, n_parents, p=relative_fitness, replace=False)
        except ValueError:
            relative_fitness[relative_fitness==0]=0.001
            relative_fitness = relative_fitness/np.sum(relative_fitness)
            indices = np.random.choice(N, n_parents, p=relative_fitness, replace=False)

    elif selection == 'tournament':
        # may get stuck in an infinite loop if there exist multiple same fitness values.
        #In that case, use k=1 to make selections randomly. 
        indices = np.array([]).astype(int)
        while len(indices) < n_parents:
            tournament = np.random.choice(N, k, replace=False)
            idx = np.where(fitnesses == fitnesses[tournament].max())[0] # [0] used since np.where return tuple. 
            # if idx[0] not in indices: # 
            indices = np.append(indices, idx[0])
    elif selection == 'rank':
        ranks = np.zeros(N)
        sort_idx = np.argsort(fitnesses)
        ranks[sort_idx] = np.arange(0, N)+1
        prob_ranked = ranks/np.sum(ranks)
        indices = np.random.choice(N, n_parents, p=prob_ranked, replace=False)
    else:
        raise ValueError('Parent selection method not found!')

    return population[indices,:]

def Crossover(parents, operator='onepoint'):
    '''Returns offsprings by mating parents
    
    INPUT:
        parents:2D np array, output of 'MatingPoolSelection'

        operator:str, default:'onepoint', crossover method

                options: onepoint,  a point is randomly selected and the tails of 
                                    the two parents are swapped to produce offsprings

                        multipoint, three points are randomly selected and segments are 
                                    combined to get a new offspring

                        uniform, each gene of two parents are separately decided whether
                                    to put into the offspring. Ex. flipping coin for each gene
    '''
    n_parents = parents.shape[0]
    n_genes = parents.shape[1]

    if operator == 'onepoint':
        offsprings = np.zeros((n_parents, n_genes))
        for i in range(n_parents):
            point = np.random.randint(1, n_genes-1)
            idx_1 = (i) % n_parents
            idx_2 = (i+1) % n_parents
            offspring_left = parents[idx_1][:point]
            offspring_right = parents[idx_2][point:]
            offspring = np.hstack((offspring_left, offspring_right)).tolist()
            offsprings[i] = offspring

    elif operator == 'multipoint':
        mid_point = n_genes/2
        offsprings = np.zeros((n_parents, n_genes))
        for i in range(n_parents):
            point_1 = np.random.randint(1, mid_point)
            point_2 = np.random.randint(mid_point, n_genes-1)
            idx_1 = (i) % n_parents
            idx_2 = (i+1) % n_parents
            offspring = parents[idx_1].copy()
            offspring[point_1:point_2] = parents[idx_2][point_1:point_2]
            offsprings[i] = offspring

    elif operator == 'uniform':   
        offsprings = np.zeros((n_parents, n_genes))
        for i in range(n_parents):
            coin = np.random.choice([True, False], n_genes)
            gene_idx = np.arange(0, n_genes)[coin]
            idx_1 = (i) % n_parents
            idx_2 = (i+1) % n_parents
            offspring = parents[idx_1].copy()
            offspring[gene_idx] = parents[idx_2][gene_idx]
            offsprings[i] = offspring

    else:
        raise ValueError('Crossover operator not found!') 
   
    return offsprings


def MutateOffspring(offsprings, method='bitflip', p=0.05):
    '''Returns mutated offsprings

    INPUT:
        offsprings : 2D np array, output of 'Crossover'

        p:float, default:0.05, mutation occurs with a probability of p [0<p<1]

        method:str, default:'bitflip' , mutation method
                    options: bitflip, bit-wise mutation
                             swap,  swaps two randomly selected genes in the chromosome
                             scrample, shuffles genes between two randomly selected points
                             inversion, inverts entire string between two randomly selected points    
    '''
    offsprings_mutated = offsprings.copy()
    # n_offsprings = offsprings.shape[0]
    n_genes = offsprings.shape[1]

    if method == 'bitflip':
        for offspring in offsprings_mutated:
            r = np.random.random()
            if r < p:
                idx = np.random.randint(0,n_genes)
                offspring[idx] = 0 if offspring[idx] else 1 # simple bit-swap

    elif method == 'swap':
        for offspring in offsprings_mutated:
            r = np.random.random()
            if r < p:
                i1, i2  = np.random.choice(n_genes,2, replace=False)
                offspring[i1], offspring[i2] = offspring[i2], offspring[i1]
    
    elif method == 'scrample':
        for offspring in offsprings_mutated:
            r = np.random.random()
            if r < p:
                i1, i2  = np.sort(np.random.choice(n_genes,2, replace=False))
                np.random.shuffle(offspring[i1:i2])

    elif method == 'inversion':
        for offspring in offsprings_mutated:
            r = np.random.random()
            if r < p:
                i1, i2  = np.sort(np.random.choice(n_genes,2, replace=False))
                offspring[i1:i2] = np.flip(offspring[i1:i2]) 
    
    else:
        raise ValueError('Mutation method not found!')
    return offsprings_mutated

def NewPopulation(parents, offsprings_mutated):
    '''Returns new population consiting of parents and their mutated offsprings'''

    return np.vstack((parents, offsprings_mutated))

