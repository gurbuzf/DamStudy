from ga.ga_hlm import *
from hlm_basic.ssn import GenerateNetwork, UpstreamArea
from hlm_basic.watershed import Watershed
from hlm_basic.tools import GetForcing, Set_InitialConditions, plot_sim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def FitnessCalculator(sim_data):
    '''Calculates fitness of the individuals'''
    fitnesses = np.array([])
    for data in sim_data:
        fitness = 0
        flow = data[0]
        storage = data[1]
        dam_over = storage[storage>=200000].count() 
        fitness -= dam_over * 50
        fitness += (len(storage)-dam_over) * 50
        if flow['0'] > 30:
                fitness -= flow['0']/30 *100
        else: fitness += 100

        if flow['81'] > 15:
            fitness -= flow['81']/15 *100
        else: fitness += 100

        if flow['162'] > 15:
            fitness -= flow['162']/15 *100
        else: fitness += 100
        # links2opt = [0,27,54,81,108,135,162,189,216]
        # fitness += 1/sum(flow[links2opt].values/A_i[links2opt])
        fitnesses = np.append(fitnesses, fitness)

    return fitnesses

def RunSimulation(object, state , t0, forcing):
    dam_params = [4.5, 5, 200000, 0.5, 0.75, 0.6, 1.66, 1, 10]
    object.dam_loc_state(states=state)
    dc_test, st_test = object.Run_255( [t0, t0+1080-1], forcing, dam_params,rtol=1e-4)
    flow_max = dc_test.max(axis=0)
    volume_max = st_test.max(axis=0)
    return [flow_max, volume_max]#, [dc_test, st_test]


def main():

    l_id, connectivity, h_order, nextlink = GenerateNetwork(5)
    n_hills = len(connectivity)
    a_hill = 0.5 * 0.5 #km2
    A_h = np.array([a_hill*10**6 for i in range(n_hills)]) #m2
    L_i = np.array([0.4*10**3 for i in range(n_hills)])  #m
    A_i = UpstreamArea(a_hill, connectivity, h_order) #km2

    dams = [9,27,36,45,63,90,108,117,126,135,144,153,171,189,198,207,216,225,234] 
    n_dams = len(dams)
    S = [100000 for _ in range(n_dams)]
    forcing, raw_data = GetForcing("C:/Users/gurbuz/Desktop/DamStudy/data/rainfall/2010_timeseries.csv", '2010-06-01','2010-08-01')
    te = len(forcing)

    SSN5 = Watershed(Model=255)
    SSN5.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN5.dam_ids = dams
    dam_params = [4.5, 5, 200000, 0.5, 0.75, 0.6, 1.66, 1, 10]
    q, s_p, s_t, s_s = Set_InitialConditions(0.5, A_i[0], A_i)

    update = 60 #mins
    t0 = 0
    fitness_overtime = []
    columns = SSN5.__columns__()
    dc_ga = pd.DataFrame(columns =columns[0])
    st_ga  = pd.DataFrame(columns =columns[1])
    while t0 < te-1080:
        if t0 !=0:
            q, S, s_p, s_t, s_s = SSN5.Get_Snapshot()
        

        # if t0==4200:
        population = InitialPopulation(8, n_dams)
        SSN5.initialize(q=q, S=S, s_p=s_p,s_t =s_t, s_s=s_s)
        data = RunSimulation(SSN5, population[0] , t0, forcing)
        flow = data[0]
        if flow['0']>30 or flow['81']>15 or flow['162']>15:
            # state = [0 for _ in range(n_dams)]

            generation = 0
            # sim_results = []
            while generation<10: # generation
                sim_data = []
                
                for state in population:
                    sim= RunSimulation(SSN5, state , t0, forcing)
                    sim_data.append(sim)
                    # sim_results.append((state,sim_result))
                fitnesses = FitnessCalculator(sim_data)

                idx = np.argmax(fitnesses)
                fitness_overtime.append(fitnesses[idx])
                if  fitnesses[idx]==1000: 
                    print(fitnesses[idx]) 
                    break
                parents = MatingPoolSelection(population, fitnesses, n_parents=None, selection='best', k=3)
                offsprings = Crossover(parents, operator='onepoint')
                offsprings = MutateOffspring(offsprings, method='bitflip', p=0.05)
                population = NewPopulation(parents, offsprings)
                generation +=1
            state = population[idx].astype(int).tolist()
        else:
            state = [1 for _ in range(n_dams)]

        print(t0, ' >>> ', state)
        SSN5.dam_loc_state(states=state)
        # SSN5.initialize(q=q, S=S, s_p=s_p,s_t =s_t, s_s=s_s)
        dc, st = SSN5.Run_255([t0, t0+update-1], forcing, dam_params,rtol=1e-4,t_eval = np.arange(t0, t0+update, 5))
        t0 += update
        dc_ga = dc_ga.append(dc)
        st_ga = st_ga.append(st)
    

    s_name = '_testGA' 
    pltKwargs = np.array([{'label':'nodam', 'color':'#1AFF1A'}, {'label':'passive', 'color':'#000000',}, 
                            {'label':'random', 'color':'#b66dff'},{'label':'elfarol', 'color':'#db6d00', 'alpha':0.7},  
                            {'label':'GA', 'color':'#924900','alpha':0.7},{'label':'extra', 'color':'#490092'}])
    pltKwargs = pltKwargs[[2]]
    dataset = [dc_ga]
    plot_sim(0, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,225,50], area=60.75, save='C:/Users/gurbuz/Desktop/Supp_DamStudy/TEST'+s_name)
    

if __name__ == "__main__":
    main()