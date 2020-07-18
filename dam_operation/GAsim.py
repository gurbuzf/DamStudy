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
        # storage = data[1]
        # dam_over = storage[storage>=200000].count() 
        # fitness -= dam_over * 10
        # fitness += -0.0454*flow['0']**2 + 1.905*flow['0'] - 10
        # fitness += -0.1653*flow['81']**2 + 3.633 * flow['81'] -10 
        # fitness += -0.1653*flow['162']**2 + 3.633 * flow['162'] -10 
        if flow[0] <1: flow[0]=1 
        fitness += 1/flow[0]
        fitnesses = np.append(fitnesses, fitness)
    return fitnesses

def RunSimulation(object, state , t0, forcing):
    dam_params = [4.5, 5, 200000, 0.5, 0.75, 0.6, 1.66, 1, 10]
    object.dam_loc_state(states=state)
    dc_temp, st_temp = object.Run_255( [t0, t0+600-1], forcing, dam_params,rtol=1e-4)
    flow_max = dc_temp.max(axis=0)
    volume_max = st_temp.max(axis=0)
    return [flow_max, volume_max]


def main():
    #Artificial catchment properties
    l_id, connectivity, h_order, nextlink = GenerateNetwork(5)
    n_hills = len(connectivity)
    a_hill = 0.5 * 0.5 #km2
    A_h = np.array([a_hill*10**6 for i in range(n_hills)]) #m2
    L_i = np.array([0.5*10**3 for i in range(n_hills)])  #m
    A_i = UpstreamArea(a_hill, connectivity, h_order) #km2

    # Location of dams and initial condition
    dams = [9,27,36,45,63,90,108,117,126,135,144,153,171,189,198,207,216,225,234] 
    n_dams = len(dams)
    S = [100000 for _ in range(n_dams)]
    
    # Forcing and simulation duration
    forcing, raw_data = GetForcing("C:/Users/gurbuz/Desktop/DamStudy/data/rainfall/2010_timeseries.csv", '2010-06-01','2010-08-01')
    te = len(forcing)

    # watershed instance
    SSN5 = Watershed(Model=255)
    SSN5.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN5.dam_ids = dams

    #dam parameters and initial conditions for channel hillslope states
    dam_params = [4.5, 5, 200000, 0.5, 0.75, 0.6, 1.66, 1, 10]
    q, s_p, s_t, s_s = Set_InitialConditions(0.5, A_i[0], A_i)

    #Simulation setup

    update = 60 #mins
    t0 = 0
    fitness_overtime = []
    columns = SSN5.__columns__()
    dc_ga = pd.DataFrame(columns =columns[0])
    st_ga  = pd.DataFrame(columns =columns[1])
    while t0 < te-600:
        if t0 !=0:
            q, S, s_p, s_t, s_s = SSN5.Get_Snapshot()

        population = InitialPopulation(10, n_dams)
        SSN5.initialize(q=q, S=S, s_p=s_p,s_t =s_t, s_s=s_s)
        data = RunSimulation(SSN5, population[0] , t0, forcing)
        flow = data[0]
        storage = data[1]

        if flow['0']>21:# or storage[storage>=200000].count() > 0 :

            generation = 0

            while generation < 10: # generation
                sim_data = []
                
                for state in population:
                    sim= RunSimulation(SSN5, state , t0, forcing)
                    sim_data.append(sim)

                fitnesses = FitnessCalculator(sim_data)
                idx = np.argmax(fitnesses)
            
                # if  fitnesses[idx]>27: 
                #     print('[+] Termination:',t0, fitnesses[idx]) 
                #     break

                parents = MatingPoolSelection(population, fitnesses, n_parents=None, selection='best', k=3)
                offsprings = Crossover(parents, operator='onepoint')
                offsprings = MutateOffspring(offsprings, method='bitflip', p=0.05)
                population = NewPopulation(parents, offsprings)
                generation +=1
            state = population[idx].astype(int).tolist()
            print(t0, ' >>> ', state, 'fitness >>>', fitnesses[idx])
        else:
            state = [1 for _ in range(n_dams)]

            print(t0, ' >>> ', state)
        # Set pre-determined gate states and run the siulation
        SSN5.dam_loc_state(states=state)
        dc, st = SSN5.Run_255([t0, t0+update-1], forcing, dam_params,rtol=1e-4,t_eval = np.arange(t0, t0+update, 5))
        t0 += update
        #Append the solutions
        dc_ga = dc_ga.append(dc)
        st_ga = st_ga.append(st)
    

    s_name = '_minOutlet_60update' 
    pltKwargs = np.array([{'label':'nodam', 'color':'#1AFF1A'}, {'label':'passive', 'color':'#000000',}, 
                            {'label':'random', 'color':'#b66dff'},{'label':'elfarol', 'color':'#db6d00', 'alpha':0.7},  
                            {'label':'GA', 'color':'#924900','alpha':0.7},{'label':'extra', 'color':'#490092'}])
    pltKwargs = pltKwargs[[3]]
    dataset = [dc_ga]
    plot_sim(0, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,225,50], area=60.75, save='C:/Users/gurbuz/Desktop/Supp_DamStudy/0'+s_name)
    plot_sim(81, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,90,20], area=20.25, save='C:/Users/gurbuz/Desktop/Supp_DamStudy/81'+s_name)
    plot_sim(162, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,90,20], area=20.25, save='C:/Users/gurbuz/Desktop/Supp_DamStudy/162'+s_name)
    plot_sim(35, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save='C:/Users/gurbuz/Desktop/Supp_DamStudy/35'+s_name)
    plot_sim(197, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save='C:/Users/gurbuz/Desktop/Supp_DamStudy/197'+s_name)
    plot_sim(143, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save='C:/Users/gurbuz/Desktop/Supp_DamStudy/143'+s_name)
    
    order_3 = [9,36,45,63,90,117,126,144,153,171,198,207,225,234] 
    order_4 = [27, 189, 216, 135, 108]
    dataset= [st_ga]
    plot_sim(order_3, forcing, dataset, pltKwargs, d_type='storage',save='C:/Users/gurbuz/Desktop/Supp_DamStudy/Order3'+s_name)
    plot_sim(order_4, forcing, dataset, pltKwargs, d_type='storage',save='C:/Users/gurbuz/Desktop/Supp_DamStudy/Order4'+s_name)

if __name__ == "__main__":
    main()