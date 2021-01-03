from ga.ga_hlm import *
from hlm_basic.ssn import GenerateNetwork, UpstreamArea
from hlm_basic.watershed import Watershed
from hlm_basic.tools import GetForcing, Set_InitialConditions, plot_sim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from fitness_scenarios import FitnessCalculator_Scenario_1_b
import pickle

def RunSimulation(args):
    ''' Used to run simulation for a given initial time and lead time.
        Output is used to calculate fitness function
    parameters:
        args:list, includes "Watershed" object, t0, forcing, dam_parameters, t_next(lead_time)
    '''
    object, state , t0, forcing, dam_parameters, t_next = args
    object.set_dam_state(states=state)
    discharge, storage = object.Run_256( [t0, t0+t_next], forcing, dam_parameters)
    flow_max = discharge.max(axis=0)
    volume_max = storage.max(axis=0)
    return [flow_max, volume_max]



if __name__ == "__main__":

    start = time.time()

    #Artificial Watershed Properties
    l_id, connectivity, h_order, nextlink = GenerateNetwork(5)
    n_hills = len(connectivity)
    a_hill = 0.5 * 0.5 #km2
    A_h = np.array([a_hill*10**6 for i in range(n_hills)]) #m2
    L_i = np.array([0.5*10**3 for i in range(n_hills)])  #m
    A_i = UpstreamArea(a_hill, connectivity, h_order) #km2

    #Precipitation data and simulation Duration(te-1)
    forcing, raw_data = GetForcing("/Users/gurbuz/DamStudy/data/rainfall/2010_timeseries.csv", '2010-06-01','2010-08-01')
    te = len(forcing)-1

    #Dam location and design properties
    dams = [9,27,36,45,63,90,108,117,126,135,144,153,171,189,198,207,216,225,234] 
    order_3 = [9,36,45,63,90,117,126,144,153,171,198,207,225,234] 
    order_4 = [27, 189, 216, 135, 108]
    n_dams = len(dams)

    # Parameters of each dam (For Model 256)
    _alpha = [0.5 for _ in range(n_dams)]
    c1 = [0.6 for _ in range(n_dams)]
    c2 = [3.0 for _ in range(n_dams)]

    H_spill = []
    H_max = []
    diam = []
    S_max = []
    L_spill = []
    L_crest = []
    for dam in dams:
        if dam in order_3:
            H_spill.append(4.5)
            H_max.append(5)
            diam.append(1.0)
            S_max.append(200000)
            L_spill.append(2.0)
            L_crest.append(5.0)
        elif dam in order_4:
            H_spill.append(4.5)
            H_max.append(5.0)
            diam.append(1.0)
            S_max.append(300000)
            L_spill.append(4.0)
            L_crest.append(10.0)

    # No dam scenario
    print('Running nodam case..')
    SSN1 = Watershed(Model=254)
    SSN1.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    q, s_p, s_t, s_s = Set_InitialConditions(0.5, A_i[0], A_i)
    SSN1.initialize(q=q, s_p=s_p, s_t=s_t, s_s=s_s)
    dc_nodam = SSN1.Run_254( [0, te],forcing, rtol=1e-6,)



    #Passive Scenario
    print('Running passive case..')
    SSN2 = Watershed(Model=256)
    SSN2.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN2.dam_ids = dams
    dam_params256 = SSN2.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
    q, s_p, s_t, s_s = Set_InitialConditions(0.5, A_i[0], A_i)
    S = [100000 for _ in range(n_dams)]
    SSN2.set_dam_state(states=[1 for _ in range(n_dams)])
    SSN2.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)
    dc_passive, st_passive = SSN2.Run_256( [0, te], forcing, dam_params256)



    #Active Control with GA
    print('Running active control case')
    # watershed object initialization
    SSN5 = Watershed(Model=256)
    SSN5.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN5.dam_ids = dams
    dam_params256 = SSN5.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
    q, s_p, s_t, s_s = Set_InitialConditions(0.5, A_i[0], A_i)
    S = [100000 for _ in range(n_dams)]
    # continuous simulation
    update = 60 #mins
    lead_time = 1080 # the time window checked if there will be any flooding 
    lead_time_opt = 360 # the time window used in optimization procedure 
    t0 = 0
    states_all = [(-60, [1 for _ in range(len(dams))])]
    columns = SSN5.__columns__()
    dc_ga = pd.DataFrame(columns =columns[0])
    st_ga  = pd.DataFrame(columns =columns[1])
    while t0 < te-lead_time:
        if t0 !=0:
            q, S, s_p, s_t, s_s = SSN5.Get_Snapshot()
        
        #define initial conditions
        SSN5.initialize(q=q, S=S, s_p=s_p,s_t =s_t, s_s=s_s)
        #check if flooding occurs
        data = RunSimulation([SSN5,[1 for _ in range(n_dams)] , t0, forcing, dam_params256, lead_time])
        flow = data[0]

        if flow['0']>32.50/2 or flow['81']>17.50/2 or  flow['162']>17.50/2:

            population = InitialPopulation2(16, n_dams)
            fitness_all = []
            generation = 0
            while generation<25: # generation
                results = []
                for dam_state in population:
                    sim = RunSimulation([SSN5, dam_state , t0, forcing, dam_params256,lead_time_opt])              
                    results.append(sim)
                state_previous = states_all[-1][1]
                fitnesses = FitnessCalculator_Scenario_1_b(results)
                idx = np.argmax(fitnesses)
                fitness_all.append(fitnesses[idx])
                parents = MatingPoolSelection(population, fitnesses, n_parents=None, selection='best')
                offsprings = Crossover(parents, operator='uniform')        
                offsprings_mutated = MutateOffspring(offsprings, method='scrample', p=0.10)
                population = NewPopulation(parents, offsprings_mutated)
                ##termination
                if generation > 8:
                    sub_fitness = fitness_all[-8:]
                    if len(set(sub_fitness)) == 1:
                        print(f'[+] Search terminated at generation {generation}')
                        break
                generation +=1
            state = population[idx].astype(float).tolist()
            print('[+]', t0, ' >>> ', state, 'fitness>>', fitnesses[idx])

        else:

            state = [1 for _ in range(n_dams)]
            print(' [-]',t0, ' >>> ', state, )

        states_all.append((t0,state))
        SSN5.set_dam_state(states=state)

        try:
            dc, st = SSN5.Run_256([t0, t0+update], forcing, dam_params256,)
        except IndexError:
            break
        
        t0 += update
        
        dc_ga = dc_ga.append(dc)
        st_ga = st_ga.append(st)
    print('Simulations completed!')



    path = '/Users/gurbuz/Supp_DamStudy/'
    s_name = '_Scenario1b_LT_1080_360_U60' 
    #Save data
    print('Saving data..')
    dc_ga.to_csv(path+'Discharge_'+s_name+'.csv', index=True)
    st_ga.to_csv(path+'Storage_'+s_name+'.csv', index=True)
    with open(path+'STATES_'+s_name+'.pickle', 'wb') as file:
        pickle.dump(states_all, file)

    #Plot and save figures
    print('Saving plots...')
    pltkwargs = np.array([{'label':'nodam', 'color':'#1AFF1A'}, {'label':'passive', 'color':'#000000',}, 
                        {'label':'GA', 'color':'magenta','linewidth':2.5, 'linestyle':'dashdot'}])
    pltKwargs = pltkwargs[[0,1,2]]
    dataset = [dc_nodam, dc_passive, dc_ga]
    plot_sim(0, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,225,50], area=60.75, save=path + 'Hydro0'+s_name)
    plot_sim(81, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,90,20], area=20.25, save=path + 'Hydro81'+s_name)
    plot_sim(162, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,90,20], area=20.25, save=path + 'Hydro162'+s_name)
    plot_sim(35, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save=path + 'Hydro35'+s_name)
    plot_sim(197, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save=path + 'Hydro197'+s_name)
    plot_sim(224, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save=path + 'Hydro224'+s_name)
    plot_sim(143, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save=path + 'Hydro143'+s_name)


    pltKwargs = pltkwargs[[1, 2]]
    dataset = [st_passive,  st_ga]
    plot_sim(order_3, forcing, dataset, pltKwargs, d_type='storage', max_storage=200000, save=path + 'DAM_3'+s_name)
    plot_sim(order_4, forcing, dataset, pltKwargs, d_type='storage', max_storage=300000, save=path + 'DAM_4'+s_name)

    end = time.time()
    print(f'***Done in {end - start} seconds.')