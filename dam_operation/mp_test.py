from ga.ga_hlm import *
from hlm_basic.ssn import GenerateNetwork, UpstreamArea
from hlm_basic.watershed import Watershed
from hlm_basic.tools import GetForcing, Set_InitialConditions, plot_sim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

def Quad_Params2(threshold, reward):
    '''Parameters of a concave quadratic equation 
    eq: ax2 + bx + c
    returns the parameters a, b, c 
    '''
    a = -8 * reward / threshold ** 2
    b = 8 * reward / threshold
    c = -reward
    return a, b, c

eq235 = Quad_Params2(21.2,10)
eq217 = Quad_Params2(20.2, 10)
eq181 = Quad_Params2(18.2, 10)
eq73 = Quad_Params2(10.5, 10)
eq55 = Quad_Params2(8, 10)
eq19 = Quad_Params2(5, 10)

def FitnessCalculator1(sim_data):
    global eq235, eq217, eq181, eq73, eq55, eq19
    fitnesses = np.array([])
    for data in sim_data:
        fitness = 0
        flow = data[0]
        storage = data[1]
        order_3 = ['9','36','45','63','90','117','126','144','153','171','198','207','225','234'] 
        order_4 = ['27', '189', '216', '135', '108']
        dam3_over = (storage[order_3]>200000).values.sum()
        dam4_over = (storage[order_4]>300000).values.sum()

        fitness += -dam3_over*120 -dam4_over*120
        fitness += eq235[0]*flow['8']**2 + eq235[1]*flow['8'] + eq235[2]
        fitness += eq217[0]*flow['26']**2 + eq217[1]*flow['26'] + eq217[2]
        fitness += eq181[0]*flow['62']**2 + eq181[1]*flow['62'] + eq181[2]
        fitness += eq73[0]*flow['89']**2 + eq73[1]*flow['89'] + eq73[2]
        fitness += eq73[0]*flow['170']**2 + eq73[1]*flow['170'] + eq73[2]
        fitness += eq55[0]*flow['107']**2 + eq55[1]*flow['107'] + eq55[2]
        fitness += eq55[0]*flow['188']**2 + eq55[1]*flow['188'] + eq55[2]
        fitness += eq19[0]*flow['35']**2 + eq19[1]*flow['35'] + eq19[2]
        fitness += eq19[0]*flow['143']**2 + eq19[1]*flow['143'] + eq19[2]
        fitness += eq19[0]*flow['116']**2 + eq19[1]*flow['116'] + eq19[2]
        fitness += eq19[0]*flow['197']**2 + eq19[1]*flow['197'] + eq19[2]
        fitness += eq19[0]*flow['234']**2 + eq19[1]*flow['234'] + eq19[2]

        fitnesses = np.append(fitnesses, fitness)
        # print('fitness in Calculator: ', fitness )
    return fitnesses

def RunSimulation(args):

    object, state, t0, forcing, dam_parameters, t_next = args
    object.set_dam_state(states=state)
    dc_test, st_test = object.Run_256( [t0, t0+t_next], forcing, dam_parameters)
    flow_max = dc_test.max(axis=0)
    # volume_max = st_test.max(axis=0)
    volume_last = st_test.iloc[-1]
    return [flow_max, volume_last]#, [dc_test, st_test]



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
    SSN1 = Watershed(Model=254)
    SSN1.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    q, s_p, s_t, s_s = Set_InitialConditions(0.5, A_i[0], A_i)
    SSN1.initialize(q=q, s_p=s_p, s_t=s_t, s_s=s_s)
    dc_nodam = SSN1.Run_254( [0, te],forcing, rtol=1e-6,)



    #Passive Scenario
    SSN3 = Watershed(Model=256)
    SSN3.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN3.dam_ids = dams
    dam_params256 = SSN3.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
    q, s_p, s_t, s_s = Set_InitialConditions(0.5, A_i[0], A_i)
    S = [100000 for _ in range(n_dams)]
    SSN3.set_dam_state(states=[1 for _ in range(n_dams)])
    SSN3.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)
    dc_passive, st_passive = SSN3.Run_256( [0, te], forcing, dam_params256)



    #Active Control with GA
    SSN5 = Watershed(Model=256)
    SSN5.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN5.dam_ids = dams
    dam_params256 = SSN5.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
    q, s_p, s_t, s_s = Set_InitialConditions(0.5, A_i[0], A_i)
    S = [100000 for _ in range(n_dams)]

    update = 60#mins
    t0 = 0
    fitness_overtime = []
    columns = SSN5.__columns__()
    dc_ga = pd.DataFrame(columns =columns[0])
    st_ga  = pd.DataFrame(columns =columns[1])

    while t0 < te-300:

        if t0 !=0:
            q, S, s_p, s_t, s_s = SSN5.Get_Snapshot()
        
        population = InitialPopulation2(10, n_dams)
        SSN5.initialize(q=q, S=S, s_p=s_p,s_t =s_t, s_s=s_s)
        data = RunSimulation([SSN5, population[0] , t0, forcing, dam_params256, 300])
        flow = data[0]

        if flow['8']>31.80/2 or flow['26']>30.40/2 or flow['62']>27.4/2 or flow['89']>16.33/2 or flow['170']>16.33/2:
    
            generation = 0
            while generation<10: # generation
               
                arguments = [[SSN5, dam_state, t0, forcing, dam_params256, 120] for dam_state in population]
                results = []
                for argument in arguments:
                    results.append(RunSimulation(argument))
                fitnesses = FitnessCalculator1(results)
                idx = np.argmax(fitnesses)
                fitness_overtime.append(fitnesses[idx])
                parents = MatingPoolSelection(population, fitnesses, n_parents=None, selection='best', k=3)
                offsprings = Crossover(parents, operator='uniform')           
                offsprings = MutateOffspring(offsprings, method='scrample', p=0.10)
                population = NewPopulation(parents, offsprings)
                generation +=1

            state = population[idx].astype(int).tolist()
            print('[+]', t0, ' >>> ', state, 'fitness>>', fitnesses[idx])
        else:
            state = [1 for _ in range(n_dams)]
            print(' [-]',t0, ' >>> ', state, )
        
        SSN5.set_dam_state(states=state)
        try:
            dc, st = SSN5.Run_256([t0, t0+update], forcing, dam_params256,)
        except IndexError:
            break
        t0 += update
        dc_ga = dc_ga.append(dc)
        st_ga = st_ga.append(st)


    end= time.time()
    print(f'***Simulations Completed in {end -start} seconds.')

    path = '/Users/gurbuz/Supp_DamStudy/'

    s_name = '_test1_LD_300_120_U60' 
    pltkwargs = np.array([{'label':'nodam', 'color':'#1AFF1A'}, {'label':'passive', 'color':'#000000',}, 
                        {'label':'GA', 'color':'magenta','alpha':1.0, 'linestyle':'dashdot', 'linewidth':3}])
    pltKwargs = pltkwargs[[0,1,2]]
    dataset = [dc_nodam, dc_passive, dc_ga]
    plot_sim(0, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,225,50], area=60.75, save=path + 'Hydro0'+s_name)
    plot_sim(81, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,90,20], area=20.25, save=path + 'Hydro81'+s_name)
    plot_sim(162, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,90,20], area=20.25, save=path + 'Hydro162'+s_name)
    plot_sim(35, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save=path + 'Hydro35'+s_name)
    plot_sim(197, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save=path + 'Hydro197'+s_name)
    plot_sim(224, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save=path + 'Hydro224'+s_name)
    plot_sim(143, forcing, dataset, pltKwargs, d_type='discharge', discharge_axis=[0,20,5], area=4.75, save=path + 'Hydro143'+s_name)


    order_3 = [9,36,45,63,90,117,126,144,153,171,198,207,225,234] 
    order_4 = [27, 189, 216, 135, 108]
    pltKwargs = pltkwargs[[1, 2]]
    dataset = [ st_passive,  st_ga]
    plot_sim(order_3, forcing, dataset, pltKwargs, d_type='storage', max_storage=200000, save=path + 'DAM_3'+s_name)
    plot_sim(order_4, forcing, dataset, pltKwargs, d_type='storage', max_storage=300000, save=path + 'DAM_4'+s_name)