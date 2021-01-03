from ga.ga_hlm import *
from hlm_basic.ssn import GenerateNetwork, UpstreamArea
from hlm_basic.watershed import Watershed
from hlm_basic.tools import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import pickle
from multiprocessing import Pool

def RunSimulation(args):
    ''' Used to run simulation for a given initial time and lead time.
        Output is used to calculate fitness function
    parameters:
        args:list, includes "Watershed" object, t0, forcing, dam_parameters, t_next(lead_time)
    '''
    object, state , t0, forcing, dam_parameters, t_next = args
    object.set_dam_state(states=state)
    discharge, storage = object.Run_256( [t0, t0+t_next], forcing, dam_parameters)
    flow_max = discharge.iloc[-1]##max(axis=0)#
    volume_max = storage.max(axis=0) #iloc[-1]#
    return [flow_max, volume_max]

def Quad_Params2(threshold, reward):
    '''Parameters of a concave quadratic equation 
    -c is the reward corresponding to the mean annual flow.
    eq: ax2 + bx + c
    returns the parameters a, b, c 
    '''
    a = -8 * reward / threshold ** 2
    b = 8 * reward / threshold
    c = -reward
    return a, b, c


# eq1_5 = Quad_Params2(2, 10)
# eq3_5 = Quad_Params2(4, 10)
# eq7 = Quad_Params2(8, 10)
# eq15 = Quad_Params2(18, 10)


eq1_5 = Quad_Params2(3, 10)
eq3_5 = Quad_Params2(5, 10)
eq7 = Quad_Params2(10, 10)
eq15 = Quad_Params2(20, 10)


def FitnessCalculator_final(sim_data, population, previous_state):
    ''' Fitness function for scenario 3(b). A concave equation is used to determine fitnesses.
    The objective is to maintain streamflow at a steady level. This level is the half of 
    Mean Annual Flood at the location of interest. All the links right downstream of the dams 
    are used to calculate fitness. 

    As input, use maximum streamflow in the pre-defined lead time.
    
    Note:Dam overtopping is penalized.
    '''
    global eq1_5, eq3_5, eq7, eq15 
  
    fitnesses = np.array([])
    for i, data in enumerate(sim_data):
        fitness = 0
        flow = data[0]

        fitness += eq1_5[0]*flow['35']**2 + eq1_5[1]*flow['35'] + eq1_5[2]
        fitness += eq1_5[0]*flow['116']**2 + eq1_5[1]*flow['116'] + eq1_5[2]
        fitness += eq1_5[0]*flow['143']**2 + eq1_5[1]*flow['143'] + eq1_5[2]
        fitness += eq1_5[0]*flow['197']**2 + eq1_5[1]*flow['197'] + eq1_5[2]
        fitness += eq1_5[0]*flow['224']**2 + eq1_5[1]*flow['224'] + eq1_5[2]
        
        fitness += eq3_5[0]*flow['107']**2 + eq3_5[1]*flow['107'] + eq3_5[2]
        fitness += eq3_5[0]*flow['188']**2 + eq3_5[1]*flow['188'] + eq3_5[2]
        
        fitness += eq7[0]*flow['81']**2 + eq7[1]*flow['81'] + eq7[2]
        fitness += eq7[0]*flow['162']**2 + eq7[1]*flow['162'] + eq7[2]

        fitness += eq15[0]*flow['0']**2 + eq15[1]*flow['0'] + eq15[2]
   
        #NO PENALTY IN FINAL RESULTS
        # ref = np.array(previous_state) / 0.25
        # statein = np.array(population[i]) / 0.25
        # diff = np.abs(statein-ref) - 1
        # diff[diff<0] = 0
        # fitness -= np.sum(diff) *5 # Penalty for unstable state changes
        fitnesses = np.append(fitnesses, fitness)
    return fitnesses


def InitialPop(n_chromosomes, n_genes):
    '''
    Returns a 2D numpy array consisting of numbers in [0, 0.25, 0.50, 0.75, 1.0] 
    with a size of (n_chromosomes, n_genes)
    '''
    init_pop = np.random.choice([0, 0.25, 0.50, 0.75, 1], size=(n_chromosomes, n_genes)) #,np.arange(0,1.1,0.1).round(1),
    np.random.shuffle(init_pop)
    # init_pop[0] = np.array([1 for _ in range(n_genes)])
    return init_pop


if __name__ == "__main__":
    
    # Properties of artificial watershed
    l_id, connectivity, h_order, nextlink = GenerateNetwork(5)
    n_hills = len(connectivity)
    a_hill = 0.5 * 0.5 #km2
    A_h = np.array([a_hill*10**6 for i in range(n_hills)]) #m2
    L_i = np.array([0.5*10**3 for i in range(n_hills)])  #m
    A_i = UpstreamArea(a_hill, connectivity, h_order) #km2

#############################################################################################################################
    d_storm = 175
    save_ext = f'{d_storm}_e6'
    rate10 = [ 12,  30,  63,  75,  84,  87 ,90,  93.5,  95,  96.01,  97.44,  97.44 ,  99.42,  99.42, 99.67,  100] # resolution 30min
    rate60 =np.array([  0,   5.95,   7.03,   7.03,   7.3 ,  10.27,  15.95,  15.95,
        19.46,  19.46,  22.97, 30.54,  33,  33,  33, 33,
        33.24,  35.41,  35.41,  38.11,  38.11,  38.92,  50  ,  53,
        53.59,  54.76,  54.76,  54.76,  62, 65,  66,  70,
        71,  71.78,  73,  73.22,  73.73,  74.08,  80,  86,
        87.35,  89.44,  89.44,  95,  98.33  ,  100,  100, 100.  ])
    forcing, cum_forcing, forcing_hour = Generate_SyntheticStorm(d_storm, 120, rate=(0.01*np.diff(rate10, prepend=0)).tolist(), timescale=30)
    te = len(forcing)-1
#######################################################################################################
#Initial conditions
    with open('/Users/gurbuz/DamStudy/data/supplement/initial_condition.pickle', 'rb') as ini_file:
        initial_condition = pickle.load(ini_file)

#DAMS in scenario-5
    dams5 = [27, 189, 216, 135, 108,] + [9,36,45,63,90,117,126,144,153,171,198,207,225,234]   ## dams on order_3 and order_4 Note: dams on the links closer to the outlet included (SCENARIO-5)
    n_dams5 = len(dams5)

###################################################################################################
    start = time.time()
    print('Running No Dam..')
    SSN = Watershed(Model=256)
    SSN.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams5) #!! TODO: make the code run without this parameters when no dam introduced
    dam_params256 = SSN.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest) # !!
    q = initial_condition['q']
    s_p = initial_condition['s_p']
    s_t = initial_condition['s_t']
    s_s = initial_condition['s_s']
    SSN.initialize(q=q, s_p=s_p, s_t=s_t, s_s=s_s)
    SSN.set_dam_state()
    dc_nodam, _ = SSN.Run_256( [0, te],forcing, dam_params256,rtol=1e-6, )
    dc_nodam.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/dc_nodam_{save_ext}.csv')
    print(f'Done nodam in {time.time()-start}')
###################################################################################################
    start = time.time()
    print('Running Passive..')
    SSN5 = Watershed(Model=256)
    SSN5.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN5.dam_ids = dams5
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams5)
    dam_params256 = SSN5.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
    q = initial_condition['q']
    s_p = initial_condition['s_p']
    s_t = initial_condition['s_t']
    s_s = initial_condition['s_s']
    fill_percent = np.repeat([0.0001], n_dams5)
    S = (S_max * fill_percent).tolist()
    SSN5.set_dam_state(states=[1 for _ in range(n_dams5)])
    SSN5.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)
    dc_passive_S5, st_passive_S5 = SSN5.Run_256( [0, te], forcing, dam_params256)
    out_passive_S5 = SSN5.CalculateOutflow(dam_params256, st_passive_S5)
    dc_passive_S5.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/dc_passive_{save_ext}.csv')
    st_passive_S5.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/st_passive_{save_ext}.csv')
    out_passive_S5.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/out_passive_{save_ext}.csv')

    print(f'Done passive in {time.time()-start}')
############################################################
    start = time.time()
    print('Running Random..')
    SSN5_r = Watershed(Model=256)
    SSN5_r.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN5_r.dam_ids = dams5
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams5)
    dam_params256 = SSN5_r.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
    q = initial_condition['q']
    s_p = initial_condition['s_p']
    s_t = initial_condition['s_t']
    s_s = initial_condition['s_s']
    fill_percent = np.repeat([0.0001], n_dams5)
    S = (S_max * fill_percent).tolist()

    t0=0
    update = 30 #mins
    columns = SSN5_r.__columns__()
    dc_active_S5r = pd.DataFrame(columns =columns[0])
    st_active_S5r  = pd.DataFrame(columns =columns[1])
    out_active_S5r = pd.DataFrame(columns =columns[1])
    while t0 < te:
        if t0 !=0:
            q, S, s_p, s_t, s_s = SSN5_r.Get_Snapshot()
        
        SSN5_r.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)

        SSN5_r.set_dam_state(states=np.random.choice([0, 0.25, 0.50, 0.75, 1], size=(n_dams5)))

        try:
            dc_S5, st_S5 = SSN5_r.Run_256( [t0, t0+update], forcing, dam_params256)
            out_S5 = SSN5_r.CalculateOutflow(dam_params256, st_S5)
        except IndexError:
            pass

        dc_active_S5r = dc_active_S5r.append(dc_S5)
        st_active_S5r = st_active_S5r.append(st_S5)
        out_active_S5r = out_active_S5r.append(out_S5)
        t0 += update
    dc_active_S5r.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/dc_active_Random_{save_ext}.csv')
    st_active_S5r.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/st_active_Random_{save_ext}.csv')
    out_active_S5r.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/out_active_Random_{save_ext}.csv')
    print(f'Done random in {time.time()-start}')


 #########################################################################################################

    print('Runnning GA')
    start = time.time()
    SSN5_g = Watershed(Model=256)
    SSN5_g.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN5_g.dam_ids = dams5
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams5)
    dam_params256 = SSN5_g.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
    q = initial_condition['q']
    s_p = initial_condition['s_p']
    s_t = initial_condition['s_t']
    s_s = initial_condition['s_s']
    fill_percent = np.repeat([0.0001], n_dams5)
    S = (S_max * fill_percent).tolist()

    t0=0
    update = 30 #mins
    t_opt = 60
    states_all = [(-60, [1 for _ in range(n_dams5)])]

    columns = SSN5_g.__columns__()
    dc_active_S5g = pd.DataFrame(columns =columns[0])
    st_active_S5g  = pd.DataFrame(columns =columns[1])
    out_active_S5g = pd.DataFrame(columns =columns[1])
    while t0 < te-t_opt:
        if t0 !=0:
            q, S, s_p, s_t, s_s = SSN5_g.Get_Snapshot()
        SSN5_g.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)
        
        population = InitialPop(20, n_dams5)
        fitness_all = []
        generation = 0
        
        while generation<45: # generation
            time_gen = time.time()
            results = []
            # for dam_state in population:
            #     sim = RunSimulation([SSN1_g, dam_state , t0, forcing, dam_params256,60])              
            #     results.append(sim)
            # print(f'Population >\n {population}')
            arguments = [[SSN5_g, dam_state, t0, forcing, dam_params256, t_opt] for dam_state in population]
            results = []
            pool = Pool(processes=20)
            results = pool.map(RunSimulation, arguments, chunksize=1)
            pool.close()
            # for argument in arguments:
            #     results.append(RunSimulation(argument))
            state_previous = states_all[-1][1]
            fitnesses =  FitnessCalculator_final(results, population, state_previous)
            idx = np.argmax(fitnesses)
            fitness_all.append(fitnesses[idx])
            parents = MatingPoolSelection(population, fitnesses, n_parents=None, selection='best')
            offsprings = Crossover(parents, operator='onepoint')        
            offsprings_mutated = MutateOffspring(offsprings, method='scrample', p=0.05)

            population = NewPopulation(parents, offsprings_mutated)
            #termination
            if generation > 15:
                sub_fitness = fitness_all[-15:]
                if len(set(sub_fitness)) == 1:
                    print(f'[+] Search terminated at generation {generation}')
                    break
            print(f'generation:{generation} Fitness:{fitnesses[idx]} Time: {time.time()-time_gen}')
            generation +=1
        
        state = population[idx].astype(float).tolist()
        print('[+]', t0, ' >>> ', state, 'fitness>>', fitnesses[idx])
        states_all.append((t0,state))    
        SSN5_g.set_dam_state(states=state)

        try:
            dc_S5, st_S5 = SSN5_g.Run_256( [t0, t0+update], forcing, dam_params256)
            out_S5 = SSN5_g.CalculateOutflow(dam_params256, st_S5)
        except IndexError:
            pass

        dc_active_S5g = dc_active_S5g.append(dc_S5)
        st_active_S5g = st_active_S5g.append(st_S5)
        out_active_S5g = out_active_S5g.append(out_S5)
        t0 += update
    
    dc_active_S5g.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/dc_active_GA_{save_ext}.csv')
    st_active_S5g.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/st_active_GA_{save_ext}.csv')
    out_active_S5g.to_csv(f'/Users/gurbuz/Supp_DamStudy/final_ActiveControl/out_active_GA_{save_ext}.csv')
    print(f'Done GA in {time.time()-start}')






# #################################################################
#     SSN1 = Watershed(Model=256)
#     SSN1.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
#     SSN1.dam_ids = dams1
#     H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams1)
#     dam_params256 = SSN1.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
#     q = initial_condition['q']
#     s_p = initial_condition['s_p']
#     s_t = initial_condition['s_t']
#     s_s = initial_condition['s_s']
#     S = [50 for _ in range(len(dams1))]
#     SSN1.set_dam_state(states=[1 for _ in range(len(dams1))])
#     SSN1.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)
#     dc_passive_S1, st_passive_S1 = SSN1.Run_256( [0, te], forcing, dam_params256)
#     out_passive_S1 = SSN1.CalculateOutflow(dam_params256, st_passive_S1)
#     dc_passive_S1.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/dc_passive_S1.csv')
#     st_passive_S1.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/st_passive_S1.csv')
#     out_passive_S1.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/out_passive_S1.csv')
# ############################################################
#     start = time.time()
#     print('Running Random..')
#     SSN1_r = Watershed(Model=256)
#     SSN1_r.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
#     SSN1_r.dam_ids = dams1
#     H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams1)
#     dam_params256 = SSN1_r.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
#     q = initial_condition['q']
#     s_p = initial_condition['s_p']
#     s_t = initial_condition['s_t']
#     s_s = initial_condition['s_s']
#     S = [50 for _ in range(len(dams1))]

#     t0=0
#     update = 60 #mins
#     columns = SSN1_r.__columns__()
#     dc_passive_S1r = pd.DataFrame(columns =columns[0])
#     st_passive_S1r  = pd.DataFrame(columns =columns[1])
#     out_passive_S1r = pd.DataFrame(columns =columns[1])
#     while t0 < te:
#         if t0 !=0:
#             q, S, s_p, s_t, s_s = SSN1_r.Get_Snapshot()
        
#         SSN1_r.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)

#         SSN1_r.set_dam_state(states=np.random.choice([0, 0.25, 0.50, 0.75, 1], size=(n_dams1)))

#         try:
#             dc_S1, st_S1 = SSN1_r.Run_256( [t0, t0+update], forcing, dam_params256)
#             out_S1 = SSN1_r.CalculateOutflow(dam_params256, st_S1)
#         except IndexError:
#             pass

#         dc_passive_S1r = dc_passive_S1r.append(dc_S1)
#         st_passive_S1r = st_passive_S1r.append(st_S1)
#         out_passive_S1r = out_passive_S1r.append(out_S1)
#         t0 += update
#     dc_passive_S1r.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/dc_passive_S1r.csv')
#     st_passive_S1r.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/st_passive_S1r.csv')
#     out_passive_S1r.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/out_passive_S1r.csv')
#     print(f'Done random in {time.time()-start}')


#  #########################################################################################################

#     print('Runnning GA')
#     start = time.time()
#     SSN1_g = Watershed(Model=256)
#     SSN1_g.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
#     SSN1_g.dam_ids = dams1
#     H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams1)
#     dam_params256 = SSN1_g.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)
#     q = initial_condition['q']
#     s_p = initial_condition['s_p']
#     s_t = initial_condition['s_t']
#     s_s = initial_condition['s_s']
#     S = [50 for _ in range(len(dams1))]

#     t0=0
#     update = 30 #mins
#     columns = SSN1_g.__columns__()
#     dc_passive_S1g = pd.DataFrame(columns =columns[0])
#     st_passive_S1g  = pd.DataFrame(columns =columns[1])
#     out_passive_S1g = pd.DataFrame(columns =columns[1])
#     while t0 < te-360:
#         if t0 !=0:
#             q, S, s_p, s_t, s_s = SSN1_g.Get_Snapshot()
#         SSN1_g.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)
        
#         population = InitialPop(16, n_dams1)
#         fitness_all = []
#         generation = 0
        
#         while generation<25: # generation
#             time_gen = time.time()
#             results = []
#             # for dam_state in population:
#             #     sim = RunSimulation([SSN1_g, dam_state , t0, forcing, dam_params256,60])              
#             #     results.append(sim)
#             # print(f'Population >\n {population}')
#             arguments = [[SSN1_g, dam_state, t0, forcing, dam_params256, 30] for dam_state in population]
#             results = []
#             pool = Pool(processes=16)
#             results = pool.map(RunSimulation, arguments, chunksize=1)
#             pool.close()
#             # for argument in arguments:
#             #     results.append(RunSimulation(argument))

#             fitnesses = FitnessCalculator(results)
#             idx = np.argmax(fitnesses)
#             fitness_all.append(fitnesses[idx])
#             parents = MatingPoolSelection(population, fitnesses, n_parents=None, selection='best')
#             offsprings = Crossover(parents, operator='uniform')        
#             offsprings_mutated = MutateOffspring(offsprings, method='scrample', p=0.10)
#             population = NewPopulation(parents, offsprings_mutated)
#             ##termination
#             if generation > 8:
#                 sub_fitness = fitness_all[-8:]
#                 if len(set(sub_fitness)) == 1:
#                     print(f'[+] Search terminated at generation {generation}')
#                     break
#             print(f'generation:{generation} Time: {time.time()-time_gen}')
#             generation +=1
            
#         state = population[idx].astype(float).tolist()
#         print('[+]', t0, ' >>> ', state, 'fitness>>', fitnesses[idx])
#         SSN1_g.set_dam_state(states=state)

#         try:
#             dc_S1, st_S1 = SSN1_g.Run_256( [t0, t0+update], forcing, dam_params256)
#             out_S1 = SSN1_g.CalculateOutflow(dam_params256, st_S1)
#         except IndexError:
#             pass

#         dc_passive_S1g = dc_passive_S1g.append(dc_S1)
#         st_passive_S1g = st_passive_S1g.append(st_S1)
#         out_passive_S1g = out_passive_S1g.append(out_S1)
#         t0 += update
    
#     dc_passive_S1g.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/dc_passive_S1g.csv')
#     st_passive_S1g.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/st_passive_S1g.csv')
#     out_passive_S1g.to_csv('/Users/gurbuz/Supp_DamStudy/activecontrol/out_passive_S1g.csv')
#     print(f'Done random in {time.time()-start}')

# def FitnessCalculator(sim_data):
#     ''' Fitness function for scenario 3(a). A concave equation is used to determine fitness.
#     the objective is to maintain streamflow at a steady level. This level is the half of 
#     Mean Annual Flood at the location of interest. All the links right downstream of the dams 
#     are used to calculate fitness. 

#     As input, use maximum streamflow in the pre-defined lead time.
    
#     Note: No Penalty for dam overtopping
#     '''
#     order_5 =  ['81', '162']
#     global a ,b ,c
    
#     fitnesses = np.array([])
#     for data in sim_data:
#         fitness = 0
#         flow = data[0]
#         storage = data[1]
#         dam5_over = (storage[order_5]>325000).values.sum()
#         fitness += a*flow['80']**2 + b*flow['80'] + c
#         fitness -= dam5_over * 10
#         fitnesses = np.append(fitnesses, fitness)
#     return fitnesses
# def Reward(x, y, flow_norm, storage_norm, a=1, b=1):
#     return 10*np.exp(-a*(x/flow_norm)**2 - b*(y/storage_norm)**2)

# def FitnessCalculator1(sim_data):
#     ''' Fitness function for scenario 3(a). A concave equation is used to determine fitness.
#     the objective is to maintain streamflow at a steady level. This level is the half of 
#     Mean Annual Flood at the location of interest. All the links right downstream of the dams 
#     are used to calculate fitness. 

#     As input, use maximum streamflow in the pre-defined lead time.
    
#     Note: No Penalty for dam overtopping
#     '''
#     order_3 = ['9','36','45','63','90','117','126','144','153','171','198','207','225','234'] 
#     order_4 = ['27', '189', '216', '135', '108']

#     fitnesses = np.array([])
#     for data in sim_data:
#         flow = data[0]
#         storage = data[1]
#         fitness = 0
#         fitness += np.sum(Reward(flow[order_3].values, storage[order_3].values, 7.6,32500))
#         fitness += np.sum(Reward(flow[order_4].values, storage[order_4].values, 9.5,86000))
#         fitnesses = np.append(fitnesses, fitness)
       
#     return fitnesses

# def FitnessCalculator_FP(sim_data, population, previous_state):
#     ''' Fitness function for scenario 3(b). A concave equation is used to determine fitnesses.
#     The objective is to maintain streamflow at a steady level. This level is the half of 
#     Mean Annual Flood at the location of interest. All the links right downstream of the dams 
#     are used to calculate fitness. 

#     As input, use maximum streamflow in the pre-defined lead time.
    
#     Note:Dam overtopping is penalized.
#     '''
#     global eq235, eq217, eq181, eq73, eq55,eq19
#     order_3 = ['9','36','45','63','90','117','126','144','153','171','198','207','225','234'] 
#     order_4 = ['27', '189', '216', '135', '108']

#     fitnesses = np.array([])
#     for i, data in enumerate(sim_data):
#         fitness = 0
#         flow = data[0]
#         storage = data[1]
        
#         dam3 = storage[order_3].values/32500
#         dam4 = storage[order_4].values/85000

#         fitness -= np.sum(dam3)+np.sum(dam4)

#         # fitness -= flow['8']/31.80 + flow['26']/30.40 + flow['35']/7.58+flow['62']/27.4+flow['89']/16.33+flow['107']/13.89 +flow['116']/7.58+flow['142']/7.58+flow['170']/16.33+flow['188']/13.89+flow['197']/7.58+flow['224']/7.58

#         for j in order_3:
#             fitness -= flow[j]/0.75
#         for j in order_4:
#             fitness -= flow[j]/2
#         # ref = np.array(previous_state) / 0.1
#         # statein = np.array(population[i]) / 0.1
#         # diff = np.abs(statein-ref) - 1
#         # diff[diff<0] = 0
#         # fitness -= np.sum(diff) # Penalty for unstable state changes
#         fitnesses = np.append(fitnesses, fitness)
#     return fitnesses

# def FitnessCalculator_Scenario_3_b(sim_data, population, previous_state):
#     ''' Fitness function for scenario 3(b). A concave equation is used to determine fitnesses.
#     The objective is to maintain streamflow at a steady level. This level is the half of 
#     Mean Annual Flood at the location of interest. All the links right downstream of the dams 
#     are used to calculate fitness. 

#     As input, use maximum streamflow in the pre-defined lead time.
    
#     Note:Dam overtopping is penalized.
#     '''
#     global eq235, eq217, eq181, eq73, eq55,eq19
#     order_3 = ['9','36','45','63','90','117','126','144','153','171','198','207','225','234'] 
#     order_4 = ['27', '189', '216', '135', '108']

#     fitnesses = np.array([])
#     for i, data in enumerate(sim_data):
#         fitness = 0
#         flow = data[0]
#         storage = data[1]
        
#         # dam3_over = (storage[order_3]>200000).values.sum()
#         # dam4_over = (storage[order_4]>300000).values.sum()

#         # fitness -= (dam3_over*120 +dam4_over*120)
#         fitness += eq235[0]*flow['8']**2 + eq235[1]*flow['8'] + eq235[2]
#         fitness += eq217[0]*flow['26']**2 + eq217[1]*flow['26'] + eq217[2]
#         fitness += eq181[0]*flow['62']**2 + eq181[1]*flow['62'] + eq181[2]
#         fitness += eq73[0]*flow['89']**2 + eq73[1]*flow['89'] + eq73[2]
#         fitness += eq73[0]*flow['170']**2 + eq73[1]*flow['170'] + eq73[2]
#         fitness += eq55[0]*flow['107']**2 + eq55[1]*flow['107'] + eq55[2]
#         fitness += eq55[0]*flow['188']**2 + eq55[1]*flow['188'] + eq55[2]
#         fitness += eq19[0]*flow['35']**2 + eq19[1]*flow['35'] + eq19[2]
#         fitness += eq19[0]*flow['143']**2 + eq19[1]*flow['143'] + eq19[2]
#         fitness += eq19[0]*flow['116']**2 + eq19[1]*flow['116'] + eq19[2]
#         fitness += eq19[0]*flow['197']**2 + eq19[1]*flow['197'] + eq19[2]
#         fitness += eq19[0]*flow['224']**2 + eq19[1]*flow['224'] + eq19[2]

#         ref = np.array(previous_state) / 0.1
#         statein = np.array(population[i]) / 0.1
#         diff = np.abs(statein-ref) - 1
#         diff[diff<0] = 0
#         fitness -= np.sum(diff) * 10 # Penalty for unstable state changes
#         fitnesses = np.append(fitnesses, fitness)
#     return fitnesses