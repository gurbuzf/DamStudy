from hlm_basic.ssn import GenerateNetwork, UpstreamArea
from hlm_basic.watershed import Watershed
from hlm_basic.tools import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime
import pickle

# # Several scenarios for dam spatial configurations are created:
# 
# <ul>
#   <li>Scenario-1</li>
#   <p> There are only two big ponds on Links 81 and 162 (order-5 streams). Total drainage area of each dam is 20.25 km<sup>2</sup></p>
#   <li>Scenario-2</li>
#   <p> There are four small ponds on order 4 links. Total drainage area of each dam is 6.75 km<sup>2</sup></p>
#   <li>Scenario-3</li>
#   <p> There are ten small ponds on order 3 links. Total drainage area of each dam is 2.25 km<sup>2</sup></p>
#   <li>Scenario-4</li>
#   <p> The dams in scenario-2 and scenario-3 placed all together .</p>
#   <li>Scenario-5</li>
#   <p> in addition to the dams in scenario-4, five more dams are placed on order 4 and order3 streams that are closer to the outlet to control more area. </p>
# 
# </ul>
## First simulations were run with [0.001, 0.03,0.18,0.7,0.04,0.02,0.01,0.005, 0.004,0.01,]

if __name__ == "__main__":
    

    # Properties of artificial watershed
    l_id, connectivity, h_order, nextlink = GenerateNetwork(5)
    n_hills = len(connectivity)
    a_hill = 0.5 * 0.5 #km2
    A_h = np.array([a_hill*10**6 for i in range(n_hills)]) #m2
    L_i = np.array([0.5*10**3 for i in range(n_hills)])  #m
    A_i = UpstreamArea(a_hill, connectivity, h_order) #km2

    dams1  = [81, 162] # two big ponds (SCENARIO-1)
    dams2 = [189, 216, 135, 108,] # dams on order_4 Note: No dams on the links closer to the outlet (SCENARIO-2)
    dams3 = [117,126,144,153,198,207,225,234] ## dams on order_3 Note: No dams on the links closer to the outlet (SCENARIO-3)
    dams4 = dams2 + dams3 ## dams on order_3 and order_4 Note: No dams on the links closer to the outlet  (SCENARIO-4)
    dams5 = [27, 189, 216, 135, 108,] + [9,36,45,63,90,117,126,144,153,171,198,207,225,234]   ## dams on order_3 and order_4 Note: dams on the links closer to the outlet included (SCENARIO-5)
    n_dams1 = len(dams1)
    n_dams2 = len(dams2)
    n_dams3 = len(dams3)
    n_dams4 = len(dams4)
    n_dams5 = len(dams5)

    ## INITIAL CONDITIONS
    with open('/Users/gurbuz/DamStudy/data/initial_conditions.pickle', 'rb') as ini_file:
        initial_conditions = pickle.load(ini_file)
    ## DESIGN STORMs
    dstorms = [67, 112, 163, 189]
    rate10 = [ 12,  30,  63,  75,  84,  87 ,90,  93.5,  95,  96.01,  97.44,  97.44 ,  99.42,  99.42, 99.67,  100] 

    print(f'Simulations started  at {datetime.datetime.now().time()}')
    ## NODAM CASE #############################################################################
    # SSN1 = Watershed(Model=256)
    # SSN1.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    # H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams([]) #!! TODO: make the code run without this parameters when no dam introduced
    # dam_params256 = SSN1.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest) # !!
    # SSN1.set_dam_state()

    # dc_nodam_Peak = pd.DataFrame(columns = SSN1.__columns__()[0])

    # for dstorm in dstorms:
    #     forcing, cum_forcing, forcing_hour = Generate_SyntheticStorm(dstorm,24, rate=(0.01*np.diff(rate10, prepend=0)).tolist(), timescale=30)
    #     te = len(forcing)-1
    #     for key in initial_conditions.keys():
    #         initial_condition = initial_conditions[key] 
    #         q = initial_condition['q']
    #         s_p = initial_condition['s_p']
    #         s_t = initial_condition['s_t']
    #         s_s = initial_condition['s_s']
        
    #         SSN1.initialize(q=q, s_t =s_t, s_p =s_p, s_s=s_s)
    #         try:
    #             dc_nodam, _ = SSN1.Run_256( [0, te], forcing, dam_params256,)

    #             temp = dc_nodam[dc_nodam.index>120].max()
    #             dc_nodam_Peak = dc_nodam_Peak.append(temp,ignore_index = True) 
    #         except IndexError:
    #             pass

    #     print(f'(NODAM)Storm:{dstorm} is simulated with given initial conditions!')
    # dc_nodam_Peak.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/dc_nodam_Peak_rand08.csv')
    # print('No dam scenario is done and results are saved!\n')
    # print(f'Time: {datetime.datetime.now().time()}')


    ## SCENARIO-1 #############################################################################

    SSN2 = Watershed(Model=256)
    SSN2.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN2.dam_ids = dams1
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams1)
    dam_params256 = SSN2.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)

    dc_passive_Peak_S1 = pd.DataFrame(columns = SSN2.__columns__()[0])
    out_passive_Peak_S1 = pd.DataFrame(columns = SSN2.__columns__()[1])

    for dstorm in dstorms:
        forcing, cum_forcing, forcing_hour = Generate_SyntheticStorm(dstorm,24, rate=(0.01*np.diff(rate10, prepend=0)).tolist(), timescale=30)
        te = len(forcing)-1
        for key in initial_conditions.keys():
            initial_condition = initial_conditions[key] 
            q = initial_condition['q']
            s_p = initial_condition['s_p']
            s_t = initial_condition['s_t']
            s_s = initial_condition['s_s']
            fill_percent = np.repeat([.99],n_dams1) #()np.random.uniform(0,0.8,n_dams1).round(2)
            S = (S_max * fill_percent).tolist()
            SSN2.set_dam_state(states=[1 for _ in range(n_dams1)])
            SSN2.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)

            dc_passive_S1, st_passive_S1 = SSN2.Run_256( [0, te], forcing, dam_params256)
            out_passive_S1 = SSN2.CalculateOutflow(dam_params256, st_passive_S1)

            temp1 = dc_passive_S1.max()#dc_passive_S1[dc_passive_S1.index>120].max()
            temp2 = out_passive_S1.max()#out_passive_S1[out_passive_S1.index>120].max()

            dc_passive_Peak_S1 = dc_passive_Peak_S1.append(temp1,ignore_index = True)
            out_passive_Peak_S1 = out_passive_Peak_S1.append(temp2,ignore_index = True)  
            
        print(f'(Scenario-1)Storm:{dstorm} is simulated with given initial conditions!')

    dc_passive_Peak_S1.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/dc_passive_Peak_S1_full.csv')
    out_passive_Peak_S1.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/out_passive_Peak_S1_full.csv')
    print('Scenario-1 is done and results are saved!\n')
    print(f'Time: {datetime.datetime.now().time()}')


    ## SCENARIO-2 #############################################################################

    SSN3 = Watershed(Model=256)
    SSN3.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN3.dam_ids = dams2
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams2)
    dam_params256 = SSN3.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)

    dc_passive_Peak_S2 = pd.DataFrame(columns = SSN3.__columns__()[0])
    out_passive_Peak_S2 = pd.DataFrame(columns = SSN3.__columns__()[1])

    for dstorm in dstorms:
        forcing, cum_forcing, forcing_hour = Generate_SyntheticStorm(dstorm,24, rate=(0.01*np.diff(rate10, prepend=0)).tolist(), timescale=30)
        te = len(forcing)-1
        for key in initial_conditions.keys():
            initial_condition = initial_conditions[key] 
            q = initial_condition['q']
            s_p = initial_condition['s_p']
            s_t = initial_condition['s_t']
            s_s = initial_condition['s_s']
            fill_percent = np.repeat([.99],n_dams2)#np.random.uniform(0,0.8,n_dams2).round(2)
            S = (S_max * fill_percent).tolist()
            SSN3.set_dam_state(states=[1 for _ in range(n_dams2)])
            SSN3.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)

            dc_passive_S2, st_passive_S2 = SSN3.Run_256( [0, te], forcing, dam_params256)
            out_passive_S2 = SSN3.CalculateOutflow(dam_params256, st_passive_S2)

            temp1 = dc_passive_S2.max()#dc_passive_S2[dc_passive_S2.index>120].max()
            temp2 = out_passive_S2.max()#out_passive_S2[out_passive_S2.index>120].max()

            dc_passive_Peak_S2 = dc_passive_Peak_S2.append(temp1,ignore_index = True)
            out_passive_Peak_S2 = out_passive_Peak_S2.append(temp2,ignore_index = True)  
            
        print(f'(Scenario-2)Storm:{dstorm} is simulated with given initial conditions!')

    dc_passive_Peak_S2.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/dc_passive_Peak_S2_full.csv')
    out_passive_Peak_S2.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/out_passive_Peak_S2_full.csv')
    print('Scenario-2 is done and results are saved!\n')
    print(f'Time: {datetime.datetime.now().time()}')


    ## SCENARIO-3 #############################################################################

    SSN4 = Watershed(Model=256)
    SSN4.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN4.dam_ids = dams3
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams3)
    dam_params256 = SSN4.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)

    dc_passive_Peak_S3 = pd.DataFrame(columns = SSN4.__columns__()[0])
    out_passive_Peak_S3 = pd.DataFrame(columns = SSN4.__columns__()[1])

    for dstorm in dstorms:
        forcing, cum_forcing, forcing_hour = Generate_SyntheticStorm(dstorm,24, rate=(0.01*np.diff(rate10, prepend=0)).tolist(), timescale=30)
        te = len(forcing)-1
        for key in initial_conditions.keys():
            
            initial_condition = initial_conditions[key] 
            q = initial_condition['q']
            s_p = initial_condition['s_p']
            s_t = initial_condition['s_t']
            s_s = initial_condition['s_s']
            fill_percent = np.repeat([.99],n_dams3)#np.random.uniform(0,0.8,n_dams3).round(2)
            S = (S_max * fill_percent).tolist()
            SSN4.set_dam_state(states=[1 for _ in range(n_dams3)])
            SSN4.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)

            dc_passive_S3, st_passive_S3 = SSN4.Run_256( [0, te], forcing, dam_params256)
            out_passive_S3 = SSN4.CalculateOutflow(dam_params256, st_passive_S3)

            temp1 = dc_passive_S3.max()#dc_passive_S3[dc_passive_S3.index>120].max()
            temp2 = out_passive_S3.max()#out_passive_S3[out_passive_S3.index>120].max()

            dc_passive_Peak_S3 = dc_passive_Peak_S3.append(temp1,ignore_index = True)
            out_passive_Peak_S3 = out_passive_Peak_S3.append(temp2,ignore_index = True)  
            
        print(f'(Scenario-3)Storm:{dstorm} is simulated with given initial conditions!')

    dc_passive_Peak_S3.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/dc_passive_Peak_S3_full.csv')
    out_passive_Peak_S3.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/out_passive_Peak_S3_full.csv')
    print('Scenario-3 is done and results are saved!\n')
    print(f'Time: {datetime.datetime.now().time()}')


    ## SCENARIO-4 #############################################################################

    SSN5 = Watershed(Model=256)
    SSN5.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN5.dam_ids = dams4
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams4)
    dam_params256 = SSN5.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)

    dc_passive_Peak_S4 = pd.DataFrame(columns = SSN5.__columns__()[0])
    out_passive_Peak_S4 = pd.DataFrame(columns = SSN5.__columns__()[1])

    for dstorm in dstorms:
        forcing, cum_forcing, forcing_hour = Generate_SyntheticStorm(dstorm,24, rate=(0.01*np.diff(rate10, prepend=0)).tolist(), timescale=30)
        te = len(forcing)-1
        for key in initial_conditions.keys():
            
            initial_condition = initial_conditions[key] 
            q = initial_condition['q']
            s_p = initial_condition['s_p']
            s_t = initial_condition['s_t']
            s_s = initial_condition['s_s']
            fill_percent = np.repeat([.99],n_dams4)#np.random.uniform(0,0.8,n_dams4).round(2)
            S = (S_max * fill_percent).tolist()
            SSN5.set_dam_state(states=[1 for _ in range(n_dams4)])
            SSN5.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)

            dc_passive_S4, st_passive_S4 = SSN5.Run_256( [0, te], forcing, dam_params256)
            out_passive_S4 = SSN5.CalculateOutflow(dam_params256, st_passive_S4)

            temp1 = dc_passive_S4.max()#dc_passive_S4[dc_passive_S4.index>120].max()
            temp2 = out_passive_S4.max()#out_passive_S4[out_passive_S4.index>120].max()

            dc_passive_Peak_S4 = dc_passive_Peak_S4.append(temp1,ignore_index = True)
            out_passive_Peak_S4 = out_passive_Peak_S4.append(temp2,ignore_index = True)  
            
        print(f'(Scenario-4)Storm:{dstorm} is simulated with given initial conditions!')

    dc_passive_Peak_S4.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/dc_passive_Peak_S4_full.csv')
    out_passive_Peak_S4.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/out_passive_Peak_S4_full.csv')
    print('Scenario-4 is done and results are saved!\n')
    print(f'Time: {datetime.datetime.now().time()}')


    ## SCENARIO-5 #############################################################################

    SSN6 = Watershed(Model=256)
    SSN6.init_custom(links=l_id, connectivity=connectivity, A_i=A_i, L_i=L_i, A_h=A_h)
    SSN6.dam_ids = dams5
    H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest = PrepareDamParams(dams5)
    dam_params256 = SSN6.init_dam_params256(H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest)

    dc_passive_Peak_S5 = pd.DataFrame(columns = SSN6.__columns__()[0])
    out_passive_Peak_S5 = pd.DataFrame(columns = SSN6.__columns__()[1])

    for dstorm in dstorms:
        forcing, cum_forcing, forcing_hour = Generate_SyntheticStorm(dstorm,24, rate=(0.01*np.diff(rate10, prepend=0)).tolist(), timescale=30)
        te = len(forcing)-1
        for key in initial_conditions.keys():
            
            initial_condition = initial_conditions[key] 
            q = initial_condition['q']
            s_p = initial_condition['s_p']
            s_t = initial_condition['s_t']
            s_s = initial_condition['s_s']
            fill_percent = np.repeat([.99],n_dams5)#np.random.uniform(0,0.8,n_dams5).round(2)
            S = (S_max * fill_percent).tolist()
            SSN6.set_dam_state(states=[1 for _ in range(n_dams5)])
            SSN6.initialize(q=q, S = S, s_t =s_t, s_p =s_p, s_s=s_s)

            dc_passive_S5, st_passive_S5 = SSN6.Run_256( [0, te], forcing, dam_params256)
            out_passive_S5 = SSN6.CalculateOutflow(dam_params256, st_passive_S5)

            temp1 = dc_passive_S5.max()#dc_passive_S5[dc_passive_S5.index>120].max()
            temp2 = out_passive_S5.max()#out_passive_S5[out_passive_S5.index>120].max()

            dc_passive_Peak_S5 = dc_passive_Peak_S5.append(temp1,ignore_index = True)
            out_passive_Peak_S5 = out_passive_Peak_S5.append(temp2,ignore_index = True)  
            
        print(f'(Scenario-5)Storm:{dstorm} is simulated with given initial conditions!')

    dc_passive_Peak_S5.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/dc_passive_Peak_S5_full.csv')
    out_passive_Peak_S5.to_csv('/Users/gurbuz/Supp_DamStudy/Dam_Configuration/out_passive_Peak_S5_full.csv')
    print('Scenario-5 is done and results are saved!\n')
    print(f'Time: {datetime.datetime.now().time()}')


