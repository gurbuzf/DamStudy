def FitnessCalculator_Scenario_1_a(sim_data):
    ''' Fitness function for scenario 1(a). The objective functions is composed of 
    only minimizing the discharge at the outlet.
    As input, use the streamflow at the outlet at the end of the pre-defined lead time.
    Note: There is no penalty for dam overtopping 
    '''
    fitnesses = np.array([])
    for data in sim_data:
        fitness = 0
        flow = data[0]
        storage = data[1]
        fitness += 1/flow['0']
        fitnesses = np.append(fitnesses, fitness)
        
    return fitnesses

def FitnessCalculator_Scenario_1_b(sim_data):     
    ''' Fitness function for scenario 1(b). The objective functions is composed of 
    only minimizing the discharge at the outlet.
    As input, use the streamflow at the outlet at the end of the pre-defined lead time.
    
    Note: Dam volume is added into the fitness values
    '''
    fitnesses = np.array([])
    for data in sim_data:
        fitness = 0
        flow = data[0]
        storage = data[1]
        fitness += 1/flow['0']
        dam3_over = (storage[order_3]>200000).values.sum()
        dam4_over = (storage[order_4]>300000).values.sum()
        fitness -= (dam3_over + dam4_over) # Total number of dams overtopping
        fitnesses = np.append(fitnesses, fitness)
    return fitnesses

def FitnessCalculator_Scenario_2(sim_data):
    ''' Fitness function for scenario 2. The objective functions is composed of 
    minimizing discharge at the links of 0, 81, 162
    As input, use the streamflow at the end of the pre-defined lead time.
    
    Note: Dam volume is added into the fitness values
    '''
    fitnesses = np.array([])
    for data in sim_data:
        fitness = 0
        flow = data[0]
        storage = data[1]
        fitness += 32.15 / flow['0']
        fitness += 17.5 / flow['81']
        fitness += 17.5 / flow['162']
        dam3_over = (storage[order_3]>200000).values.sum()
        dam4_over = (storage[order_4]>300000).values.sum()
        fitness -= 3*(dam3_over + dam4_over) # Total number of dams overtopping
        fitnesses = np.append(fitnesses, fitness)
    return fitnesses


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

eq235 = Quad_Params2(31.80,10)
eq217 = Quad_Params2(30.40, 10)
eq181 = Quad_Params2(27.4, 10)
eq73 = Quad_Params2(16.33, 10)
eq55 = Quad_Params2(13.89, 10)
eq19 = Quad_Params2(7.58, 10)

def FitnessCalculator_Scenario_3_a(sim_data):
    ''' Fitness function for scenario 3(a). A concave equation is used to determine fitness.
    the objective is to maintain streamflow at a steady level. This level is the half of 
    Mean Annual Flood at the location of interest. All the links right downstream of the dams 
    are used to calculate fitness. 

    As input, use maximum streamflow in the pre-defined lead time.
    
    Note: No Penalty for dam overtopping
    '''
    global eq235, eq217, eq181, eq73, eq55,eq19

    fitnesses = np.array([])
    for data in sim_data:
        fitness = 0
        flow = data[0]
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
       
    return fitnesses



def FitnessCalculator_Scenario_3_b(sim_data):
    ''' Fitness function for scenario 3(b). A concave equation is used to determine fitnesses.
    The objective is to maintain streamflow at a steady level. This level is the half of 
    Mean Annual Flood at the location of interest. All the links right downstream of the dams 
    are used to calculate fitness. 

    As input, use maximum streamflow in the pre-defined lead time.
    
    Note:Dam overtopping is penalized.
    '''
    global eq235, eq217, eq181, eq73, eq55,eq19

    fitnesses = np.array([])
    for data in sim_data:
        fitness = 0
        flow = data[0]
        storage = data[1]
        order_3 = ['9','36','45','63','90','117','126','144','153','171','198','207','225','234'] 
        order_4 = ['27', '189', '216', '135', '108']
        dam3_over = (storage[order_3]>200000).values.sum()
        dam4_over = (storage[order_4]>300000).values.sum()

        fitness -= (dam3_over*120 +dam4_over*120)
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
    return fitnesses

def Linear_Penalty(penalty, s_max, s_spill):
    ''' A Linear penalty function for dam overtopping.
    The dam starts penalizing when it reaches spillway level.

    penalty: penalty value when dam is at maximum level
    '''
    a = penalty / (s_spill-s_max)
    b = - a * s_spill
    return a , b
a3, b3 = Linear_Penalty(100, 200000, 162000)
a4, b4 = Linear_Penalty(100, 300000, 243000)



def FitnessCalculator_Scenario_4(sim_data):
    ''' Fitness function for scenario 4. A concave equation is used to determine fitnesses.
    The objective is to maintain streamflow at a steady level. This level is the half of 
    Mean Annual Flood at the location of interest. All the links right downstream of the dams 
    are used to calculate fitness. 

    As input, use maximum streamflow in the pre-defined lead time.
    
    Note: A linear function is used to penalize dam overtopping
    '''
    global eq235, eq217, eq181, eq73, eq55, eq19 
    global a3, b3, a4, b4

    fitnesses = np.array([])
    for data in sim_data:
        fitness = 0
        flow = data[0]
        storage = data[1]
        order_3 = ['9','36','45','63','90','117','126','144','153','171','198','207','225','234'] 
        order_4 = ['27', '189', '216', '135', '108']
        
        for dam_id in order_3:
            if storage[dam_id].values >162000:
                fitness += a3*storage[dam_id].values + b3
        for dam_id in order_4:
            if storage[dam_id].values >243000:
                fitness += a4*storage[dam_id].values + b4

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
    return fitnesses