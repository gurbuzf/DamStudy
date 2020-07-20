import numpy as np
import math
import warnings

def Model_190(t, y_i, forcing, global_params, params, connectivity):
    dim = len(connectivity)
    lambda_1 = global_params[1]
    A_h = params[2]
    k2 = params[3]
    k3 = params[4]
    invtau = params[5]
    c_1 = params[6]
    c_2 = params[7]
    t = int(round(t))
    print(t, end='\r')
    forcing = forcing[t]

    q = y_i[0:dim]  #Channel Discharge
    q[q<0] = 1.000000e-6
    s_p = y_i[dim:2*dim]  #Ponded Storage
    s_s = y_i[2*dim:]  #Subsurface Storage

    q_pl = k2 * s_p
    q_sl = k3 * s_s

    ##Discharge
    dq = -q + (q_pl+q_sl)*A_h/60.0
    q_in = np.array([sum(q[parents]) for parents in connectivity])
    dq = dq + q_in
    dq = invtau*(q**lambda_1)*dq

    ##Hillslope
    ds_p = forcing *c_1 - q_pl
    ds_s = forcing *c_2 -q_sl

    return np.concatenate((dq, ds_p, ds_s))

def Model_254(t, y_i, forcing, global_params, params, connectivity):
    dim = len(connectivity)
    
    lambda_1 = global_params[1]     #[-]
    k_3 = global_params[4]          #[1/min]
    # h_b = global_params[6]          #[m]
    S_L = global_params[7]          #[m]
    A = global_params[8]            #[-]
    B =  global_params[9]           #[-]
    exponent = global_params[10]    #[-]
    # v_B = global_params[11]         #[m/s]

    # L = params[1]	#[m]
    # A_h = params[2]	#[m^2]
    invtau = params[3]	#[1/min]
    k_2 = params[4] 	#[1/min]
    k_i = params[5] 	#[1/min]
    c_1 = params[6] 
    c_2 = params[7] 
    
    t = int(round(t))
    print(t, end='\r')
    forcing = forcing[t]

    
    q = y_i[0:dim]          #[m^3/s]
    q[q<0] = 0.0
    s_p = y_i[dim:2*dim]     #[m]
    s_t = y_i[2*dim:3*dim]   #[m]
    s_s = y_i[3*dim:]        #[m] 
    
    pow_term =np.array([(1-st/S_L)**exponent if (1-st/S_L)>0 else 0 for st in s_t])
    k_t = (A + B*pow_term)*k_2
    
    #Fluxes
    q_pl = k_2 * s_p
    q_pt = k_t * s_p
    q_ts = k_i * s_t
    q_sl = k_3 * s_s	#[m/min]

    
    q_in = np.array([sum(q[childs]) for childs in connectivity])
    dq = -q + (q_pl + q_sl) * c_2  + q_in
    dq = invtau * pow(q, lambda_1) * dq

    ds_p = forcing * c_1 - q_pl - q_pt 
    ds_t = q_pt - q_ts 
    ds_s = q_ts - q_sl 
    
    dx = np.concatenate((dq, ds_p, ds_t,ds_s))
    return dx

def Model_254_dam(t, y_i, forcing, global_params, params,dam_params, connectivity, nextlink, gate_state):
    dim = len(connectivity)
    
    lambda_1 = global_params[1]     #[-]
    k_3 = global_params[4]          #[1/min]
    # h_b = global_params[6]          #[m]
    S_L = global_params[7]          #[m]
    A = global_params[8]            #[-]
    B =  global_params[9]           #[-]
    exponent = global_params[10]    #[-]
    # v_B = global_params[11]         #[m/s]

    # L = params[1]	#[m]
    # A_h = params[2]	#[m^2]
    invtau = params[3]	#[1/min]
    k_2 = params[4] 	#[1/min]
    k_i = params[5] 	#[1/min]
    c_1 = params[6] 
    c_2 = params[7] 
    
    
    t = int(round(t))
    print(t, end='\r')

    forcing = forcing[t]
    

    q = y_i[0:dim]		 #[m^3/s]
    q[q<0] = 0.0
    S = y_i[dim:2*dim]  # Channel Storage [m^3]
    S[S<0] = 0.0
    s_p = y_i[2*dim:3*dim]    #[m]
    s_t = y_i[3*dim:4*dim]    #[m]
    s_s = y_i[4*dim:]    #[m] 
    
    pow_term =np.array([(1-st/S_L)**exponent if (1-st/S_L)>0 else 0 for st in s_t])
    k_t = (A + B*pow_term)*k_2
    #Fluxes
    q_pl = k_2 * s_p
    q_pt = k_t * s_p
    q_ts = k_i * s_t
    q_sl = k_3 * s_s	#[m/min]

    
    q_in = np.array([sum(q[childs]) for childs in connectivity])
    dq = -q + (q_pl + q_sl) * c_2  + q_in
    dq = invtau * pow(q, lambda_1) * dq

    ds_p = forcing * c_1 - q_pl - q_pt 
    ds_t = q_pt - q_ts 
    ds_s = q_ts - q_sl 
    
    #Channel Storage
    dS = np.zeros(dim)
    dam = dam_params[0]
    H_max = dam_params[2]
    S_max = dam_params[3]
    alpha = dam_params[4]   #Exponent for bankfull
    for idx in range(dim):
        if dam[idx] == 0:pass
        else:
            h = H_max * pow(S[idx] / S_max, alpha) #non-Linear Storage-Discharge
            q_in = 60 * q[idx]       #[m3/s->m3/min]
            q_out = 60 * dam_q(h, gate_state[idx], dam_params) #[m3/s->m3/min]
            dS[idx] = q_in - q_out
            # print('next',nextlink[idx])
            if nextlink[idx] !=-1:
                dq[nextlink[idx]] = dq[nextlink[idx]] - invtau[nextlink[idx]]*(q[nextlink[idx]]**lambda_1)*(q[idx]-q_out/60)

    return np.concatenate((dq, dS, ds_p, ds_t, ds_s))


def Model_254_dam_varParam(t, y_i, forcing, global_params, params, dam_params, connectivity, nextlink, gate_state):
    dim = len(connectivity)
    
    lambda_1 = global_params[1]     #[-]
    k_3 = global_params[4]          #[1/min]
    # h_b = global_params[6]          #[m]
    S_L = global_params[7]          #[m]
    A = global_params[8]            #[-]
    B =  global_params[9]           #[-]
    exponent = global_params[10]    #[-]
    # v_B = global_params[11]         #[m/s]

    # L = params[1]	#[m]
    # A_h = params[2]	#[m^2]
    invtau = params[3]	#[1/min]
    k_2 = params[4] 	#[1/min]
    k_i = params[5] 	#[1/min]
    c_1 = params[6] 
    c_2 = params[7] 
    
    
    t = int(round(t))
    print(t, end='\r')
    
    forcing = forcing[t]

    q = y_i[0:dim]		 #[m^3/s]
    q[q<0] = 0.0
    S = y_i[dim:2*dim]  # Channel Storage [m^3]
    S[S<0] = 0.0
    s_p = y_i[2*dim:3*dim]    #[m]
    s_t = y_i[3*dim:4*dim]    #[m]
    s_s = y_i[4*dim:]    #[m] 
    
    pow_term =np.array([(1-st/S_L)**exponent if (1-st/S_L)>0 else 0 for st in s_t])
    k_t = (A + B*pow_term)*k_2
    #Fluxes
    q_pl = k_2 * s_p
    q_pt = k_t * s_p
    q_ts = k_i * s_t
    q_sl = k_3 * s_s	#[m/min]

    
    q_in = np.array([sum(q[childs]) for childs in connectivity])
    dq = -q + (q_pl + q_sl) * c_2  + q_in
    dq = invtau * pow(q, lambda_1) * dq

    ds_p = forcing * c_1 - q_pl - q_pt 
    ds_t = q_pt - q_ts 
    ds_s = q_ts - q_sl 
    
    #Channel Storage
    dS = np.zeros(dim)
    dam = dam_params[0]
    H_spill = dam_params[1] #Height of the spillway [m]
    H_max = dam_params[2]  #Height of the dam [m]
    S_max = dam_params[3]  #Maximum volume of water the dam can hold [m3] 
    alpha = dam_params[4]   #Exponent for bankfull
    diam = dam_params[5]    #Diameter of dam orifice [m]
    C1 = dam_params[6]      #Coefficient for discharge from dam
    C2 = dam_params[7]      #Coefficient for discharge from dam
    L_spill = dam_params[8] #Length of the spillway [m].
    L_crest = dam_params[9]
    for idx in range(dim):
        if dam[idx] == 0:pass    
        else:
            h_spill = H_spill[idx]
            h_max = H_max[idx]
            s_max = S_max[idx]
            diameter = diam[idx]
            c1 = C1[idx]
            c2 = C2[idx]
            l_spill = L_spill[idx]
            l_crest = L_crest[idx]
            h = h_max * pow(S[idx] / s_max, alpha[idx])
            q_in = 60 * q[idx]      #[m3/s->m3/min]
            q_out = 60 * dam_q_varParam_varState(h, gate_state[idx],h_spill, h_max, diameter, c1, c2, l_spill, l_crest) #[m3/s->m3/min]
            dS[idx] = q_in - q_out
            # print('next',nextlink[idx])
            if nextlink[idx] !=-1:
                dq[nextlink[idx]] = dq[nextlink[idx]] - invtau[nextlink[idx]]*(q[nextlink[idx]]**lambda_1)*(q[idx]-q_out/60)

    return np.concatenate((dq, dS, ds_p, ds_t, ds_s))

def Model_190_dam(t, y_i, forcing, global_params, params,dam_params, connectivity, nextlink, gate_state):
    dim = len(connectivity)
    lambda_1 = global_params[1]
    A_h = params[2]     #Area of the hillslope of this link [m2]
    k2 = params[3]      #[1/min]  k2  
    k3 = params[4]      #[1/min]  k3
    invtau = params[5]  #[1/min]  invtau
    c_1 = params[6]     #m/min
    c_2 = params[7]     #m/min
    dam = dam_params[0]
    alpha = dam_params[4]   #Exponent for bankfull
    # print(t)
    t = int(round(t))
    print(t, end='\r')
    forcing = forcing[t]

    q = y_i[0:dim]  #Channel Discharge
    q[q<0] = 0
    S = y_i[dim:2*dim]  # Channel Storage [m^3]
    S[S<0] = 0
    s_p = y_i[2*dim:3*dim]  #Ponded Storage
    s_s = y_i[3*dim:]  #Subsurface Storage

    q_pl = k2 * s_p
    q_sl = k3 * s_s
    
    ##Discharge
    dq = -q + (q_pl+q_sl)*A_h/60.0 
    qin = np.array([sum(q[parents]) for parents in connectivity])
    dq = dq + qin
#     with warnings.catch_warnings ():
#         warnings.filterwarnings ('error')
#         try:
#             dq = invtau*(q**lambda_1)*dq
#         except RuntimeWarning:
#             print('dq >>> ', dq)
#             print('q  >>> ', q)
#             print('invtau >>> ', invtau)
    dq = invtau*(q**lambda_1)*dq
            

    ##Hillslope
    ds_p = forcing *c_1 - q_pl
    ds_s = forcing *c_2 -q_sl

    #Channel Storage
    dS = np.zeros(dim)
    H_max = dam_params[2]
    S_max = dam_params[3]
    for idx in range(dim):
        if dam[idx] == 0:pass
        else:
            h = H_max * pow(S[idx] / S_max, alpha) #non-Linear Storage-Discharge
            q_in = 60 * q[idx]       #[m3/s->m3/min]
            q_out = 60 * dam_q(h,gate_state[idx], dam_params) #[m3/s->m3/min]
            dS[idx] = q_in - q_out
            # print('next',nextlink[idx])
            if nextlink[idx] !=-1:
                dq[nextlink[idx]] = dq[nextlink[idx]] - invtau[nextlink[idx]]*(q[nextlink[idx]]**lambda_1)*(q[idx]-q_out/60)

    return np.concatenate((dq, dS, ds_p, ds_s))

def dam_q(h,gate_state,dam_params):    
    
    H_spill = dam_params[1] #Height of the spillway [m]
    H_max = dam_params[2]   #Height of the dam [m]
    # S_max = dam_params[3]   #Maximum volume of water the dam can hold [m3] 
    # alpha = dam_params[4]   #Exponent for bankfull
    diam = dam_params[5]    #Diameter of dam orifice [m]
    c1 = dam_params[6]      #Coefficient for discharge from dam
    c2 = dam_params[7]      #Coefficient for discharge from dam
    L_spill = dam_params[8] #Length of the spillway [m].
    L_crest = dam_params[9]
    orifice_area = np.pi*pow(diam,2)/4
    g = 9.80665

    qs0 = qs1 = qs2 = qs3 = 0
    if h<0:h=0 # to ensure numerical stability

    if h < diam:  ## the case of h < Pipe Diameter
        if gate_state:
            r = diam / 2.0
            frac = (h-r) / r
            # frac = (h < 2 * r) ? (h - r) / r : 1.0; # From Asynch C code. ????
            A = -r*r*(math.acos(frac) - pow(1 - frac*frac, .5)*frac - np.pi)
            qs0 = c1*A*pow(2 * g*h, .5)

    elif h >= diam and h <= H_spill:
        if gate_state:
            qs1 = c1 * orifice_area * pow(2 * g*h, .5)

    elif H_spill < h <= H_max:     
        if gate_state:
            qs1 = c1 * orifice_area * pow(2 * g*h, .5)
        qs2 = c2 * L_spill * pow(h-H_spill, 1.5)

    elif h>H_max:
        if gate_state:
            qs1 = c1 * orifice_area * pow(2 * g*h, .5)
        qs2 = L_spill * c2 * pow(h-H_spill, 1.5)
        qs3 = (L_crest-L_spill) * c2 * pow(h-H_max, 1.5)  #!!!!!!!!!!!!!!!!
        
    return qs0+qs1+qs2+qs3


def dam_q_varParam(h,gate_state,h_spill, h_max, diameter, c_1, c_2, l_spill, l_crest):    
    '''
    h_spill: Height of the spillway [m]
    h_max : Height of the dam [m]
    s_max : dam_params[3]   #Maximum volume of water the dam can hold [m3] 
    alpha : Exponent for bankfull
    diameter : #Diameter of dam orifice [m]
    c_1 : Coefficient for discharge from dam
    c_2 :Coefficient for discharge from dam
    l_spill :Length of the spillway [m].
    l_crest : dam_params[9]
    '''
    open_rate = state * diameter
    orifice_area = np.pi*pow(diameter,2)/4
    g = 9.80665

    qs0 = qs1 = qs2 = qs3 = 0
    if h<0:h=0 # to ensure numerical stability

    if h < diameter:  ## the case of h < Pipe Diameter
        if gate_state:
            r = diameter / 2.0
            frac = (h-r) / r
            # frac = (h < 2 * r) ? (h - r) / r : 1.0; # From Asynch C code. ????
            A = -r*r*(math.acos(frac) - pow(1 - frac*frac, .5)*frac - np.pi)
            qs0 = c_1*A*pow(2 * g*h, .5)

    elif h >= diameter and h <= h_spill:
        if gate_state:
            qs1 = c_1 * orifice_area * pow(2 * g*h, .5)

    elif h_spill < h <= h_max:     
        if gate_state:
            qs1 = c_1 * orifice_area * pow(2 * g*h, .5)
        qs2 = c_2 * l_spill * pow(h-h_spill, 1.5)

    elif h>h_max:
        if gate_state:
            qs1 = c_1 * orifice_area * pow(2 * g*h, .5)
        qs2 = l_spill * c_2 * pow(h-h_spill, 1.5)
        qs3 = (l_crest-l_spill) * c_2 * pow(h-h_max, 1.5)  #!!!!!!!!!!!!!!!!
        


        # qs1 = c1 * orifice_area * pow(2 * g*h, .5)
        # qs2 = L_spill * c2 * pow(h-H_spill, 1.5)
        # qs3 = (L_crest-L_spill) * c2 * pow(h-H_max, 1.5)  #!!!!!!!!!!!!!!!!
    return qs0+qs1+qs2+qs3

def dam_q_varParam_varState(h,gate_state,h_spill, h_max, diam, c_1, c_2, l_spill, l_crest):    
    '''
    h_spill: Height of the spillway [m]
    h_max : Height of the dam [m]
    s_max : dam_params[3]   #Maximum volume of water the dam can hold [m3] 
    alpha : Exponent for bankfull
    diameter : #Diameter of dam orifice [m]
    c_1 : Coefficient for discharge from dam
    c_2 :Coefficient for discharge from dam
    l_spill :Length of the spillway [m].
    l_crest : dam_params[9]
    '''
    

    diameter = gate_state * diam

    orifice_area = np.pi*pow(diameter,2)/4
    g = 9.80665

    qs0 = qs1 = qs2 = qs3 = 0
    if h<0:h=0 # to ensure numerical stability

    if h < diameter:  ## the case of h < Pipe Diameter
        if gate_state:
            r = diameter / 2.0
            frac = (h-r) / r
            # frac = (h < 2 * r) ? (h - r) / r : 1.0; # From Asynch C code. ????
            A = -r*r*(math.acos(frac) - pow(1 - frac*frac, .5)*frac - np.pi)
            qs0 = c_1*A*pow(2 * g*h, .5)

    elif h >= diameter and h <= h_spill:
        if gate_state:
            qs1 = c_1 * orifice_area * pow(2 * g*h, .5)

    elif h_spill < h <= h_max:     
        if gate_state:
            qs1 = c_1 * orifice_area * pow(2 * g*h, .5)
        qs2 = c_2 * l_spill * pow(h-h_spill, 1.5)

    elif h>h_max:
        if gate_state:
            qs1 = c_1 * orifice_area * pow(2 * g*h, .5)
        qs2 = l_spill * c_2 * pow(h-h_spill, 1.5)
        qs3 = (l_crest-l_spill) * c_2 * pow(h-h_max, 1.5)  #!!!!!!!!!!!!!!!!
        


        # qs1 = c1 * orifice_area * pow(2 * g*h, .5)
        # qs2 = L_spill * c2 * pow(h-H_spill, 1.5)
        # qs3 = (L_crest-L_spill) * c2 * pow(h-H_max, 1.5)  #!!!!!!!!!!!!!!!!
    return qs0+qs1+qs2+qs3


