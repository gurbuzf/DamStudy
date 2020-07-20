#-----------------------------------------------------#
# Author: Faruk Gurbuz
# Date: 06/21/2020
# solve_ivp from scipy.integrate is used to solve
# differential equations. 
# SCIPY version : 1.4.1 
#-----------------------------------------------------#

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from hlm_basic.hlm_models import Model_190, Model_254, Model_190_dam, Model_254_dam, Model_254_dam_varParam
from hlm_basic.tools import read_prm, read_rvr


global_params_190 = [0.33, 0.2, -0.1, 0.33, 0.2, 2.0425e-06]
                   #v_0 lambda_1 lambda_2 v_h  k_3 k_I_factor h_b S_L  A   B    exponent vb
global_params_254 = [0.33, 0.2, -0.1, 0.02, 2.0425e-6, 0.02, 0.5, 0.10, 0.0, 99.0, 3.0, 0.75]

class Watershed:
    def __init__(self, Model=190):
        self.indexed_connectivity = None
        self.dam_ids = []
        self.dam_index = []
        global global_params_190, global_params_254
        
        if Model == 190:
            self.global_params = global_params_190.copy()
            self.modeltype = 190
            print('Model 190 is being used!')
            
        elif Model == 191:
            self.global_params = global_params_190.copy()
            self.modeltype = 191
            print('Model 191(190 with dams) is being used!')
            
        elif Model == 254:
            self.global_params = global_params_254.copy()
            self.modeltype = 254
            print('Model 254 is being used!')
            
        elif Model == 255:
            self.global_params = global_params_254.copy()
            self.modeltype = 255
            print('Model 255 (254 with_dams) is being used!')
        elif Model == 256:
            self.global_params = global_params_254.copy()
            self.modeltype = 256
            print('Model 256 (254 with_dams) is being used!')
        else: 
            self.global_params = None

    @staticmethod
    def index_connectivity(connectivity, links):
        indexed_connectivity =[]
        for childs in connectivity:
            if childs==[]:
                indexed_connectivity.append([])
            else:
                l_idx = []
                for link in childs:
                    l_idx.append(links.index(link))
                indexed_connectivity.append(l_idx)
        return indexed_connectivity

    @staticmethod
    def next_link(connectivity, dim):
        nextlink = np.zeros(dim)-1 
        for i in range(0, dim):
            nextlink[connectivity[i][:]]=i
            nextlink = nextlink.astype(int) 
        return nextlink

    def __params__(self, A_i, L_i, A_h):
        
        if self.modeltype == 190 or self.modeltype == 191:
            global_params = self.global_params
            v_r = global_params[0]      #Channel reference velocity [m/s]
            lambda_1 = global_params[1] #Exponent of channel velocity discharge []
            lambda_2 = global_params[2] #Exponent of channel velocity area []
            RC = global_params[3]       #Runoff coefficient []
            v_h = global_params[4]      #Velocity of water on the hillslope [m/s]
            v_g = global_params[5]      #Velocity of water in the subsurface [m/s]

            k2 = v_h * L_i / A_h * 60.0	#[1/min]  k2
            k3 = v_g * L_i / A_h * 60.0	#[1/min]  k3
            invtau = 60.0*v_r*(A_i**lambda_2) / ((1.0 - lambda_1)*L_i)	#[1/min]  invtau
            c_1 = RC*(0.001 / 60.0)		#(mm/hr->m/min)  c_1
            c_2 = (1.0 - RC)*(0.001 / 60.0)	#(mm/hr->m/min)  c_2
            self.params = [A_i, L_i, A_h, k2, k3, invtau, c_1, c_2]
            
        elif self.modeltype == 254 or self.modeltype == 255 or self.modeltype == 256:
            global_params = self.global_params
            v_0 = global_params[0]          #[m/s]
            lambda_1 = global_params[1]     #[-]
            lambda_2 = global_params[2]     #[-]
            v_h = global_params[3]          #[m/s]
            k_i_factor =global_params[5]    #[-]

            invtau = 60.0*v_0*pow(A_i, lambda_2) / ((1.0 - lambda_1)*L_i)	 # [1/min]  invtau
            k_2 = v_h * (L_i / A_h) * 60.0                            # [1/min] k_2
            k_i = k_2 * k_i_factor                                    # [1/min] k_i
            c_1 = (0.001 / 60.0)                                      # (mm/hr->m/min)  c_1
            c_2 = A_h / 60.0                                          # c_2
            self.params = [A_i, L_i, A_h, invtau, k_2, k_i, c_1, c_2]

    def init_from_file(self, path_rvr, path_prm):
        ''' Reads rvr and prm files and extracts connectivity ot the river network and 
            parameters:
            WARNING: the order of links must be the same in rvr and prm files.
            INPUT:
                path_rvr:str, directory of rvr file
                path_prm:str, directory of prm file
        '''
        self.links, self.connectivity = read_rvr(path_rvr) 
        self.A_i, self.L_i, self.A_h = read_prm(path_prm) 
        self.dim = len(self.links)
        self.indexed_connectivity = self.index_connectivity(self.connectivity, self.links)
        self.nextlink = self.next_link(self.indexed_connectivity, self.dim)
        self.__params__(self.A_i, self.L_i, self.A_h)
        
    
    def init_custom(self, links, connectivity, A_i, L_i, A_h,):
        ''' initialize watershed object with user defined parameters
        INPUT:
            links:list, list of links
            connectivity:list, list of child links (must be the same order with links)
            A_i:list, total upstream area [km2]
            L_i:list, length of links [m]
            A_h:list, hillslope area [m2] 
        '''
        self.links = links
        self.connectivity = connectivity
        self.A_i = A_i   #[km2]
        self.L_i = L_i   #[m]
        self.A_h = A_h   #[m2]
        self.dim = len(self.links)
        self.indexed_connectivity = self.index_connectivity(self.connectivity, self.links)
        self.nextlink = self.next_link(self.indexed_connectivity, self.dim)
        self.__params__(self.A_i, self.L_i, self.A_h)
    
    def initialize(self, q, s_p, s_t =None, s_s=None, S=None):
        ''' Initial condition of the system
        MODEL-190
            Input:
                q:list, initial discharge of the links[m3/s]
                s_p: list, water ponded on the hillslopes [m]
                s_s: list, water depth in the hillslope subsurface[m]
        MODEL-191
            Input:
                q:list, initial discharge of the links[m3/s]
                S:list, initial volume of the dams [m3]
                s_p: list, water ponded on the hillslopes [m]
                s_s: list, water depth in the hillslope subsurface[m]
         MODEL-254
             Input:
                q:list, initial discharge of the links[m3/s]
                s_p: list, water ponded on the hillslopes [m]
                s_t: list, water depth in the top layer[m]
                s_s:list,  water depth in the subsurface [m]
          MODEL-255
             Input:
                q:list, initial discharge of the links[m3/s]
                S:list, initial volume of the dams [m3]
                s_p: list, water ponded on the hillslopes [m]
                s_t: list, water depth in the top layer[m]
                s_s:list,  water depth in the subsurface [m]
        '''
        
        if self.modeltype == 190:
            if s_s == None:
                print('s_s is not given. All initial conditions for s_s is set to 0!')
                s_s = [0 for _ in range(self.dim)]
            self.__yi = np.array(q + s_p +  s_s)
        
        elif self.modeltype == 191:     
            S_dams = np.zeros(self.dim)
            if self.dam_ids != []:
                j = 0
                for i in self.dam_ids:
                    idx = self.links.index(i)
                    S_dams[idx] = S[j]
                    j +=1
            else:
                print('INFO: No dam is set!')
                pass
            self.__yi = np.array(q + S_dams.tolist() + s_p + s_s )
            
        elif self.modeltype == 254:
            if s_t == None:
                print('s_t is set to None. All initial conditions for s_t is set to 0!')
                s_t = [0 for _ in range(self.dim)]
            self.__yi = np.array(q + s_p + s_t + s_s)
        
        elif self.modeltype == 255 or self.modeltype == 256:     
            S_dams = np.zeros(self.dim)
            if self.dam_ids != []:
                j = 0
                for i in self.dam_ids:
                    idx = self.links.index(i)
                    S_dams[idx] = S[j]
                    j +=1
            else:
                print('INFO: No dam is set!')
                pass
            self.__yi = np.array(q + S_dams.tolist() + s_p + s_t + s_s)

    def dam_loc_state(self,states = None):
        ''' Gets previously given dam ids(link ids where dams are located) and given states and 
            convert it into arrays to be able to run the simulations 
        INPUT:
            states:list, a list of states(0 or 1), must follow the order in dam_ids
        '''
        self.__dam = np.zeros(self.dim).astype(int)
        self.__state = np.zeros(self.dim).astype(int)
        self.dam_index = []
        j=0
        if self.dam_ids != []:
            for i in self.dam_ids:
                idx = self.links.index(i)
                self.dam_index.append(idx)
                self.__dam[idx] = 1        
                self.__state[idx] = states[j]
                j+=1
        else:
            if states is not None:
                print("No dam is set! Given states will be ignored.")
            else:    
                print('No dam is set!')
            pass

    def set_dam_state(self, states=None):
        ''' Gets previously given dam ids(link ids where dams are located) and given states and 
            convert it into arrays to be able to run the simulations 
        INPUT:
            states:list, a list of states(0 or 1), must follow the order in dam_ids
        '''
        if self.modeltype ==256:
            self.__dam = np.zeros(self.dim).astype(float)
        else:
            self.__dam = np.zeros(self.dim).astype(int)

        self.__state = np.zeros(self.dim).astype(int)
        self.dam_index = []
        j=0
        if self.dam_ids != []:
            for i in self.dam_ids:
                idx = self.links.index(i)
                self.dam_index.append(idx)
                self.__dam[idx] = 1        
                self.__state[idx] = states[j]
                j+=1
        else:
            if states is not None:
                print("No dam is set! Given states will be ignored.")
            else:    
                print('No dam is set!')
            pass

    def init_dam_params256(self, h_spill,h_max, s_max, alpha, diameter, c_1, c_2, l_spill, l_crest):
        ''' Returns an nested list of dam parameters to be used in the model runs

        '''
        H_spill = np.zeros(self.dim)
        H_max = np.zeros(self.dim)
        S_max = np.zeros(self.dim)
        _alpha = np.zeros(self.dim)
        diam = np.zeros(self.dim)
        c1 = np.zeros(self.dim)
        c2 = np.zeros(self.dim)
        L_spill = np.zeros(self.dim)
        L_crest = np.zeros(self.dim)
        j = 0
        if self.dam_ids != []:
            for i in self.dam_ids:
                idx = self.links.index(i)
                H_spill[idx] = h_spill[j] 
                H_max[idx] = h_max[j]
                S_max[idx] = s_max[j]
                _alpha[idx] = alpha[j]
                diam[idx] = diameter[j]
                c1[idx] = c_1[j]
                c2[idx] = c_2[j]
                L_spill[idx] = l_spill[j]
                L_crest[idx] = l_crest[j]
                j += 1
        else:
            print('No dam is set!')
        dam_params = [H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest]
        return dam_params

        
    def Run_190(self, t_span, forcing, t_eval=None, rtol=1e-6):
        
        self.sol = solve_ivp(Model_190, t_span, t_eval=t_eval, y0=self.__yi,
                            args=(forcing, self.global_params, self.params, self.indexed_connectivity))
        discharge = self.sol.y.T[:, 0:self.dim]
        
        col1, _ = self.__columns__()
        streamflow  = pd.DataFrame(discharge, index = self.sol.t, columns=col1)
        
        return streamflow
        
    def Run_191(self, t_span, forcing, dam_params, t_eval=None, rtol=1e-6):
        
        if self.__state is None and self.dam_ids==[]:
            self.__state = np.zeros(self.dim)

        dam_params = [self.__dam] + dam_params

        self.sol = solve_ivp(Model_190_dam, t_span, t_eval = t_eval, y0 = self.__yi, 
                        args=(forcing, self.global_params, self.params, dam_params, 
                        self.indexed_connectivity, self.nextlink, self.__state), rtol = rtol)
        
        discharge = self.sol.y.T[:, 0:self.dim]
        dam_storage = self.sol.y[self.dim:2*self.dim, :][self.dam_index].T
        
        col1, col2 = self.__columns__()
        streamflow  = pd.DataFrame(discharge, index = self.sol.t, columns=col1)
        storage  = pd.DataFrame(dam_storage, index = self.sol.t, columns=col2)

        return streamflow, storage
    
    
    def Run_254(self, t_span, forcing, t_eval=None, rtol=1e-6):
        
        self.sol = solve_ivp(Model_254, t_span, t_eval=t_eval, y0=self.__yi,
                            args=(forcing, self.global_params, self.params, self.indexed_connectivity))
    
        discharge = self.sol.y.T[:, 0:self.dim]
        
        col1, _ = self.__columns__()
        streamflow  = pd.DataFrame(discharge, index = self.sol.t, columns=col1)
        
        return streamflow
    
    def Run_255(self, t_span, forcing, dam_params, t_eval=None, rtol=1e-6, method='RK45'):
        
        if self.__state is None and self.dam_ids==[]:
            self.__state = np.zeros(self.dim)

        dam_params = [self.__dam] + dam_params

        self.sol = solve_ivp(Model_254_dam, t_span, t_eval = t_eval, y0 = self.__yi, 
                        args=(forcing, self.global_params, self.params, dam_params, 
                        self.indexed_connectivity, self.nextlink, self.__state), rtol = rtol, method=method)
        
        discharge = self.sol.y.T[:, 0:self.dim]
        dam_storage = self.sol.y[self.dim:2*self.dim, :][self.dam_index].T
        
        col1, col2 = self.__columns__()
        streamflow  = pd.DataFrame(discharge, index = self.sol.t, columns=col1)
        storage  = pd.DataFrame(dam_storage, index = self.sol.t, columns=col2)

        return streamflow, storage

    def Run_256(self, t_span, forcing, dam_params, t_eval=None, rtol=1e-6, method='RK45'):
        
        if self.__state is None and self.dam_ids==[]:
            self.__state = np.zeros(self.dim)

        dam_params = [self.__dam] + dam_params

        self.sol = solve_ivp(Model_254_dam_varParam, t_span, t_eval = t_eval, y0 = self.__yi, 
                        args=(forcing, self.global_params, self.params, dam_params, 
                        self.indexed_connectivity, self.nextlink, self.__state), rtol = rtol, method=method)
        
        discharge = self.sol.y.T[:, 0:self.dim]
        dam_storage = self.sol.y[self.dim:2*self.dim, :][self.dam_index].T
        
        col1, col2 = self.__columns__()
        streamflow  = pd.DataFrame(discharge, index = self.sol.t, columns=col1)
        storage  = pd.DataFrame(dam_storage, index = self.sol.t, columns=col2)

        return streamflow, storage
    
    
    def __columns__(self):
        
        return [str(self.links[i]) for i in range(self.dim)],\
        [str(self.links[i]) for i in self.dam_index]
    

    def Get_Snapshot(self):
        ''' Gets last condition of the states from solve_ivp output
        
        OUTPUT:
         For model 191, the order of states q, S, s_p, s_s
         For model 255, the orer of states q, S, s_p, s_t, s_s
        '''
        if self.sol == None:
            print('Please run the simulation to be able get a snaphot!')
        else:
            if self.modeltype == 191:
                q = self.sol.y.T[:, 0:self.dim][-1].tolist()
                S = self.sol.y[self.dim:2*self.dim, :][self.dam_index].T[-1].tolist()
                s_p = self.sol.y.T[:, 2*self.dim:3*self.dim][-1].tolist()
                s_s = self.sol.y.T[:, 3*self.dim:4*self.dim][-1].tolist()
                return q, S, s_p, s_s
            
            elif self.modeltype == 255 or self.modeltype==256:
                q = self.sol.y.T[:, 0:self.dim][-1].tolist()
                S = self.sol.y[self.dim:2*self.dim, :][self.dam_index].T[-1].tolist()
                s_p = self.sol.y.T[:, 2*self.dim:3*self.dim][-1].tolist()
                s_t = self.sol.y.T[:, 3*self.dim:4*self.dim][-1].tolist()
                s_s = self.sol.y.T[:, 4*self.dim:5*self.dim][-1].tolist()
                return q, S, s_p, s_t, s_s
