import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib import rcParams
from matplotlib.pyplot import savefig
from matplotlib.lines import Line2D

def read_rvr(path):
    rvr = []
    with open(path) as _:
        for line in _:
            line = line.strip()
            if line:
                rvr.append(line)
    # n_links = int(rvr[0])
    links = []
    connectivity = []
    for i in range(1,len(rvr)):
        line = rvr[i]
        if i%2 != 0:
            links.append(int(line))
        else:
            conn = list(map(int, rvr[i].split(' ')))
            if conn==[0]:conn=[]
            connectivity.append(conn[1:]) #First element, number of parents,not included
    return links, connectivity

def read_prm(path):
    '''Returns infromation in rvr file
    input:
        path = rvr directory
    returns:
        A_i:list,  Area of total upstream area of a link [km^2]
        L_i:list, Length of the link [m]
        A_h:list, Area of the hillslope [m^2]
    '''
    prm = []
    with open(path) as _:
        for line in _:
            line = line.strip()
            if line:
                prm.append(line)
    links = []
    params = []
    for i in range(1, len(prm)):
        line = prm[i]
        if i%2 != 0:
            links.append(int(line))
        else:
            prms = list(map(float, prm[i].split(' ')))
            params.append(prms)#
    params =np.array(params)
    A_i = params[:, 0]
    L_i = params[:, 1]*10**3
    A_h = params[:, 2]*10**6
    return A_i, L_i, A_h

def GetForcing(path, start, end):
    ''' Reads raw data and generates precipitation time series for hlm_basic
        This function basicly repeats  hourly data 60 times 
    
    INPUT: 
        path:str,directory of hourly rainfall data (must be pd.Series)
        start:str, start date (included)
        end: str, end date (not included)
    OUTPUT:
        forcing: a list,  input of model
        subset: pd.Series, raw data 
    '''

    raw_data = pd.read_csv(path, index_col=0, parse_dates=True)
    subset = raw_data[(raw_data.index >= start) & (raw_data.index < end)]
    timeseries = subset.to_numpy().flatten()
    
    forcing  = []
    for prcp in timeseries:
        forcing.extend([prcp for _ in range(60)])
    return forcing, subset

def Set_InitialConditions(qmin, At_up, A_up, k3=340):
    ''' Returns a dictionary including initial conditions for states(i.e channels, )

    INPUT:
        qmin:float,  baseflow observed at the outlet [m3/s]
        At_up :float,  Total upstream area [km2]
        A_up:np.array, upstram area of all links [km2]
        k3:float, number of days ground water flow reaches adjacent stream 
    OUTPUT:
        q: initial condition od channel flow
        s_p: initial condition of ponding, set zero for all
        s_t: initial condition of top layer, set 1.000000e-6 for all
        s_s: initial condition of subsurface
    '''
    dim  = len(A_up)
    k3 = 1/(k3 * 24 * 60)
    factor = 60/1e6
    q = ((qmin / At_up) * A_up).tolist()
    s_p = [0.0 for _ in range(dim)]
    s_t = [1.000000e-6 for _ in range(dim)]
    ss = qmin / (At_up * k3) * factor
    ss = round(ss, 5)
    s_s = [ss for _ in range(dim)]

    return q, s_p, s_t, s_s



def plot_sim(link_ids, forcing, results, plt_kwargs,d_type='discharge',discharge_axis=None, area=None, save=None, max_storage=None):
    '''Plots simulation results
    
    INPUT:
        links_ids:int or list, link ids for which hydrograph or storage (if a dam exists) to be plotted
        forcing:list, minute-based precipitation data
        results:list, includes pd.DataFrames which are output of hlm_basic
        plt_kwargs:list, includes pyplot kwargs. kwargs dictionaries must follow the order of results
        d_type:'discharge' or 'storage', type of the data
        discharge_axis:(optional) list, to customize discharge axis [min, max, stepsize]
        area:(optional) float, upstream aream of correspoding link,use to show mean annual flood level
        save:(optional) str, save name (with or without full path)
    '''
    
    rcParams.update({'font.size': 13,'axes.labelweight':'bold','axes.labelsize':14,\
                            'ytick.major.size':6,'xtick.major.size':6,'xtick.direction':'in','ytick.direction':'in',\
                            'lines.linewidth':3.5})
    kwargs = plt_kwargs.copy()
    fig, ax = plt.subplots(2, 1,figsize=(20, 6), gridspec_kw={'height_ratios':[1, 3]}, sharex=True)

    ax[0].plot(range(len(forcing)), forcing, alpha=1, color='b')
    ax[0].set_ylim([max(forcing)*1.1, 0])
    ax[0].set_ylabel('Rainfall \n[mm/hr]')
    ax[0].grid()
    j=0
    if type(link_ids) != list: link_ids =[link_ids]
    for result in results:
        for link_id in link_ids:
            ax[1].plot(result.index, result[str(link_id)].values, **kwargs[j])
        j += 1
    if d_type == 'discharge':
        ax[1].set(xlabel='Time[min]', ylabel='Discharge[m$^3$/s]')
        ax[1].set_xlim([0, len(forcing)])
        # leg_title = 'LINK'
        if discharge_axis is not None:
            start = discharge_axis[0]
            end = discharge_axis[1]
            step =discharge_axis[2]
            ax[1].set_ylim([start, end])
            ax[1].set_yticks(np.arange(start, end, step))
        if area is not None:
            maf = round(3.12 * area**0.57 , 2)
            ax[1].axhline(y=maf, c='r', linestyle='dashed', linewidth=2)
            ax[1].text(10000, maf, f'Mean Annual Flood = {maf} m$^3$/s', va='bottom', ha='center')

    elif d_type == 'storage':
        ax[1].set(xlabel='Time[min]', ylabel='Storage[10$^3$ m$^3$]')
        ax[1].set_xlim([0, len(forcing)])
        if max_storage is not None:
            ax[1].set_ylim([-5, max_storage+50000])
            ax[1].axhline(y=max_storage, c='r', linestyle='dashed', linewidth=2)
            ax[1].set_yticks(np.arange(0,max_storage+50000,100000))
            ax[1].set_yticklabels(np.arange(0,int((max_storage+50000)/1000), 100))

    colors = []
    labels = []
    for i in range(len(plt_kwargs)):
        colors.append(plt_kwargs[i]['color'])
        labels.append(plt_kwargs[i]['label'])
    lines = [Line2D([0], [0], color=c) for c in colors]
    ax[1].legend(lines, labels, loc='upper left')

    ax[1].grid()
    plt.subplots_adjust(hspace=0)
    if save is not None:
        fig.savefig(save + '.png',bbox_inches = 'tight', pad_inches = 0.5)