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

def GetForcing(path, start, end, cumulative=False):
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
    sub_cumsum = subset.cumsum().values.flatten()
    timeseries = subset.to_numpy().flatten()
    
    forcing  = []
    for prcp in timeseries:
        forcing.extend([prcp for _ in range(60)])
    if cumulative==True:
        cumsum = []
        for prcp in sub_cumsum:
            cumsum.extend([prcp for _ in range(60)])
        return forcing, subset, cumsum
    else:
        return forcing, subset

def Generate_SyntheticStorm(dStorm, duration, rate = None, timescale=60):
    ''' Generates a synthetic storm for a given design storm
        
        Parameters:
            dstorm:float, design storm
            duration:int, number of hours
            rate:array like, design storm distribution in time(optional) 
            timescale:int, the timescale of given rates in minutes (USE 30 or 60 depending on the timescale of rates)
    
    '''
    n_hour  = 60/timescale
    duration = int(duration * n_hour)
    if rate == None:
        rate =  [0.001, 0.01,0.03,0.08,0.8,0.04,0.02,0.01,0.005, 0.004] 
    
    forcing_h = []
    for i in range(duration):
        try:
            forcing_h.append(rate[i]*dStorm)
        except IndexError:
            forcing_h.append(0)
    cum_forcing = np.cumsum(forcing_h)
    cum_forcing = np.repeat(cum_forcing, timescale)
    forcing = np.repeat(forcing_h, timescale)
    
    return forcing, cum_forcing, forcing_h

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

def PrepareDamParams(dam_links):
    ''' Returns each dam parameters in the order of dams given in dam_links

        parameters:
            dam_links:array like, the links where dams located

        returns:
            _alpha, c1, c2, H_spill, H_max, diam, S_max, L_spill, L_crest
    '''

    order_3 = [9,36,45,63,90,117,126,144,153,171,198,207,225,234] 
    order_4 = [27, 189, 216, 135, 108,]
    order_5 = [81,162]
    n_dams = len(dam_links)
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
    for dam in dam_links:
        if dam in order_3:
            H_spill.append(4.25)
            H_max.append(5)
            diam.append(0.50)
            S_max.append(45000)
            L_spill.append(3.0)
            L_crest.append(8.0)
        elif dam in order_4:
            H_spill.append(4.0)
            H_max.append(5.0)
            diam.append(0.75)
            S_max.append(135000)
            L_spill.append(4.0)
            L_crest.append(10.0)
        elif dam in order_5:
            H_spill.append(10)
            H_max.append(11)
            diam.append(1.00)
            S_max.append(405000)
            L_spill.append(6.0)
            L_crest.append(15.0)
    return H_spill, H_max, S_max, _alpha, diam, c1, c2, L_spill, L_crest




def plot_sim(link_ids, forcing, results, plt_kwargs,d_type='discharge',discharge_axis=None, area=None, save=None, 
            max_storage=None, storage_int=5, fig_size=(20, 6), show_rain=True, x_timescale='hour', x_stepsize=5, legend=True):
    '''Plots simulation results
    
    Parameters:
        links_ids:int or list, link ids for which hydrograph or storage (if a dam exists) to be plotted
        forcing:list, minute-based precipitation data
        results:list, includes pd.DataFrames which are output of hlm_basic
        plt_kwargs:list, includes pyplot kwargs. kwargs dictionaries must follow the order of results
        d_type:'discharge' or 'storage', type of the data
        discharge_axis:(optional) list, to customize discharge axis [min, max, stepsize]
        area:(optional) float, upstream aream of correspoding link,use to show mean annual flood level
        save:(optional) str, save name (with or without full path)
        max_storage:float, draws a line at maximum capacity of a dam
        storage_int:float,  steps for axis ticks when storage is shown
        fig_size:tuple, (x, y) the size of figure
        show_rain:boolean, shows rainfall axes if true
        x_timescale:str, if None, 'minute' else 'hour', 'day'
        x_stepsize;int, the intervals of x ticks
        legend:boolean, shows legend if true   

    '''
    
    rcParams.update({'font.size': 13,'font.family':'sans-serif','font.sans-serif':['Arial'],'axes.labelweight':'bold','axes.labelsize':14,\
                            'ytick.major.size':6,'xtick.major.size':6,'xtick.direction':'in','ytick.direction':'in',\
                            'lines.linewidth':3.5})
    kwargs = plt_kwargs.copy()
    init_time = np.floor(results[0].index[0])
    if x_timescale=='hour':
        min2newScale = 60
        xlabel = 'Time[hour]'
    elif x_timescale=='day':
        min2newScale = 24*60
        xlabel = 'Time[day]'
    elif x_timescale==None:
        min2newScale = 1
        xlabel = 'Time[min]'


    
    fig, ax = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios':[1, 3]}, sharex=True)

    ax[0].plot(np.arange(init_time, init_time+len(forcing)), forcing, alpha=1, color='b')
    ax[0].set_ylim([max(forcing)*1.1, 0])
    ax[0].set_ylabel('Rainfall \n[mm]')
    ax[0].set_xlim([init_time, init_time+len(forcing)])
    ax[0].set_xticks(np.arange(init_time, init_time+len(forcing), x_stepsize*min2newScale))
    ax[0].set_xticklabels(np.arange(init_time/(min2newScale), (init_time+len(forcing))/(min2newScale), x_stepsize).astype(int))
    ax[0].grid()
    ax[0].set_visible(show_rain)
    j=0
    if type(link_ids) != list: link_ids =[link_ids]
    for result in results:
        for link_id in link_ids:
            ax[1].plot(result.index, result[str(link_id)].values, **kwargs[j])
        j += 1
    if d_type == 'discharge':
        ax[1].set(xlabel=xlabel, ylabel='Discharge[m$^3$/s]')
        # ax[1].set_xlim([0, len(forcing)])
        # leg_title = 'LINK'
        if discharge_axis is not None:
            start = discharge_axis[0]
            end = discharge_axis[1]
            step = discharge_axis[2]
            ax[1].set_ylim([start, end])
            ax[1].set_yticks(np.arange(start, end, step))
        if area is not None:
            maf = round(3.12 * area**0.57 , 2)
            ax[1].axhline(y=maf, c='r', linestyle='dashed', linewidth=2)
            ax[1].text(results[0].index[0]+len(forcing)/4, maf, f'Mean Annual Flood = {maf} m$^3$/s', va='bottom', ha='center')

    elif d_type == 'storage':
        ax[1].set(xlabel=xlabel, ylabel='Storage[10$^3$ m$^3$]')
        # ax[1].set_xlim([0, len(forcing)])
        if max_storage is not None:
            step = len(str(max_storage))-2
            ax[1].set_ylim([-5, max_storage*1.15])
            ax[1].axhline(y=max_storage, c='r', linestyle='dashed', linewidth=2)
            ax[1].set_yticks(np.arange(0,max_storage*1.1,storage_int*10**step))
            ax[1].set_yticklabels(np.arange(0,int((max_storage*1.1)/1000), storage_int*10**step/1000).astype(int))

    if legend == True: 
        colors = []
        labels = []
        linestyles = []
        for i in range(len(plt_kwargs)):
            colors.append(plt_kwargs[i]['color'])
            labels.append(plt_kwargs[i]['label'])
            try:
                linestyles.append(plt_kwargs[i]['linestyle'])
            except KeyError:
                linestyles.append('solid')
        lines = [Line2D([0], [0], color=c, linestyle=l) for c, l in zip(colors, linestyles)]
        ax[1].legend(lines, labels, loc='upper right',framealpha=1, edgecolor='k')

    ax[1].grid()
    plt.subplots_adjust(hspace=0)
    if save is not None:
        fig.savefig(save + '.png',bbox_inches = 'tight', pad_inches = 0.5)



def Partial_PLOT(start, end,link,forcing_data, dataset,plt_kwargs, d_type='discharge',discharge_axis=None, 
                    area=None, save=None, max_storage=None, storage_int=5, fig_size=(20, 6), show_rain=True):

    start = start * 24 *60
    end = end * 24 *60
    forcing = forcing_data[start:end].copy()
    _dataset = dataset.copy()
    for i, data in enumerate(dataset):
        _dataset[i] = data[(data.index>=start)&(data.index<=end)]
    plot_sim(link, forcing, _dataset, plt_kwargs,d_type,discharge_axis, area, save, max_storage, storage_int, fig_size, show_rain)


