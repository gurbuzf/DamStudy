import itertools
import numpy as np

def generator(num_digit,iterable):
    d = itertools.product(iterable, repeat=num_digit)
    result = []
    for i in d:
        result.append(list(i))
    return result


def link_number(num_digit, link_idset):
    ID_number =0
    for i in range(num_digit):
        power = (num_digit-1)-i
        ID_number = ID_number + (3**power)*link_idset[i]
    return ID_number

def conn_find(ID_set):
    if ID_set[-1] == 1:
        c = []
    elif ID_set[-1] == 0:
        c1 = list(ID_set)
        c2 = list(ID_set)
        c1[-1] = 1
        c2[-1] = 2
        c = [link_number(len(c1),c1),
              link_number(len(c2),c2)]

    i=-1
    reserve = list(ID_set)
    
    while ID_set[i]==2:
        i-=1
        if i == -1-len(ID_set):
            break
        else:
            pass
        if ID_set[i] !=0:
            c = []
        elif ID_set[i]==0:
            for k in range(i, 0, 1):
                if ID_set[k] == 2:
                    d1 = reserve
                    d1[k]=0
            c1 = list(d1)
            c2 = list(d1)
            c1[i] = 1   #TODO: to be made some changes here 
            c2[i] = 2
            c = [link_number(len(c1),c1),
                 link_number(len(c2),c2)]  
            
    return c

#H_order represents Horton order of each link
def Horton_order(conn_array):
    length = len(conn_array)
    H_order = np.zeros(length)  
    n = list(H_order).count(0)
    while n>0:
        for i in range(length):
            if conn_array[i] == []:
                H_order[i] = int(1)
            elif conn_array[i] !=[]:
                if H_order[conn_array[i][0]]!=0 and H_order[conn_array[i][1]]!=0:
                    if H_order[conn_array[i][0]] != H_order[conn_array[i][1]]:
                        s = max(H_order[conn_array[i][0]],H_order[conn_array[i][1]])
                        H_order[i] = s
                    else:
                        s = H_order[conn_array[i][0]] 
                        H_order[i] = s+1        
        n = list(H_order).count(0)
    return H_order

        
def GenerateNetwork(n):
    ''' Returns a list 
    [link_id, connectivity, link_order, nextlink]
    '''
    l= [0,1,2]
    #Generate digits for each link
    Link_ID_set = generator(n, l)
    n_hills = len(Link_ID_set)
    #Link_id and connectivity
    connectivity = []
    link_id = []
    for idx in range(n_hills):
        connectivity.append(conn_find(Link_ID_set[idx]))
        link_id.append(link_number(n, Link_ID_set[idx]))
    
    #Horton order 
    link_order = Horton_order(connectivity)
    link_order = link_order.astype(int)
    link_order = list(link_order)
    #Next_Link
    nextlink = np.zeros(n_hills)-1 
    for i in range(n_hills):
        nextlink[connectivity[i][:]] = i
    nextlink = nextlink.astype(int) 
    nextlink = list(nextlink)
    return [link_id, connectivity, link_order, nextlink]

def UpstreamArea(a_hill, connectivity, h_order):
    n_hills = len(connectivity)
    area =np.zeros(n_hills)
    orders = sorted(list(set(h_order)))
    for order in orders:

        indices = sorted([i for i,val in enumerate(h_order) if val==order],reverse=True)
        if order == 1:
            for idx in indices:
                area[idx] = 1
        else:
            for idx in indices:
                area[idx] = sum(area[connectivity[idx]])+1
    A_i = (a_hill*area).astype(np.float32)
    return A_i
    