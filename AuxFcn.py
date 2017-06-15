# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 00:58:23 2017

@author: Wayne
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
#def create_dataset(dataset, look_back=6):
#	 dataX, dataY = [], []
#	 for i in range(len(dataset)-look_back-1):
# 		  a = dataset[i:(i+look_back), 0] 
#		  dataX.append(a)
#		  dataY.append(dataset[i + look_back, 0])
#	 return np.array(dataX), np.array(dataY)
#%%
def create_catgorical_dataset(data, feature_dim,time_step=6,ix_list=None):
    """
    data: (n,) int32, each elem is in {0,1,2,...,feature_dim}
    Output: (n,time_step,feature_dim)
    The feature will be represented as one-hot representation.
    
    Assume dataset is z_1,z_2,...z_n, and time_step=3, feature_dim=2
    This function generate:
        x1,x2,x3
        x2,x3,x4
        x3,x4,x5
        x4,x5,x6
        ..., where xi's are (2,) booling array.
    And
        x4,
        x5,
        x6,
        ...
    """
    if ix_list is None:
        start = 0
        end = len(data)-1
    else:
        start, end = ix_list
    
    n = end-start+1
    if n-time_step<1:
        raise ValueError('Not enough sample for this time_step=%d'%time_step)
    x_all = np.zeros((n-time_step,time_step,feature_dim),dtype = bool) # encode in 1-of-k representation
    x_tmp = np.zeros((n,feature_dim),dtype=bool)
    for i in range(start,end):
        x_tmp[i,data[i]]=True

    for j in range(time_step):
        x_all[:,j,:]=x_tmp[j:n-time_step+j,:]
    y = x_tmp[time_step:n,:]
    return x_all,y

def translate(x_kp, diction):
    '''
    x_kp: (k,p) boolean, 
    diction: dictionary with p keys
    '''
    y_k = np.argmax(x_kp,axis=1)
    word = ''
    for i in range(len(y_k)):
        word+=diction[y_k[i]]
    return word
#%%
def data_reducing(data):
    """
    the input is raw text data in string, e.g.
    'I hope you will be ready to own publicly, whenev.....'
    
    This function will replace capital characters with the lower case ones.
    Also only retain ',' '.' '?' '!' ' ' '\n'
    
    """
    #%%
    cap = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    low = 'abcdefghijklmnopqrstuvwxyz'
    apos= ",.?! \n:;-'"
    
    dic1 = {cap[i]:low[i] for i in range(26)}
    dic2 = {low[i]:low[i] for i in range(26)}
    dic3 = {apos[i]:apos[i] for i in range(len(apos))}
    M = {**dic1,**dic2,**dic3}
    #%%
    c = M[data[0]]
    for i in range(1,len(data)):
        try:
            c+= M[data[i]]
        except:
            c+=' '
    return c
    #%%
def txt_gen(model,initial_x,n,diction=None,seed = 2):
    """
    initial_x_t: (time_step,dim), dim is the dimension of features of a word.
    n: number of word
    diction: dictionary to convert integer to word
    """
    np.random.seed(seed)
    x = initial_x
    time_step,dim = x.shape
    y_gen = np.zeros((n,),dtype='int32')
    i=0
    while i<n:    
        y = model.predict(x[None,:,:]) 
        # y is (dim,), y is the probability of each possible words
        # Also note, sum(y)=1
        yi = np.random.choice(dim,1, p=y.flatten())[0]
        y_gen[i]=yi
        x[0:time_step-1,:] = x[1:time_step,:]
        x[-1,:]      = np.zeros((dim,),dtype='bool') 
        x[-1,yi] = True
        i+=1
    if diction:
        words = ''
        for i in range(n):
            words+=diction[y_gen[i]]
        return words
    return y    
                    
def print_model_history(history,plot_val=False):
    history_dict = history.history
    #history_dict.keys()
    epo = np.arange(0,len(history_dict['acc']))
    plt.plot(epo,history_dict['acc'],'g-')
    if plot_val:
        plt.plot(epo,history_dict['val_acc'],'r-')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    