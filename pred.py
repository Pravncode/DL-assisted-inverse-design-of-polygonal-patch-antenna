import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# %reload_ext autoreload
from os import system
# import subprocess
import itertools as itt
# import multiprocessing
import time
import math
# import de_nn as de_nn
from scipy.stats import qmc
from numpy import *
import random
# from matplotlib.patches import Circle, Wedge, Polygon
# from matplotlib.collections import PatchCollection
from itertools import chain
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
model1=load_model('best_model2.h5')
# model1=load_model('model_combined2.h5')
# model1=load_model('model_combined.h5')
# import matlab.engine

def polar_to_cartesian(pop):

    rad = pop[:,0:4]
    phi = pop[:,4:8]
    alp = pop[:,8]
    rad1 = np.zeros((np.shape(pop)[0],8))
    phi1 = np.zeros((np.shape(pop)[0],8))
    
    for i in range (np.shape(pop)[0]):
        for j in range(4):
            rad1[i,j] = rad[i,j]
            phi1[i,j] = phi[i,j]
            rad1[i, j+4] = rad[i, 3-j]
            phi1[i, j+4] = phi[i, 3-j]
    
    tht = np.zeros((np.shape(pop)[0],8))

    phi_sum = np.cumsum(phi1, axis=1)
    # print(alp, '\n', phi1)
    for i in range (np.shape(pop)[0]):
        for j in range(8):
            tht[i,j] = phi_sum[i,j]*alp[i]/phi_sum[i,-1]
   
    X=[]
    Y=[]
    for i in range(np.shape(pop)[0]):
        r, t = rad1[i], tht[i]
        xc = np.zeros(8)
        yc = np.zeros(8)

        for j in range (8):
            xc[j] = r[j]*math.cos(t[j])
            yc[j] = r[j]*math.sin(t[j])
        x1 = xc[0:4]
        x2 = np.flip(x1)
        y1 = yc[0:4]
        y2 = -y1
        y2 = np.flip(y2)
        x3 = np.append(x1,x2)
        y3 = np.append(y1,y2)
        if (np.all(x3<0) or np.all(x3>0)):
            false_ind = np.where(np.logical_and(x3<1,x3>-1))[0]
            for ind in false_ind:
                if x3[ind] > 0:
                    x3[ind] = -1.2
                else:
                    x3[ind] = 1.2
        
        X.append(x3/1000)
        Y.append(y3/1000)
    XY=np.concatenate((X,Y),axis=1)
    return (XY)

def prediction(pop):
    xy = polar_to_cartesian(pop)
    lower = np.array([-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,
                  -0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03,-0.03])
                  

    upper = np.array([0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,
                  0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03])
                  
    for i in range (np.shape(xy)[0]):
        xy[i] = (xy[i]-lower)/(upper-lower)

    p = model1.predict(xy)
    return(p)


def get_mse(pop, tar,n_i,n_p):
    isl = np.array(list(chain.from_iterable(pop)))
    tar = np.tile(tar, (n_i*n_p,1))    
    pp = prediction(isl)
    mse = pp[:,52]
#     mse = np.mean((tar - pp)**2, axis=1)
    return np.reshape(mse,(n_i,n_p))

def get_mse3(pop, tar,n_i,n_p):
    isl = np.array(list(chain.from_iterable(pop)))
    Gtar = tar[0:41600]
    Gtar = np.reshape(Gtar,(128,25,13))
    Gtar = Gtar[51,:,:]
    
    Star = tar[41600:41728]
    Star = Star[51]
    Rtar = tar[41728:41856]
    Rtar = Rtar[51]
    
    tar1 = np.tile(Gtar, (n_i*n_p,1,1,1))
    tar2 = np.tile(Star, (n_i*n_p,1))
    tar3 = np.tile(Rtar, (n_i*n_p,1))
    
    y = prediction(isl)
    y1=y[:,0:41600]
    y1=np.reshape(y1,(n_i*n_p,13,25,128))
    y1=np.swapaxes(y1,1,3)
    y1=y1[:,48:54,:,:]
    

    y2=y[:,41600:41728]
    y2=y2[:,48:54]
    y3=y[:,41728:41856]
    y3=y3[:,48:54]
#     print(np.shape(Gtar),np.shape(tar1),np.shape(y1))
    m_1 = np.mean(np.mean(np.mean((y1-tar1)**2, axis = 3), axis = 2), axis =1)
#     m_1 = np.reshape(m_1,(n_i*n_p,325))
    m_2 = np.mean((y2-tar2)**2, axis = 1)
    m_3 = np.mean((y3-tar3)**2, axis = 1)
#     print(np.shape(m_1),np.shape(m_2))
    mse = 0.5*m_1 + m_2 + 0.01*m_3
#     mse = 0.05*m_1 + 1*m_2 + 0.0001*m_3
#     print(m)
    return np.reshape(mse,(n_i,n_p))

def get_msed(pop, tar,n_i,n_p):
    isl = np.array(list(chain.from_iterable(pop)))
    Gtar = tar[0:41600]
    Gtar = np.reshape(Gtar,(128,25,13))
    Ltar = Gtar[50:54,:,:]
    Mtar = Gtar[102:106,:,:]
    
    Ptar = tar[41600:41728]
    Star = Ptar[50:54]
    Qtar = Ptar[102:106]
    Btar = tar[41728:41856]
    Rtar = Btar[50:54]
    Ctar = Btar[102:106]
    
    tar1 = np.tile(Ltar, (n_i*n_p,1,1,1))
    tar6 = np.tile(Mtar, (n_i*n_p,1,1,1))
    tar2 = np.tile(Star, (n_i*n_p,1))
    tar3 = np.tile(Rtar, (n_i*n_p,1))
    tar4 = np.tile(Qtar, (n_i*n_p,1))
    tar5 = np.tile(Ctar, (n_i*n_p,1))
    
    y = prediction(isl)
    y1=y[:,0:41600]
    y1=np.reshape(y1,(n_i*n_p,13,25,128))
    y1=np.swapaxes(y1,1,3)
    ya=y1[:,50:54,:,:]
    yb=y1[:,102:106,:,:]

    Y2=y[:,41600:41728]
    y2=Y2[:,50:54]
    Y1=Y2[:,102:106]
    Y3=y[:,41728:41856]
    y3=Y3[:,50:54]
    Y4=Y3[:,102:106]
#     print(np.shape(Gtar),np.shape(tar1),np.shape(y1))
    m_0 = np.mean(np.mean(np.mean((ya-tar1)**2, axis = 3), axis = 2), axis =1)
    m_1 = np.mean(np.mean(np.mean((yb-tar6)**2, axis = 3), axis = 2), axis =1)
#     m_1 = np.reshape(m_1,(n_i*n_p,325))
    m_2 = np.mean((y2-tar2)**2, axis = 1)
    m_4 = np.mean((Y1-tar4)**2, axis = 1)
    m_3 = np.mean((y3-tar3)**2, axis = 1)
    m_5 = np.mean((Y4-tar5)**2, axis = 1)
#     print(np.shape(m_1),np.shape(m_2))
    mse = 0.5*(m_1 + m_0) + (m_2 + m_4) + 0.01*(m_3 + m_5)
#     print(m)
    return np.reshape(mse,(n_i,n_p))

## for matlab prediction

def polar_to_cartesian_mat(pop):
    rad = pop[:,0:4]
    phi = pop[:,4:8]
    alp = pop[:,8]
    rad1 = np.zeros((np.shape(pop)[0],8))
    phi1 = np.zeros((np.shape(pop)[0],8))
    
    for i in range (np.shape(pop)[0]):
        for j in range(4):
            rad1[i,j] = rad[i,j]
            phi1[i,j] = phi[i,j]
            rad1[i, j+4] = rad[i, 3-j]
            phi1[i, j+4] = phi[i, 3-j]
    
    tht = np.zeros((np.shape(pop)[0],8))

    phi_sum = np.cumsum(phi1, axis=1)
    # print(alp, '\n', phi1)
    for i in range (np.shape(pop)[0]):
        for j in range(8):
            tht[i,j] = phi_sum[i,j]*alp[i]/phi_sum[i,-1]
   
    X=[]
    Y=[]
    for i in range(np.shape(pop)[0]):
        r, t = rad1[i], tht[i]
        xc = np.zeros(8)
        yc = np.zeros(8)

        for j in range (8):
            xc[j] = r[j]*math.cos(t[j])
            yc[j] = r[j]*math.sin(t[j])
        x1 = xc[0:4]
        x2 = np.flip(x1)
        y1 = yc[0:4]
        y2 = -y1
        y2 = np.flip(y2)
        x3 = np.append(x1,x2)
        y3 = np.append(y1,y2)
        if (np.all(x3<0) or np.all(x3>0)):
            false_ind = np.where(np.logical_and(x3<1,x3>-1))[0]
            for ind in false_ind:
                if x3[ind] > 0:
                    x3[ind] = -1.2
                else:
                    x3[ind] = 1.2
        
        X.append(x3/1000)
        Y.append(y3/1000)
    
        np.savetxt('X_DE_mat.txt',X,fmt='%5.5f', delimiter = '\t')
        np.savetxt('Y_DE_mat.txt',Y,fmt='%5.5f', delimiter = '\t')
    return (X,Y)

# def mat_run(pop):
#     eng = matlab.engine.start_matlab()
# #     eng.gpuDevice(1)
#     polar_to_cartesian_mat(pop)
#     eng.ant128_GPU(nargout=0)
#     S11=np.loadtxt('ant128_S11_DE.txt',delimiter=',')
#     patt = np.loadtxt('ant128_patt_DE.txt',delimiter=',')
#     imp = np.loadtxt('ant128_imp_DE.txt',delimiter=',')
#     zo = imp[:,0:128]
# #     
#     return(S11,patt,zo)

def get_mse_Mat(pop, tar,n_i,n_p):
    isl = np.array(list(chain.from_iterable(pop)))
    Gtar = tar[0:41600]
    Gtar = np.reshape(Gtar,(128,25,13))
    Gtar = Gtar[51,:,:]
    
    Star = tar[41600:41728]
    Star = Star[51]
    Rtar = tar[41728:41856]
    Rtar = Rtar[51]
    
    tar1 = np.tile(Gtar, (n_i*n_p,1,1,1))
    tar2 = np.tile(Star, (n_i*n_p,1))
    tar3 = np.tile(Rtar, (n_i*n_p,1))
    # for matlab prediction
    eng = matlab.engine.start_matlab()
#     eng.gpuDevice(1)
    polar_to_cartesian_mat(isl)
    eng.ant128_GPU1(nargout=0)
    
    S11=np.loadtxt('ant128_S11_DE.txt',delimiter=',')
    patt = np.loadtxt('ant128_patt_DE.txt',delimiter=',')
    imp = np.loadtxt('ant128_imp_DE.txt',delimiter=',')
#     zo = imp[:,0:128]
    y1 = np.reshape(patt,(n_i*n_p,13,25,128))
    y1=np.swapaxes(y1,1,3)
    y1=y1[:,48:54,:,:]
    
    y2=S11[:,48:54]
    
    y3=imp[:,48:54]
    
    m_1 = np.mean(np.mean(np.mean((y1-tar1)**2, axis = 3), axis = 2), axis =1)
#     m_1 = np.reshape(m_1,(n_i*n_p,325))
    m_2 = np.mean((y2-tar2)**2, axis = 1)
    m_3 = np.mean((y3-tar3)**2, axis = 1)
#     print(np.shape(m_1),np.shape(m_2))
    mse = 0.5*m_1 + m_2 + 0.01*m_3
#     print(m)    
    return np.reshape(mse,(n_i,n_p))
