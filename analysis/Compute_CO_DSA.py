# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:10:00 2026

@author: Administrator
"""

# 计算DSA
import numpy as np
fr_MC = np.load("Data/CO/fr_MC_5.npy")
fr_MM = np.load("Data/CO/fr_MM_5.npy")
fr_SR_Dense = np.load("Data/CO/fr_SR_Dense.npy")[:,10:-10]
fr_SR_Moderate= np.load("Data/CO/fr_SR_Moderate.npy")[:,10:-10]
fr_SR_Sparse = np.load("Data/CO/fr_SR_Sparse.npy")[:,10:-10]
fr_CO_Single = np.load("Data/CO/fr_CO_Single.npy")[:,10:-10]
fr_CO_Multi = np.load("Data/CO/fr_CO_Multi.npy")[:,10:-10]
fr_DR_Single = np.load("Data/CO/fr_DR_Single.npy")[:,10:-10]
fr_DR_Multi = np.load("Data/CO/fr_DR_Multi.npy")[:,10:-10]
dataset = [fr_MC, fr_MM, fr_SR_Dense, fr_SR_Moderate, fr_SR_Sparse, fr_CO_Single, fr_CO_Multi, fr_DR_Single, fr_DR_Multi]

from DSA import DSA
K_d_C = np.zeros((9,4))
K_d_M = np.zeros((9,4))
n_delays = 5
delay_interval = 2
rank = 10
d = 'cuda'
for i in range(4):
    fr_MC_i = np.load(f"Data/CO/fr_MC_{i+1}.npy")
    dsa = DSA(dataset, fr_MC_i,n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=False,device=d,iters=2000,lr=1e-2)
    similarities = dsa.fit_score()
    K_d_C[:,i] = similarities[:,0]
    fr_MM_i = np.load(f"Data/CO/fr_MM_{i+1}.npy")
    dsa = DSA(dataset, fr_MM_i,n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=False,device=d,iters=2000,lr=1e-2)
    similarities = dsa.fit_score()
    K_d_M[:,i] = similarities[:,0]
np.save("Data/CO/DSA_C.npy", K_d_C)
np.save("Data/CO/DSA_M.npy", K_d_M)