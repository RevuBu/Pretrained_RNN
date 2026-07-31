# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:45:06 2026

@author: Administrator
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import auc
    
def cca_analysis(fr1,fr2,dim):
    fr1_flatten = np.reshape(fr1,(160, fr1.shape[2]))
    pca = PCA(n_components=10)
    X = pca.fit_transform(fr1_flatten, dim)
    fr2_flatten = np.reshape(fr2,(160, fr2.shape[2]))
    pca = PCA(n_components=10)
    Y = pca.fit_transform(fr2_flatten, dim)
    
    cca = CCA(n_components=dim)    
    cca.fit(X,Y)
    X_c, Y_c = cca.transform(X,Y) 
    r = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]     
    return r

with open("Data/CO/CCA_C.pkl", 'rb') as f:
    CCA_C = pickle.load(f)
with open("Data/CO/AUC_C.pkl", 'rb') as f:
    AUC_C = pickle.load(f)
    
# r_Cb_15_if = np.zeros((9,10))
# r_Cb_20_if = np.zeros((9,10))
# r_Cb_25_if = np.zeros((9,10))
r_Cb_20 = np.zeros((9,10))
r_Cb_25 = np.zeros((9,10))
# auc_Cb_15_if = np.zeros(9)
# auc_Cb_20_if = np.zeros(9)
# auc_Cb_25_if = np.zeros(9)
auc_Cb_20 = np.zeros(9)
auc_Cb_25 = np.zeros(9)
pc = np.arange(1,11)

# fr_b_15_if = np.load("Data/CenterOut/fr_b_15_if.npy")
# fr_b_20_if = np.load("Data/CenterOut/fr_b_20_if.npy")
# fr_b_25_if = np.load("Data/CenterOut/fr_b_25_if.npy")
fr_b_15 = np.load("Data/CO/fr_b.npy")
fr_b_20 = np.load("Data/CO/fr_b_20.npy")
fr_b_25 = np.load("Data/CO/fr_b_25.npy")

for i in range(9):
    fr_MC_i = np.load(f"Data/CO/fr_MC_{i+1}.npy")
    # r_Cb_15_if[i] = cca_analysis(fr_MC_i, fr_b_15_if, 10)
    # auc_Cb_15_if[i] = auc(pc, r_Cb_15_if[i])
    # r_Cb_20_if[i] = cca_analysis(fr_MC_i, fr_b_20_if, 10)
    # auc_Cb_20_if[i] = auc(pc, r_Cb_20_if[i])
    # r_Cb_25_if[i] = cca_analysis(fr_MC_i, fr_b_25_if, 10)
    # auc_Cb_25_if[i] = auc(pc, r_Cb_25_if[i])
    r_Cb_20[i] = cca_analysis(fr_MC_i, fr_b_20, 10)
    auc_Cb_20[i] = auc(pc, r_Cb_20[i])
    r_Cb_25[i] = cca_analysis(fr_MC_i, fr_b_25, 10)
    auc_Cb_25[i] = auc(pc, r_Cb_25[i])


CCA_C_all = {"vs Monkey C":CCA_C["vs Monkey C"],
             "vs SR Model 1":CCA_C["vs Basic Model"],
             "vs SR Model 2":r_Cb_20,
             "vs SR Model 3":r_Cb_25,
             "vs CO Model 1":CCA_C["vs CO Model 1"],
             "vs CO Model 2":CCA_C["vs CO Model 2"]}
AUC_C_all = {"vs Monkey C":AUC_C["vs Monkey C"],
             "vs SR Model 1":AUC_C["vs Basic Model"],
             "vs SR Model 2":auc_Cb_20,
             "vs SR Model 3":auc_Cb_25,
             "vs CO Model 1":AUC_C["vs CO Model 1"],
             "vs CO Model 2":AUC_C["vs CO Model 2"]}
with open("Data/CO/CCA_C_all.pkl", 'wb') as f:
    pickle.dump(CCA_C_all, f)
with open("Data/CO/AUC_C_all.pkl", 'wb') as f:
    pickle.dump(AUC_C_all, f)


# with open("Data/CenterOut/CCA_M.pkl", 'rb') as f:
#     CCA_M = pickle.load(f)
with open("Data/CO/AUC_M.pkl", 'rb') as f:
    AUC_M = pickle.load(f)
    
fr = np.load("Data/CO/fr_MC_1.npy")
fr_co1 = np.load("Data/CO/fr_co1.npy")
fr_co2 = np.load("Data/CO/fr_co12.npy")
r_M = np.zeros((5,10))
r_Mb_15 = np.zeros((5,10))
r_Mb_20 = np.zeros((5,10))
r_Mb_25 = np.zeros((5,10))
r_M_CO_1 = np.zeros((5,10))
r_M_CO_2 = np.zeros((5,10))
auc_M = np.zeros(5)
auc_Mb_20 = np.zeros(5)
auc_Mb_25 = np.zeros(5)
for i in range(5):
    fr_MM_i = np.load(f"Data/CO/fr_MM_{i+1}.npy")
    r_M[i] = cca_analysis(fr_MM_i, fr, 10)
    auc_M[i] = auc(pc, r_M[i])
    r_Mb_15[i] = cca_analysis(fr_MM_i, fr_b_15, 10)
    r_Mb_20[i] = cca_analysis(fr_MM_i, fr_b_20, 10)
    auc_Mb_20[i] = auc(pc, r_Mb_20[i])
    r_Mb_25[i] = cca_analysis(fr_MM_i, fr_b_25, 10)
    auc_Mb_25[i] = auc(pc, r_Mb_25[i])
    r_M_CO_1[i] = cca_analysis(fr_MM_i, fr_co1, 10)
    r_M_CO_2[i] = cca_analysis(fr_MM_i, fr_co2, 10)

CCA_M_all = {"vs Monkey M":r_M,
             "vs SR Model 1":r_Mb_15,
             "vs SR Model 2":r_Mb_20,
             "vs SR Model 3":r_Mb_25,
             "vs CO Model 1":r_M_CO_1,
             "vs CO Model 2":r_M_CO_2}
AUC_M_all = {"vs Monkey M":auc_M,
             "vs SR Model 1":AUC_M["vs Basic Model"],
             "vs SR Model 2":auc_Mb_20,
             "vs SR Model 3":auc_Mb_25,
             "vs CO Model 1":AUC_M["vs CO Model 1"],
             "vs CO Model 2":AUC_M["vs CO Model 2"]}
with open("Data/CO/CCA_M_all.pkl", 'wb') as f:
    pickle.dump(CCA_M_all, f)
with open("Data/CO/AUC_M_all.pkl", 'wb') as f:
    pickle.dump(AUC_M_all, f)