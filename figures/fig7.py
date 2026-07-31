# -*- coding: utf-8 -*-
"""
Created on Tue May  5 06:37:07 2026

@author: Administrator
"""

import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import motornet as mn
from ..model.rnn import RNNCell
from ..tasks.grid_reach import CenterOutTask, BasicTaskEnv
from ..training.utils import run_episode
import pickle
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import auc
from ..training.params import default_params
model_params, task_params, training_params = default_params()
model_params["noise_std"] = 0.
device = th.device("cuda")
    
from scipy import signal
def downsample_resample_poly(data, target_length):
    """使用resample_poly进行重采样"""
    n, t, k = data.shape
    downsampled = np.zeros((n, target_length, k))

    for i in range(n):
        for j in range(k):
            downsampled[i, :, j] = signal.resample(data[i, :, j], target_length)

    return downsampled

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

options = {"batch_size" : 8, "center_joint" : [[38., 113.3]], "angle_interval" : [45], "target_radius" : [0.12], 
           "target_on" : 0.1, "delay_durtion" : 0.6, "reach_durtion" : 0.4, "hold_durtion" : 0.1, 
           "catch_trial_proportion" : 0.,}
options["obs_class"]="prop+vis"
options["ini_std"] = 0.
effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
max_ep_duration = round(options["target_on"] + options["delay_durtion"] + options["reach_durtion"] + options["hold_durtion"],3)
env = BasicTaskEnv(effector=effector, max_ep_duration=max_ep_duration).to(device)
task=CenterOutTask()
trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train = task.genReach(options)
options["trajactory"] = trajactory
options["velocity"] = velocity
options["marker"] = marker
options["movement_timepoints"] = marker[:,-1]
options["gocue"] = gocue
options["goal"] = goal_for_train
options["joint_state"] = joint_state

w_1 = th.load("model/260331_CO_Model_from_DR/rnn_co_1.pth")
DR_CO = RNNCell(model_params)
DR_CO.load_state_dict(w_1)
DR_CO.to(device)
data_1,_ = run_episode(env, DR_CO, options, device=device, detach=True)
fr_1 = data_1["all_hidden"].to("cpu").numpy()
fr_1 = downsample_resample_poly(fr_1,50)
fr_dr_co = fr_1[:,20:40]

w_2 = th.load("model/260330_CO_Model_from_multi_DR/rnn_co_1.pth")
DR_CO_multi = RNNCell(model_params)
DR_CO_multi.load_state_dict(w_2)
DR_CO_multi.to(device)
data_2,_ = run_episode(env, DR_CO_multi, options, device=device, detach=True)
fr_2 = data_2["all_hidden"].to("cpu").numpy()
fr_2 = downsample_resample_poly(fr_2,50)
fr_dr_multi_co = fr_2[:,20:40]
fr_MC = np.load("Data/CO/fr_MC_1.npy")
fr_MM = np.load("Data/CO/fr_MM_4.npy")

# 与真实数据进行比较 CCA
pc = np.arange(1,11)
r_C_1 = cca_analysis(fr_dr_co, fr_MC, 10)
auc_C_1 = auc(pc, r_C_1)
r_C_2 = cca_analysis(fr_dr_multi_co, fr_MC, 10)
auc_C_2 = auc(pc, r_C_2)


with open("Data/CO/CCA_C_all.pkl", 'rb') as f:
    CCA_C = pickle.load(f)
with open("Data/CO/AUC_C_all.pkl", 'rb') as f:
    AUC_C = pickle.load(f)

CCA_D2C_MC = {"vs Monkey C":CCA_C["vs Monkey C"].mean(axis=0),
              "vs SR Model":CCA_C["vs SR Model 1"].mean(axis=0),
              "vs DR Model":r_C_1,
              "vs DR multi Model":r_C_2,
              "vs CO Model 1":CCA_C["vs CO Model 1"].mean(axis=0),
              "vs CO Model 2":CCA_C["vs CO Model 2"].mean(axis=0)}
AUC_D2C_MC = {"vs Monkey C":AUC_C["vs Monkey C"].mean(axis=0),
              "vs SR Model":AUC_C["vs SR Model 1"].mean(axis=0),
              "vs DR Model":auc_C_1,
              "vs DR multi Model":auc_C_2,
              "vs CO Model 1":AUC_C["vs CO Model 1"].mean(axis=0),
              "vs CO Model 2":AUC_C["vs CO Model 2"].mean(axis=0)}
    
# colors=["#3B4252","#75C4AC","#BF616A","#A3BE8C","#EBCB8B","#457B9B"]
colors=["#3B4252","#BF616A","#75C4AC","#457B9B","#EBCB8B","#BA68C8"]
labels = ["vs Monkey C: ","vs SR_Dense: ","vs DR_Single: ","vs DR_Multi: ","vs CO_Single: ","vs CO_Multi: "]
AUC_data = list(AUC_D2C_MC.values())
for ii, l in enumerate(labels):
    labels[ii] = l+f"{AUC_data[ii]:.3f}"
pc = np.arange(1,11)
fig,ax = plt.subplots(figsize=(6,4))
CCA_data = list(CCA_D2C_MC.values())
for i in range(6):
    ax.plot(pc, CCA_data[i],'.-',linewidth=3,markersize=12,color=colors[i],label=labels[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(list(range(1,11)))
plt.xlabel('neural mode',fontsize=14)
plt.ylabel('CC score',fontsize=16)
plt.legend(fontsize=12)
plt.subplots_adjust(bottom=0.15)
plt.show()
#plt.savefig('E:/Modelling/BasicModel/Results/Fig4_E.svg', format='svg', dpi=300)

# AUC_data = list(AUC_D2C_MC.values())
# fig,ax = plt.subplots(figsize=(6,4))
# bar = ax.bar(labels, AUC_data,color=colors,edgecolor='black', width=0.6, alpha=0.8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xticks(fontsize=9, rotation=45)
# plt.ylabel('AUC of CCA', fontsize=14)
# plt.subplots_adjust(bottom=0.26)
# plt.subplots_adjust(top=0.96)
# plt.show()
#plt.savefig('E:/Modelling/BasicModel/Results/Fig4_F.svg', format='svg', dpi=300)

r_M_1 = cca_analysis(fr_dr_co, fr_MM, 10)
auc_M_1 = auc(pc, r_M_1)
r_M_2 = cca_analysis(fr_dr_multi_co, fr_MM, 10)
auc_M_2 = auc(pc, r_M_2)
with open("Data/CO/CCA_M_all.pkl", 'rb') as f:
    CCA_M = pickle.load(f)
with open("Data/CO/AUC_M_all.pkl", 'rb') as f:
    AUC_M = pickle.load(f)
    
CCA_D2C_MM = {"vs Monkey M":CCA_C["vs Monkey C"].mean(axis=0),
              "vs SR Model":CCA_M["vs SR Model 1"].mean(axis=0),
              "vs DR Model":r_M_1,
              "vs DR multi Model":r_M_2,
              "vs CO Model 1":CCA_M["vs CO Model 1"].mean(axis=0),
              "vs CO Model 2":CCA_M["vs CO Model 2"].mean(axis=0)}
AUC_D2C_MM = {"vs Monkey M":AUC_M["vs Monkey M"].mean(axis=0),
              "vs SR Model":AUC_M["vs SR Model 1"].mean(axis=0),
              "vs DR Model":auc_M_1,
              "vs DR multi Model":auc_M_2,
              "vs CO Model 1":AUC_M["vs CO Model 1"].mean(axis=0),
              "vs CO Model 2":AUC_M["vs CO Model 2"].mean(axis=0)}
colors=["#3B4252","#BF616A","#75C4AC","#457B9B","#EBCB8B","#BA68C8"]
labels = ["vs Monkey M: ","vs SR_Dense: ","vs DR_Single: ","vs DR_Multi: ","vs CO_Single: ","vs CO_Multi: "]
AUC_data = list(AUC_D2C_MM.values())
for ii, l in enumerate(labels):
    labels[ii] = l+f"{AUC_data[ii]:.3f}"

pc = np.arange(1,11)
fig,ax = plt.subplots(figsize=(6,4))
CCA_data = list(CCA_D2C_MM.values())
for i in range(6):
    ax.plot(pc, CCA_data[i],'.-',linewidth=3,markersize=12,color=colors[i],label=labels[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(list(range(1,11)))
plt.xlabel('neural mode',fontsize=14)
plt.ylabel('CC score',fontsize=16)
plt.legend(fontsize=12)
plt.subplots_adjust(bottom=0.15)
plt.show()
#plt.savefig('E:/Modelling/BasicModel/Results/Fig4_G.svg', format='svg', dpi=300)

# AUC_data = list(AUC_D2C_MM.values())
# fig,ax = plt.subplots(figsize=(6,4))
# bar = ax.bar(labels, AUC_data,color=colors,edgecolor='black', width=0.6, alpha=0.8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xticks(fontsize=9, rotation=45)
# plt.ylabel('AUC of CCA', fontsize=14)
# plt.subplots_adjust(bottom=0.26)
# plt.subplots_adjust(top=0.96)
# plt.show()