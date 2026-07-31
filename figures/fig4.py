# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 12:57:55 2025

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
from ..training.params import default_params
import pickle
from jPCA import jPCA
from jPCA.util import plot_projections
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = th.device("cuda")


model_params, task_params, training_params = default_params()
model_params["noise_std"] = 0.
w = th.load("model/250916_basicModel/rnn__joint_interval_15.pth")
Basic_model = RNNCell(model_params)
Basic_model.load_state_dict(w)
Basic_model.to(device)
w_co_1 = th.load("model/251105_CO_Model/rnn_co_1_ft.pth")
CO_model_1 = RNNCell(model_params)
CO_model_1.load_state_dict(w_co_1)
CO_model_1.to(device)
w_co_12 = th.load("model/251105_CO_Model/rnn_co_12_ft.pth")
CO_model_12 = RNNCell(model_params)
CO_model_12.load_state_dict(w_co_12)
CO_model_12.to(device)

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

data,_ = run_episode(env, Basic_model, options, device=device, detach=True)
traj_b = data['xy'].to('cpu').numpy()
fr = data["all_hidden"].to("cpu").numpy()

cmap = plt.get_cmap('RdYlGn')
colors = cmap(np.linspace(0, 1, 8))
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(8):
    ax.plot(traj_b[i,:marker[i,-1],0], traj_b[i,:marker[i,-1],1],color=colors[i])
    # ax.plot(goal[i,0], goal[i,1], marker='o', markersize=10, mfc=colors[i], markeredgewidth=0)
ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('equal')
plt.show()

# PSTH
def normalize(data):
    for i in range(data.shape[-1]):
        tmp = data[:,:,i]
        min_val = tmp.min()
        max_val = tmp.max()
        data[:,:,i] = (tmp - min_val) / (max_val - min_val)  
    return data
fr=normalize(fr)
fig,ax=plt.subplots(2,2,figsize=(14,8))
for i in range(8):
    ax[0,0].plot(fr[i,marker[0,0]:marker[0,2],99],color=colors[i],linewidth=2.5)
    ax[0,0].spines[:].set_visible(False)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,1].plot(fr[i,marker[0,0]:marker[0,2],12],color=colors[i],linewidth=2.5)
    ax[0,1].spines[:].set_visible(False)
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[1,0].plot(fr[i,marker[0,0]:marker[0,2],31],color=colors[i],linewidth=2.5)
    ax[1,0].spines[:].set_visible(False)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,1].plot(fr[i,marker[0,0]:marker[0,2],27],color=colors[i],linewidth=2.5)
    ax[1,1].spines[:].set_visible(False)
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
plt.tight_layout()
plt.show()

# jPCA
fr_prep = []
fr_exec = []
for i in range(8): 
    fr_prep.append(fr[i,marker[i,0]:marker[i,1]])
    fr_exec.append(fr[i,marker[i,1]:marker[i,2]])
Fr_prep = np.concatenate(fr_prep)
Fr_exec = np.concatenate(fr_exec)

times=list(range(0,400,10))
jpca = jPCA.JPCA(num_jpcs=6)
(projected, 
 full_data_var,
 pca_var_capt,
 jpca_var_capt) = jpca.fit(fr_exec,tstart=0,tend=390,times=times)
print(pca_var_capt/full_data_var)
print(jpca_var_capt/full_data_var)
fig, ax = plt.subplots(figsize=(6, 6))
ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plot_projections(projected,axis=ax,cm=cmap,arrow_size=0.04,circle_size=0.02)

# CCA
with open("Data/CO/CCA_C.pkl", 'rb') as f:
    CCA_C = pickle.load(f)
colors=["#3B4252","#BF616A","#75C4AC","#457B9B","#EBCB8B","#BA68C8"]
labels = list(CCA_C.keys())
pc = np.arange(1,11)
CCA_data = list(CCA_C.values())
fig,ax = plt.subplots(figsize=(6,4))
for i in range(6):
    ax.plot(pc, CCA_data[i].mean(axis=0),'.-',linewidth=3,markersize=12,color=colors[i],label=labels[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(list(range(1,11)))
plt.xlabel('neural mode',fontsize=14)
plt.ylabel('CC score',fontsize=16)
plt.legend(fontsize=12)
plt.subplots_adjust(bottom=0.15)
plt.show()

with open("Data/CO/CCA_M.pkl", 'rb') as f:
    CCA_M = pickle.load(f)
colors=["#3B4252","#BF616A","#75C4AC","#457B9B","#EBCB8B","#BA68C8"]
labels = list(CCA_M.keys())
pc = np.arange(1,11)
CCA_data = list(CCA_M.values())
fig,ax = plt.subplots(figsize=(6,4))
for i in range(6):
    ax.plot(pc, CCA_data[i].mean(axis=0),'.-',linewidth=3,markersize=12,color=colors[i],label=labels[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(list(range(1,11)))
plt.xlabel('neural mode',fontsize=14)
plt.ylabel('CC score',fontsize=16)
plt.legend(fontsize=12)
plt.subplots_adjust(bottom=0.15)
plt.show()


with open("Data/CO/CCA_C_all.pkl", 'rb') as f:
    CCA_C = pickle.load(f)
with open("Data/CO/AUC_C_all.pkl", 'rb') as f:
    AUC_C = pickle.load(f)
# colors=["#3B4252","#75C4AC","#BF616A","#A3BE8C","#EBCB8B","#457B9B"]
# labels = list(CCA_C.keys())
# labels = ["vs Monkey C","vs SR_Dense","vs SR_Moderate","vs SR_Sparse","vs CO_Single","vs CO_Multi"]
AUC_data = list(AUC_C.values())
AUC_mean = [AUC_data[i].mean() for i in range(len(AUC_data))]
colors=["#3B4252","#BF616A","#75C4AC","#457B9B","#EBCB8B","#BA68C8"]
labels = ["vs Monkey C: ","vs SR_Dense: ","vs SR_Moderate: ","vs SR_Sparse: ","vs CO_Single: ","vs CO_Multi: "]
for ii, l in enumerate(labels):
    labels[ii] = l+f"{AUC_mean[ii]:.3f}"
pc = np.arange(1,11)
fig,ax = plt.subplots(figsize=(6,4))
CCA_data = list(CCA_C.values())
for i in range(6):
    ax.plot(pc, CCA_data[i].mean(axis=0),'.-',linewidth=3,markersize=12,color=colors[i],label=labels[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(list(range(1,11)))
plt.xlabel('neural mode',fontsize=14)
plt.ylabel('CC score',fontsize=16)
plt.legend(fontsize=12)
plt.subplots_adjust(bottom=0.15)
plt.show()
plt.savefig('Results/Fig4_E.svg', format='svg', dpi=300)

AUC_data = list(AUC_C.values())
AUC_mean = [AUC_data[i].mean() for i in range(len(AUC_data))]
AUC_std = [AUC_data[i].std() for i in range(len(AUC_data))]
fig,ax = plt.subplots(figsize=(6,4))
bar = ax.bar(labels, AUC_mean, yerr=AUC_std,capsize=10,color=colors,edgecolor='black', width=0.6, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=9, rotation=45)
plt.ylabel('AUC of CCA', fontsize=14)
plt.subplots_adjust(bottom=0.26)
plt.subplots_adjust(top=0.96)
plt.show()
plt.savefig('Results/Fig4_F.svg', format='svg', dpi=300)


with open("Data/CO/CCA_M_all.pkl", 'rb') as f:
    CCA_M = pickle.load(f)
with open("Data/CO/AUC_M_all.pkl", 'rb') as f:
    AUC_M = pickle.load(f)

CCA_M["vs Monkey M"] = CCA_C["vs Monkey C"]
colors=["#3B4252","#BF616A","#75C4AC","#457B9B","#EBCB8B","#BA68C8"]
labels = ["vs Monkey C: ","vs SR_Dense: ","vs SR_Moderate: ","vs SR_Sparse: ","vs CO_Single: ","vs CO_Multi: "]
AUC_data = list(AUC_M.values())
AUC_mean = [AUC_data[i].mean() for i in range(len(AUC_data))]
for ii, l in enumerate(labels):
    labels[ii] = l+f"{AUC_mean[ii]:.3f}"

pc = np.arange(1,11)
fig,ax = plt.subplots(figsize=(6,4))
CCA_data = list(CCA_M.values())
for i in range(6):
    ax.plot(pc, CCA_data[i].mean(axis=0),'.-',linewidth=3,markersize=12,color=colors[i],label=labels[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(list(range(1,11)))
plt.xlabel('neural mode',fontsize=14)
plt.ylabel('CC score',fontsize=16)
plt.legend(fontsize=12)
plt.subplots_adjust(bottom=0.15)
plt.show()
plt.savefig('Results/Fig4_G.svg', format='svg', dpi=300)

AUC_data = list(AUC_M.values())
AUC_mean = [AUC_data[i].mean() for i in range(len(AUC_data))]
AUC_std = [AUC_data[i].std() for i in range(len(AUC_data))]
fig,ax = plt.subplots(figsize=(6,4))
bar = ax.bar(labels, AUC_mean, yerr=AUC_std,capsize=10,color=colors,edgecolor='black', width=0.6, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=9, rotation=45)
plt.ylabel('AUC of CCA', fontsize=14)
plt.subplots_adjust(bottom=0.26)
plt.subplots_adjust(top=0.96)
plt.show()
plt.savefig('Results/Fig4_H.svg', format='svg', dpi=300)


