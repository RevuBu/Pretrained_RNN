# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 02:58:25 2025

@author: Administrator
"""
import os
import json
import matplotlib.pyplot as plt
from ..training.utils import run_episode
import torch as th
import pickle
import numpy as np
import motornet as mn
from ..tasks.sequential_reach import RandomTargetReach, RTTEnv
from ..model.build import build_model
from DSA import DSA
device = th.device("cuda")

def cal_vel(vel):
    return np.sqrt(vel[:,0]**2 + vel[:,1]**2)

def cal_p_loss(data, goal, bhv_marker, device=th.device("cuda")):
    trial_num = goal.shape[0]
    e_pos = th.zeros(trial_num).to(device)
    for trial in range(trial_num):
        end = bhv_marker[trial, -1]
        for i in range(goal.shape[1]-1):
            tt = bhv_marker[trial, 2*i+1]
            e_pos[trial] += th.mean(th.abs(data['xy'][trial, tt-5:tt+5, :] - goal[trial, i, :]))
        e_pos[trial] += th.mean(th.abs(data['xy'][trial, end-5:end, :] - goal[trial, -1, :]))
        e_pos[trial] = e_pos[trial] / goal.shape[1]
        
    return th.mean(e_pos).to("cpu").numpy()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
param = {"model"           : "rnn",
         "n_rec"           : 100,
         "nonlinearity"    : "tanh",
         "noise_std"       : 0.10,
         "ini_std"         : 0.0,
         "obs_class"       : "prop+vis",
         "n_out_task"      : 12,
         }
effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
model = build_model(param)
model_name = "model/RTT/RTT_1.pth"
w = th.load(model_name)
model.load_state_dict(w)

task = RandomTargetReach()
trial_num = 128
seq_len = 4
dis_range = [0.10,0.15]
options = {"batch_size" : trial_num,
           "obs_class" : "prop+vis", 
           "ini_std" : 0.0,
           "dis_range" : dis_range,
           "hold_durtion" : 0.1,
           "ave_vel" : 0.5
           }
options["seq_len"] = seq_len
targets, joint_state = task.genTargets(trial_num, seq_len, dis_range)
options["targets"] = targets
options["joint_state"] = joint_state

trajactory, velocity, marker, gocue, joint_state, goal = task.genReach(options)
options["trajactory"]=trajactory
options["velocity"]=velocity
options["marker"] = marker
options["movement_timepoints"] = marker[:,-1]
options["gocue"] = gocue
options["goal"] = goal
max_ep_duration = round(marker[:,-1].max() * effector.dt, 3)
env = RTTEnv(effector=effector, max_ep_duration=max_ep_duration).to(device)
data, _ = run_episode(env, model, options, device=device, detach=True)
traj = data["xy"].to("cpu").numpy()
vel = data["vel"].to("cpu").numpy()
# 画轨迹和速度图
colors=["red","orange","blue"]
labels=["Start point","1st target","2nd target","3rd target"]

i = 43
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(traj[i,0,0],traj[i,0,1],marker='o',markersize=10,color='green',label=labels[0])
for j in range(3):
    ax.plot(goal[i,j,0],goal[i,j,1],marker='o',markersize=16,color=colors[j],label=labels[j+1])
ax.plot(traj[i,:marker[i,-1],0],traj[i,:marker[i,-1],1],linewidth=3)
ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.legend(fontsize=12)
plt.show()

fig,ax = plt.subplots(figsize=(6,4))
vel_proj=np.sqrt(vel[i,:marker[i,-1],0]**2 + vel[i,:marker[i,-1],1]**2)
ax.plot(np.arange(0,10*marker[i,-1],10),vel_proj,linewidth=3)
ax.plot(np.arange(10*marker[i,0],10*marker[i,1],10),vel_proj[marker[i,0]:marker[i,1]],color=colors[0],linewidth=3,label="to 1st target")
ax.plot(np.arange(10*marker[i,2],10*marker[i,3],10),vel_proj[marker[i,2]:marker[i,3]],color=colors[1],linewidth=3,label="to 2nd target")
ax.plot(np.arange(10*marker[i,4],10*marker[i,5],10),vel_proj[marker[i,4]:marker[i,5]],color=colors[2],linewidth=3,label="to 3rd target")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Time(s)',fontsize=16)
plt.ylabel('Vel(m/s)',fontsize=16)
plt.legend(fontsize=14,loc='upper left', bbox_to_anchor=(0.65, 1.37))
plt.show()

# 画loss图
loss_file = "model/251105_RTT/rnn_RTT_"
position_loss = []
velocity_loss = []
for i in range(3,8):
    file_name = f"{loss_file}{i}__loss.json"
    with open(file_name, 'r', encoding='utf-8') as f:
        loss = json.load(f)
        position_loss+=loss["losses"]["position"]
pos_loss_1 = np.array(position_loss)

lf_rtt_3 = "model/251130_RTT_3/rnn_RTT_3__loss.json"
with open(lf_rtt_3, 'r') as f:
    loss_rtt_3 = json.load(f)
pos_loss_2 = np.array(loss_rtt_3['losses']['position'])

lf_rtt_4 = "model/251130_RTT_4/rnn_RTT_4__loss.json"
with open(lf_rtt_4, 'r') as f:
    loss_rtt_4 = json.load(f)
pos_loss_3 = np.array(loss_rtt_4['losses']['position'])

lf_rtt_3_fs = "model/251130_RTT_3_fs/rnn_RTT_3__loss.json"
with open(lf_rtt_3_fs, 'r') as f:
    loss_rtt_3_fs = json.load(f)
pos_loss_4 = np.array(loss_rtt_3_fs['losses']['position'])

lf_rtt_4_fs = "model/251130_RTT_4_fs/rnn_RTT_4__loss.json"
with open(lf_rtt_4_fs, 'r') as f:
    loss_rtt_4_fs = json.load(f)
pos_loss_5 = np.array(loss_rtt_4_fs['losses']['position'])

# colors=["black","#0CBCC0","#099396","#F0786A","#AE2012"]
# colors=["black","#75c4ac","#e79e67","#fffdb9","#ed7bb6"]
colors=["#3B4252","#75C4AC","#BF616A","#A3BE8C","#EBCB8B"]
labels = ["RTT_continue_learning","RTT_2_fine_tune","RTT_3_fine_tune","RTT_2_de_novo","RTT_3_de_novo"]
fig,ax = plt.subplots(figsize=(6,4))
ax.semilogy(pos_loss_1, color=colors[0], lw=1.2, label=labels[0])
ax.semilogy(pos_loss_2, color=colors[1], lw=1.2, label=labels[1])
ax.semilogy(pos_loss_3, color=colors[2], lw=1.2, label=labels[2])
ax.semilogy(pos_loss_4, color=colors[3], lw=1.2, label=labels[3])
ax.semilogy(pos_loss_5, color=colors[4], lw=1.2, label=labels[4])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Loss',fontsize=14)
ax.set_xlabel('Epoch',fontsize=14)
ax.legend(fontsize=10)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
plt.show()
plt.savefig('Results/Fig5_D.svg', format='svg', dpi=300)

with open("Data/RTT/eval_rtt.pkl","rb") as f:
    eval_rtt = pickle.load(f)

fig,ax=plt.subplots(figsize=(6,4))
model_list = list(eval_rtt.keys())
for i, e in enumerate(eval_rtt.values()):
    e_mean=e.mean(axis=1)
    e_std=e.std(axis=1)
    ax.plot(range(2,10),e_mean,'o-',color=colors[i],lw=3,markersize=8, label=labels[i])
    ax.fill_between(range(2,10),e_mean-2*e_std,e_mean+2*e_std,color=colors[i],alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(2,10))
ax.set_xlabel('Target number',fontsize=14)
ax.set_ylabel('Position error',fontsize=14)
plt.ylim([0.004,0.014])
plt.legend(loc="upper right", fontsize=10)
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.15)
plt.show()
plt.savefig('Results/Fig5_E.svg', format='svg', dpi=300)

K_d = np.zeros((5,10))
n_delays = 5
delay_interval = 2
rank = 10
d = 'cuda'
RTT_data = np.load("Data/RTT/MM_M1.npy")
models = ["RTT","RTT_2","RTT_2_fs","RTT_3","RTT_3_fs"]
for i, m in enumerate(models):
    for j in range(2):
        w = th.load(f"model/RTT/{m}_{j+1}.pth")
        model.load_state_dict(w)
        model.to(device)
        for t in range(5):
            targets, joint_state = task.genTargets(trial_num, seq_len, dis_range)
            options["targets"] = targets
            options["joint_state"] = joint_state
            trajactory, velocity, marker, gocue, joint_state, goal = task.genReach(options)
            options["trajactory"]=trajactory
            options["velocity"]=velocity
            options["marker"] = marker
            options["movement_timepoints"] = marker[:,-1]
            options["gocue"] = gocue
            options["goal"] = goal
            max_ep_duration = round(marker[:,-1].max() * effector.dt, 3)
            env = RTTEnv(effector=effector, max_ep_duration=max_ep_duration).to(device)
            data,_ = run_episode(env, model, options, device=device, detach=True)
            fr=data["all_hidden"].to("cpu").numpy()
            dsa = DSA(fr,RTT_data,n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=False,device=d,iters=2000,lr=1e-2)
            similarities = dsa.fit_score()
            if j == 0 :
                K_d[i,t] = similarities
            else:
                K_d[i,t+5] = similarities
np.save("Data/RTT/RTT_DSA.npy",K_d)

K_d = np.load("Data/RTT/rtt_DSA_260505.npy")
fig,ax = plt.subplots(figsize=(4,4))
x=np.arange(1,12,2)
K_mean = [K_d[i].mean() for i in range(4)]
K_std = [K_d[i].std() for i in range(4)]
labels = ["vs SR_Dense_ft","vs SR_Moderate_ft","vs SR_Sparse_ft","RTT"]
colors = ["#3B4252","#457B9B","#EBCB8B","#BA68C8"]
bar = ax.bar(labels, K_mean, yerr=K_std,capsize=10,color=colors,edgecolor='black', width=0.6, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=8, rotation=45)
plt.subplots_adjust(bottom=0.27)
plt.subplots_adjust(left=0.13)
plt.ylabel('Dissimilarity Score', fontsize=14)
plt.show()
plt.savefig('Results/Fig5_F.svg', format='svg', dpi=300)
