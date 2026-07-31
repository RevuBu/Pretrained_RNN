# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 13:37:11 2025

@author: Administrator
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import motornet as mn
from ..model.rnn import RNNCell
from ..model.build import build_task
from ..training.utils import run_episode
from ..tasks.grid_reach import BasicTask, BasicTaskEnv
from ..training.params import default_params
from sklearn.decomposition import PCA
device = th.device("cuda")

model_file = "model/250916_basicModel/rnn__joint_interval_15.pth"
model_params,task_params,_ = default_params()

effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
model = RNNCell(model_params)
w = th.load(model_file)
model.load_state_dict(w)
model.sigma=0.
model.to(device)

param = {"obs_class"       : "prop+vis",
         "ini_std"         : 0.10,
         "session_num"     : 5,
         "joint_interval"  : [25,20,15],
         "grid_interval"   : [0.12, 0.10, 0.08],
         "test_batch_size" : [128, 256, 512],
         "vel_range"       : [0.3,0.4,0.5,0.6],
         }
task_info = {"task" : "grid",
             "workspace":"joint_space",
             "trial_num" : 1,
             "to_range" : [0.1,0.11],
             "dis_range" : [0.05, 0.35],
             "delay_range": [0.2, 0.6],
             "hold_durtion" : 0.05,
             "catch_trial_proportion":0.0
             }

task_info["joint_interval"] = 15
task_info["vel_range"] = [0.4]
task, options, traj, vel = build_task(param, task_info)
max_ep_duration = round(options["movement_timepoints"].max() * effector.dt, 3)
env = BasicTaskEnv(effector=effector, max_ep_duration=max_ep_duration).to(device)
data, _ = run_episode(env, model, options, device=device, detach=True)
fr=data["all_hidden"].to("cpu").numpy()
traj_pred=data["xy"].to("cpu").numpy()
vel_pred=data["vel"].to("cpu").numpy()
ntps = options["movement_timepoints"]
marker = options["marker"]
goal = options["g"]
trial_num = traj_pred.shape[0]

trial_index_g = []
targets, idx, reaches = np.unique(goal, axis=0, return_index=True, return_counts=True)
for i, target in enumerate(targets):
    trial_index_g.append(np.arange(idx[i],idx[i]+reaches[i]))

trial_index_s = []
start_points = traj[:,0,:]
start_point, inverse = np.unique(start_points, axis=0, return_inverse=True)
for i, sp in enumerate(start_point):
    trial_index_s.append(np.where(inverse == i)[0])

fr_list=[]
for i in trial_index_g[10]: 
    fr_list.append(fr[i,marker[i,0]:marker[i,2]+1])
for i in trial_index_s[10]: 
    fr_list.append(fr[i,marker[i,0]:marker[i,2]+1])
Fr = np.concatenate(fr_list)

cmap = plt.get_cmap('RdYlGn')
colors = cmap(np.linspace(0, 1, len(trial_index_g[10])))

# 画joint grid轨迹图
fig, ax = plt.subplots(figsize=(8, 4))
targets, idx, reaches = np.unique(goal, axis=0, return_index=True, return_counts=True)
for i, target in enumerate(targets):
    for j in range(reaches[i]-1):
        trial = int(j + idx[i])
        ax.plot(traj_pred[trial, :ntps[trial], 0],traj_pred[trial, :ntps[trial], 1], color='b', alpha=0.2, linewidth=0.6)
ax.plot(goal[trial_index_g[10][0],0], goal[trial_index_g[10][0],1], marker='o', markersize=16, mfc='green', markeredgewidth=0)
for i,trial in enumerate(trial_index_g[10]):
    ax.plot(traj_pred[trial,marker[trial,0]+10:marker[trial,2]+1,0], traj_pred[trial,marker[trial,0]+10:marker[trial,2]+1,1], color=colors[i])
    ax.plot(traj_pred[trial,0,0], traj_pred[trial,0,1], marker='o', markersize=13, mfc=colors[i], markeredgewidth=0)
    ax.text(traj_pred[trial,0,0], traj_pred[trial,0,1],str(i+1),fontsize=9, ha='center', va='center', color='black', fontweight='bold')
for i,trial in enumerate(trial_index_s[10]):
    ax.plot(traj_pred[trial,marker[trial,0]+10:marker[trial,2]+1,0], traj_pred[trial,marker[trial,0]+10:marker[trial,2]+1,1],'--', color=colors[i])    
plt.xlim(-0.35,0.35)
plt.ylim(0.18, 0.52)
ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('equal')
plt.show()

# 画PCA结果
pca = PCA(n_components=10)
pca.fit(Fr)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_ratio = np.cumsum(explained_variance_ratio)

# 准备期
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
for i,trial in enumerate(trial_index_g[10]):
    activity_pc = pca.transform(fr[trial,:marker[trial,2]+1])
    ax.plot(activity_pc[marker[trial,0]:marker[trial,1]+1,0], 
            activity_pc[marker[trial,0]:marker[trial,1]+1,1], 
            activity_pc[marker[trial,0]:marker[trial,1]+1,2],color=colors[i])
    ax.plot(activity_pc[marker[trial,1],0], activity_pc[marker[trial,1],1], activity_pc[marker[trial,1],2], marker='s',markersize=15, mfc=colors[i], markeredgewidth=0)
    ax.text(activity_pc[marker[trial,1],0], activity_pc[marker[trial,1],1], activity_pc[marker[trial,1],2],str(i+1),fontsize=11, ha='center', va='center', color='black', fontweight='bold')
for i,trial in enumerate(trial_index_s[10]):
    activity_pc = pca.transform(fr[trial,:marker[trial,2]+1])
    ax.plot(activity_pc[marker[trial,0]:marker[trial,1]+1,0], 
            activity_pc[marker[trial,0]:marker[trial,1]+1,1], 
            activity_pc[marker[trial,0]:marker[trial,1]+1,2],'--',color=colors[i])
    ax.plot(activity_pc[marker[trial,1],0], activity_pc[marker[trial,1],1], activity_pc[marker[trial,1],2], marker='^',markersize=15, mfc=colors[i], markeredgewidth=0)
    ax.text(activity_pc[marker[trial,1],0], activity_pc[marker[trial,1],1], activity_pc[marker[trial,1],2],str(i+1),fontsize=11, ha='center', va='center', color='black', fontweight='bold')
azim = ax.azim    # 方位角(水平旋转角度, 0.77)
elev = ax.elev    # 方位角(竖直旋转角度, 55.65)
ax.view_init(elev=elev, azim=azim)
ax.set_axis_off()
plt.savefig('Results/prep.svg', format='svg', dpi=300)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=elev, azim=azim)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.savefig('Results/axis.svg', format='svg', dpi=300)

# 执行期
p = np.min(marker[:,2]-marker[:,1])
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
for i,trial in enumerate(trial_index_g[10]):
    activity_pc = pca.transform(fr[trial,:marker[trial,2]+1])
    ax.plot(activity_pc[marker[trial,0]:marker[trial,1]+p,0], 
            activity_pc[marker[trial,0]:marker[trial,1]+p,1], 
            activity_pc[marker[trial,0]:marker[trial,1]+p,2],color=colors[i])
    ax.plot(activity_pc[marker[trial,1],0], activity_pc[marker[trial,1],1], activity_pc[marker[trial,1],2], marker='s',markersize=12, mfc=colors[i], markeredgewidth=0)
    ax.text(activity_pc[marker[trial,1],0], activity_pc[marker[trial,1],1], activity_pc[marker[trial,1],2],str(i+1),fontsize=10, ha='center', va='center', color='black', fontweight='bold')

for i,trial in enumerate(trial_index_s[10]):
    activity_pc = pca.transform(fr[trial,:marker[trial,2]+1])
    ax.plot(activity_pc[marker[trial,0]:marker[trial,1]+p,0], 
            activity_pc[marker[trial,0]:marker[trial,1]+p,1], 
            activity_pc[marker[trial,0]:marker[trial,1]+p,2],'--',color=colors[i])
    ax.plot(activity_pc[marker[trial,1],0], activity_pc[marker[trial,1],1], activity_pc[marker[trial,1],2], marker='^',markersize=12, mfc=colors[i], markeredgewidth=0)
    ax.text(activity_pc[marker[trial,1],0], activity_pc[marker[trial,1],1], activity_pc[marker[trial,1],2],str(i+1),fontsize=10, ha='center', va='center', color='black', fontweight='bold')
ax.view_init(elev=elev, azim=azim)
ax.set_axis_off()
plt.savefig('Results/exec.svg', format='svg', dpi=300)


with open("Data/eval_results.json","r") as f:
    eval_results = json.load(f)

t_cc = eval_results["joint_grid"]["traj_cc"]
v_cc = eval_results["joint_grid"]["vel_cc"]
e = eval_results["joint_grid"]["epe"]
fig, ax = plt.subplots(figsize=(4, 6))
ax.errorbar([1.,2.],
    [np.mean(t_cc), np.mean(v_cc)], 
    yerr=[np.std(t_cc), np.std(v_cc)],
    fmt='o',
    markersize=12,
    color='blue',
    ecolor='red',  # 误差线颜色
    elinewidth=2,     # 误差线宽度
    capthick=3,       # 端帽厚度
    alpha=0.8,        # 透明度
    capsize=10
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x1=[1.,2.]
elabels=["Traj","Vel"]
ax.set_xticks(x1, elabels,fontsize=18)
ax.set_yticks([0.9,0.95,1.0])
ax.set_ylabel('Cross Correlation',fontsize=18)
plt.xlim(0.5,2.5)
plt.ylim(0.9,1)
plt.subplots_adjust(left=0.25)
plt.show()
plt.savefig('Results/Fig3_D.svg')

fig, ax = plt.subplots(figsize=(2, 6))
ax.errorbar(1.,
    np.mean(e), 
    yerr=np.std(e),
    fmt='o',
    markersize=12,
    color='blue',
    ecolor='red',  # 误差线颜色
    elinewidth=2,     # 误差线宽度
    capthick=3,       # 端帽厚度
    alpha=0.8,        # 透明度
    capsize=10
)
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
x1=[1.]
elabels=["Joint grid"]
ax.set_xticks(x1, elabels,fontsize=18)
ax.set_ylabel('End point error (m)',fontsize=20)
plt.xlim(0.5,1.5)
plt.ylim(0.005,0.015)
plt.show()

with open("Results/eval_results_15.json","r") as f:
    er_15 = json.load(f)
with open("Results/eval_results_20.json","r") as f:
    er_20 = json.load(f)
with open("Results/eval_results_25.json","r") as f:
    er_25 = json.load(f)

task_group = ["joint_grid","cartesian_grid","random"]
data_group = [er_15,er_20,er_25]
colors = ["#7986CB","#3F51B5","#303F9F"]
markers = ['o','^','d']
labels = ['SR_Dense','SR_Moderate','SR_Sparse']
for task in task_group:
    pos1 = [.5,1.0,1.5]
    pos2 = [2.8,3.3,3.8]
    fig, ax = plt.subplots(figsize=(4, 6))
    for i in range(3):
        tcc_mean = np.mean(data_group[i][task]["traj_cc"])
        tcc_std = np.std(data_group[i][task]["traj_cc"])
        ax.errorbar(pos1[i],
        tcc_mean, 
        yerr=tcc_std,
        fmt=markers[i],
        markersize=12,
        color='#303F9F',
        ecolor='black',  # 误差线颜色
        elinewidth=1.5,     # 误差线宽度
        capthick=3,       # 端帽厚度
        alpha=0.8,        # 透明度
        capsize=8,
        label=labels[i]
    )
        vcc_mean = np.mean(data_group[i][task]["vel_cc"])
        vcc_std = np.std(data_group[i][task]["vel_cc"])
        ax.errorbar(pos2[i],
        vcc_mean, 
        yerr=vcc_std,
        fmt=markers[i],
        markersize=12,
        color='#303F9F',
        ecolor='black',  # 误差线颜色
        elinewidth=1.5,     # 误差线宽度
        capthick=3,       # 端帽厚度
        alpha=0.8,        # 透明度
        capsize=8
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    x1=[1.,3.3]
    elabels=["Traj","Vel"]
    ax.set_xticks(x1, elabels,fontsize=18)
    ax.set_yticks([0.8,1.0])
    ax.set_ylabel('Cross Correlation',fontsize=18)
    plt.xlim(0.,4.5)
    plt.ylim(0.8,1)
    plt.subplots_adjust(left=0.25)
    ax.legend()
    plt.show()
    fig, ax = plt.subplots(figsize=(2, 6))
    for i in range(3):
        epe_mean=np.mean(data_group[i][task]["epe"])
        epe_std=np.std(data_group[i][task]["epe"])
        ax.errorbar(pos1[i],
        epe_mean, 
        yerr=epe_std,
        fmt=markers[i],
        markersize=12,
        color='#303F9F',
        ecolor='black',  # 误差线颜色
        elinewidth=1.5,     # 误差线宽度
        capthick=2,       # 端帽厚度
        alpha=0.8,        # 透明度
        capsize=8
    )
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    x1=[1.]
    elabels=["Joint grid"]
    ax.set_xticks(x1, elabels,fontsize=18)
    ax.set_ylabel('End point error (m)',fontsize=20)
    plt.xlim(0.,2.)
    plt.ylim(0.0,0.03)
    plt.show()


import matplotlib.pyplot as plt
with open("Results/eval_results_15.json","r") as f:
    er_100 = json.load(f)
with open("Results/eval_results_15_200.json","r") as f:
    er_200 = json.load(f)
with open("Results/eval_results_15_50.json","r") as f:
    er_50 = json.load(f)

task_group = ["joint_grid","cartesian_grid","random"]
data_group = [er_50,er_100,er_200]
colors = ["#7986CB","#3F51B5","#303F9F"]
markers = ['o','^','d']
labels = ['50_neurons','100_neurons','200_neurons']
for task in task_group:
    pos1 = [.5,1.0,1.5]
    pos2 = [2.8,3.3,3.8]
    fig, ax = plt.subplots(figsize=(4, 6))
    for i in range(3):
        tcc_mean = np.mean(data_group[i][task]["traj_cc"])
        tcc_std = np.std(data_group[i][task]["traj_cc"])
        ax.errorbar(pos1[i],
        tcc_mean, 
        yerr=tcc_std,
        fmt=markers[i],
        markersize=12,
        color='#303F9F',
        ecolor='black',  # 误差线颜色
        elinewidth=1.5,     # 误差线宽度
        capthick=3,       # 端帽厚度
        alpha=0.8,        # 透明度
        capsize=8,
        label=labels[i]
    )
        vcc_mean = np.mean(data_group[i][task]["vel_cc"])
        vcc_std = np.std(data_group[i][task]["vel_cc"])
        ax.errorbar(pos2[i],
        vcc_mean, 
        yerr=vcc_std,
        fmt=markers[i],
        markersize=12,
        color='#303F9F',
        ecolor='black',  # 误差线颜色
        elinewidth=1.5,     # 误差线宽度
        capthick=3,       # 端帽厚度
        alpha=0.8,        # 透明度
        capsize=8
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    x1=[1.,3.3]
    elabels=["Traj","Vel"]
    ax.set_xticks(x1, elabels,fontsize=18)
    ax.set_yticks([0.8,1.0])
    ax.set_ylabel('Cross Correlation',fontsize=18)
    plt.xlim(0.,4.5)
    plt.ylim(0.8,1)
    plt.subplots_adjust(left=0.25)
    ax.legend()
    plt.show()
    fig, ax = plt.subplots(figsize=(2, 6))
    for i in range(3):
        epe_mean=np.mean(data_group[i][task]["epe"])
        epe_std=np.std(data_group[i][task]["epe"])
        ax.errorbar(pos1[i],
        epe_mean, 
        yerr=epe_std,
        fmt=markers[i],
        markersize=12,
        color='#303F9F',
        ecolor='black',  # 误差线颜色
        elinewidth=1.5,     # 误差线宽度
        capthick=2,       # 端帽厚度
        alpha=0.8,        # 透明度
        capsize=8
    )
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    x1=[1.]
    elabels=["Joint grid"]
    ax.set_xticks(x1, elabels,fontsize=18)
    ax.set_ylabel('End point error (m)',fontsize=20)
    plt.xlim(0.,2.)
    plt.ylim(0.0,0.03)
    plt.show()







# task="random"
# tcc1=np.array(data_group[0][task]["traj_cc"])
# tcc2=np.array(data_group[1][task]["traj_cc"])
# tcc3=np.array(data_group[2][task]["traj_cc"])
# u_stat, p_value = stats.mannwhitneyu(tcc1, tcc2, alternative='two-sided')
# print(f"p-value: {p_value}")
# u_stat, p_value = stats.mannwhitneyu(tcc1, tcc3, alternative='two-sided')
# print(f"p-value: {p_value}")
# u_stat, p_value = stats.mannwhitneyu(tcc2, tcc3, alternative='two-sided')
# print(f"p-value: {p_value}")
# vcc1=data_group[0][task]["vel_cc"]
# vcc2=data_group[1][task]["vel_cc"]
# vcc3=data_group[2][task]["vel_cc"]
# u_stat, p_value = stats.mannwhitneyu(vcc1, vcc2, alternative='two-sided')
# print(f"p-value: {p_value}")
# u_stat, p_value = stats.mannwhitneyu(vcc1, vcc3, alternative='two-sided')
# print(f"p-value: {p_value}")
# u_stat, p_value = stats.mannwhitneyu(vcc2, vcc3, alternative='two-sided')
# print(f"p-value: {p_value}")
# epe1=data_group[0][task]["epe"]
# epe2=data_group[1][task]["epe"]
# epe3=data_group[2][task]["epe"]
# u_stat, p_value = stats.mannwhitneyu(epe1, epe2, alternative='two-sided')
# print(f"p-value: {p_value}")
# u_stat, p_value = stats.mannwhitneyu(epe1, epe3, alternative='two-sided')
# print(f"p-value: {p_value}")
# u_stat, p_value = stats.mannwhitneyu(epe2, epe3, alternative='two-sided')
# print(f"p-value: {p_value}")