# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 20:44:01 2025

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt


# colors=["#FFBC40","#EE9B00","#0CBCC0","#099396","#F0786A","#AE2012"]
# labels = ["DR","DR_multi","DR_cl","DR_multi_cl","DR_fs","DR_multi_fs"]
# fig, ax = plt.subplots(figsize=(6,4))
# ml=loss.mean(axis=1)
# ml_fs=loss_fs.mean(axis=1)
# for i in range(6):
#     if i < 4:
#         ax.semilogy(ml[3-i], color=colors[3-i], linewidth=1.8, label=labels[3-i])
#     else:
#         ax.semilogy(ml_fs[i-4], color=colors[i], linewidth=1.8, label=labels[i])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks([0,10000,20000,30000])
# plt.legend(loc="upper right", bbox_to_anchor=(0.9, 1.1), fontsize=13)
# plt.tight_layout()
# plt.xlabel('Epoch',fontsize=16)
# plt.ylabel('Loss',fontsize=16)
# plt.show()

# fig,ax = plt.subplots(figsize=(6,4))
# for i in range(6):
#     ax.plot(pc, r_CCA[i].mean(axis=0),'.-',linewidth=2.5,markersize=10,color=colors[i],label=labels[i])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks(list(range(1,11)))
# plt.xlabel('neural mode',fontsize=16)
# plt.ylabel('CC score',fontsize=16)
# plt.legend(fontsize=13)
# plt.show()

# fig,ax = plt.subplots(figsize=(6,4))
# x=np.arange(1,12,2)
# auc_mean = [auc_cca[i].mean() for i in range(auc_cca.shape[0])]
# auc_std = [auc_cca[i].std() for i in range(auc_cca.shape[0])]
# bar = ax.bar(labels, auc_mean, yerr=auc_std,capsize=12,color=colors,edgecolor='black', width=0.6, alpha=0.8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xticks(fontsize=12, rotation=45)
# plt.ylabel('AUC of CCA', fontsize=14)
# plt.show()

# fig,ax = plt.subplots(figsize=(6,4))
# x=np.arange(1,12,2)
# K_mean = [K_d[i].mean() for i in range(auc_cca.shape[0])]
# K_std = [K_d[i].std() for i in range(auc_cca.shape[0])]
# bar = ax.bar(labels, K_mean, yerr=K_std,capsize=12,color=colors,edgecolor='black', width=0.6, alpha=0.8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xticks(fontsize=12, rotation=45)
# plt.ylabel('Dynamical Similarity', fontsize=14)
# plt.show()

# fig,ax = plt.subplots(figsize=(6,4))
# for i in range(6):
#     ax.plot(pc, r_CCA[i].mean(axis=0),'.-',linewidth=2.5,markersize=10,color=colors[i],label=labels[i])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xticks(list(range(1,11)))
# plt.xlabel('neural mode',fontsize=14)
# plt.ylabel('CC score',fontsize=14)
# plt.legend(fontsize=10)
# plt.show()

# fig,ax = plt.subplots(figsize=(6,4))
# x=np.arange(1,12,2)
# auc_data = [auc_cca[i] for i in range(auc_cca.shape[0])]
# box = ax.boxplot(auc_data,positions=x,label=labels,patch_artist=True,showfliers=False)
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# ax.set_xticks(x, labels, fontsize=8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# #plt.ylim([5.5,8.0])
# plt.ylabel("AUC")
# plt.show()

# fig,ax = plt.subplots(figsize=(6,4))
# x=np.arange(1,12,2)
# k = [K_d[i] for i in range(K_d.shape[0])]
# box = ax.boxplot(k,positions=x,label=labels,patch_artist=True,showfliers=False)
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
# ax.set_xticks(x, labels, fontsize=8)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# #plt.ylim([5.5,8.0])
# plt.ylabel("Distance")
# plt.show()

traj = np.load("Data/DR/traj_DR.npy")
vel = np.load("Data/DR/vel_DR.npy")

fig,ax = plt.subplots(figsize=(6,4))
for i in range(6):
    ax.plot(traj[i,0,0],traj[i,0,1],marker='o',markersize=10,color='green',label=labels[0])
    for j in range(3):
        ax.plot(goal[i,j,0],goal[i,j,1],marker='o',markersize=16,color=colors[j],label=labels[j+1])
    ax.plot(traj[i,:marker[i,-1],0],traj[i,:marker[i,-1],1],linewidth=3)
ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.legend(fontsize=12)
plt.show()

r_CCA = np.load("Data/DR/r_CCA.npy")
auc_cca = np.load("Data/DR/AUC.npy")
K_d = np.load("Data/DR/DSA.npy")

labels = ["SR_Dense_ft_S: ","SR_Moderate_ft_S: ","SR_Sparse_ft_S: ", "SR_Dense_ft_M: ","SR_Moderate_ft_M: ","SR_Sparse_ft_M: ", "DR_S: ", "DR_M: "]
for ii, l in enumerate(labels):
    labels[ii] = l + f"{auc_cca[ii]:.3f}"
colors = ["#CC247C","#E95351","#F7A24F","#FBEB66","#4EA660","#79CAFB","#5292F7","#AA77E9"]
pc = np.arange(1,11)
fig,ax = plt.subplots(figsize=(6,4))
for i in range(8):
    ax.plot(pc, r_CCA[i],'.-',linewidth=2.5,markersize=10,color=colors[i],label=labels[i])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(list(range(1,11)))
plt.xlabel('neural mode',fontsize=14)
plt.ylabel('CC score',fontsize=14)
plt.subplots_adjust(bottom=0.13)
plt.legend(fontsize=9)
plt.show()

fig,ax = plt.subplots(figsize=(6,4))
x=np.arange(1,16,2)
bar = ax.bar(labels, auc_cca,color=colors,edgecolor='black', width=0.6, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=8, rotation=45)
plt.ylabel('AUC of CCA', fontsize=14)
plt.subplots_adjust(bottom=0.27)
plt.show()
enumerate()
fig,ax = plt.subplots(figsize=(6,4))
x=np.arange(1,16,2)
bar = ax.bar(labels, K_d,color=colors,edgecolor='black', width=0.6, alpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=8, rotation=45)
plt.ylabel('Dissimilarity Score', fontsize=14)
plt.subplots_adjust(bottom=0.27)
plt.subplots_adjust(left=0.13)
plt.show()
