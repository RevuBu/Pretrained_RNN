# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:49:56 2026

@author: Administrator
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import motornet as mn
from ..model.rnn import RNNCell
from ..tasks.sequential_reach import RandomTargetReach, RTTEnv
from ..training.utils import run_episode
from ..training.params import default_params
from DSA import DSA

device = th.device("cuda")
model_params, task_params, _ = default_params()
model_params["noise_std"] = 0.10
w1 = "well_trained_model/251220_RTT_4/rnn_RTT_4.pth"
w2 = "well_trained_model/260330_RTT_20/rnn_RTT_4.pth"
w3 = "well_trained_model/260330_RTT_25/rnn_RTT_4.pth"
w4 = "well_trained_model/251130_RTT_4_fs/rnn_RTT_4.pth"
weight_list = [w1,w2,w3,w4]
effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
task = RandomTargetReach()
trial_num = 128
seq_len=5
dis_range=[0.05,0.15]
targets_model, joint_state = task.genTargets(trial_num, seq_len, dis_range)
options = {"batch_size" : trial_num,
           "obs_class" : "prop+vis", 
           "ini_std" : 0.0,
           "seq_len" : seq_len,
           "dis_range" : dis_range,
           "hold_durtion" : 0.1,
           "ave_vel" : 0.5
           }
options["targets"] = targets_model
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
K_d = np.zeros((4,10))
n_delays = 5
delay_interval = 2
rank = 10
d = 'cuda'
RTT_data = np.load("Data/RTT/MM_M1.npy")
fr_list = []
for i, weight in enumerate(weight_list):
    model = RNNCell(model_params)
    w = th.load(weight)
    model.load_state_dict(w)
    model.to(device)
    for t in range(10):
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
        K_d[i,t] = similarities
np.save("Data/RTT/rtt_DSA_260505",K_d)