# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 13:07:23 2025

@author: Administrator
"""

import json
import numpy as np
import torch as th
import motornet as mn
from ..model.rnn import RNNCell
from ..model.build import build_task
from ..training.utils import run_episode, evalPrediction, compute_epe, cal_vel
from ..tasks.grid_reach import BasicTask, BasicTaskEnv
from ..training.params import default_params

device = th.device("cuda")

# fig3F,G:计算格点对和随机点对端点误差和运动轨迹和速度的CC
k = ["joint_grid", "cartesian_grid", "random"]
eval_results = {}

model_file = "model/260708_basicModel_200/"
model_name = "rnn__joint_interval_15.pth"
# with open(model_file+"param.json", "r") as f:
#     model_params = json.load(f)
model_params,task_params,_ = default_params()

effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
model = RNNCell(model_params)
w = th.load(model_file+model_name)
model.load_state_dict(w)
model.to(device)

param = {"obs_class"       : "prop+vis",
         "ini_std"         : 0.05,
         "session_num"     : 5,
         "joint_interval"  : [25,20,15],
         "grid_interval"   : [0.12, 0.10, 0.08],
         "test_batch_size" : [128, 256, 512, 1024],
         "center_joint"    : [[56.6, 107.35],[47.3, 111.],[38, 113.3],[30.,114.]],
         "reach_distance"  : [0.10, 0.13],
         "vel_range"       : [0.3,0.4,0.5,0.6],
         }
task_info = {"task" : "grid",
             "workspace":"joint_space",
             "trial_num" : 1,
             "to_range" : [0.2, 0.3],
             "dis_range" : [0.05, 0.35],
             "delay_range": [0.2, 0.6],
             "hold_durtion" : 0.10,
             "catch_trial_proportion":0.0
             }

#关节空间格点
t_j=[]
v_j=[]
e_j=[]
for ind, j in enumerate(param["joint_interval"]):
    task_info["joint_interval"] = j
    for v in param["vel_range"]:
        task_info["vel_range"] = [v]
        task, options, traj, vel = build_task(param, task_info)
        max_ep_duration = round(options["movement_timepoints"].max() * effector.dt, 3)
        env = BasicTaskEnv(effector=effector, 
                           max_ep_duration=max_ep_duration, 
                           action_noise=task_params["action_noise"],
                           proprioception_noise=task_params["proprioception_noise"],
                           vision_noise=task_params["vision_noise"],
                           proprioception_delay=task_params["proprioception_delay"],
                           vision_delay=task_params["vision_delay"]).to(device)
        for t in range(param["session_num"]):
            data, _ = run_episode(env, model, options, device=device, detach=True)

            traj_pred=data["xy"].to("cpu").numpy()
            vel_pred=data["vel"].to("cpu").numpy()
            ntps = options["movement_timepoints"]
            marker = options["marker"]
            goal = options["g"]
            trial_num = traj_pred.shape[0]
        
            t_p=np.vstack([traj_pred[t,:ntps[t]] for t in range(trial_num)])
            t=np.vstack([traj[t,:ntps[t]] for t in range(trial_num)])
            v_p=np.vstack([vel_pred[t,:ntps[t]] for t in range(trial_num)])
            v=np.vstack([vel[t,:ntps[t]] for t in range(trial_num)])
            tcc = evalPrediction(t, t_p, 'CC')
            vcc = evalPrediction(v, v_p, 'CC')
            epe = compute_epe(goal, traj_pred, marker)
            t_j.append(tcc.mean())
            v_j.append(vcc.mean())
            e_j.append(epe.mean())

joint_grid_er = {"traj_cc":t_j,"vel_cc":v_j,"epe":e_j}
eval_results["joint_grid"] = joint_grid_er

#笛卡尔空间格点
task_info["workspace"] = "cartesian_space"
t_g=[]
v_g=[]
e_g=[]
for ind, j in enumerate(param["grid_interval"]):
    task_info["grid_interval"] = j
    for v in param["vel_range"]:
        task_info["vel_range"] = [v]
        task, options, traj, vel = build_task(param, task_info)
        max_ep_duration = round(options["movement_timepoints"].max() * effector.dt, 3)
        env = BasicTaskEnv(effector=effector, 
                           max_ep_duration=max_ep_duration, 
                           action_noise=task_params["action_noise"],
                           proprioception_noise=task_params["proprioception_noise"],
                           vision_noise=task_params["vision_noise"],
                           proprioception_delay=task_params["proprioception_delay"],
                           vision_delay=task_params["vision_delay"]).to(device)
        for t in range(param["session_num"]):
            data, _ = run_episode(env, model, options, device=device, detach=True)

            traj_pred=data["xy"].to("cpu").numpy()
            vel_pred=data["vel"].to("cpu").numpy()
            ntps = options["movement_timepoints"]
            marker = options["marker"]
            goal = options["g"]
            trial_num = traj_pred.shape[0]
        
            t_p=np.vstack([traj_pred[t,:ntps[t]] for t in range(trial_num)])
            t=np.vstack([traj[t,:ntps[t]] for t in range(trial_num)])
            v_p=np.vstack([vel_pred[t,:ntps[t]] for t in range(trial_num)])
            v=np.vstack([vel[t,:ntps[t]] for t in range(trial_num)])
            tcc = evalPrediction(t, t_p, 'CC')
            vcc = evalPrediction(v, v_p, 'CC')
            epe = compute_epe(goal, traj_pred, marker)
            t_g.append(tcc.mean())
            v_g.append(vcc.mean())
            e_g.append(epe.mean())

cartesian_grid_er = {"traj_cc":t_g,"vel_cc":v_g,"epe":e_g}
eval_results["cartesian_grid"] = cartesian_grid_er

#随机点
task_info["task"] = "random"
t_r=[]
v_r=[]
e_r=[]
for ind, j in enumerate(param["test_batch_size"]):
    task_info["trial_num"] = j
    for v in param["vel_range"]:
        task_info["vel_range"] = [v]
        task, options, traj, vel = build_task(param, task_info)
        max_ep_duration = round(options["movement_timepoints"].max() * effector.dt, 3)
        env = BasicTaskEnv(effector=effector, 
                           max_ep_duration=max_ep_duration, 
                           action_noise=task_params["action_noise"],
                           proprioception_noise=task_params["proprioception_noise"],
                           vision_noise=task_params["vision_noise"],
                           proprioception_delay=task_params["proprioception_delay"],
                           vision_delay=task_params["vision_delay"]).to(device)
        for t in range(param["session_num"]):
            data, _ = run_episode(env, model, options, device=device, detach=True)

            traj_pred=data["xy"].to("cpu").numpy()
            vel_pred=data["vel"].to("cpu").numpy()
            ntps = options["movement_timepoints"]
            marker = options["marker"]
            goal = options["g"]
            trial_num = traj_pred.shape[0]
        
            t_p=np.vstack([traj_pred[t,:ntps[t]] for t in range(trial_num)])
            t=np.vstack([traj[t,:ntps[t]] for t in range(trial_num)])
            v_p=np.vstack([vel_pred[t,:ntps[t]] for t in range(trial_num)])
            v=np.vstack([vel[t,:ntps[t]] for t in range(trial_num)])
            tcc = evalPrediction(t, t_p, 'CC')
            vcc = evalPrediction(v, v_p, 'CC')
            epe = compute_epe(goal, traj_pred, marker)
            t_r.append(tcc.mean())
            v_r.append(vcc.mean())
            e_r.append(epe.mean())

random_er = {"traj_cc":t_r,"vel_cc":v_r,"epe":e_r}
eval_results["random"] = random_er
with open('Results/eval_results_15_200.json', 'w', encoding='utf-8') as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=4)

