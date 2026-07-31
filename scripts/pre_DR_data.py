import json
import torch as th
import torch.nn as nn
import numpy as np
import motornet as mn
from ..tasks.sequential_reach import DoubleReachTask, DoubleReachEnv
from ..model.build import build_model

device = th.device("cuda")


def normalize(data):
    for i in range(data.shape[-1]):
        tmp = data[:, :, i]
        min_val = tmp.min()
        max_val = tmp.max()
        data[:, :, i] = (tmp - min_val) / (max_val - min_val)
    return data


def run_episode(env, rnn, options, device=th.device("cuda"), detach=False):
    batch_size = options["batch_size"]
    std = options["ini_std"]
    ntps = int(round(env.max_ep_duration / env.dt, 3))
    obs, info = env.reset(options=options)

    x = rnn.init_hidden(batch_size, std=std)
    terminated = False

    data = {
        "xy": th.zeros(batch_size, ntps, 2).to(device),
        "vel": th.zeros(batch_size, ntps, 2).to(device),
        "all_actions": th.zeros(batch_size, ntps, env.muscle.n_muscles).to(device),
        "all_hidden": th.zeros(batch_size, ntps, rnn.hidden_size).to(device),
        "all_joint": th.zeros(batch_size, ntps, info["states"]["joint"].shape[-1]).to(device),
        "all_muscle": th.zeros(batch_size, ntps, info["states"]["muscle"].shape[-1]).to(device),
        "all_force": th.zeros(batch_size, ntps, info["states"]["muscle"].shape[-1]).to(device),
        "muscle_length": th.zeros(batch_size, ntps, info["states"]["muscle"].shape[-1]).to(device),
    }

    while not terminated:
        tps = int(round(env.elapsed / env.dt, 2))
        action, x = rnn(x, obs[:, -6:], obs[:, :-6])
        obs, _, terminated, _, info = env.step(action=action, options=options)
        data["all_hidden"][:, tps, :] = x
        data["all_muscle"][:, tps, :] = info["states"]["muscle"][:, 0, :]
        data["all_joint"][:, tps, :] = info["states"]["joint"]
        data["xy"][:, tps, :] = info["states"]["fingertip"]
        data["vel"][:, tps, :] = info["states"]["cartesian"][:, 2:]
        data["all_actions"][:, tps, :] = action
        data["all_force"][:, tps, :] = info["states"]["muscle"][:, 6, :]
        data["muscle_length"][:, tps, :] = info["states"]["muscle"][:, 1, :]

    if detach:
        for key in data:
            data[key] = th.detach(data[key])
    return data


param = {
    "model": "rnn", "n_rec": 100, "nonlinearity": "tanh",
    "noise_std": 0.0, "ini_std": 0.0, "obs_class": "prop+vis", "n_out_task": 12,
}
effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
model = build_model(param)
model.tasknet = nn.Sequential(
    nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, model.task_size)
)

task = DoubleReachTask()
options = {"task": "DR", "batch_size": 30, "catch_trial_proportion": 0.0}
trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train = task.genReach(options=options)
options["goal_class"] = 1
options["obs_class"] = param["obs_class"]
options["ini_std"] = param["ini_std"]
options["trajactory"] = trajactory
options["velocity"] = velocity
options["marker"] = marker
options["movement_timepoints"] = marker[-1]
options["gocue"] = gocue
options["goal"] = goal
options["joint_state"] = joint_state
max_ep_duration = round(marker[-1].max() * effector.dt, 3)
env = DoubleReachEnv(effector=effector, max_ep_duration=max_ep_duration).to(device)

models = ["DR_1", "DR_2", "DR_3", "DR_multi_1", "DR_multi_2", "DR_multi_3", "DR_fs", "DR_multi_fs"]
fr = np.zeros((8, options["batch_size"], marker[-1], param["n_rec"]))
for i, m in enumerate(models):
    w = th.load(f"model/DR/rnn_{m}.pth")
    model.load_state_dict(w)
    model.to(device)
    data = run_episode(env, model, options, device=device, detach=True)
    fr[i] = data["all_hidden"].to("cpu").numpy()

np.save("Data/DR/fr.npy", fr)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import auc
from DSA import DSA


def cca_analysis(fr1,fr2,dim):
    c1,t1,n1 = fr1.shape
    fr1_flatten = np.reshape(fr1,(c1*t1, n1))
    pca = PCA(n_components=10)
    X = pca.fit_transform(fr1_flatten, dim)
    
    c2,t2,n2 = fr2.shape
    fr2_flatten = np.reshape(fr2,(c2*t2, n2))
    pca = PCA(n_components=10)
    Y = pca.fit_transform(fr2_flatten, dim)
    
    cca = CCA(n_components=dim)    
    cca.fit(X,Y)
    X_c, Y_c = cca.transform(X,Y) 
    r = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(X_c.shape[1])]     
    return r

n_delays = 5
delay_interval = 2
rank = 10
d = 'cuda'
dr_data = np.load("Data/DR/fr_dr_data.npy")
r_CCA = np.zeros((4,5,10))
auc_cca = np.zeros((4,5))
K_d = np.zeros((4,5))

labels = ["DR","DR_multi","DR_fs","DR_multi_fs"]
pc = np.arange(1,11)
for i,m in enumerate(labels):
    fr = np.load(f"Data/DR/fr_{m}.npy")
    for j in range(5):
        r = cca_analysis(fr[j], dr_data, 10)
        r_CCA[i,j] = r
        auc_cca[i,j] = auc(pc, r)
        dsa = DSA(fr[j],dr_data,n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=False,device=d,iters=2000,lr=1e-2)
        similarities = dsa.fit_score()
        K_d[i,j] = similarities

np.save("Data/DR/r_CCA.npy",r_CCA)
np.save("Data/DR/AUC.npy",auc_cca)
np.save("Data/DR/DSA.npy",K_d)