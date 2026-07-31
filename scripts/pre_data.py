import torch as th
import numpy as np
import motornet as mn
from ..model.rnn import RNNCell
from ..tasks.grid_reach import CenterOutTask, BasicTaskEnv
from ..training.utils import run_episode
from ..training.params import default_params

device = th.device("cuda")
model_params, task_params, _ = default_params()
model_params["noise_std"] = 0.0

options = {
    "batch_size": 8,
    "center_joint": [[38.0, 113.3]],
    "angle_interval": [45],
    "target_radius": [0.12],
    "target_on": 0.1,
    "delay_durtion": 0.4,
    "reach_durtion": 0.4,
    "hold_durtion": 0.1,
    "catch_trial_proportion": 0.0,
}
options["obs_class"] = "prop+vis"
options["ini_std"] = 0.0
effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
max_ep_duration = round(
    options["target_on"] + options["delay_durtion"] + options["reach_durtion"] + options["hold_durtion"], 3
)
env = BasicTaskEnv(effector=effector, max_ep_duration=max_ep_duration).to(device)
task = CenterOutTask()
trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train = task.genReach(options)
options["trajactory"] = trajactory
options["velocity"] = velocity
options["marker"] = marker
options["movement_timepoints"] = marker[:, -1]
options["gocue"] = gocue
options["goal"] = goal_for_train
options["joint_state"] = joint_state

data_name = [
    "fr_SR_Dense", "fr_SR_Moderate", "fr_SR_Sparse",
    "fr_CO_Single", "fr_CO_Multi", "fr_DR_Single", "fr_DR_Multi",
]
model_list = [
    "model/250916_basicModel/rnn__joint_interval_15.pth",
    "model/260326_basicModel/rnn__joint_interval_20.pth",
    "model/260327_basicModel/rnn__joint_interval_25.pth",
    "model/251105_CO_Model/rnn_co_1.pth",
    "model/251105_CO_Model/rnn_co_12.pth",
    "model/260331_CO_Model_from_DR/rnn_co_1.pth",
    "model/260330_CO_Model_from_multi_DR/rnn_co_12.pth",
]

model = RNNCell(model_params)
for i, model_name in enumerate(model_list):
    w = th.load(model_name)
    model.load_state_dict(w)
    model.to(device)
    data, _ = run_episode(env, model, options, device=device, detach=True)
    fr = data["all_hidden"].to("cpu").numpy()
    np.save(f"Data/CO/{data_name[i]}.npy", fr)
