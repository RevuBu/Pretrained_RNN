import os
import json
import numpy as np
import torch as th
import torch.nn as nn
import motornet as mn
from tqdm import tqdm
from ..tasks.sequential_reach import RandomTargetReach, RTTEnv
from ..model.build import build_model
from .utils import run_episode, print_losses


def cal_loss(data, goal, velocity, bhv_marker, training_params, device=th.device("cuda")):
    loss_weights = training_params["loss_weights"]
    loss_to_compute = training_params["loss_to_compute"]
    loss = {}
    losses_weighted = {}
    trial_num = goal.shape[0]

    e_pos = th.zeros(trial_num).to(device)
    e_vel = th.zeros(trial_num).to(device)
    e_jerk = th.zeros(trial_num).to(device)
    e_muscle = th.zeros(trial_num).to(device)

    for trial in range(trial_num):
        end = bhv_marker[trial, -1]
        for i in range(goal.shape[1] - 1):
            go = bhv_marker[trial, 2 * i]
            tt = bhv_marker[trial, 2 * i + 1]
            e_pos[trial] += th.mean(
                th.abs(data["xy"][trial, tt - 5 : tt + 5, :] - goal[trial, i, :])
            )
            e_vel[trial] += th.mean(
                th.square(data["vel"][trial, go:tt, :] - velocity[trial, go:tt, :])
            )

        e_pos[trial] += th.mean(
            th.abs(data["xy"][trial, end - 5 : end, :] - goal[trial, -1, :])
        )
        e_pos[trial] = e_pos[trial] / goal.shape[1]
        e_vel[trial] += th.mean(
            th.square(
                data["vel"][trial, bhv_marker[trial, 2 * i] : end, :]
                - velocity[trial, bhv_marker[trial, 2 * i] : end, :]
            )
        )
        e_vel[trial] = e_vel[trial] / goal.shape[1]
        e_jerk[trial] = th.mean(th.square(th.diff(data["vel"][trial, :end, :], n=2, dim=0)))
        e_muscle[trial] = th.mean(data["all_force"][trial, :end, :])

    loss["position"] = th.mean(e_pos)
    loss["vel"] = th.mean(e_vel)
    loss["jerk"] = th.mean(e_jerk)
    loss["muscle"] = th.mean(e_muscle)

    for i, key in enumerate(loss.keys()):
        losses_weighted[key] = loss_weights[i] * loss[key]

    overall_loss = 0.0
    if loss_to_compute is None:
        for l in losses_weighted.keys():
            overall_loss += losses_weighted[l]
    else:
        for l in loss_to_compute:
            overall_loss += losses_weighted[l]
    return overall_loss, losses_weighted, loss


def train(model, env, options, training_params, model_file, device):
    goal = th.tensor(options["goal"]).to(device)
    vel = th.tensor(options["velocity"]).to(device)
    marker = options["marker"]

    optimizer = th.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=training_params["lr"]
    )
    losses = {"overall": [], "position": [], "vel": [], "jerk": [], "muscle": []}
    original_loss = {"position": [], "vel": [], "jerk": [], "muscle": []}
    n_batch = training_params["n_batch"]
    batch_size = goal.shape[0]

    for batch in tqdm(range(n_batch), desc=f"Training {n_batch} batches of {batch_size}", unit="batch"):
        data, _ = run_episode(env, model, options, device=device)
        loss, losses_weighted, uwl = cal_loss(
            data=data, goal=goal, velocity=vel,
            bhv_marker=marker, training_params=training_params,
        )

        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses["overall"].append(loss.item())
        losses["position"].append(losses_weighted["position"].item())
        losses["vel"].append(losses_weighted["vel"].item())
        losses["jerk"].append(losses_weighted["jerk"].item())
        losses["muscle"].append(losses_weighted["muscle"].item())

        for key in uwl.keys():
            original_loss[key].append(uwl[key].item())

        log_file = model_file + "___losses.txt"
        if (batch % training_params["interval"] == 0) and (batch != 0):
            print_losses(
                overall_loss=loss, losses_weighted=losses_weighted,
                log_file=log_file, batch=batch,
            )

        loss_file = model_file + "__loss.json"
        with open(loss_file, "w") as file:
            json.dump({"losses": original_loss}, file)

    return model


if __name__ == "__main__":
    loss_weights = np.array([1e+2, 1e+2, 1e+4, 1e-2])
    loss_to_compute = ["position", "vel", "jerk", "muscle"]

    training_params = {
        "n_batch": 5001,
        "interval": 1000,
        "lr": 1e-4,
        "loss_weights": loss_weights,
        "loss_to_compute": loss_to_compute,
    }

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    param = {
        "model": "rnn",
        "n_rec": 200,
        "nonlinearity": "tanh",
        "noise_std": 0.10,
        "ini_std": 0.0,
        "obs_class": "prop+vis",
        "n_out_task": 12,
    }
    device = th.device("cuda")
    model_name = "well_trained_model/260710_RTT_200"
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    with open(model_name + "/param.json", "w") as file:
        json.dump(param, file)

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    model = build_model(param)

    Basic_model_file = "well_trained_model/260708_basicModel_200/"
    Basic_model_name = "rnn__joint_interval_15.pth"
    w = th.load(Basic_model_file + Basic_model_name)
    model.load_state_dict(w)
    frozen_params = ["recurrent_layer.weight", "output_layer.weight", "output_layer.bias"]
    for name, model_param in model.named_parameters():
        if name in frozen_params:
            model_param.requires_grad = False
    model.to(device)

    task = RandomTargetReach()
    trial_num = 128
    dis_range = [0.10, 0.15]
    options = {
        "batch_size": trial_num,
        "obs_class": "prop+vis",
        "ini_std": 0.0,
        "dis_range": dis_range,
        "hold_durtion": 0.1,
        "ave_vel": 0.5,
    }
    for seq_len in range(4, 5):
        options["seq_len"] = seq_len
        targets, joint_state = task.genTargets(trial_num, seq_len, dis_range)
        options["targets"] = targets
        options["joint_state"] = joint_state
        trajactory, velocity, marker, gocue, joint_state, goal = task.genReach(options)
        options["trajactory"] = trajactory
        options["velocity"] = velocity
        options["marker"] = marker
        options["movement_timepoints"] = marker[:, -1]
        options["gocue"] = gocue
        options["goal"] = goal
        max_ep_duration = round(marker[:, -1].max() * effector.dt, 3)
        env = RTTEnv(effector=effector, max_ep_duration=max_ep_duration).to(device)
        model_file = model_name + "/" + param["model"] + "_RTT_" + str(seq_len)
        model = train(model, env, options, training_params, model_file, device)
        th.save(model.state_dict(), model_file + ".pth")
