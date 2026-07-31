import os
import json
import numpy as np
import torch as th
import torch.nn as nn
import motornet as mn
from tqdm import tqdm
from ..tasks.sequential_reach import DoubleReachTask, DoubleReachEnv
from ..model.build import build_model
from .utils import print_losses
from .params import default_params


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


def cal_loss(data, goal, trajactory, velocity, bhv_marker, training_params, device=th.device("cuda")):
    loss_weights = training_params["loss_weights"]
    loss_to_compute = training_params["loss_to_compute"]
    loss = {}
    losses_weighted = {}

    if training_params["task"] == "SR":
        go = bhv_marker[1]
        tt = bhv_marker[2]
        end = bhv_marker[-1]
        loss["position"] = th.mean(th.abs(data["xy"][:, tt:end, :] - goal[:, None, :2]))
        loss["hold"] = th.mean(th.abs(data["xy"][:, :go] - trajactory[:, 0:1]))
        loss["traj"] = th.mean(th.square(data["xy"][:, go:tt] - trajactory[:, go:tt]))
        loss["vel"] = th.mean(th.square(data["vel"][:, go:tt] - velocity[:, go:tt]))
        loss["jerk"] = th.mean(th.square(th.diff(data["vel"][:, :end, :], n=2, dim=0)))
        loss["muscle"] = th.mean(data["all_force"][:, :end, :])

    elif training_params["task"] == "DR":
        go1 = bhv_marker[1]
        tt1 = bhv_marker[2]
        go2 = bhv_marker[3]
        tt2 = bhv_marker[4]
        end = bhv_marker[-1]
        loss["position"] = th.mean(
            th.abs(data["xy"][:, tt1:go2, :] - goal[:, None, :2])
            + th.abs(data["xy"][:, tt2:end, :] - goal[:, None, 2:])
        )
        loss["hold"] = th.mean(th.abs(data["xy"][:, :go1] - trajactory[:, 0:1]))
        loss["traj"] = th.mean(th.square(data["xy"][:, go1:tt2] - trajactory[:, go1:tt2]))
        loss["vel"] = th.mean(th.square(data["vel"][:, go1:tt2] - velocity[:, go1:tt2]))
        loss["jerk"] = th.mean(th.square(th.diff(data["vel"][:, :end, :], n=2, dim=0)))
        loss["muscle"] = th.mean(data["all_force"][:, :end, :])

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


def train(model, env, task, options, training_params, model_file, device):
    optimizer = th.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=training_params["lr"]
    )
    losses = {
        "overall": [], "position": [], "hold": [], "traj": [],
        "vel": [], "jerk": [], "muscle": [],
    }
    original_loss = {
        "position": [], "hold": [], "traj": [], "vel": [], "jerk": [], "muscle": [],
    }
    n_batch = training_params["n_batch"]
    batch_size = options["batch_size"]

    for batch in tqdm(range(n_batch), desc=f"Training {n_batch} batches of {batch_size}", unit="batch"):
        if (batch % training_params["interval"]) == 0:
            if training_params["task"] == "SR":
                options["task"] = "SR"
                trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train = (
                    task.genSingleReach(options=options)
                )
            else:
                options["task"] = "DR"
                trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train = (
                    task.genReach(options=options)
                )

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

            gft = th.tensor(goal_for_train).to(device)
            traj = th.tensor(trajactory).to(device)
            vel = th.tensor(velocity).to(device)

        data = run_episode(env, model, options, device=device)
        loss, losses_weighted, uwl = cal_loss(
            data=data, goal=gft, trajactory=traj, velocity=vel,
            bhv_marker=marker, training_params=training_params,
        )

        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses["overall"].append(loss.item())
        losses["position"].append(losses_weighted["position"].item())
        losses["hold"].append(losses_weighted["hold"].item())
        losses["traj"].append(losses_weighted["traj"].item())
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
    loss_weights = np.array([1e+2, 3e+2, 1e+3, 5e+2, 1e+1, 1e-2])
    loss_to_compute = ["position", "hold", "traj", "vel", "jerk", "muscle"]

    training_params = {
        "task": "SR",
        "n_batch": 10001,
        "interval": 1000,
        "lr": 1e-4,
        "loss_weights": loss_weights,
        "loss_to_compute": loss_to_compute,
    }

    task_params = {
        "catch_trial_proportion": 0.0,
        "threshold": 0.02,
        "action_noise": 0.01,
        "proprioception_noise": 0.01,
        "vision_noise": 0.01,
        "proprioception_delay": 0.0,
        "vision_delay": 0.0,
    }

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    param = {
        "model": "rnn",
        "n_rec": 100,
        "nonlinearity": "tanh",
        "noise_std": 0.0,
        "ini_std": 0.0,
        "obs_class": "prop+vis",
        "n_out_task": 12,
    }
    device = th.device("cuda")
    model_name = "well_trained_model/251216_DR"
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    with open(model_name + "/param.json", "w") as file:
        json.dump(param, file)

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    model = build_model(param)
    model.tasknet = nn.Sequential(
        nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, model.task_size)
    )

    # Load pretrained basic model
    Basic_model_file = "well_trained_model/250916_basicModel/"
    Basic_model_name = "rnn__joint_interval_15.pth"
    pretrained_dict = th.load(Basic_model_file + Basic_model_name)
    new_model_dict = model.state_dict()
    for param_name in new_model_dict.keys():
        if "tasknet.0" not in param_name:
            new_model_dict[param_name] = pretrained_dict[param_name]
    model.load_state_dict(new_model_dict)

    frozen_params = ["recurrent_layer.weight", "output_layer.weight", "output_layer.bias"]
    for name, model_param in model.named_parameters():
        if name in frozen_params:
            model_param.requires_grad = False
    model.to(device)

    task = DoubleReachTask()

    # Train DR
    options = {"batch_size": 30, "catch_trial_proportion": 0.0}
    _, _, marker, _, _, _, _ = task.genReach(options=options)
    max_ep_duration = round(marker[-1].max() * effector.dt, 3)
    env = DoubleReachEnv(effector=effector, max_ep_duration=max_ep_duration).to(device)

    training_params["task"] = "DR"
    training_params["n_batch"] = 10001
    training_params["interval"] = 1000
    model_file = model_name + "/" + param["model"] + "_DR_6dir"
    model = train(model, env, task, options, training_params, model_file, device)
    th.save(model.state_dict(), model_file + ".pth")
