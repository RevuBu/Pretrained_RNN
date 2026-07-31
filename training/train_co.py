import os
import json
import torch as th
from tqdm import tqdm
import motornet as mn
from ..model.rnn import RNNCell
from ..tasks.grid_reach import CenterOutTask, BasicTaskEnv
from .utils import run_episode, cal_loss, print_losses
from .params import default_params


def train(model, env, task, options, task_params, training_params, model_file, device):
    optimizer = th.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=training_params["lr"]
    )

    losses = {
        "overall": [],
        "position": [],
        "hold": [],
        "trajactory": [],
        "velocity": [],
        "jerk": [],
        "muscle": [],
    }
    original_loss = {
        "position": [],
        "hold": [],
        "trajactory": [],
        "velocity": [],
        "jerk": [],
        "muscle": [],
    }
    n_batch = training_params["n_batch"]
    batch_size = options["batch_size"]

    for batch in tqdm(range(n_batch), desc=f"Training {n_batch} batches of {batch_size}", unit="batch"):
        if (batch % training_params["interval"]) == 0:
            trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train = (
                task.genReach(options)
            )
            options["trajactory"] = trajactory
            options["velocity"] = velocity
            options["marker"] = marker
            options["movement_timepoints"] = marker[:, -1]
            options["gocue"] = gocue
            options["goal"] = goal_for_train
            options["joint_state"] = joint_state

            gft = th.tensor(goal_for_train).to(device)
            traj = th.tensor(trajactory).to(device)
            vel = th.tensor(velocity).to(device)

        data, _ = run_episode(env, model, options, device=device)
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
        losses["trajactory"].append(losses_weighted["trajactory"].item())
        losses["velocity"].append(losses_weighted["velocity"].item())
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
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = th.device("cuda")
    model_name = "well_trained_model/260329_CO_Model_fine_tuned_15"
    if not os.path.exists(model_name):
        os.mkdir(model_name)

    model_params, task_params, training_params = default_params()
    param = {
        "model_params": model_params,
        "task_params": task_params,
        "training_params": training_params,
    }
    with open(model_name + "/param.json", "w") as file:
        json.dump(param, file)

    options = {
        "batch_size": 80,
        "center_joint": [[38.0, 113.3]],
        "angle_interval": [45],
        "target_radius": [0.12],
        "target_on": 0.2,
        "delay_durtion": 0.6,
        "reach_durtion": 0.4,
        "hold_durtion": 0.1,
        "catch_trial_proportion": 0.1,
    }
    options["obs_class"] = "prop+vis"
    options["ini_std"] = model_params["ini_std"]
    task = CenterOutTask()

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    max_ep_duration = round(
        options["target_on"] + options["delay_durtion"] + options["reach_durtion"] + options["hold_durtion"], 3
    )
    env = BasicTaskEnv(
        effector=effector,
        max_ep_duration=max_ep_duration,
        action_noise=task_params["action_noise"],
        proprioception_noise=task_params["proprioception_noise"],
        vision_noise=task_params["vision_noise"],
        proprioception_delay=task_params["proprioception_delay"],
        vision_delay=task_params["vision_delay"],
    ).to(device)

    # Load pretrained model and fine-tune
    w = th.load("model/250916_basicModel/rnn__joint_interval_15.pth")

    model_file_1 = model_name + "/rnn_co_1"
    model_1 = RNNCell(model_params)
    model_1.load_state_dict(w)
    for name, model_param in model_1.named_parameters():
        if "tasknet" not in name:
            model_param.requires_grad = False
    model_1.to(device)
    model_1 = train(model_1, env, task, options, task_params, training_params, model_file_1, device)
    th.save(model_1.state_dict(), model_file_1 + ".pth")

    # Train multi-condition model
    options["batch_size"] = 960
    options["center_joint"] = [[56.6, 107.35], [47.3, 111.0], [38, 113.3], [30.0, 114.0]]
    options["target_radius"] = [0.08, 0.1, 0.12]
    model_file_2 = model_name + "/rnn_co_12"
    model_2 = RNNCell(model_params)
    model_2.load_state_dict(w)
    for name, model_param in model_2.named_parameters():
        if "tasknet" not in name:
            model_param.requires_grad = False
    model_2.to(device)
    model_2 = train(model_2, env, task, options, task_params, training_params, model_file_2, device)
    th.save(model_2.state_dict(), model_file_2 + ".pth")
