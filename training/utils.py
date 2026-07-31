import torch as th
import numpy as np
import json
import warnings
from sklearn.metrics import r2_score, mean_squared_error


def run_episode(env, rnn, options, device=th.device("cuda"), detach=False):
    batch_size = options["batch_size"]
    move_ntps = options["movement_timepoints"]
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
        running_trials = move_ntps > tps
        action, x = rnn(x, obs[:, -4:], obs[:, :-4])
        obs, _, terminated, truncated, info = env.step(action=action, options=options)
        x = x[truncated == False]

        data["all_hidden"][running_trials == True, tps, :] = x
        data["all_muscle"][running_trials == True, tps, :] = info["states"]["muscle"][:, 0, :]
        data["all_joint"][running_trials == True, tps, :] = info["states"]["joint"]
        data["xy"][running_trials == True, tps, :] = info["states"]["fingertip"]
        data["vel"][running_trials == True, tps, :] = info["states"]["cartesian"][:, 2:]
        data["all_actions"][running_trials == True, tps, :] = action[truncated == False, :]
        data["all_force"][running_trials == True, tps, :] = info["states"]["muscle"][:, 6, :]
        data["muscle_length"][running_trials == True, tps, :] = info["states"]["muscle"][:, 1, :]

    if detach:
        for key in data:
            data[key] = th.detach(data[key])

    W = rnn.getWeight(detach=detach)
    return data, W


def cal_loss(data, goal, trajactory, velocity, bhv_marker, training_params, device=th.device("cuda")):
    loss_weights = np.array(training_params["loss_weights"])
    loss_to_compute = training_params["loss_to_compute"]
    loss = {}
    losses_weighted = {}
    trial_num = goal.shape[0]

    e_pos = th.zeros(trial_num).to(device)
    e_hold = th.zeros(trial_num).to(device)
    e_traj = th.zeros(trial_num).to(device)
    e_vel = th.zeros(trial_num).to(device)
    e_jerk = th.zeros(trial_num).to(device)
    e_muscle = th.zeros(trial_num).to(device)

    for trial in range(trial_num):
        go = bhv_marker[trial, 1]
        tt = bhv_marker[trial, 2]
        end = bhv_marker[trial, 3]
        e_pos[trial] = th.mean(
            th.abs(data["xy"][trial, tt:end, :] - goal[trial, end - 1 : end, :])
        )
        e_hold[trial] = th.mean(
            th.abs(data["xy"][trial, :go, :] - trajactory[trial, :go, :])
        )
        e_traj[trial] = th.mean(
            th.square(data["xy"][trial, go:tt, :] - trajactory[trial, go:tt, :])
        )
        e_vel[trial] = th.mean(
            th.square(data["vel"][trial, go:tt, :] - velocity[trial, go:tt, :])
        )
        e_jerk[trial] = th.mean(th.square(th.diff(data["vel"][trial, go:tt, :], n=2, dim=0)))
        e_muscle[trial] = th.mean(data["all_force"][trial, :end, :])

    loss["position"] = th.mean(e_pos)
    loss["hold"] = th.mean(e_hold)
    loss["trajactory"] = th.mean(e_traj)
    loss["velocity"] = th.mean(e_vel)
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


def print_losses(overall_loss, losses_weighted, log_file, batch):
    fstring = f"batch: {batch:5d}, over_loss: {overall_loss:9.5f}, "
    for l in losses_weighted.keys():
        fstring = fstring + f"{l}: {losses_weighted[l]:9.5f}, "
    with open(log_file, "a") as f:
        print(fstring[:-2], file=f)


def save_model(env, rnn, losses, model_name, quiet=False):
    model_file = model_name + "/" + model_name + "_rnn"
    log_file = model_name + "/" + model_name + "_log.json"
    cfg_file = model_name + "/" + model_name + "_cfg.json"

    th.save(rnn.state_dict(), model_file)
    with open(log_file, "w") as file:
        json.dump({"losses": losses}, file)
    cfg = env.get_save_config()
    with open(cfg_file, "w") as file:
        json.dump(cfg, file)
    if not quiet:
        print(f"saved {model_file}")
        print(f"saved {log_file}")
        print(f"saved {cfg_file}")


def cal_vel(vel, ntps):
    trial_num, trial_length = vel.shape[0], vel.shape[1]
    vel_proj = np.zeros((trial_num, trial_length))
    for i in range(trial_num):
        vel_proj[i, : ntps[i]] = np.sqrt(
            vel[i, : ntps[i], 0] ** 2 + vel[i, : ntps[i], 1] ** 2
        )
    return vel_proj


def evalPrediction(trueValue, prediction, measure):
    if prediction.shape[0] == 0:
        perf = np.empty(trueValue.shape[1])
        perf[:] = np.nan
        return perf

    if measure == "CC":
        n = trueValue.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = np.corrcoef(trueValue, prediction, rowvar=False)
        perf = np.diag(R[n:, :n])
    elif measure == "R2":
        perf = r2_score(trueValue, prediction, multioutput="raw_values")
    elif measure == "MSE":
        perf = mean_squared_error(trueValue, prediction, multioutput="raw_values")
    elif measure == "RMSE":
        MSE = evalPrediction(trueValue, prediction, "MSE")
        perf = np.sqrt(MSE)
    else:
        raise (Exception(f'Performance measure "{measure}" is not supported.'))
    return perf


def compute_epe(goal, traj_pred, marker):
    batch = goal.shape[0]
    epe = np.zeros(batch)
    for i in range(batch):
        tt = marker[i, 2]
        end = marker[i, 3]
        dx = goal[i, 0] - traj_pred[i, tt:end, 0]
        dy = goal[i, 1] - traj_pred[i, tt:end, 1]
        epe[i] = np.mean(np.sqrt(dx**2 + dy**2))
    return epe
