import torch as th
from ..tasks.grid_reach import BasicTask
from ..model.rnn import RNNCell
from ..training.params import default_params


def build(param):
    task_list = []
    option_list = []

    model_params, task_params, training_params = default_params()
    model_params["nonlinearity"] = param["nonlinearity"]
    model_params["n_rec"] = param["n_rec"]
    model_params["n_out_task"] = param["n_out_task"]
    model_params["noise_std"] = param["noise_std"]

    if param["obs_class"] == "prop":
        model_params["n_fb"] = 12
    elif param["obs_class"] == "vis":
        model_params["n_fb"] = 2
    else:
        model_params["n_fb"] = 14

    if param["model"] == "rnn":
        model = RNNCell(model_params)
    else:
        raise ValueError("RNN type error")

    for n in param["joint_interval"]:
        task = BasicTask()
        task.genJointset(joint_interval=n)
        task_list.append(task)
        traj, vel, marker, gocue, joint_state, goal = task.genReachTask(
            task="grid",
            trial_num=1,
            dis_range=task_params["dis_range"],
            vel_range=task_params["vel_range"],
            delay_range=task_params["delay_range"],
            hold_dur=task_params["hold_durtion"],
            catch_trial_proportion=task_params["catch_trial_proportion"],
        )
        options = {
            "obs_class": param["obs_class"],
            "ini_std": param["ini_std"],
            "batch_size": goal.shape[0],
            "trajactory": traj,
            "velocity": vel,
            "joint_state": joint_state,
            "marker": marker,
            "movement_timepoints": marker[:, -1],
            "goal": goal,
            "gocue": gocue,
        }
        option_list.append(options)

    task_params["joint_interval"] = param["joint_interval"]
    task_params["grid_interval"] = param["grid_interval"]
    task_params["test_batch_size"] = param["test_batch_size"]
    task_params["center_joint"] = param["center_joint"]
    task_params["reach_distance"] = param["reach_distance"]

    return model, task_list, option_list, task_params, training_params


def build_model(param, tasknet=None):
    model_params, _, _ = default_params()
    model_params["nonlinearity"] = param["nonlinearity"]
    model_params["n_rec"] = param["n_rec"]
    model_params["n_out_task"] = param["n_out_task"]
    model_params["noise_std"] = param["noise_std"]

    if param["obs_class"] == "prop":
        model_params["n_fb"] = 12
    elif param["obs_class"] == "vis":
        model_params["n_fb"] = 2
    else:
        model_params["n_fb"] = 14

    model = RNNCell(model_params, tasknet=tasknet)
    return model


def build_task(param, task_info):
    task = BasicTask()
    options = {"obs_class": param["obs_class"], "ini_std": param["ini_std"]}

    if task_info["workspace"] == "joint_space":
        task.genJointset(joint_interval=task_info["joint_interval"])
    elif task_info["workspace"] == "cartesian_space":
        task.genCartesianGridSet(grid_interval=task_info["grid_interval"])
    else:
        raise ValueError("Workspace type error")

    if task_info["task"] == "grid" or task_info["task"] == "random":
        trajactory, velocity, marker, gocue, joint_state, goal, goal_for_train = (
            task.genReachTask(
                task=task_info["task"],
                trial_num=task_info["trial_num"],
                dis_range=task_info["dis_range"],
                vel_range=task_info["vel_range"],
                delay_range=task_info["delay_range"],
                to_range=task_info["to_range"],
                hold_dur=task_info["hold_durtion"],
                catch_trial_proportion=task_info["catch_trial_proportion"],
            )
        )
        traj = th.tensor(trajactory)
        vel = th.tensor(velocity)
    elif task_info["task"] == "centerOut":
        co_options = {
            "center_joint": task_info["center_joint"],
            "angle_interval": task_info["angle_interval"],
            "reach_distance": task_info["reach_distance"],
            "average_velocity": task_info["vel_range"][-1],
            "delay_range": task_info["delay_range"],
            "hold_durtion": task_info["hold_durtion"],
        }
        traj, vel, marker, gocue, joint_state, goal = task.centerOut(co_options)
        traj = th.tensor(traj)
        vel = th.tensor(vel)
    else:
        raise ValueError("Task type error")

    options["batch_size"] = goal.shape[0]
    options["marker"] = marker
    options["movement_timepoints"] = marker[:, -1]
    options["joint_state"] = joint_state
    options["g"] = goal
    options["goal"] = goal_for_train
    options["gocue"] = gocue

    return task, options, traj, vel
