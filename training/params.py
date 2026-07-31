import numpy as np


def default_params():
    density = {"rec": 0.8, "in": 0.5, "out": 0.5}

    model_params = {
        "device": "cuda",
        "nonlinearity": "tanh",
        "n_fb": 14,
        "n_rec": 200,
        "n_out": 6,
        "n_out_task": 12,
        "density": density,
        "radius": 1.5,
        "ini_std": 0.10,
        "dt": 10,
        "tau": 50,
        "noise_std": 0.15,
        "dale's law": True,
    }

    task_params = {
        "task": "grid",
        "batch_size": 1,
        "joint_interval": 15,
        "dis_range": [0.05, 0.35],
        "vel_range": [0.25, 0.4, 0.5],
        "delay_range": [0.3, 0.6],
        "to_range": [0.2, 0.3],
        "hold_durtion": 0.10,
        "catch_trial_proportion": 0.1,
        "threshold": 0.02,
        "action_noise": 0.01,
        "proprioception_noise": 0.01,
        "vision_noise": 0.01,
        "proprioception_delay": 0.0,
        "vision_delay": 0.0,
    }

    loss_weights = [1e+2, 5e+2, 1e+3, 5e+2, 1e+3, 1e-2]
    loss_to_compute = ["position", "hold", "trajactory", "velocity", "jerk", "muscle"]
    weight_to_opt = ["", ""]

    training_params = {
        "n_batch": 20001,
        "interval": 1000,
        "lr": 1e-4,
        "loss_weights": loss_weights,
        "loss_to_compute": loss_to_compute,
        "weight_reg": "l1",
        "weight_to_opt": weight_to_opt,
        "cuda": True,
        "loss_fn": "mse",
        "optimizer": "adam",
    }

    return model_params, task_params, training_params
