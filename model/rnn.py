import torch as th
import torch.nn as nn
import numpy as np
from scipy.sparse import random
from scipy import stats
from numpy import linalg


class RNNCell(nn.Module):
    def __init__(self, params, tasknet=None):
        super(RNNCell, self).__init__()
        self.device = th.device(params["device"])
        self.task_size = params["n_out_task"]
        self.feedback_size = params["n_fb"]
        self.input_size = self.feedback_size + self.task_size
        self.hidden_size = params["n_rec"]
        self.output_size = params["n_out"]
        self.dale = params["dale's law"]
        self.sigma = params["noise_std"]
        self.dt = params["dt"]
        self.tau = params["tau"]
        self.alpha = th.tensor(self.dt / self.tau).to(self.device)
        self.nonlinearity = set_nonlinearity(params)

        if tasknet is None:
            self.tasknet = nn.Sequential(
                nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, self.task_size)
            )
        else:
            self.tasknet = tasknet

        self.input_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.recurrent_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size, bias=True)
        nn.init.constant_(self.output_layer.bias, -0.6)

        if self.dale:
            (self.recurrent_layer.weight.data,
             self.input_layer.weight.data,
             self.output_layer.weight.data,
             self.dale_mask,
             self.output_mask,
             self.input_mask) = init_connectivity(
                self.hidden_size, self.input_size, self.output_size,
                params["density"], radius=params["radius"]
            )

        self.connectivity_constraints()
        self.to(self.device)
        self.input_mask = self.input_mask.to(self.device)
        self.dale_mask = self.dale_mask.to(self.device)
        self.output_mask = self.output_mask.to(self.device)

    def forward(self, x, task_input, fb):
        task_output = self.tasknet(task_input)
        inputs = self.input_layer(th.cat([fb, task_output], dim=1))
        noise_t = th.sqrt(2 * self.alpha) * self.sigma * th.randn_like(x)
        x = (1 - self.alpha) * x + self.alpha * self.nonlinearity(
            self.recurrent_layer(x) + inputs + noise_t
        )
        outputs = th.sigmoid(self.output_layer(x))
        return outputs, x

    def init_hidden(self, batch_size, std=0.1):
        return th.normal(
            th.zeros((batch_size, self.hidden_size)), std=std
        ).to(self.device)

    def connectivity_constraints(self):
        self.input_layer.weight.data = th.relu(self.input_layer.weight.data)
        self.output_layer.weight.data = th.relu(self.output_layer.weight.data)
        if self.dale:
            self.input_layer.weight.data = self.input_mask * th.relu(
                self.input_layer.weight.data
            )
            self.output_layer.weight.data = self.output_mask * th.relu(
                self.output_layer.weight.data
            )
            self.recurrent_layer.weight.data = th.relu(
                self.recurrent_layer.weight.data * self.dale_mask
            ) * self.dale_mask

    def getWeight(self, detach=False):
        w_in = self.input_layer.weight.data
        w_out = self.output_layer.weight.data
        w_rec = self.recurrent_layer.weight.data
        if detach:
            W = {
                "w_in": w_in.detach().clone().to("cpu"),
                "w_rec": w_rec.detach().clone().to("cpu"),
                "w_out": w_out.detach().clone().to("cpu"),
            }
        else:
            W = {"w_in": w_in.clone(), "w_rec": w_rec.clone(), "w_out": w_out.clone()}
        return W


def init_connectivity(N, input_size, output_size, density, radius=1.5):
    Ne = int(N * 0.8)
    Ni = int(N * 0.2)
    W_rec = th.empty([0, N])
    mu_E = 1 / np.sqrt(N)
    mu_I = 4 / np.sqrt(N)
    var = 1 / N
    rowE = th.empty([Ne, 0])
    rowI = th.empty([Ni, 0])
    rowE = th.cat(
        (
            rowE,
            th.tensor(
                random(
                    Ne, Ne, density=density["rec"],
                    data_rvs=stats.norm(scale=var, loc=mu_E).rvs
                ).toarray()
            ).float(),
        ),
        1,
    )
    rowE = th.cat(
        (
            rowE,
            -th.tensor(
                random(
                    Ne, Ni, density=density["rec"],
                    data_rvs=stats.norm(scale=var, loc=mu_I).rvs
                ).toarray()
            ).float(),
        ),
        1,
    )
    rowI = th.cat(
        (
            rowI,
            th.tensor(
                random(
                    Ni, Ne, density=density["rec"],
                    data_rvs=stats.norm(scale=var, loc=mu_E).rvs
                ).toarray()
            ).float(),
        ),
        1,
    )
    rowI = th.cat(
        (
            rowI,
            -th.tensor(
                random(
                    Ni, Ni, density=density["rec"],
                    data_rvs=stats.norm(scale=var, loc=mu_I).rvs
                ).toarray()
            ).float(),
        ),
        1,
    )
    W_rec = th.cat((W_rec, rowE), 0)
    W_rec = th.cat((W_rec, rowI), 0)
    W_rec = W_rec - th.diag(th.diag(W_rec))
    w, v = linalg.eig(W_rec)
    spec_radius = np.max(np.absolute(w))
    W_rec = radius * W_rec / spec_radius

    W_in = th.zeros([N, input_size]).float()
    W_in[:, :] = radius * th.tensor(
        random(
            N, input_size, density=density["in"],
            data_rvs=stats.norm(scale=var, loc=mu_E).rvs
        ).toarray()
    ).float()

    W_out = th.zeros([output_size, N])
    W_out[:, :Ne] = th.tensor(
        random(
            output_size, Ne, density=density["out"],
            data_rvs=stats.norm(scale=var, loc=mu_E).rvs
        ).toarray()
    ).float()

    dale_mask = th.sign(W_rec).float()
    output_mask = (W_out != 0).float()
    input_mask = (W_in != 0).float()
    return W_rec.float(), W_in.float(), W_out.float(), dale_mask, output_mask, input_mask


def set_nonlinearity(params):
    if params["nonlinearity"] == "tanh":
        return th.tanh
    elif params["nonlinearity"] == "identity":
        return lambda x: x
    elif params["nonlinearity"] == "relu":
        return nn.ReLU()
    elif params["nonlinearity"] == "softplus":
        softplus_scale = 1
        nonlinearity = (
            lambda x: th.log(1.0 + th.exp(softplus_scale * x)) / softplus_scale
        )
        return nonlinearity
    elif type(params["nonlinearity"]) == str:
        print("Nonlinearity not yet implemented. Continuing with identity")
        return lambda x: x
    else:
        return params["nonlinearity"]
