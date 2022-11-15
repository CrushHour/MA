import torch
import math
from torch import nn
import numpy as np
from model import PyTorchFingerModel
from rich.progress import track
from itertools import chain


class ForwardKinematics(torch.nn.Module):
    def __init__(self, template, parameter):
        super().__init__()
        self.model = PyTorchFingerModel(template, parameter)
        self.kin_parameters = []
        for k in self.model.parameters:
            p, q = self.model.parameters[k]
            self.kin_parameters.append(p)
            self.kin_parameters.append(q)
        self.kin_parameters = torch.nn.ParameterList(self.kin_parameters)

        init = torch.ones((32,)) * math.log(5e-2)
        self.log_stddev = nn.Parameter(init, requires_grad=True)

    def forward(self, angles):
        return self.model.forward_kinematic(angles)


class InverseKinematics(torch.nn.Module):
    def __init__(self, n_obs, n_joints):
        super().__init__()
        self.n_joints = n_joints
        self.n_obs = n_obs
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_obs, 512),
            nn.Softplus(),
            nn.Linear(512, 512),
            nn.Softplus(),
            nn.Linear(512, self.n_joints * 2),
        )

    def forward(self, x):
        stt = self.linear_relu_stack(x)
        mu = stt[:, : self.n_joints]
        std = torch.exp(stt[:, self.n_joints :] / 2) + 1e-5
        return self.sample(mu, std)

    def sample(self, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z


model = ForwardKinematics("tendon_finger.template.xml", "generated_parameters.yaml")
train_model = ForwardKinematics("tendon_finger.template.xml", "generated_parameters_disturbed.yaml")


# N = 500
# angles = torch.Tensor(np.random.uniform(0.0, np.pi / 4.0, size=(N, 3)))
# obs = model(angles)
# obs = torch.hstack(obs).detach()
# print("original shape:", obs.shape)

obs = np.load("rollout.npy").astype("float32")
angles = np.load("rollout_angles.npy").astype("float32")
obs = torch.tensor(obs.reshape((-1, 32))).detach()
angles = torch.tensor(angles.reshape((-1, 3))).detach()
print("new shape:", obs.shape)

encoder = InverseKinematics(obs.shape[1], 3)

optimizer = torch.optim.Adam(
    chain(
        encoder.parameters(),
        train_model.parameters(),
    ),
    lr=0.0005,
)


for n in track(range(1, 2000), description="Training..."):
    # zero the parameter gradients
    optimizer.zero_grad()
    train_model.zero_grad()
    encoder.zero_grad()

    # forward + backward + optimize
    p, q, pred_angles = encoder(obs)
    pred_angles = torch.tanh(pred_angles) * np.pi / 4.0
    pred_obs = train_model(pred_angles)
    pred_obs = torch.hstack(pred_obs)

    kl = torch.distributions.kl_divergence(q, p)
    kl = kl.mean()

    dec = torch.distributions.Normal(pred_obs, torch.exp(train_model.log_stddev) + 1e-5)
    # loss = kl + torch.mean((pred_obs - obs) ** 2 / torch.exp(train_model.log_stddev))
    loss = kl - torch.mean(dec.log_prob(obs))
    loss.backward()

    print(torch.mean((pred_angles - angles) ** 2))

    # for param in train_model.model.parameters:
    #    print(param, param[0].grad)

    # for param in train_model.parameters():
    #    print(param.grad)

    optimizer.step()

with np.printoptions(precision=2, suppress=True):
    print("Final parameters:")
    for k in train_model.model.parameters:
        print(k, ": ", train_model.model.parameters[k][0].detach().numpy())
        print(k, ": ", train_model.model.parameters[k][1].detach().numpy())

    print("Original parameters:")
    for k in train_model.model.parameters:
        print(k, ": ", model.model.parameters[k][0].detach().numpy())
        print(k, ": ", model.model.parameters[k][1].detach().numpy())
