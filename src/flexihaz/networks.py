from __future__ import annotations

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers=(64, 64), activation="relu", dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for width in hidden_layers:
            layers.append(nn.Linear(in_dim, int(width)))
            layers.append(make_activation(activation))
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlexiHazardNet(nn.Module):
    def __init__(self, x_dim: int, z_dim: int | None = None, hidden_layers=(64, 64), activation="relu", dropout: float = 0.0):
        super().__init__()
        self.x_dim = int(x_dim)
        self.z_dim = None if z_dim is None else int(z_dim)
        self.g_net = MLP(input_dim=1 + self.x_dim, hidden_layers=hidden_layers, activation=activation, dropout=dropout)
        self.linear = nn.Linear(self.z_dim, 1, bias=False) if self.z_dim is not None else None

    def forward(self, t, X, Z=None):
        inp = torch.cat([t, X], dim=1)
        g = self.g_net(inp)
        if self.linear is None:
            eta = g
        else:
            if Z is None:
                raise ValueError("Z is required for a model with a linear branch.")
            eta = g + self.linear(Z)
        return eta, g

    def theta(self):
        if self.linear is None:
            return None
        return self.linear.weight.detach().cpu().numpy().reshape(-1)


class ProjectionNet(nn.Module):
    """Network used to estimate one coordinate of g*(T, X) for theta covariance."""

    def __init__(self, x_dim: int, hidden_layers=(64, 64), activation="relu", dropout: float = 0.0):
        super().__init__()
        self.net = MLP(input_dim=1 + int(x_dim), hidden_layers=hidden_layers, activation=activation, dropout=dropout)

    def forward(self, t, X):
        return self.net(torch.cat([t, X], dim=1))
