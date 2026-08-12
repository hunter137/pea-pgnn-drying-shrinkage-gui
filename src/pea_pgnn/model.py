"""Frozen 104,200-parameter PEA-PGNN architecture."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.03):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")

    def forward(self, inputs):
        return self.dropout(self.activation(self.norm(self.linear(inputs)))) + self.residual(inputs)


class PEAPGNN(nn.Module):
    def __init__(
        self,
        n_features=39,
        eps_anchor_index=34,
        tau_anchor_index=33,
        hidden=(256, 128, 64),
        dropout=0.03,
        delta_eps_range=(-0.5, 1.2),
        delta_tau_range=(-0.8, 2.0),
        additive_scale=200.0,
        eps_min=100.0,
        eps_max=1500.0,
        tau_min=5.0,
        tau_max=1000.0,
    ):
        super().__init__()
        self.eps_anchor_index = int(eps_anchor_index)
        self.tau_anchor_index = int(tau_anchor_index)
        self.delta_eps_range = tuple(float(v) for v in delta_eps_range)
        self.delta_tau_range = tuple(float(v) for v in delta_tau_range)
        self.additive_scale = float(additive_scale)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        layers = []
        width = int(n_features)
        for output_width in hidden:
            layers.append(ResidualBlock(width, int(output_width), float(dropout)))
            width = int(output_width)
        self.backbone = nn.Sequential(*layers)
        self.eps_multiplicative_head = nn.Linear(width, 1)
        self.eps_additive_head = nn.Linear(width, 1)
        self.tau_head = nn.Linear(width, 1)
        self.alpha_head = nn.Linear(width, 1)
        self.weight_head = nn.Linear(width, 4)
        for head in (self.eps_multiplicative_head, self.eps_additive_head, self.tau_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def parameter_state(self, raw_features, scaled_features):
        hidden = self.backbone(scaled_features)
        eps_anchor = torch.clamp(
            raw_features[:, self.eps_anchor_index], 50.0, self.eps_max
        )
        tau_anchor = torch.clamp(
            raw_features[:, self.tau_anchor_index], self.tau_min, self.tau_max
        )
        eps_low, eps_high = self.delta_eps_range
        tau_low, tau_high = self.delta_tau_range
        delta_eps = eps_low + (eps_high - eps_low) * torch.sigmoid(
            self.eps_multiplicative_head(hidden).squeeze(-1)
        )
        delta_add = torch.tanh(self.eps_additive_head(hidden).squeeze(-1)) * self.additive_scale
        eps_inf = torch.clamp(
            eps_anchor * (1.0 + delta_eps) + delta_add, self.eps_min, self.eps_max
        )
        delta_tau = tau_low + (tau_high - tau_low) * torch.sigmoid(
            self.tau_head(hidden).squeeze(-1)
        )
        tau = torch.clamp(tau_anchor * (1.0 + delta_tau), self.tau_min, self.tau_max)
        alpha = torch.sigmoid(self.alpha_head(hidden).squeeze(-1)) * 0.8 + 0.1
        weights = torch.softmax(self.weight_head(hidden), dim=-1)
        return {
            "eps_anchor": eps_anchor,
            "tau_anchor": tau_anchor,
            "eps_inf": eps_inf,
            "tau": tau,
            "alpha": alpha,
            "weights": weights,
            "delta_eps": delta_eps,
            "delta_tau": delta_tau,
            "delta_add": delta_add,
        }

    @staticmethod
    def evolution(age, tau, alpha, weights):
        age = torch.clamp(age, min=0.01)
        tau = torch.clamp(tau, min=0.1)
        law_1 = torch.tanh(torch.sqrt(age / tau))
        law_2 = (age / (age + tau)) ** alpha
        law_3 = torch.sqrt(age / (age + tau**2 / 100.0))
        denominator = torch.log1p(torch.tensor(1e4, dtype=age.dtype, device=age.device))
        law_4 = torch.clamp(torch.log1p(age / tau) / denominator, 0.0, 1.0)
        return (
            weights[:, 0] * law_1
            + weights[:, 1] * law_2
            + weights[:, 2] * law_3
            + weights[:, 3] * law_4
        )

    def forward(self, raw_features, scaled_features, age, return_details=False):
        state = self.parameter_state(raw_features, scaled_features)
        fraction = self.evolution(age, state["tau"], state["alpha"], state["weights"])
        prediction = torch.clamp(state["eps_inf"] * fraction, min=0.0)
        if return_details:
            details = dict(state)
            details["evolution_fraction"] = fraction
            return prediction, details
        return prediction

    def n_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def model_from_config(config):
    return PEAPGNN(
        n_features=39,
        eps_anchor_index=34,
        tau_anchor_index=33,
        hidden=tuple(config["hidden"]),
        dropout=config["dropout"],
        delta_eps_range=tuple(config["delta_eps_range"]),
        delta_tau_range=tuple(config["delta_tau_range"]),
        additive_scale=config["additive_scale"],
        eps_min=config["eps_min"],
        eps_max=config["eps_max"],
        tau_min=config["tau_min"],
        tau_max=config["tau_max"],
    )

