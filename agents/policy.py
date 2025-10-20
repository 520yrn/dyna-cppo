# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:53:53 2025

@author: Ruining
"""

import torch
import torch.nn as nn
from typing import Tuple


class PolicyNet(nn.Module):
    """CPPO-style Gaussian policy without explicit mu_head name."""

    def __init__(self, obs_dim: int, act_dim: int, log_std_init: float = -0.5):
        super().__init__()

        # Use a single Sequential MLP to compute μ(s)
        self.mu_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )

        # State-independent log std
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        # PPO-style initialization
        for layer in self.mu_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.constant_(layer.bias, 0.0)

        # Small gain for final output layer
        nn.init.orthogonal_(self.mu_net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean μ and std σ."""
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def sample_action(self, obs: torch.Tensor):
        """Sample an action using the reparameterization trick."""
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Compute log πθ(a|s) and entropy for E-step / M-step."""
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, mu

