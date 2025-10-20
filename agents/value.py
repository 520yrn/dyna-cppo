# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:54:03 2025

@author: Ruining
"""

from typing import Tuple
import torch
import torch.nn as nn


def _orth_init_sequential(sequential: nn.Sequential, last_gain: float = 1.0) -> None:
    """Apply orthogonal init to all Linear layers in a Sequential block.
    Args:
        sequential: the nn.Sequential module to initialize
        last_gain:  gain for the final Linear (default 1.0; value head commonly uses 1.0)
    """
    # Iterate all submodules
    for i, layer in enumerate(sequential):
        # Only initialize Linear layers
        if isinstance(layer, nn.Linear):
            # For hidden layers use tanh gain; for last layer use last_gain
            gain = nn.init.calculate_gain("tanh") if i < len(sequential) - 1 else last_gain
            nn.init.orthogonal_(layer.weight, gain=gain)  # orthogonal weight init
            nn.init.constant_(layer.bias, 0.0)            # zero bias is common & stable


class ValueNet(nn.Module):
    """Reward value function V_r(s)."""

    def __init__(self, obs_dim: int, hidden_dims: Tuple[int, int] = (64, 64)):
        super().__init__()
        # Define a 2-layer MLP with tanh activations and a scalar output head
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.Tanh(), 
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(), 
            nn.Linear(hidden_dims[1], 1),
            )
        # PPO-style initialization
        _orth_init_sequential(self.net, last_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute V_r(s) for a batch of observations. Shape: [B, 1]."""
        return self.net(obs)


class CostNet(nn.Module):
    """Cost value function V_c(s)."""

    def __init__(self, obs_dim: int, hidden_dims: Tuple[int, int] = (64, 64)):
        super().__init__()
        # Same architecture as ValueNet to keep symmetry between critics
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1), 
            )
        _orth_init_sequential(self.net, last_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute V_c(s) for a batch of observations. Shape: [B, 1]."""
        return self.net(obs)
