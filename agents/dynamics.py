# agents/dynamics.py
# -*- coding: utf-8 -*-
"""
dynamics.py

Lightweight dynamics model and a simple model-based rollout wrapper
for Dyna-style CPPO.

Main components:
- DynamicsModel:       learns s_{t+1} ≈ f_φ(s_t, a_t), optionally reward and cost
- ModelEnv:            a minimal "environment" that steps using DynamicsModel

Author: Ruining
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper functions for numerical stability / rescue
# ---------------------------------------------------------------------------

def _safe_numpy_obs(obs: np.ndarray) -> np.ndarray:
    """
    Ensure a finite numpy observation (replace NaN/Inf with 0).
    This is a light-weight rescue, so the rollout does not crash even if
    the model starts to diverge.
    """
    if not np.all(np.isfinite(obs)):
        # Replace NaN, +Inf, -Inf with zeros
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs.astype(np.float32, copy=True)


def _safe_tensor(x: torch.Tensor, name: str = "") -> torch.Tensor:
    """
    Ensure a finite tensor (replace NaN/Inf with 0).

    You can uncomment the print statements below if you want to log
    when things go bad, but by default we keep them silent to avoid
    spamming the terminal.
    """
    if not torch.isfinite(x).all():
        # Debug logging (optional):
        # print(f"[DynamicsModel] Non-finite tensor detected in {name}:",
        #       x.detach().cpu().numpy())
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# ---------------------------------------------------------------------------
# Dynamics model
# ---------------------------------------------------------------------------

class DynamicsModel(nn.Module):
    """
    Simple feedforward dynamics model:

        s_{t+1}, r_t, c_t ≈ f_φ(s_t, a_t)

    By default only predicts the next state s_{t+1}. You can enable reward
    and/or cost prediction by setting predict_reward / predict_cost to True.

    Shapes:
        obs:       [N, obs_dim]
        act:       [N, act_dim]
        next_obs:  [N, obs_dim]
        reward:    [N, 1]
        cost:      [N, 1]
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        predict_reward: bool = False,
        predict_cost: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.predict_reward = predict_reward
        self.predict_cost = predict_cost

        in_dim = obs_dim + act_dim
        # Output: next_state (+ optional reward/cost scalars)
        out_dim = obs_dim
        if predict_reward:
            out_dim += 1
        if predict_cost:
            out_dim += 1

        layers = []
        last_dim = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Use a small, stable initialization to reduce early explosions
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward dynamics prediction.

        Args:
            obs: [N, obs_dim] current states
            act: [N, act_dim] actions

        Returns:
            A dict with keys:
                - "next_obs": predicted next states [N, obs_dim]
                - "reward":   (optional) predicted reward [N, 1]
                - "cost":     (optional) predicted cost [N, 1]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if act.dim() == 1:
            act = act.unsqueeze(0)

        # Ensure finite inputs
        obs = _safe_tensor(obs, name="obs")
        act = _safe_tensor(act, name="act")

        x = torch.cat([obs, act], dim=-1)
        out = self.net(x)

        # Ensure finite outputs
        out = _safe_tensor(out, name="dyn_out")

        # Split outputs
        idx = 0
        next_obs = out[:, idx : idx + self.obs_dim]
        idx += self.obs_dim

        result: Dict[str, torch.Tensor] = {"next_obs": next_obs}

        if self.predict_reward:
            reward = out[:, idx : idx + 1]
            idx += 1
            result["reward"] = reward

        if self.predict_cost:
            cost = out[:, idx : idx + 1]
            result["cost"] = cost

        return result

    def one_step_loss(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        next_obs_target: torch.Tensor,
        reward_target: Optional[torch.Tensor] = None,
        cost_target: Optional[torch.Tensor] = None,
        obs_weight: float = 1.0,
        reward_weight: float = 1.0,
        cost_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        One-step supervised loss:

            L(φ) = w_s * ||s_{t+1} - ŝ_{t+1}||^2
                 + w_r * ||r_t - r̂_t||^2
                 + w_c * ||c_t - ĉ_t||^2

        This matches the form in your report but only for k=1.
        For multi-step prediction, call this repeatedly on rolled-out
        model predictions.

        Args:
            obs, act:           [N, ...] current states and actions
            next_obs_target:    [N, obs_dim] true next states
            reward_target:      [N, 1] (optional) true rewards
            cost_target:        [N, 1] (optional) true costs
            obs_weight:         weight for state prediction loss
            reward_weight:      weight for reward prediction loss
            cost_weight:        weight for cost prediction loss

        Returns:
            Scalar loss tensor.
        """
        pred = self.forward(obs, act)
        loss = torch.tensor(0.0, device=obs.device)

        # State loss
        loss_state = F.mse_loss(pred["next_obs"], next_obs_target)
        loss = loss + obs_weight * loss_state

        # Reward loss
        if self.predict_reward and reward_target is not None:
            loss_reward = F.mse_loss(pred["reward"], reward_target)
            loss = loss + reward_weight * loss_reward

        # Cost loss
        if self.predict_cost and cost_target is not None:
            loss_cost = F.mse_loss(pred["cost"], cost_target)
            loss = loss + cost_weight * loss_cost

        return loss

    @torch.no_grad()
    def rollout(
        self,
        policy: nn.Module,
        init_obs: np.ndarray,
        horizon: int,
        device: torch.device,
        deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Roll out a trajectory of length `horizon` under the learned
        dynamics model and current policy.

        This is a purely model-based rollout:
            s_{t+1} = f_φ(s_t, a_t)

        Args:
            policy:       your PolicyNet, expected to take obs -> (mu, std)
            init_obs:     [obs_dim] initial state (numpy, from real env)
            horizon:      number of model steps
            device:       torch device
            deterministic: if True, use mu as action instead of sampling

        Returns:
            A dict of numpy arrays:
                'obs':      [T, obs_dim]
                'act':      [T, act_dim]
                'next_obs': [T, obs_dim]
                'reward':   (if predict_reward) [T]
                'cost':     (if predict_cost) [T]
        """
        obs = _safe_numpy_obs(init_obs)

        obs_buf = []
        act_buf = []
        next_obs_buf = []
        rew_buf: list[float] = []
        cost_buf: list[float] = []

        for t in range(horizon):
            obs_buf.append(obs.copy())

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_t = _safe_tensor(obs_t, name="rollout_obs")

            # ---- Policy forward (no hard crash) ----
            mu, std = policy(obs_t)

            mu = _safe_tensor(mu, name="rollout_mu")
            std = _safe_tensor(std, name="rollout_std")

            # Clamp std to avoid zero or exploding values
            eps = 1e-6
            std = torch.clamp(std, min=eps, max=1e2)

            if deterministic:
                act_t = mu
            else:
                dist = torch.distributions.Normal(mu, std)
                act_t = dist.sample()

            act_t = _safe_tensor(act_t, name="rollout_act")

            pred = self.forward(obs_t, act_t)
            next_obs_t = _safe_tensor(pred["next_obs"], name="rollout_next_obs")

            act_np = act_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            next_obs_np = next_obs_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

            act_buf.append(act_np)
            next_obs_buf.append(next_obs_np)

            if self.predict_reward and "reward" in pred:
                rew = float(pred["reward"].item())
                rew_buf.append(rew)
            if self.predict_cost and "cost" in pred:
                c = float(pred["cost"].item())
                cost_buf.append(c)

            obs = _safe_numpy_obs(next_obs_np)

        result: Dict[str, np.ndarray] = {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "act": np.asarray(act_buf, dtype=np.float32),
            "next_obs": np.asarray(next_obs_buf, dtype=np.float32),
        }
        if self.predict_reward:
            result["reward"] = np.asarray(rew_buf, dtype=np.float32)
        if self.predict_cost:
            result["cost"] = np.asarray(cost_buf, dtype=np.float32)
        return result


# ---------------------------------------------------------------------------
# ModelEnv: light-weight environment-like wrapper
# ---------------------------------------------------------------------------

class ModelEnv:
    """
    A minimal environment-like wrapper around DynamicsModel.

    This is not a full Gym env, just a helper to make model-based
    rollouts easier to integrate into your Dyna-CPPO code.

    Usage:
        menv = ModelEnv(dyn_model, policy, device)
        obs = menv.reset(real_init_obs)
        for t in range(T):
            next_obs, reward, cost = menv.step()
    """

    def __init__(
        self,
        dyn_model: DynamicsModel,
        policy: nn.Module,
        device: torch.device,
        deterministic: bool = False,
    ):
        self.dyn_model = dyn_model
        self.policy = policy
        self.device = device
        self.deterministic = deterministic

        self._obs: Optional[np.ndarray] = None

    def reset(self, init_obs: np.ndarray) -> np.ndarray:
        """
        Reset internal state to a given initial observation.

        Args:
            init_obs: [obs_dim] numpy array from real env

        Returns:
            The internal copy of the initial observation.
        """
        self._obs = _safe_numpy_obs(init_obs)
        return self._obs

    @torch.no_grad()
    def step(self) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """
        Take one model-based step:

            a_t ~ π( s_t )
            s_{t+1}, r_t, c_t = f_φ( s_t, a_t )

        Returns:
            next_obs: [obs_dim]
            reward:  float or None
            cost:    float or None
        """
        assert self._obs is not None, "Call reset(init_obs) before step()."

        obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_t = _safe_tensor(obs_t, name="menv_obs")

        mu, std = self.policy(obs_t)

        mu = _safe_tensor(mu, name="menv_mu")
        std = _safe_tensor(std, name="menv_std")

        # Clamp std for stability
        eps = 1e-6
        std = torch.clamp(std, min=eps, max=1e2)

        if self.deterministic:
            act_t = mu
        else:
            dist = torch.distributions.Normal(mu, std)
            act_t = dist.sample()

        act_t = _safe_tensor(act_t, name="menv_act")

        pred = self.dyn_model(obs_t, act_t)
        next_obs_t = _safe_tensor(pred["next_obs"], name="menv_next_obs")

        next_obs = next_obs_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        next_obs = _safe_numpy_obs(next_obs)

        rew: Optional[float] = None
        cost: Optional[float] = None
        if self.dyn_model.predict_reward and "reward" in pred:
            rew = float(pred["reward"].item())
        if self.dyn_model.predict_cost and "cost" in pred:
            cost = float(pred["cost"].item())

        self._obs = next_obs
        return next_obs, rew, cost
