# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:54:48 2025

@author: Ruining
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _to_tensor(x, device):
    """Convert array-like to float32 tensor on `device` without grad."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def mstep_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs,                 # [N, obs_dim]  (torch or numpy)
    actions,             # [N, act_dim]  (torch or numpy)
    old_logp,            # [N]           log π_old(a|s) from rollout
    v_target,            # [N]           target ratio v from E-step (stop-grad)
    clip_lower: float = 0.6,     # lower bound for r_θ to control fwd-KL (Appendix C)
    entropy_coef: float = 0.0,   # optional entropy bonus weight
    norm_coef: float = 0.01,     # small penalty on (E[r]-1)^2 to keep mean ratio near 1
    max_grad_norm: Optional[float] = 0.5,  # gradient clipping (None to disable)
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    One M-step gradient update.
    Returns a dict of scalars for logging.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # --- move inputs to device (no grad for data) ---
    obs_t      = _to_tensor(obs, device)
    act_t      = _to_tensor(actions, device)
    old_logp_t = _to_tensor(old_logp, device).view(-1)
    v_t        = _to_tensor(v_target, device).view(-1)

    # --- forward: current log π_θ(a|s) ---
    # Expect your PolicyNet to provide either:
    #  (a) a `log_prob(obs, actions)` method; OR
    #  (b) `dist = policy.distribution(obs)` returning a torch.distributions object.
    if hasattr(policy, "log_prob"):
        new_logp = policy.log_prob(obs_t, act_t).view(-1)
        entropy  = getattr(policy, "entropy", lambda o: None)(obs_t)
    else:
        # generic fallback: assume `policy.forward(obs)` returns a dist
        dist = policy(obs_t)  # should return a distribution with .log_prob() / .entropy()
        new_logp = dist.log_prob(act_t).sum(-1)  # sum over action dims if needed
        # entropy averaged over batch if available
        try:
            entropy = dist.entropy()
            # some distributions return per-dimension entropy; reduce to [N]
            if entropy.ndim > 1:
                entropy = entropy.sum(-1)
        except Exception:
            entropy = None

    # --- importance ratio r_θ = exp(log π_θ - log π_old) ---
    ratio = torch.exp(new_logp - old_logp_t)  # [N]

    # --- lower-bound clipping on ratio (Appendix C) ---
    # Only clip the lower side to enforce a floor, which helps control forward-KL.
    ratio_clipped = torch.clamp_min(ratio, clip_lower)

    # --- tracking loss: make r_θ match the target non-parametric ratio v (stop-grad on v) ---
    track_loss = 0.5 * F.mse_loss(ratio_clipped, v_t)

    # --- encourage normalization E[r] ≈ 1 to maintain E-step constraint E[v]=1 ---
    norm_loss = (ratio.mean() - 1.0).pow(2)

    # --- entropy bonus for exploration/stability ---
    ent_term = -entropy_coef * (entropy.mean() if (entropy is not None) else torch.tensor(0.0, device=device))

    # --- total loss ---
    loss = track_loss + norm_coef * norm_loss + ent_term

    # --- optimize ---
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    # --- stats for logging ---
    with torch.no_grad():
        # approximate reverse-KL: E[ v * log(v / r_θ) ] is costly; report useful proxies
        mean_ratio = ratio.mean().item()
        frac_lb    = (ratio < clip_lower).float().mean().item()
        mse_raw    = F.mse_loss(ratio, v_t).item()
        mse_clip   = (2.0 * track_loss).item()  # because track_loss = 0.5 * MSE

        stats = dict(
            loss=float(loss.item()),
            track_loss=float(track_loss.item()),
            norm_loss=float(norm_coef * norm_loss.item()),
            entropy=float((entropy.mean().item() if entropy is not None else 0.0)),
            mean_ratio=float(mean_ratio),
            frac_ratio_below_lb=float(frac_lb),
            mse_ratio_raw=float(mse_raw),
            mse_ratio_clipped=float(mse_clip),
        )
    return stats
