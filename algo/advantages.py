# -*- coding: utf-8 -*-
"""
algo/advantages.py

Author: Ruining
"""
from typing import Dict
import numpy as np
import torch


def _to_torch(x, device):
    """Helper to convert numpy arrays to float32 torch tensors on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def compute_gae_dual(
    rewards: np.ndarray,
    costs: np.ndarray,
    values_r: np.ndarray,
    values_c: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    cost_gamma: float = 0.99,
    cost_lam: float = 0.95,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute GAE for reward (A_r) and cost (A_c), plus bootstrap returns (R_r, R_c).

    Args:
        rewards:    [T] step rewards r_t
        costs:      [T] step costs   c_t
        values_r:   [T+1] reward value predictions V_r(s_t) with a bootstrapped V_r(s_{T})
        values_c:   [T+1] cost   value predictions V_c(s_t) with a bootstrapped V_c(s_{T})
        terminated: [T] boolean flags (True if episode terminated at step t)
        truncated:  [T] boolean flags (True if episode truncated  at step t)
        gamma, lam:         reward discount and GAE lambda
        cost_gamma, cost_lam: cost discount and GAE lambda (often same as reward)
        device:     torch device

    Returns:
        dict with torch tensors on `device`:
            'adv_r': [T] reward advantages
            'ret_r': [T] reward returns (targets for V_r)
            'adv_c': [T] cost   advantages
            'ret_c': [T] cost   returns (targets for V_c)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    r  = _to_torch(rewards, device)
    c  = _to_torch(costs, device)
    Vr = _to_torch(values_r, device)   # length T+1
    Vc = _to_torch(values_c, device)   # length T+1
    term = _to_torch(terminated, device)  # length T
    trunc= _to_torch(truncated,  device)  # length T

    T = r.shape[0]
    assert Vr.shape[0] == T + 1 and Vc.shape[0] == T + 1, "values must be length T+1 for bootstrapping"

    # Mask that stops bootstrapping on either terminal event
    done = torch.clamp(term + trunc, 0.0, 1.0)

    # ------- Reward GAE -------
    adv_r = torch.zeros(T, dtype=torch.float32, device=device)
    gae = torch.zeros((), dtype=torch.float32, device=device)
    for t in reversed(range(T)):
        delta = r[t] + gamma * Vr[t + 1] * (1.0 - done[t]) - Vr[t]
        gae = delta + gamma * lam * (1.0 - done[t]) * gae
        adv_r[t] = gae
    # Target returns for value regression
    ret_r = adv_r + Vr[:-1]

    # ------- Cost GAE (symmetric computation) -------
    adv_c = torch.zeros(T, dtype=torch.float32, device=device)
    gae_c = torch.zeros((), dtype=torch.float32, device=device)
    for t in reversed(range(T)):
        delta_c = c[t] + cost_gamma * Vc[t + 1] * (1.0 - done[t]) - Vc[t]
        gae_c = delta_c + cost_gamma * cost_lam * (1.0 - done[t]) * gae_c
        adv_c[t] = gae_c
    ret_c = adv_c + Vc[:-1]

    # ---- Normalize reward advantages (common in PPO/CPPO); keep cost-adv raw ----
    adv_r = (adv_r - adv_r.mean()) / (adv_r.std(unbiased=False) + 1e-8)

    return dict(adv_r=adv_r, ret_r=ret_r, adv_c=adv_c, ret_c=ret_c)


# Convenience wrapper for numpy-first code paths
def compute_gae_dual_numpy(
    rewards: np.ndarray,
    costs: np.ndarray,
    values_r: np.ndarray,
    values_c: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    cost_gamma: float = 0.99,
    cost_lam: float = 0.95,
) -> Dict[str, np.ndarray]:
    """Same as compute_gae_dual but returning numpy arrays (CPU)."""
    out = compute_gae_dual(
        rewards, costs, values_r, values_c, terminated, truncated,
        gamma, lam, cost_gamma, cost_lam, device=torch.device("cpu"),
    )
    return {k: v.cpu().numpy() for k, v in out.items()}
