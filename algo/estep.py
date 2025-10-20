# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:54:38 2025

@author: Ruining
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def _center(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center a vector and return (x - mean, mean)."""
    m = float(x.mean()) if x.size > 0 else 0.0
    return x - m, m


def _qr_2d(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthonormalize two centered vectors via Gram-Schmidt (equivalent to QR in 2D).
    Returns orthonormal basis (u, v) with:
      u = a / ||a||,
      v = (b - (b·u)u) / ||b - (b·u)u||
    If any norm is ~0 (degenerate), we fall back to unit axis to avoid NaN.
    """
    eps = 1e-12
    na = np.linalg.norm(a)
    if na < eps:
        u = np.ones_like(a) / np.sqrt(max(len(a), 1))
    else:
        u = a / na

    b_proj = b - (u * (b @ u))
    nb = np.linalg.norm(b_proj)
    if nb < eps:
        v = u.copy()
    else:
        v = b_proj / nb
    return u, v


def _solve_theta_for_cost(u_c: np.ndarray, u_r: np.ndarray,
                          rhs: float) -> float:
    cos_theta = np.clip(rhs, -1.0, 1.0) # cosθ = clamp(rhs, -1, 1)
    return float(np.arccos(cos_theta))


def estep_optimal_v(
    adv_r: np.ndarray,           # A   : reward advantages (length N)
    adv_c: np.ndarray,           # Ac  : cost   advantages (length N)
    Jc_pi: float,                # J_c(π): current policy's discounted cost estimate
    d_constraint: float,         # d    : cost limit
    gamma: float,                # discount for cost margin scaling
    delta_kl: float,             # δ    : reverse-KL budget in Eq.(3)/(4)
    lower_bound: float = 0.0,    # element-wise lower bound b for v (Theorem 3.7 & App.C)
) -> Dict[str, np.ndarray]:
    """
    Compute the optimal non-parametric ratios v on sampled data (E-step).

    Returns:
        {
          'v': (N,) optimal probability ratios (≥0),
          'mask_clipped': boolean mask of which entries were clipped to the lower bound,
          'delta_used': δ' used after masking (for logs),
          'feasible_cost': float, estimated average cost under v,
        }
    """
    assert adv_r.ndim == adv_c.ndim == 1 and adv_r.shape == adv_c.shape
    N = int(adv_r.shape[0])
    if N == 0:
        return dict(v=np.array([]), mask_clipped=np.array([]),
                    delta_used=0.0, feasible_cost=0.0)

    # ----- Step 0: prepare margins & budgets -----
    
    d_prime = (1.0 - gamma) * (d_constraint - Jc_pi) # d' = (1-γ) (d - J_c(π)) under Eq.(4)
    l2_budget = 2.0 * N * max(delta_kl, 0.0) # ||v-1||_2^2 ≤ 2 N δ on vector v-1 under Eq.(5)

    # current *active* index set (mask==True means still being optimized, not clipped yet)
    active = np.ones(N, dtype=bool)
    v = np.ones(N, dtype=np.float64)   # start from uniform ratio 1

    # mask-clip-recenter-reQR: (Algorithm 1 in Appendix C)
    #   solve on active set in A–Ac plane; 
    #   clip elements below lower_bound; 
    #   mask and re-solve.
    while True:
        idx = np.where(active)[0]
        M = idx.size
        if M == 0: # all constraint are masked
            break

        # ---- 1) Center A and Ac on active subset (subtract means)
        A_act, mean_A = _center(adv_r[idx])
        Ac_act, mean_Ac = _center(adv_c[idx]) # centering step
        u_c, u_r = _qr_2d(Ac_act, A_act) # Orthonormalize centered (Ac, A) -> (Âc, Â)

        # ---- 2) Compute the radius under current L2 budget on active set
        l2_budget_active = l2_budget * (M / N)   # simple proportional split after masking
        R = np.sqrt(max(l2_budget_active, 0.0)) # R = sqrt(l2_budget_active), v' = R (cosθ Âc + sinθ Â) under Sec 3.3, optimal v'

        # ---- 3) Enforce the linear cost constraint on active set:
        norm_Ac = np.linalg.norm(Ac_act) + 1e-12
        rhs = 0.0
        if R > 0.0 and norm_Ac > 0.0:
            rhs = (M * (d_prime - mean_Ac)) / (R * norm_Ac) # solve rhs:= M*(d' - mean_Ac) / (R * ||Ac_act||)
        theta = _solve_theta_for_cost(u_c, u_r, rhs)

        # ---- 4) Construct v'' and then v on active set
        direction = np.cos(theta) * u_c + np.sin(theta) * u_r # unit vector in the 2D subspace
        v_pp = R * direction # v'' direction to feasible region
        v_new = 1.0 + v_pp + mean_A * 0.0  # E[v]=1, thus mean_A is removed

        # ---- 5) Apply element-wise lower bound on *entire* vector
        v[idx] = v_new
        clipped_mask = (v < lower_bound)
        if not np.any(clipped_mask[active]):
            # no active elements violate the lower bound -> finished
            break

        # ---- 6) Record & mask the clipped entries as Appendix C suggests (vm, Am_c)
        v[clipped_mask] = lower_bound
        # remove (mask out) those entries and re-solve on the remaining active indices
        active = active & (~clipped_mask)

        # loop continues with smaller active set, recentering A, Ac, and re-allocating l2 budget

    # Estimated feasible cost under v (sample-average form)
    feasible_cost = (v * adv_c).mean()

    return dict(
        v=v.astype(np.float32, copy=False),
        mask_clipped=(v <= lower_bound + 1e-12),
        delta_used=float(delta_kl),
        feasible_cost=float(feasible_cost),
    )
