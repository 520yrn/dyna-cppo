# -*- coding: utf-8 -*-
"""
main.py

Minimal training loop for CPPO (paper reproduction):
- Collect on-policy rollout under current policy (Gymnasium 0.28, Safety-Gymnasium return: obs, reward, cost, terminated, truncated, info)
- Compute dual-branch GAE: reward advantages A_r, cost advantages A_c, and bootstrapped returns for V_r, V_c
- E-step: solve for non-parametric optimal ratios v (q / p_pi) under cost & reverse-KL constraints
- M-step: update parametric policy π_θ to track v with a lower-bound clipped ratio (forward-KL control)
- Critic update: regress V_r to ret_r and V_c to ret_c

Author: Ruining
"""

from __future__ import annotations
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import safety_gymnasium  # environment backend

# ---- your local modules (adjust import paths if your tree is different) ----
from agents.policy import PolicyNet            # must expose: forward(obs)->dist OR log_prob(obs, act)
from agents.value  import ValueNet, CostNet    # simple MLP regressors for V_r and V_c
from algo.advantages import compute_gae_dual   # dual-branch GAE (reward & cost)
from algo.estep      import estep_optimal_v    # solve target ratios v on a batch
from algo.mstep      import mstep_update       # track v with lower-bound clipped ratio
from utils.seed      import set_seed           # reproducibility
from utils.log       import Logger             # simple console/file logger


# -------------------------------
# helper: evaluate critics on obs
# -------------------------------
@torch.no_grad()
def _critic_forward(vr: nn.Module, vc: nn.Module, obs: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run V_r and V_c over a numpy batch of observations and return numpy arrays (on CPU)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    vr_pred = vr(obs_t).squeeze(-1)   # [N]
    vc_pred = vc(obs_t).squeeze(-1)   # [N]
    return vr_pred.cpu().numpy(), vc_pred.cpu().numpy()


# ---------------------------------------
# rollout: collect one on-policy trajectory
# ---------------------------------------
def collect_rollout(
    env,
    policy: nn.Module,
    vr: nn.Module,
    vc: nn.Module,
    horizon: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Collect one on-policy rollout of length T = horizon using the current policy π.
    Assumes Safety-Gymnasium step signature: (obs, reward, cost, terminated, truncated, info).

    Returns a dict of numpy arrays:
      'obs'        : [T, obs_dim] observations
      'act'        : [T, act_dim] actions sampled from π
      'old_logp'   : [T]          log π_old(a|s) at sampling time
      'rew'        : [T]          rewards
      'cost'       : [T]          costs
      'term'       : [T]          terminated flags
      'trunc'      : [T]          truncated flags
      'val_r'      : [T+1]        V_r(s_t) with bootstrap V_r(s_T)
      'val_c'      : [T+1]        V_c(s_t) with bootstrap V_c(s_T)
    """
    # reset episode if needed
    obs, _ = env.reset()
    obs_buf   = []
    act_buf   = []
    logp_buf  = []
    rew_buf   = []
    cost_buf  = []
    term_buf  = []
    trunc_buf = []

    # first value predictions for s_0.. we'll append bootstrap after the loop
    v_r_list = []
    v_c_list = []

    for t in range(horizon):
        # record current obs
        obs_buf.append(obs.copy())

        # critics on current state s_t (we also need V_r/V_c for each step and bootstrap)
        with torch.no_grad():
            vr_t = vr(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).item()
            vc_t = vc(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).item()
        v_r_list.append(vr_t)
        v_c_list.append(vc_t)

        # sample action from current policy
        # policy should return a distribution OR expose log_prob(obs, act)
        with torch.no_grad():
            mu, std = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))  # expect dist
            dist = torch.distributions.Normal(mu, std)
            act_t = dist.sample()                                     # [1, act_dim]
            logp_t = dist.log_prob(act_t).sum(-1)                     # [1]
            act_np = act_t.squeeze(0).cpu().numpy()
            logp_np = logp_t.item()

        # env step: Safety-Gymnasium -> six elements
        next_obs, rew, cost, terminated, truncated, info = env.step(act_np)

        # store step
        act_buf.append(act_np)
        logp_buf.append(logp_np)
        rew_buf.append(float(rew))
        cost_buf.append(float(cost))
        term_buf.append(bool(terminated))
        trunc_buf.append(bool(truncated))

        # move on
        obs = next_obs
        if terminated or truncated:
            # if episode ends early, reset to keep collecting until horizon
            obs, _ = env.reset()

    # bootstrap value at s_T for both heads
    with torch.no_grad():
        vr_T = vr(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).item()
        vc_T = vc(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).item()

    # pack arrays
    obs_arr   = np.asarray(obs_buf,   dtype=np.float32)
    act_arr   = np.asarray(act_buf,   dtype=np.float32)
    logp_arr  = np.asarray(logp_buf,  dtype=np.float32)
    rew_arr   = np.asarray(rew_buf,   dtype=np.float32)
    cost_arr  = np.asarray(cost_buf,  dtype=np.float32)
    term_arr  = np.asarray(term_buf,  dtype=np.bool_)
    trunc_arr = np.asarray(trunc_buf, dtype=np.bool_)
    val_r_arr = np.asarray(v_r_list + [vr_T], dtype=np.float32)  # length T+1
    val_c_arr = np.asarray(v_c_list + [vc_T], dtype=np.float32)  # length T+1

    return dict(
        obs=obs_arr, act=act_arr, old_logp=logp_arr,
        rew=rew_arr, cost=cost_arr, term=term_arr, trunc=trunc_arr,
        val_r=val_r_arr, val_c=val_c_arr,
    )


def train_once(
    env_id: str = "SafetyPointGoal1-v0",
    total_iters: int = 1000,
    horizon: int = 1024,              # rollout length T (on-policy)
    gamma: float = 0.99,
    lam: float = 0.95,
    cost_gamma: float = 0.99,
    cost_lam: float = 0.95,
    d_constraint: float = 25.0,       # cost limit (adjust per task spec if needed)
    delta_kl: float = 0.02,           # reverse-KL budget (E-step)
    ratio_lower_bound: float = 0.6,   # lower bound on r_theta (M-step)
    pi_lr: float = 3e-4,
    v_lr: float  = 3e-4,
    seed: int = 0,
    log_dir: str = "./logs",
):
    """One simple training loop over `total_iters` iterations."""
    # device & seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    logger = Logger(log_dir)

    # build env (no render for headless training)
    env = safety_gymnasium.make(env_id, render_mode=None)

    # infer dimensions
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    # build networks
    policy = PolicyNet(obs_dim=obs_dim, act_dim=act_dim).to(device)
    v_r    = ValueNet(obs_dim=obs_dim).to(device)
    v_c    = CostNet(obs_dim=obs_dim).to(device)

    # optimizers
    pi_opt = optim.Adam(policy.parameters(), lr=pi_lr)
    vr_opt = optim.Adam(v_r.parameters(),    lr=v_lr)
    vc_opt = optim.Adam(v_c.parameters(),    lr=v_lr)

    # training loop
    global_step = 0
    start_time = time.time()
    for it in range(1, total_iters + 1):
        # ---------- (1) collect on-policy rollout ----------
        batch = collect_rollout(env, policy, v_r, v_c, horizon=horizon, device=device)

        # ---------- (2) compute dual GAE (A_r, A_c) & returns ----------
        gae = compute_gae_dual(
            rewards=batch["rew"],
            costs=batch["cost"],
            values_r=batch["val_r"],
            values_c=batch["val_c"],
            terminated=batch["term"],
            truncated=batch["trunc"],
            gamma=gamma, lam=lam,
            cost_gamma=cost_gamma, cost_lam=cost_lam,
            device=device,
        )
        adv_r = gae["adv_r"]   # [T] normalized
        ret_r = gae["ret_r"]   # [T]
        adv_c = gae["adv_c"]   # [T] raw (not normalized)
        ret_c = gae["ret_c"]   # [T]

        # estimate current discounted cost J_c(pi) ~ mean(ret_c) as a proxy
        Jc_pi = float(ret_c.mean().item())

        # ---------- (3) E-step: solve optimal non-parametric ratios v ----------
        est = estep_optimal_v(
            adv_r=adv_r.cpu().numpy(),
            adv_c=adv_c.cpu().numpy(),
            Jc_pi=Jc_pi,
            d_constraint=d_constraint,
            gamma=gamma,
            delta_kl=delta_kl,
            lower_bound=ratio_lower_bound,
        )
        v_target = est["v"]  # numpy [T]

        # ---------- (4) M-step: track v_target with lower-bound clipped ratio ----------
        mstats = mstep_update(
            policy=policy,
            optimizer=pi_opt,
            obs=batch["obs"],
            actions=batch["act"],
            old_logp=batch["old_logp"],
            v_target=v_target,
            clip_lower=ratio_lower_bound,
            entropy_coef=0.0,
            norm_coef=0.01,
            max_grad_norm=0.5,
            device=device,
        )

        # ---------- (5) Critic updates (squared error to returns) ----------
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        ret_r_t = torch.as_tensor(ret_r, dtype=torch.float32, device=device).view(-1, 1)
        ret_c_t = torch.as_tensor(ret_c, dtype=torch.float32, device=device).view(-1, 1)

        vr_pred = v_r(obs_t)
        vc_pred = v_c(obs_t)
        loss_vr = 0.5 * ((vr_pred - ret_r_t) ** 2).mean()
        loss_vc = 0.5 * ((vc_pred - ret_c_t) ** 2).mean()

        vr_opt.zero_grad(set_to_none=True)
        loss_vr.backward()
        torch.nn.utils.clip_grad_norm_(v_r.parameters(), 0.5)
        vr_opt.step()

        vc_opt.zero_grad(set_to_none=True)
        loss_vc.backward()
        torch.nn.utils.clip_grad_norm_(v_c.parameters(), 0.5)
        vc_opt.step()

        # ---------- (6) Logging ----------
        global_step += len(batch["rew"])
        ep_rew = float(np.sum(batch["rew"]))
        ep_cost = float(np.sum(batch["cost"]))
        elapsed = time.time() - start_time
        logger.log(
            episode=it,
            reward=ep_rew,
            cost=ep_cost,
            constraint_violation=max(0.0, ep_cost - d_constraint),
        )
        log_path = os.path.join(log_dir, "train_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[it {it:04d}] step={global_step} | "
                f"EpRew={ep_rew:.1f} EpCost={ep_cost:.1f} | "
                f"Vr={loss_vr.item():.3f} Vc={loss_vc.item():.3f} | "
                f"PiLoss={mstats['loss']:.3f} Track={mstats['track_loss']:.3f} "
                f"MeanRatio={mstats['mean_ratio']:.3f} LB%={mstats['frac_ratio_below_lb']*100:.1f}% | "
                f"time={elapsed/60:.1f}m\n"
                )

    env.close()


if __name__ == "__main__":
    # You can tweak these defaults or wire up argparse later.
    train_once(
        env_id="SafetyPointGoal1-v0",
        total_iters=10000,          # keep small for a smoke test; increase for real training
        horizon=1024,
        gamma=0.99, lam=0.95,
        cost_gamma=0.99, cost_lam=0.95,
        d_constraint=25.0,       # TODO: adjust to the task's actual cost limit if needed
        delta_kl=0.02,
        ratio_lower_bound=0.6,
        pi_lr=3e-4,
        v_lr=3e-4,
        seed=0,
        log_dir="./logs",
    )
